"""Sort analysed photos into per-recommendation folders (copy / move / symlink).

Safe by construction: plans are pure, nothing is overwritten, copies are atomic, and
sources that vanished or changed since the report was written are skipped, not guessed at.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from media_culler.models import GroupRole, ImageRecord
from media_culler.reporting import Report

log = logging.getLogger(__name__)

DUPLICATES_BUCKET = "similar_alternates"


class Mode(str, Enum):
    COPY = "copy"
    MOVE = "move"
    SYMLINK = "symlink"


@dataclass(frozen=True)
class Operation:
    source: Path
    destination: Path
    bucket: str
    size_bytes: int


@dataclass
class OrganizeResult:
    done: list[Operation] = field(default_factory=list)
    skipped: list[tuple[Operation, str]] = field(default_factory=list)
    failed: list[tuple[Operation, str]] = field(default_factory=list)


def bucket_for(record: ImageRecord, separate_duplicates: bool) -> str | None:
    if record.status != "valid" or record.recommendation is None:
        return None
    if separate_duplicates and record.group_role == GroupRole.SIMILAR:
        return DUPLICATES_BUCKET
    return record.recommendation.value


def plan_operations(
    report: Report,
    destination: Path,
    *,
    separate_duplicates: bool = True,
    only: set[str] | None = None,
) -> list[Operation]:
    """Deterministic plan. Filename collisions get a short content-hash suffix."""
    claimed: set[Path] = set()
    ops: list[Operation] = []
    for record in sorted(report.images, key=lambda r: r.path):
        bucket = bucket_for(record, separate_duplicates)
        if bucket is None or (only is not None and bucket not in only):
            continue
        target = destination / bucket / record.filename
        if target in claimed:
            suffix = (record.sha256 or "dup")[:8]
            target = target.with_name(f"{target.stem}_{suffix}{target.suffix}")
        counter = 1
        while target in claimed:  # pathological: identical name AND hash prefix
            target = target.with_name(f"{target.stem}_{counter}{target.suffix}")
            counter += 1
        claimed.add(target)
        ops.append(Operation(Path(record.path), target, bucket, record.size_bytes))
    return ops


def _execute_one(op: Operation, mode: Mode) -> str | None:
    """Returns a skip reason, or None if the operation was performed."""
    if not op.source.exists():
        return "source missing"
    if mode is not Mode.SYMLINK and op.source.stat().st_size != op.size_bytes:
        return "source changed since analysis"
    if op.destination.exists() or op.destination.is_symlink():
        return "destination exists"

    op.destination.parent.mkdir(parents=True, exist_ok=True)
    if mode is Mode.SYMLINK:
        os.symlink(op.source, op.destination)
    elif mode is Mode.COPY:
        partial = op.destination.with_name(op.destination.name + ".part")
        try:
            shutil.copy2(op.source, partial)
            os.replace(partial, op.destination)
        finally:
            partial.unlink(missing_ok=True)
    else:
        shutil.move(str(op.source), str(op.destination))
    return None


def execute(ops: list[Operation], mode: Mode, destination: Path | None = None) -> OrganizeResult:
    result = OrganizeResult()
    for op in ops:
        try:
            reason = _execute_one(op, mode)
        except OSError as error:
            log.warning("Failed %s -> %s: %s", op.source, op.destination, error)
            result.failed.append((op, str(error)))
            continue
        if reason:
            result.skipped.append((op, reason))
        else:
            result.done.append(op)

    if destination is not None and result.done:
        manifest = destination / ".culler_manifest.json"
        entries = [
            {"source": str(o.source), "destination": str(o.destination), "mode": mode.value}
            for o in result.done
        ]
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return result
