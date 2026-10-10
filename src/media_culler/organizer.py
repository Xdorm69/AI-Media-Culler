"""Sort analysed photos into per-recommendation folders (copy / move / symlink), and undo it.

Safe by construction:

* plans are pure (the only filesystem reads are directory listings to find sidecars);
* nothing is ever overwritten;
* copies are written to a ``.part`` file, checked against the SHA-256 recorded at analysis
  time, and only then renamed into place; a cross-device move deletes its source only after
  that check passes;
* sources that vanished or changed since the report was written are skipped, not guessed at;
* every change is journalled to ``<dest>/.culler/undo/<run>.jsonl`` so ``culler undo`` can
  reverse it, and undo refuses to delete a copy it cannot prove is the one it made.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from media_culler.imaging import sha256_file
from media_culler.models import GroupRole, ImageRecord
from media_culler.reporting import Report

log = logging.getLogger(__name__)

DUPLICATES_BUCKET = "similar_alternates"
JOURNAL_DIR = Path(".culler") / "undo"
UNDONE_SUFFIX = ".undone"

# Files that belong to a photo and must travel with it. Matched case-insensitively as
# ``<stem><suffix>`` (Lightroom, Apple, RawTherapee...) or ``<full name><suffix>``
# (darktable: ``IMG_1.CR2.xmp``).
SIDECAR_SUFFIXES = (".xmp", ".aae", ".pp3", ".dop", ".on1", ".thm")
# An iPhone Live Photo is a still plus a same-named .mov.
_LIVE_PHOTO_STILLS = frozenset({".heic", ".heif", ".jpg", ".jpeg"})
_LIVE_PHOTO_VIDEO = ".mov"


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
    sha256: str | None = None  # content hash recorded at analysis; None for sidecars
    kind: str = "photo"  # or "sidecar"


@dataclass
class OrganizeResult:
    done: list[Operation] = field(default_factory=list)
    skipped: list[tuple[Operation, str]] = field(default_factory=list)
    failed: list[tuple[Operation, str]] = field(default_factory=list)
    journal: Path | None = None


def bucket_for(record: ImageRecord, separate_duplicates: bool) -> str | None:
    if record.status != "valid" or record.recommendation is None:
        return None
    if separate_duplicates and record.group_role == GroupRole.SIMILAR:
        return DUPLICATES_BUCKET
    return record.recommendation.value


def _group_folder(record: ImageRecord, by_path: dict[str, ImageRecord]) -> str | None:
    """``burst_0007`` etc. for a grouped photo; a RAW's JPEG companion follows its RAW."""
    owner = by_path.get(record.paired_with or "", record)
    if owner.duplicate_group is None or owner.group_kind is None:
        return None
    return f"{owner.group_kind.value}_{owner.duplicate_group:04d}"


class _Listings:
    """Case-insensitive directory listings, read once each."""

    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, Path]] = {}

    def names(self, directory: Path) -> dict[str, Path]:
        if directory not in self._cache:
            try:
                self._cache[directory] = {
                    e.name.lower(): Path(e.path) for e in os.scandir(directory)
                }
            except OSError:
                self._cache[directory] = {}
        return self._cache[directory]


def find_sidecars(photo: Path, listings: _Listings | None = None) -> list[Path]:
    """Sidecar files sitting next to ``photo`` (existing files only, sorted)."""
    names = (listings or _Listings()).names(photo.parent)
    wanted = [photo.stem.lower() + suffix for suffix in SIDECAR_SUFFIXES]
    wanted += [photo.name.lower() + suffix for suffix in SIDECAR_SUFFIXES]
    if photo.suffix.lower() in _LIVE_PHOTO_STILLS:
        wanted.append(photo.stem.lower() + _LIVE_PHOTO_VIDEO)
    found = {names[w] for w in wanted if w in names and names[w] != photo}
    return sorted(found)


def _sidecar_target(photo: Path, target: Path, sidecar: Path) -> Path:
    """Name the sidecar after wherever its photo ended up (collision suffixes included)."""
    if sidecar.name.lower().startswith(photo.name.lower()):  # IMG_1.CR2.xmp
        return target.with_name(target.name + sidecar.name[len(photo.name) :])
    return target.with_name(target.stem + sidecar.suffix)  # IMG_1.xmp


def plan_operations(
    report: Report,
    destination: Path,
    *,
    separate_duplicates: bool = True,
    only: set[str] | None = None,
    include_sidecars: bool = True,
    group_folders: bool = False,
) -> list[Operation]:
    """Deterministic plan. Filename collisions get a short content-hash suffix.

    ``include_sidecars`` brings along XMP/AAE/... files and Live Photo videos.
    ``group_folders`` puts members of a burst/duplicate group in ``<bucket>/<kind>_<id>/``.
    """
    claimed: set[Path] = set()
    claimed_sources: set[Path] = set()
    listings = _Listings()
    by_path = {r.relative_path: r for r in report.images}
    ops: list[Operation] = []
    for record in sorted(report.images, key=lambda r: r.path):
        bucket = bucket_for(record, separate_duplicates)
        if bucket is None or (only is not None and bucket not in only):
            continue
        folder = destination / bucket
        if group_folders and (name := _group_folder(record, by_path)):
            folder = folder / name
        target = folder / record.filename
        if target in claimed:
            suffix = (record.sha256 or "dup")[:8]
            target = target.with_name(f"{target.stem}_{suffix}{target.suffix}")
        counter = 1
        while target in claimed:  # pathological: identical name AND hash prefix
            target = target.with_name(f"{target.stem}_{counter}{target.suffix}")
            counter += 1
        claimed.add(target)
        source = Path(record.path)
        claimed_sources.add(source)
        ops.append(Operation(source, target, bucket, record.size_bytes, record.sha256))

        if not include_sidecars:
            continue
        for sidecar in find_sidecars(source, listings):
            if sidecar in claimed_sources:  # e.g. one IMG_1.xmp shared by a RAW and its JPEG
                continue
            side_target = _sidecar_target(source, target, sidecar)
            if side_target in claimed:
                continue
            try:
                size = sidecar.stat().st_size
            except OSError:
                continue
            claimed.add(side_target)
            claimed_sources.add(sidecar)
            ops.append(Operation(sidecar, side_target, bucket, size, None, "sidecar"))
    return ops


# ---------------------------------------------------------------------------- execute


class _Journal:
    """Append-only JSON Lines record of what a run did, flushed line by line."""

    def __init__(self, destination: Path, mode: Mode) -> None:
        directory = destination / JOURNAL_DIR
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = directory / f"{stamp}.jsonl"
        counter = 1
        while self.path.exists() or self.path.with_name(self.path.name + UNDONE_SUFFIX).exists():
            self.path = directory / f"{stamp}_{counter}.jsonl"
            counter += 1
        self._handle = self.path.open("a", encoding="utf-8")
        self.write({"event": "run", "version": 1, "mode": mode.value, "created_at": stamp})

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


def _verified_copy(source: Path, destination: Path, expected: str | None) -> str:
    """Copy via a temp file, check it against ``expected``, then rename. Returns the hash."""
    if expected is None:  # sidecars: no analysis-time hash, so hash the source first
        expected = sha256_file(source)
    partial = destination.with_name(destination.name + ".part")
    try:
        shutil.copy2(source, partial)
        actual = sha256_file(partial)
        if actual != expected:
            raise OSError(
                "copy does not match the analysed file (it changed, or the copy is corrupt)"
            )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return expected


def _execute_one(op: Operation, mode: Mode, journal: _Journal | None) -> str | None:
    """Returns a skip reason, or None if the operation was performed."""
    if not op.source.exists():
        return "source missing"
    if mode is not Mode.SYMLINK and op.source.stat().st_size != op.size_bytes:
        return "source changed since analysis"
    if op.destination.exists() or op.destination.is_symlink():
        return "destination exists"

    op.destination.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "source": str(op.source),
        "destination": str(op.destination),
        "mode": mode.value,
        "kind": op.kind,
        "size": op.size_bytes,
    }
    if journal:
        journal.write({"event": "begin", **entry})  # before acting: a crash must be undoable

    digest: str | None = None
    if mode is Mode.SYMLINK:
        os.symlink(op.source, op.destination)
    elif mode is Mode.COPY:
        digest = _verified_copy(op.source, op.destination, op.sha256)
    else:
        try:
            os.rename(op.source, op.destination)  # same filesystem: atomic, no data moves
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            digest = _verified_copy(op.source, op.destination, op.sha256)
            op.source.unlink()  # only reached once the copy has been verified
    if journal:
        journal.write({"event": "done", **entry, "sha256": digest})
    return None


def execute(ops: list[Operation], mode: Mode, destination: Path | None = None) -> OrganizeResult:
    """Run a plan. With ``destination`` set, journal it so ``undo_run`` can reverse it."""
    result = OrganizeResult()
    journal = _Journal(destination, mode) if destination is not None and ops else None
    try:
        for op in ops:
            try:
                reason = _execute_one(op, mode, journal)
            except OSError as error:
                log.warning("Failed %s -> %s: %s", op.source, op.destination, error)
                result.failed.append((op, str(error)))
                continue
            if reason:
                result.skipped.append((op, reason))
            else:
                result.done.append(op)
    finally:
        if journal:
            journal.close()
    if journal:
        if result.done:
            result.journal = journal.path
        else:
            journal.path.unlink(missing_ok=True)  # nothing happened: nothing to undo
    return result


# ------------------------------------------------------------------------------ undo


@dataclass
class UndoResult:
    reverted: list[Path] = field(default_factory=list)  # destinations undone
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    journal: Path | None = None


def latest_journal(destination: Path) -> Path | None:
    directory = destination / JOURNAL_DIR
    candidates = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    return candidates[-1] if candidates else None


def _read_journal(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Mode and the performed entries. A ``begin`` with no ``done`` only counts for moves
    (a crash after the rename leaves the file moved but unconfirmed)."""
    mode = Mode.COPY.value
    begun: dict[str, dict[str, Any]] = {}
    done: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line from a crash
        event = record.get("event")
        if event == "run":
            mode = record.get("mode", mode)
        elif event == "begin":
            begun[record["destination"]] = record
        elif event == "done":
            done[record["destination"]] = record
    entries = list(done.values())
    entries += [e for k, e in begun.items() if k not in done and e.get("mode") == "move"]
    return mode, entries


def _undo_one(entry: dict[str, Any], apply: bool) -> str | None:
    """Returns a skip reason, or None if (or, in a dry run, if it would be) reverted."""
    source, destination = Path(entry["source"]), Path(entry["destination"])
    mode = entry.get("mode")

    if mode == Mode.SYMLINK.value:
        if not destination.is_symlink() or Path(os.readlink(destination)) != source:
            return "not the link this run created"
        if apply:
            destination.unlink()
        return None

    if not destination.exists():
        return "already gone"
    if mode == Mode.COPY.value:
        if not source.exists():
            return "original is missing: keeping the copy as the only one"
        recorded = entry.get("sha256")
        if recorded is None or sha256_file(destination) != recorded:
            return "copy was modified since it was made"
        if apply:
            destination.unlink()
        return None

    # move
    if source.exists() or source.is_symlink():
        return "original location is occupied"
    if destination.stat().st_size != entry.get("size"):
        return "file was modified since it was moved"
    if entry.get("sha256") and sha256_file(destination) != entry["sha256"]:
        return "file was modified since it was moved"
    if apply:
        source.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(destination, source)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.move(str(destination), str(source))
    return None


def _prune_empty_dirs(destination: Path, touched: list[Path]) -> None:
    for directory in sorted({p.parent for p in touched}, key=lambda d: len(d.parts), reverse=True):
        current = directory
        while current != destination and destination in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def undo_run(journal: Path, destination: Path, *, apply: bool = False) -> UndoResult:
    """Reverse one organize run. Dry run unless ``apply``. Never deletes what it cannot prove."""
    _mode, entries = _read_journal(journal)
    result = UndoResult()
    for entry in reversed(entries):
        target = Path(entry["destination"])
        try:
            reason = _undo_one(entry, apply)
        except OSError as error:
            reason = f"error: {error}"
        if reason in (None,):
            result.reverted.append(target)
        else:
            result.skipped.append((target, reason))
    if apply:
        _prune_empty_dirs(destination, result.reverted)
        if not any(r for _, r in result.skipped if r.startswith("error")):
            marked = journal.with_name(journal.name + UNDONE_SUFFIX)
            journal.rename(marked)
            result.journal = marked
    return result
