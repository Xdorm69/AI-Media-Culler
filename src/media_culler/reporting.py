"""Report model, summary statistics and atomic JSON/CSV writers."""

from __future__ import annotations

import csv
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from media_culler import __version__
from media_culler.exceptions import ReportError
from media_culler.models import GroupRole, ImageRecord, Recommendation
from media_culler.pipeline import RunResult

SCHEMA_VERSION = 3
_READABLE_SCHEMAS = {2, 3}  # v3 only adds fields; v2 reports load with defaults


class TopImage(BaseModel):
    filename: str
    relative_path: str
    final_score: float
    recommendation: Recommendation


class Summary(BaseModel):
    total_files: int
    valid_images: int
    errors: int
    total_size_bytes: int
    recommendations: dict[str, int]
    duplicate_groups: int
    images_in_duplicate_groups: int
    redundant_images: int  # non-representative members of duplicate groups
    mean_score: float | None
    median_score: float | None
    analyzer_failures: dict[str, int]
    scene_counts: dict[str, int] = Field(default_factory=dict)
    closed_eyes_photos: int = 0
    top: list[TopImage]


class Report(BaseModel):
    schema_version: int = SCHEMA_VERSION
    tool_version: str = __version__
    generated_at: str
    duration_seconds: float
    input_directory: str
    skipped_analyzers: dict[str, str]
    cache_hits: int = 0
    cache_misses: int = 0
    summary: Summary
    images: list[ImageRecord]


def build_summary(records: list[ImageRecord], duplicate_groups: int, top_n: int = 10) -> Summary:
    valid = [r for r in records if r.status == "valid"]
    scored = sorted(
        (r for r in valid if r.final_score is not None),
        key=lambda r: (-(r.final_score or 0), r.path),
    )
    counts: dict[str, int] = {}
    for record in valid:
        if record.recommendation is not None:
            counts[record.recommendation.value] = counts.get(record.recommendation.value, 0) + 1

    failures: dict[str, int] = {}
    for record in valid:
        for name in record.analyzer_errors:
            failures[name] = failures.get(name, 0) + 1

    scenes: dict[str, int] = {}
    for record in valid:
        if record.scores and record.scores.scene_type:
            key = record.scores.scene_type.value
            scenes[key] = scenes.get(key, 0) + 1

    values = [r.final_score for r in scored if r.final_score is not None]
    return Summary(
        total_files=len(records),
        valid_images=len(valid),
        errors=len(records) - len(valid),
        total_size_bytes=sum(r.size_bytes for r in records),
        recommendations=dict(sorted(counts.items())),
        duplicate_groups=duplicate_groups,
        images_in_duplicate_groups=sum(1 for r in valid if r.duplicate_group is not None),
        redundant_images=sum(1 for r in valid if r.group_role == GroupRole.SIMILAR),
        mean_score=round(statistics.fmean(values), 2) if values else None,
        median_score=round(statistics.median(values), 2) if values else None,
        analyzer_failures=failures,
        scene_counts=dict(sorted(scenes.items())),
        closed_eyes_photos=sum(
            1 for r in valid if r.scores and r.scores.group and r.scores.group.closed_eyes
        ),
        top=[
            TopImage(
                filename=r.filename,
                relative_path=r.relative_path,
                final_score=r.final_score or 0.0,
                recommendation=r.recommendation or Recommendation.UNSCORED,
            )
            for r in scored[:top_n]
        ],
    )


def build_report(result: RunResult, top_n: int = 10) -> Report:
    return Report(
        generated_at=result.started_at,
        duration_seconds=result.duration_seconds,
        input_directory=str(result.input_directory),
        skipped_analyzers=result.skipped_analyzers,
        cache_hits=result.cache_hits,
        cache_misses=result.cache_misses,
        summary=build_summary(result.records, result.duplicate_groups, top_n),
        images=result.records,
    )


# ---------------------------------------------------------------------- writing


def _atomic_write(path: Path, writer: Any, *, newline: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as handle:
            writer(handle)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_json(report: Report, path: Path) -> Path:
    try:
        _atomic_write(path, lambda h: h.write(report.model_dump_json(indent=2)))
    except OSError as error:
        raise ReportError(f"Cannot write {path}: {error}") from error
    return path


CSV_COLUMNS = [
    "rank",
    "filename",
    "relative_path",
    "recommendation",
    "scene",
    "final_score",
    "technical",
    "ai",
    "subject",
    "composition",
    "faces",
    "subjects",
    "closed_eyes",
    "duplicate_group",
    "group_role",
    "width",
    "height",
    "size_bytes",
    "status",
    "error",
    "analyzer_errors",
]


def _csv_row(r: ImageRecord) -> dict[str, Any]:
    s = r.scores
    return {
        "rank": r.collection_rank,
        "filename": r.filename,
        "relative_path": r.relative_path,
        "recommendation": r.recommendation.value if r.recommendation else "",
        "scene": s.scene_type.value if s and s.scene_type else "",
        "final_score": s.final if s else None,
        "technical": s.technical if s else None,
        "ai": s.ai if s else None,
        "subject": s.subject if s else None,
        "composition": s.composition if s else None,
        "faces": r.faces.face_count if r.faces else None,
        "subjects": len(s.group.significant) if s and s.group else None,
        "closed_eyes": len(s.group.closed_eyes) if s and s.group else None,
        "duplicate_group": r.duplicate_group,
        "group_role": r.group_role.value if r.group_role else "",
        "width": r.width,
        "height": r.height,
        "size_bytes": r.size_bytes,
        "status": r.status,
        "error": r.error or "",
        "analyzer_errors": json.dumps(r.analyzer_errors) if r.analyzer_errors else "",
    }


def write_csv(records: list[ImageRecord], path: Path) -> Path:
    def writer(handle: Any) -> None:
        out = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        out.writeheader()
        for record in sorted(records, key=lambda r: (r.collection_rank or 10**9, r.path)):
            out.writerow({k: ("" if v is None else v) for k, v in _csv_row(record).items()})

    try:
        _atomic_write(path, writer, newline="")
    except OSError as error:
        raise ReportError(f"Cannot write {path}: {error}") from error
    return path


def write_reports(report: Report, directory: Path, formats: set[str]) -> dict[str, Path]:
    written: dict[str, Path] = {}
    if "json" in formats:
        written["json"] = write_json(report, directory / "inventory.json")
    if "csv" in formats:
        written["csv"] = write_csv(report.images, directory / "inventory.csv")
    return written


def load_report(path: Path) -> Report:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReportError(f"Report not found: {path}. Run `culler run` first.") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"Cannot read report {path}: {error}") from error

    if isinstance(data, list):
        raise ReportError(
            f"{path} is a v0.1 report (bare list). "
            f"Re-run `culler run` to produce a v{SCHEMA_VERSION} report."
        )
    if data.get("schema_version") not in _READABLE_SCHEMAS:
        raise ReportError(
            f"Unsupported report schema {data.get('schema_version')!r} (expected {SCHEMA_VERSION})."
        )
    try:
        return Report.model_validate(data)
    except ValidationError as error:
        raise ReportError(
            f"Report {path} is malformed: {error.error_count()} problem(s)."
        ) from error
