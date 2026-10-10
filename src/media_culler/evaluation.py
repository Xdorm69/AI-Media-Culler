"""Score the tool's decisions against your own labels, so tuning is measured, not guessed.

You label a few hundred photos in a CSV; ``culler eval REPORT LABELS`` compares them with
the report. Every column except ``path`` is optional, and blank cells are ignored, so you
can label only what you care about:

=============  =====================================================================
``path``       the report's ``relative_path`` (or a file name, if unambiguous)
``verdict``    ``keep`` or ``reject``: would you keep this photo?
``scene``      ``people`` / ``landscape`` / ``general``
``closed_eyes``  ``1``/``0`` (or yes/no): is a subject's eye closed?
``group``      any id: frames you consider one burst / near-duplicate set
``best``       ``1`` on the frame you would keep from that ``group``
=============  =====================================================================
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from media_culler.exceptions import CullerError
from media_culler.models import GroupRole, ImageRecord, Recommendation
from media_culler.reporting import Report

LABEL_COLUMNS = ("path", "verdict", "scene", "closed_eyes", "group", "best")
_TRUE = {"1", "true", "yes", "y", "closed"}
_FALSE = {"0", "false", "no", "n", "open"}
_VERDICTS = {"keep", "reject"}
_SCENES = {"people", "landscape", "general"}


class LabelError(CullerError):
    """The labels file cannot be used."""


@dataclass
class Label:
    path: str
    verdict: str | None = None
    scene: str | None = None
    closed_eyes: bool | None = None
    group: str | None = None
    best: bool = False


@dataclass
class VerdictStats:
    labelled: int = 0
    # human verdict -> tool decision ("keep" = select/usable, "review", "reject")
    matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    false_reject_rate: float | None = None  # kept by you, rejected by the tool: the costly error
    keep_recall: float | None = None  # of your keepers, share the tool also keeps
    keep_precision: float | None = None  # of the tool's keeps, share you also keep
    reject_recall: float | None = None  # of your rejects, share the tool flags
    review_rate: float | None = None  # share the tool leaves to you
    undecided: int = 0  # tool had no score


@dataclass
class DetectionStats:
    labelled: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float | None = None
    recall: float | None = None


@dataclass
class SceneStats:
    labelled: int = 0
    accuracy: float | None = None
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class BestShotStats:
    groups: int = 0
    best_kept: int = 0  # your pick survived (was not set aside as an alternate)
    picked_exactly: int = 0  # ...and it is the ONLY frame of the group the tool kept
    best_kept_rate: float | None = None
    picked_exactly_rate: float | None = None
    misses: list[str] = field(default_factory=list)  # groups where your best was dropped


@dataclass
class EvalResult:
    labelled: int
    matched: int
    unmatched: list[str]
    verdict: VerdictStats
    scene: SceneStats
    closed_eyes: DetectionStats
    best_shot: BestShotStats

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _flag(value: str, column: str, row: int) -> bool | None:
    text = value.strip().lower()
    if not text:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise LabelError(f"labels row {row}: {column} must be 1/0 or yes/no, got {value!r}")


def load_labels(path: Path) -> list[Label]:
    if not path.exists():
        raise LabelError(f"Labels file not found: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "path" not in [f.strip() for f in reader.fieldnames]:
                raise LabelError("labels file needs a header row with a 'path' column")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise LabelError(f"Cannot read labels file: {error}") from error

    labels: list[Label] = []
    seen: set[str] = set()
    for number, raw in enumerate(rows, start=2):
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        name = row.get("path", "")
        if not name:
            continue
        if name in seen:
            raise LabelError(f"labels row {number}: {name!r} is listed twice")
        seen.add(name)
        verdict = row.get("verdict", "").lower() or None
        if verdict is not None and verdict not in _VERDICTS:
            raise LabelError(
                f"labels row {number}: verdict must be keep or reject, got {verdict!r}"
            )
        scene = row.get("scene", "").lower() or None
        if scene is not None and scene not in _SCENES:
            raise LabelError(f"labels row {number}: scene must be one of {sorted(_SCENES)}")
        labels.append(
            Label(
                path=name,
                verdict=verdict,
                scene=scene,
                closed_eyes=_flag(row.get("closed_eyes", ""), "closed_eyes", number),
                group=row.get("group") or None,
                best=bool(_flag(row.get("best", ""), "best", number)),
            )
        )
    return labels


def write_template(report: Report, path: Path) -> int:
    """Labels file listing every valid image, ready to fill in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in report.images if r.status == "valid"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_COLUMNS)
        for record in sorted(rows, key=lambda r: r.relative_path):
            writer.writerow([record.relative_path, "", "", "", "", ""])
    return len(rows)


def _index(
    records: list[ImageRecord],
) -> tuple[dict[str, ImageRecord], dict[str, list[ImageRecord]]]:
    by_path = {r.relative_path: r for r in records}
    by_name: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_name[record.filename].append(record)
    return by_path, by_name


def _tool_decision(record: ImageRecord) -> str | None:
    if record.recommendation in (Recommendation.SELECT, Recommendation.USABLE):
        return "keep"
    if record.recommendation is Recommendation.REVIEW:
        return "review"
    if record.recommendation is Recommendation.REJECT:
        return "reject"
    return None


def _kept_as_best(record: ImageRecord) -> bool:
    return record.group_role is not GroupRole.SIMILAR


def evaluate(report: Report, labels: list[Label]) -> EvalResult:
    valid = [r for r in report.images if r.status == "valid"]
    by_path, by_name = _index(valid)
    matched: list[tuple[Label, ImageRecord]] = []
    unmatched: list[str] = []
    for label in labels:
        record = by_path.get(label.path)
        if record is None:
            candidates = by_name.get(Path(label.path).name, [])
            record = candidates[0] if len(candidates) == 1 else None
        if record is None:
            unmatched.append(label.path)
        else:
            matched.append((label, record))

    return EvalResult(
        labelled=len(labels),
        matched=len(matched),
        unmatched=unmatched,
        verdict=_verdicts(matched),
        scene=_scenes(matched),
        closed_eyes=_closed_eyes(matched),
        best_shot=_best_shots(matched),
    )


def _verdicts(matched: list[tuple[Label, ImageRecord]]) -> VerdictStats:
    stats = VerdictStats()
    matrix: dict[str, Counter[str]] = {"keep": Counter(), "reject": Counter()}
    for label, record in matched:
        if label.verdict is None:
            continue
        decision = _tool_decision(record)
        if decision is None:
            stats.undecided += 1
            continue
        stats.labelled += 1
        matrix[label.verdict][decision] += 1
    stats.matrix = {k: dict(v) for k, v in matrix.items()}
    keep_row, reject_row = matrix["keep"], matrix["reject"]
    keepers, rejects = sum(keep_row.values()), sum(reject_row.values())
    tool_keeps = keep_row["keep"] + reject_row["keep"]
    stats.false_reject_rate = _ratio(keep_row["reject"], keepers)
    stats.keep_recall = _ratio(keep_row["keep"], keepers)
    stats.keep_precision = _ratio(keep_row["keep"], tool_keeps)
    stats.reject_recall = _ratio(reject_row["reject"], rejects)
    stats.review_rate = _ratio(keep_row["review"] + reject_row["review"], keepers + rejects)
    return stats


def _scenes(matched: list[tuple[Label, ImageRecord]]) -> SceneStats:
    stats = SceneStats()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0
    for label, record in matched:
        if label.scene is None or record.scores is None or record.scores.scene_type is None:
            continue
        predicted = record.scores.scene_type.value
        stats.labelled += 1
        confusion[label.scene][predicted] += 1
        correct += predicted == label.scene
    stats.accuracy = _ratio(correct, stats.labelled)
    stats.confusion = {k: dict(v) for k, v in confusion.items()}
    return stats


def _closed_eyes(matched: list[tuple[Label, ImageRecord]]) -> DetectionStats:
    stats = DetectionStats()
    for label, record in matched:
        if label.closed_eyes is None or record.scores is None:
            continue
        flagged = bool(record.scores.group and record.scores.group.closed_eyes)
        stats.labelled += 1
        if label.closed_eyes and flagged:
            stats.tp += 1
        elif label.closed_eyes:
            stats.fn += 1
        elif flagged:
            stats.fp += 1
        else:
            stats.tn += 1
    stats.precision = _ratio(stats.tp, stats.tp + stats.fp)
    stats.recall = _ratio(stats.tp, stats.tp + stats.fn)
    return stats


def _best_shots(matched: list[tuple[Label, ImageRecord]]) -> BestShotStats:
    stats = BestShotStats()
    groups: dict[str, list[tuple[Label, ImageRecord]]] = defaultdict(list)
    for label, record in matched:
        if label.group:
            groups[label.group].append((label, record))
    for name, members in sorted(groups.items()):
        bests = [record for label, record in members if label.best]
        if len(bests) != 1 or len(members) < 2:
            continue  # needs exactly one human pick among at least two frames
        stats.groups += 1
        if _kept_as_best(bests[0]):
            stats.best_kept += 1
            kept = [r for _, r in members if _kept_as_best(r)]
            if len(kept) == 1:
                stats.picked_exactly += 1
        else:
            stats.misses.append(name)
    stats.best_kept_rate = _ratio(stats.best_kept, stats.groups)
    stats.picked_exactly_rate = _ratio(stats.picked_exactly, stats.groups)
    return stats
