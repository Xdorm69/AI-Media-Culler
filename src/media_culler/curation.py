"""Collection-level decisions: ranking, recommendations and best-shot selection."""

from __future__ import annotations

import statistics

from media_culler.config import SelectionConfig, ThresholdsConfig
from media_culler.models import GroupRole, ImageRecord, Recommendation, SceneType


def _scored(records: list[ImageRecord]) -> list[ImageRecord]:
    return [r for r in records if r.status == "valid" and r.final_score is not None]


def _sort_key(record: ImageRecord) -> tuple[float, str]:
    return (-(record.final_score or 0.0), record.path)  # score desc, path asc (deterministic)


def rank_collection(records: list[ImageRecord]) -> None:
    ranked = sorted(_scored(records), key=_sort_key)
    total = len(ranked)
    for index, record in enumerate(ranked, start=1):
        record.collection_rank = index
        record.collection_size = total
        record.collection_percentile = round((total - index) / total * 100, 2)


def _soft_focus_landscape(record: ImageRecord, thresholds: ThresholdsConfig) -> bool:
    """Landscape whose (tiled) sharpness sub-score is below the gate.

    Landscape-only on purpose: a portrait with creamy bokeh has low *global* sharpness
    while its face is crisp, so the same rule would wrongly demote it.
    """
    scores = record.scores
    if scores is None or scores.scene_type is not SceneType.LANDSCAPE:
        return False
    technical = scores.details.get("technical")
    sharpness = technical.get("sharpness") if isinstance(technical, dict) else None
    return sharpness is not None and sharpness < thresholds.soft_focus_sharpness


def _closed_eyes(record: ImageRecord) -> int:
    group = record.scores.group if record.scores else None
    return len(group.closed_eyes) if group else 0


def _significant_faces(record: ImageRecord) -> int:
    group = record.scores.group if record.scores else None
    return len(group.significant) if group else 0


def classify(record: ImageRecord, thresholds: ThresholdsConfig) -> tuple[Recommendation, str]:
    final = record.final_score
    technical = record.scores.technical if record.scores else None
    if final is None or technical is None:
        return Recommendation.UNSCORED, "Analysis incomplete; no score available"
    if technical < thresholds.technical_floor:
        return Recommendation.REJECT, "Very low technical quality"
    if _soft_focus_landscape(record, thresholds) and final >= thresholds.review:
        return Recommendation.REVIEW, "Landscape is soft/out of focus; check before keeping"
    closed = _closed_eyes(record)
    if thresholds.closed_eyes_gate and closed and final >= thresholds.review:
        total = _significant_faces(record)
        return Recommendation.REVIEW, f"Closed eyes on {closed} of {total} faces; check first"
    if final >= thresholds.select:
        return Recommendation.SELECT, "High overall photographic score"
    if final >= thresholds.usable:
        return Recommendation.USABLE, "Good overall photographic score"
    if final >= thresholds.review:
        return Recommendation.REVIEW, "Borderline score requires review"
    return Recommendation.REJECT, "Low overall photographic score"


def apply_recommendations(records: list[ImageRecord], thresholds: ThresholdsConfig) -> None:
    for record in records:
        if record.status == "valid":
            record.recommendation, record.recommendation_reason = classify(record, thresholds)


def _selection_value(record: ImageRecord) -> float:
    """Selection score, falling back to the final score for records that predate it."""
    scores = record.scores
    if scores is None:
        return -1.0
    if scores.selection is not None:
        return scores.selection
    return scores.final if scores.final is not None else -1.0


def select_best_shots(records: list[ImageRecord], config: SelectionConfig | None = None) -> None:
    """Pick one representative per similar-image group.

    Frames are compared on their *selection* score (no AI component). Within ``margin`` points
    of the best, the frame with the fewest problems wins, where a problem is a subject with
    closed eyes or a subject that went missing (fewer faces than the group's typical frame).
    Outside the margin, quality wins: nobody should keep a blurry frame just for open eyes.
    """
    config = config or SelectionConfig()
    groups: dict[int, list[ImageRecord]] = {}
    for record in records:
        if record.status != "valid":
            continue
        if record.duplicate_group is None:
            record.group_role = GroupRole.UNIQUE
            record.is_group_representative = True
        else:
            groups.setdefault(record.duplicate_group, []).append(record)

    for members in groups.values():
        _select_in_group(members, config)


def _select_in_group(members: list[ImageRecord], config: SelectionConfig) -> None:
    scored = [r for r in members if _selection_value(r) >= 0]
    pool = scored or members

    # The median (not max) face count, so one false detection can't make every other frame
    # look like it lost a person.
    typical = int(statistics.median_high([_significant_faces(r) for r in pool]))

    def problems(record: ImageRecord) -> int:
        return _closed_eyes(record) + max(0, typical - _significant_faces(record))

    top = max(_selection_value(r) for r in pool)
    contenders = [r for r in pool if _selection_value(r) >= top - config.margin]
    best = min(contenders, key=lambda r: (problems(r), -_selection_value(r), r.path))
    strongest = min(pool, key=lambda r: (-_selection_value(r), r.path))

    for record in members:
        record.best_shot_score = best.final_score
        if record is best:
            record.group_role = GroupRole.REPRESENTATIVE
            record.is_group_representative = True
            if best is not strongest and problems(best) < problems(strongest):
                record.selection_reason = (
                    "Chosen over a slightly higher-scoring frame because it has fewer "
                    f"problems ({problems(best)} vs {problems(strongest)}): "
                    "eyes open and everyone present"
                )
            else:
                record.selection_reason = "Highest selection score in similar-image group"
        else:
            record.group_role = GroupRole.SIMILAR
            record.is_group_representative = False
            record.selection_reason = _why_not(record, best, problems(record), problems(best))


def _why_not(record: ImageRecord, best: ImageRecord, mine: int, theirs: int) -> str:
    closed = _closed_eyes(record)
    if closed and mine > theirs:
        return f"Closed eyes on {closed} face(s); a better frame exists in this group"
    if mine > theirs:
        return "A subject is missing or turned away compared with the best frame"
    return "Lower selection score than the group's best shot"
