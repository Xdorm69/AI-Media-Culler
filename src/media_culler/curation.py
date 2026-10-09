"""Collection-level decisions: ranking, recommendations and best-shot selection."""

from __future__ import annotations

from media_culler.config import ThresholdsConfig
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


def classify(record: ImageRecord, thresholds: ThresholdsConfig) -> tuple[Recommendation, str]:
    final = record.final_score
    technical = record.scores.technical if record.scores else None
    if final is None or technical is None:
        return Recommendation.UNSCORED, "Analysis incomplete; no score available"
    if technical < thresholds.technical_floor:
        return Recommendation.REJECT, "Very low technical quality"
    if _soft_focus_landscape(record, thresholds) and final >= thresholds.review:
        return Recommendation.REVIEW, "Landscape is soft/out of focus; check before keeping"
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


def select_best_shots(records: list[ImageRecord]) -> None:
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
        best = min(members, key=_sort_key)  # unscored (None -> 0.0) naturally sort last
        best_score = best.final_score
        for record in members:
            record.best_shot_score = best_score
            if record is best:
                record.group_role = GroupRole.REPRESENTATIVE
                record.is_group_representative = True
                record.selection_reason = "Highest final score in similar-image group"
            else:
                record.group_role = GroupRole.SIMILAR
                record.is_group_representative = False
                record.selection_reason = "Lower final score than group representative"
