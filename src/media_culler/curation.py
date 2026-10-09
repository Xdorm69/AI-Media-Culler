"""Collection-level decisions: ranking, recommendations and best-shot selection."""

from __future__ import annotations

from media_culler.config import ThresholdsConfig
from media_culler.models import GroupRole, ImageRecord, Recommendation


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


def classify(record: ImageRecord, thresholds: ThresholdsConfig) -> tuple[Recommendation, str]:
    final = record.final_score
    technical = record.scores.technical if record.scores else None
    if final is None or technical is None:
        return Recommendation.UNSCORED, "Analysis incomplete; no score available"
    if technical < thresholds.technical_floor:
        return Recommendation.REJECT, "Very low technical quality"
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
