from __future__ import annotations

from media_culler.labels import AESTHETIC_SCORES, COMPOSITION_SCORES, QUALITY_SCORES
from media_culler.models import AIAssessment

_NEUTRAL = 50.0


def score_ai(assessment: AIAssessment) -> tuple[float, dict[str, float]]:
    parts = {
        "photographic_quality": QUALITY_SCORES.get(assessment.photographic_quality.label, _NEUTRAL),
        "composition": COMPOSITION_SCORES.get(assessment.composition.label, _NEUTRAL),
        "aesthetic": AESTHETIC_SCORES.get(assessment.aesthetic.label, _NEUTRAL),
    }
    total = (
        parts["photographic_quality"] * 0.35
        + parts["composition"] * 0.30
        + parts["aesthetic"] * 0.35
    )
    return round(total, 2), parts
