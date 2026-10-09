from __future__ import annotations

from collections.abc import Mapping

from media_culler.models import TechnicalMetrics
from media_culler.scoring._math import clamp, ramp


def sharpness_score(value: float) -> float:
    return ramp(value, 50, 500, 10, 100)


def exposure_score(brightness: float) -> float:
    return round(clamp(100 - abs(brightness - 125) / 125 * 100), 2)


def contrast_score(value: float) -> float:
    return ramp(value, 20, 80, 15, 100)


def resolution_score(megapixels: float) -> float:
    return ramp(megapixels, 2, 12, 20, 100)


def score_technical(
    metrics: TechnicalMetrics, overrides: Mapping[str, float] | None = None
) -> tuple[float, dict[str, float]]:
    """Weighted technical score. ``overrides`` replaces individual part scores (0-100),
    which is how scene-aware scoring swaps in better sharpness/exposure/contrast measures.
    """
    parts = {
        "sharpness": sharpness_score(metrics.sharpness),
        "exposure": exposure_score(metrics.brightness),
        "contrast": contrast_score(metrics.contrast),
        "resolution": resolution_score(metrics.megapixels),
    }
    if overrides:
        parts.update(overrides)
    total = (
        parts["sharpness"] * 0.40
        + parts["exposure"] * 0.25
        + parts["contrast"] * 0.15
        + parts["resolution"] * 0.20
    )
    return round(total, 2), {k: round(v, 2) for k, v in parts.items()}
