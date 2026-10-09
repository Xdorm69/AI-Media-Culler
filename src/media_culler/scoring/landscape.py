"""Landscape-specific scoring: horizon, tone and focus rules that suit scenery.

Faces drive subject/composition for portraits; landscapes need different questions:
is the horizon level and sensibly placed, are skies blown out, is anything actually in
focus (rather than "is the whole frame busy").
"""

from __future__ import annotations

from media_culler.config import SceneConfig
from media_culler.models import SceneMetrics, TechnicalMetrics
from media_culler.scoring._math import clamp, ramp
from media_culler.scoring.composition import rule_of_thirds_score
from media_culler.scoring.technical import sharpness_score


def horizon_level_score(tilt_deg: float) -> float:
    """100 when level, falling to ~45 at 3 degrees and ~10 at 8+ degrees."""
    tilt = abs(tilt_deg)
    if tilt <= 0.3:
        return 100.0
    if tilt <= 3.0:
        return ramp(tilt, 0.3, 3.0, 100, 45)
    return ramp(tilt, 3.0, 8.0, 45, 10)


def horizon_placement_score(y: float) -> float:
    """Favour the thirds lines; a centred horizon is acceptable, one hugging an edge is not."""
    distance = min(abs(y - 1 / 3), abs(y - 2 / 3))
    if distance <= 1 / 6:
        return ramp(distance, 0, 1 / 6, 100, 70)
    return ramp(distance, 1 / 6, 1 / 3, 70, 40)


def exposure_score(scene: SceneMetrics, brightness: float) -> float:
    """Clipping-based exposure. Mean brightness only matters at the extremes.

    The portrait rule (mean luminance near 125) punishes sunsets, night scenes and
    high-key snow. What actually ruins a landscape is blown sky or crushed shadows.
    """
    highlights = ramp(scene.highlight_clip, 0.005, 0.15, 0, 70)
    shadows = ramp(scene.shadow_clip, 0.02, 0.40, 0, 50)
    extreme = max(ramp(brightness, 180, 235, 0, 40), ramp(35 - brightness, 0, 30, 0, 40))
    return round(clamp(100 - highlights - shadows - extreme), 2)


def dynamic_range_score(dynamic_range: float) -> float:
    """Replaces global contrast: flat, hazy frames score low, full tonal range scores high."""
    return ramp(dynamic_range, 50, 180, 25, 100)


def technical_overrides(
    technical: TechnicalMetrics, scene: SceneMetrics
) -> tuple[dict[str, float], dict[str, object]]:
    """Part-score overrides for score_technical plus what changed, for the report."""
    focus = max(technical.sharpness, scene.sharpness_focus)
    overrides = {
        "sharpness": sharpness_score(focus),
        "exposure": exposure_score(scene, technical.brightness),
        "contrast": dynamic_range_score(scene.dynamic_range),
    }
    info: dict[str, object] = {
        "sharpness_source": "tiled" if scene.sharpness_focus > technical.sharpness else "global",
        "sharpness_used": round(focus, 2),
    }
    return overrides, info


def score_landscape_composition(
    scene: SceneMetrics, config: SceneConfig
) -> tuple[float, dict[str, object]]:
    parts: dict[str, tuple[float, float]] = {}  # name -> (score, weight)
    details: dict[str, object] = {}

    horizon = scene.horizon
    if horizon is not None and horizon.confidence >= config.horizon_min_confidence:
        parts["horizon_level"] = (horizon_level_score(horizon.angle_deg), 0.45)
        parts["horizon_placement"] = (horizon_placement_score(horizon.y), 0.30)
        details["horizon_tilt_deg"] = horizon.angle_deg
        details["horizon_y"] = horizon.y
    else:
        details["horizon"] = "none detected"

    parts["focal_point"] = (rule_of_thirds_score(scene.saliency_x, scene.saliency_y), 0.25)

    total_weight = sum(weight for _, weight in parts.values())
    score = sum(value * weight for value, weight in parts.values()) / total_weight
    details["parts"] = {name: round(value, 2) for name, (value, _) in parts.items()}
    return round(score, 2), details
