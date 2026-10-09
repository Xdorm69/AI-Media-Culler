from __future__ import annotations

from media_culler.config import Settings, WeightsConfig
from media_culler.models import ImageRecord, SceneType, Scores
from media_culler.scoring.ai import score_ai
from media_culler.scoring.composition import score_composition
from media_culler.scoring.landscape import score_landscape_composition, technical_overrides
from media_culler.scoring.scene import classify_scene
from media_culler.scoring.subject import score_subject
from media_culler.scoring.technical import score_technical


def combine(
    components: dict[str, float | None], weights: WeightsConfig
) -> tuple[float | None, dict[str, float]]:
    """Weighted mean over available components, with weights renormalised.

    Unavailable components (analyzer disabled/failed) are *excluded* rather than
    scored as 0, so a missing model never silently drags an image's score down.
    """
    raw = {
        name: getattr(weights, name)
        for name, value in components.items()
        if value is not None and getattr(weights, name) > 0
    }
    total = sum(raw.values())
    if "technical" not in raw or total <= 0:
        return None, {}
    normalised = {name: w / total for name, w in raw.items()}
    final = sum(components[name] * w for name, w in normalised.items())
    return round(final, 2), {k: round(v, 4) for k, v in normalised.items()}


def score_record(record: ImageRecord, settings: Settings) -> Scores:
    scores = Scores()

    scene_type, scene_reason = classify_scene(record, settings.scene)
    scores.scene_type = scene_type
    scores.details["scene"] = {"type": scene_type.value, "reason": scene_reason}
    landscape = scene_type is SceneType.LANDSCAPE

    if record.technical is not None:
        overrides: dict[str, float] = {}
        if landscape and record.scene is not None:
            overrides, info = technical_overrides(record.technical, record.scene)
            scores.details["scene"].update(info)
        scores.technical, scores.details["technical"] = score_technical(record.technical, overrides)
    if record.ai is not None:
        scores.ai, scores.details["ai"] = score_ai(record.ai)

    if landscape:
        # No faces to judge: "subject" does not apply, so it is excluded from the weighted
        # mean (not scored as a neutral 50) and composition is judged on landscape rules.
        if record.scene is not None:
            scores.composition, composition_details = score_landscape_composition(
                record.scene, settings.scene
            )
            scores.details["composition"] = composition_details
    elif record.faces is not None:
        scores.subject, scores.details["subject"] = score_subject(record.faces)
        scores.composition, scores.details["composition"] = score_composition(
            record.faces, record.width, record.height
        )

    scores.final, scores.weights_used = combine(
        {
            "technical": scores.technical,
            "ai": scores.ai,
            "subject": scores.subject,
            "composition": scores.composition,
        },
        settings.weights,
    )
    return scores
