"""Decide whether a photo should be judged as people, landscape or anything else."""

from __future__ import annotations

from media_culler.config import SceneConfig
from media_culler.labels import LANDSCAPE_LABEL
from media_culler.models import ImageRecord, SceneType


def classify_scene(record: ImageRecord, config: SceneConfig) -> tuple[SceneType, str]:
    """Returns (scene type, human-readable reason).

    Faces only make a photo "people" when they are big enough to be the subject: a hiker
    a kilometre away on a ridge does not turn a landscape into a portrait.
    """
    faces = record.faces.faces if record.faces else []
    if any(face.area_ratio >= config.people_min_face_area for face in faces):
        return SceneType.PEOPLE, "prominent face"

    content = record.ai.content if record.ai else None
    if (
        content is not None
        and content.label == LANDSCAPE_LABEL
        and content.confidence >= config.landscape_min_confidence
    ):
        return SceneType.LANDSCAPE, "AI content label"

    if config.heuristic_without_ai and record.ai is None and record.scene is not None:
        horizon = record.scene.horizon
        if horizon is not None and horizon.confidence >= config.heuristic_horizon_confidence:
            return SceneType.LANDSCAPE, "strong horizon (no AI label available)"

    return SceneType.GENERAL, "default"
