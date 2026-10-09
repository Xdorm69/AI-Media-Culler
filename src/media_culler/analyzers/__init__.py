"""Analyzer registry."""

from __future__ import annotations

from media_culler.analyzers.base import Analyzer
from media_culler.analyzers.faces import FaceAnalyzer
from media_culler.analyzers.scene import SceneAnalyzer
from media_culler.analyzers.technical import TechnicalAnalyzer
from media_culler.analyzers.vision import VisionAnalyzer
from media_culler.config import Settings


def build_analyzers(settings: Settings) -> list[Analyzer]:
    """Instantiate (but do not set up) analyzers enabled by ``settings``."""
    analyzers: list[Analyzer] = [TechnicalAnalyzer()]
    if settings.scene.enabled:
        analyzers.append(SceneAnalyzer())
    if settings.vision.enabled:
        analyzers.append(VisionAnalyzer(settings.vision))
    if settings.faces.enabled:
        analyzers.append(FaceAnalyzer(settings))
    return analyzers


__all__ = ["Analyzer", "build_analyzers"]
