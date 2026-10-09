"""Face detection + landmarks via MediaPipe Face Landmarker."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Protocol

import numpy as np

from media_culler.analyzers.base import Analyzer
from media_culler.analyzers.face_metrics import build_face_analysis
from media_culler.config import Settings
from media_culler.exceptions import AnalyzerUnavailable
from media_culler.imaging import ImageContext, resize_max_side
from media_culler.models import FaceAnalysis

log = logging.getLogger(__name__)


class FaceBackend(Protocol):
    def detect(self, rgb: np.ndarray) -> list[np.ndarray]:
        """Return one (N, 3) normalised landmark array per detected face."""


class MediaPipeBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._landmarker: Any = None
        self._mp: Any = None

    def load(self) -> None:
        model_path = self._settings.face_model_path
        if not model_path.is_file():
            raise AnalyzerUnavailable(
                f"Face model not found at {model_path}. Run: culler models download"
            )
        try:
            import mediapipe as mp
        except ImportError as error:
            raise AnalyzerUnavailable(
                "Face analysis needs mediapipe: pip install 'ai-media-culler[faces]'"
            ) from error

        cfg = self._settings.faces
        try:
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=cfg.max_faces,
                min_face_detection_confidence=cfg.min_detection_confidence,
                min_face_presence_confidence=cfg.min_presence_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception as error:
            raise AnalyzerUnavailable(f"Could not initialise face landmarker: {error}") from error
        self._mp = mp

    def detect(self, rgb: np.ndarray) -> list[np.ndarray]:
        small = np.ascontiguousarray(resize_max_side(rgb, self._settings.faces.max_side))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=small)
        result = self._landmarker.detect(mp_image)
        return [
            np.array([(p.x, p.y, p.z) for p in face], dtype=np.float64)
            for face in result.face_landmarks
        ]

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


class FaceAnalyzer(Analyzer):
    name: ClassVar[str] = "faces"
    section: ClassVar[str] = "faces"
    result_type = FaceAnalysis
    version: ClassVar[str] = "2"

    def __init__(self, settings: Settings, backend: FaceBackend | None = None) -> None:
        self._settings = settings
        self._backend = backend
        self._owns_backend = backend is None

    def setup(self) -> None:
        if self._backend is None:
            backend = MediaPipeBackend(self._settings)
            backend.load()
            self._backend = backend

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if self._owns_backend and callable(close):
            close()

    def fingerprint(self) -> str:
        c = self._settings.faces
        return (
            f"{self.version}:{c.max_faces}:{c.min_detection_confidence}:{c.min_presence_confidence}"
        )

    def analyze(self, ctx: ImageContext) -> FaceAnalysis:
        assert self._backend is not None, "setup() not called"
        return build_face_analysis(self._backend.detect(ctx.rgb), ctx)
