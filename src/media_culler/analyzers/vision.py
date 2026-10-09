"""Zero-shot photo assessment with SigLIP.

v0.1 ran the full model four times per image (once per label group). SigLIP scores each
label independently (sigmoid), so all 18 labels go through ONE forward pass and give
identical per-label probabilities: ~4x fewer image-encoder passes.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Protocol

import numpy as np

from media_culler.analyzers.base import Analyzer
from media_culler.config import VisionConfig
from media_culler.exceptions import AnalyzerUnavailable
from media_culler.imaging import ImageContext, resize_max_side
from media_culler.labels import LABEL_GROUPS
from media_culler.models import AIAssessment, LabelPrediction

log = logging.getLogger(__name__)

ALL_LABELS: list[str] = [label for group in LABEL_GROUPS.values() for label in group]


class VisionBackend(Protocol):
    def score(self, rgb: np.ndarray, labels: list[str]) -> list[float]:
        """Return one probability in [0, 1] per label, in order."""


def build_assessment(scores: list[float]) -> AIAssessment:
    """Split a flat per-label score vector into the four labelled groups."""
    if len(scores) != len(ALL_LABELS):
        raise ValueError(f"Expected {len(ALL_LABELS)} scores, got {len(scores)}")

    predictions: dict[str, LabelPrediction] = {}
    offset = 0
    for group, labels in LABEL_GROUPS.items():
        values = scores[offset : offset + len(labels)]
        offset += len(labels)
        top = max(range(len(labels)), key=lambda i: values[i])  # first max wins ties
        predictions[group] = LabelPrediction(
            label=labels[top],
            confidence=round(values[top] * 100, 2),
            scores={label: round(float(v), 6) for label, v in zip(labels, values, strict=True)},
        )
    return AIAssessment(**predictions)


class SigLipBackend:
    """Thin wrapper over Hugging Face SigLIP. Heavy imports happen lazily in ``load``."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._torch = None
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self) -> str:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise AnalyzerUnavailable(
                "AI analysis needs torch + transformers: pip install 'ai-media-culler[ai]'"
            ) from error

        requested = self._config.device
        if requested == "auto":
            if torch.cuda.is_available():
                requested = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"

        try:
            self._processor = AutoProcessor.from_pretrained(self._config.model_name)
            self._model = AutoModel.from_pretrained(self._config.model_name).to(requested).eval()
        except Exception as error:  # network, disk, bad model name, OOM, ...
            raise AnalyzerUnavailable(
                f"Could not load vision model '{self._config.model_name}': {error}"
            ) from error

        self._torch = torch
        self._device = requested
        return requested

    def score(self, rgb: np.ndarray, labels: list[str]) -> list[float]:
        from PIL import Image

        assert self._torch is not None and self._model is not None and self._processor is not None
        image = Image.fromarray(resize_max_side(rgb, self._config.max_side))
        inputs = self._processor(
            text=labels, images=image, padding="max_length", return_tensors="pt"
        ).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits_per_image[0]
            probabilities = self._torch.sigmoid(logits)
        return [float(p) for p in probabilities.detach().cpu().tolist()]

    def close(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._device == "cuda":
            self._torch.cuda.empty_cache()


class VisionAnalyzer(Analyzer):
    name: ClassVar[str] = "vision"
    section: ClassVar[str] = "ai"
    result_type = AIAssessment
    version: ClassVar[str] = "2"

    def __init__(self, config: VisionConfig, backend: VisionBackend | None = None) -> None:
        self._config = config
        self._backend = backend
        self._owns_backend = backend is None
        self.device: str | None = None

    def setup(self) -> None:
        if self._backend is None:
            siglip = SigLipBackend(self._config)
            self.device = siglip.load()
            self._backend = siglip
            log.info("Vision model %s on %s", self._config.model_name, self.device)

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if self._owns_backend and callable(close):
            close()

    def fingerprint(self) -> str:
        return f"{self.version}:{self._config.model_name}"

    def analyze(self, ctx: ImageContext) -> AIAssessment:
        assert self._backend is not None, "setup() not called"
        return build_assessment(self._backend.score(ctx.rgb, ALL_LABELS))
