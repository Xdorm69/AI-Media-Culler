"""Sharpness, exposure, contrast, resolution and perceptual hash from a single decode."""

from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np
from PIL import Image

from media_culler.analyzers.base import Analyzer
from media_culler.imaging import ImageContext
from media_culler.models import TechnicalMetrics

_HASH_SIZE = 8
_DCT_SIZE = _HASH_SIZE * 4  # 32x32 working image, as in the standard pHash


def _dct_matrix(n: int) -> np.ndarray:
    """Unnormalised DCT-II matrix (matches scipy.fftpack.dct, used by ImageHash)."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    return 2.0 * np.cos(np.pi * k * (2 * i + 1) / (2 * n))


_DCT = _dct_matrix(_DCT_SIZE)


def perceptual_hash(rgb: np.ndarray) -> str:
    """64-bit DCT perceptual hash as 16 hex chars. Bit-compatible with imagehash.phash."""
    small = (
        Image.fromarray(rgb).convert("L").resize((_DCT_SIZE, _DCT_SIZE), Image.Resampling.LANCZOS)
    )
    pixels = np.asarray(small, dtype=np.float64)
    coefficients = _DCT @ pixels @ _DCT.T
    low = coefficients[:_HASH_SIZE, :_HASH_SIZE]
    bits = (low > np.median(low)).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def laplacian_variance(gray: np.ndarray) -> float:
    # float32 output + meanStdDev (double accumulators) avoids a 64-bit copy of huge images.
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    _, std = cv2.meanStdDev(laplacian)
    return float(std[0][0] ** 2)


class TechnicalAnalyzer(Analyzer):
    name: ClassVar[str] = "technical"
    section: ClassVar[str] = "technical"
    result_type = TechnicalMetrics
    version: ClassVar[str] = "2"
    thread_safe: ClassVar[bool] = True

    def analyze(self, ctx: ImageContext) -> TechnicalMetrics:
        mean, std = cv2.meanStdDev(ctx.gray)
        return TechnicalMetrics(
            sharpness=round(laplacian_variance(ctx.gray), 2),
            brightness=round(float(mean[0][0]), 2),
            contrast=round(float(std[0][0]), 2),
            megapixels=round(ctx.width * ctx.height / 1_000_000, 2),
            perceptual_hash=perceptual_hash(ctx.rgb),
        )
