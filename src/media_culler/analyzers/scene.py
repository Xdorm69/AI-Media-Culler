"""Scene-level metrics for landscape-aware scoring.

OpenCV + NumPy only, so it is thread-safe and runs in the decode workers next to the
technical analyzer. Everything except tiled sharpness runs on a downscaled copy.
"""

from __future__ import annotations

from math import atan2, degrees, pi
from typing import ClassVar

import cv2
import numpy as np

from media_culler.analyzers.base import Analyzer
from media_culler.imaging import ImageContext, resize_max_side
from media_culler.models import HorizonInfo, SceneMetrics

_ANALYSIS_SIDE = 1024
_HORIZON_SIDE = 640
_MAX_TILT_DEG = 15.0  # steeper lines are intentional angles or other structure, not a horizon
_CLUSTER_DEG = 2.0  # segments within this of the dominant angle may belong to the horizon
_CANNY = (40, 120)
_MAX_SEGMENTS = 400
_LINE_BAND = 0.03  # ...and must also lie within this fraction of frame height of each other
_GRID = 4
_SHARPNESS_PERCENTILE = 75
_SALIENCY_SIDE = 64


def detect_horizon(gray: np.ndarray) -> HorizonInfo | None:
    """Find the dominant near-horizontal line, or None if there is no convincing one."""
    small = resize_max_side(gray, _HORIZON_SIDE)
    height, width = small.shape[:2]
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    # Fixed thresholds on purpose: median-derived ones over-trigger on dark, textured frames
    # and flood Hough with ground-texture fragments. A horizon is a long, high-contrast edge.
    edges = cv2.Canny(blurred, *_CANNY)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=pi / 720,
        threshold=max(20, int(0.2 * width)),
        minLineLength=int(0.35 * width),
        maxLineGap=max(4, int(0.04 * width)),
    )
    if lines is None:
        return None

    # angle, length, y at frame centre (extrapolated, 0-1), x start, x end
    segments: list[tuple[float, float, float, float, float]] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # (N,4) or (N,1,4) by version
        if x2 == x1:
            continue
        if x2 < x1:  # always read left to right so the sign is meaningful
            x1, y1, x2, y2 = x2, y2, x1, y1
        angle = degrees(atan2(float(y2 - y1), float(x2 - x1)))
        if abs(angle) > _MAX_TILT_DEG:
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        slope = (y2 - y1) / (x2 - x1)
        centre_y = ((y1 + y2) / 2 + slope * (width / 2 - (x1 + x2) / 2)) / height
        segments.append((angle, length, centre_y, float(x1), float(x2)))
    if not segments:
        return None

    # Textured ground produces many short near-horizontal fragments at other heights and
    # angles. Pick the mode, not a median: the (angle, height) cell with the most total
    # supporting segment length. Candidates are capped so this stays O(n^2) on small n.
    segments.sort(key=lambda s: s[1], reverse=True)
    segments = segments[:_MAX_SEGMENTS]
    total_length = sum(s[1] for s in segments)
    angles = np.array([s[0] for s in segments])
    lengths = np.array([s[1] for s in segments])
    centres = np.array([s[2] for s in segments])
    near = (np.abs(angles[:, None] - angles[None, :]) <= _CLUSTER_DEG) & (
        np.abs(centres[:, None] - centres[None, :]) <= _LINE_BAND
    )
    best = int(np.argmax(near @ lengths))
    kept = [segments[i] for i in np.flatnonzero(near[best])]
    kept_length = sum(s[1] for s in kept)
    angle = sum(s[0] * s[1] for s in kept) / kept_length
    y = sum(s[2] * s[1] for s in kept) / kept_length

    covered = np.zeros(width, dtype=bool)  # union of x-ranges: overlapping Hough lines count once
    for _, _, _, x0, x1 in kept:
        covered[int(x0) : int(x1) + 1] = True
    coverage = float(covered.mean())
    consistency = kept_length / total_length
    return HorizonInfo(
        angle_deg=round(angle, 3),
        y=round(float(np.clip(y, 0.0, 1.0)), 4),
        confidence=round(coverage * consistency, 4),
    )


def saliency_centroid(gray: np.ndarray) -> tuple[float, float]:
    """Centroid (x, y in 0-1) of the most salient region (spectral-residual saliency)."""
    small = cv2.resize(gray, (_SALIENCY_SIDE, _SALIENCY_SIDE), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(small.astype(np.float32))
    log_amplitude = np.log(np.abs(spectrum) + 1e-8).astype(np.float32)
    residual = log_amplitude - cv2.blur(log_amplitude, (3, 3))
    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (0, 0), 3)

    mask = saliency > saliency.mean() + saliency.std()
    if not mask.any():
        return 0.5, 0.5
    ys, xs = np.nonzero(mask)
    weights = saliency[mask].astype(np.float64)
    total = float(weights.sum())
    cx = float((xs + 0.5) @ weights / total / _SALIENCY_SIDE)
    cy = float((ys + 0.5) @ weights / total / _SALIENCY_SIDE)
    return round(cx, 4), round(cy, 4)


def tiled_sharpness(gray: np.ndarray, grid: int = _GRID) -> float:
    """Upper-quartile tile Laplacian variance.

    Global Laplacian variance averages in smooth sky, water and fog, so a crisp landscape
    with a big empty sky looks "blurry". The 75th-percentile tile ignores those regions
    while a genuinely soft or motion-blurred frame is soft in *every* tile.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    height, width = laplacian.shape
    tile_h, tile_w = height // grid, width // grid
    if tile_h < 8 or tile_w < 8:
        _, std = cv2.meanStdDev(laplacian)
        return float(std[0][0] ** 2)
    variances = []
    for row in range(grid):
        for col in range(grid):
            tile = laplacian[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w]
            _, std = cv2.meanStdDev(tile)
            variances.append(float(std[0][0] ** 2))
    return float(np.percentile(variances, _SHARPNESS_PERCENTILE))


class SceneAnalyzer(Analyzer):
    name: ClassVar[str] = "scene"
    section: ClassVar[str] = "scene"
    result_type = SceneMetrics
    version: ClassVar[str] = "1"
    thread_safe: ClassVar[bool] = True

    def analyze(self, ctx: ImageContext) -> SceneMetrics:
        small_rgb = resize_max_side(ctx.rgb, _ANALYSIS_SIDE)
        small_gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
        brightest = small_rgb.max(axis=2)
        low, high = np.percentile(small_gray, [1, 99])
        cx, cy = saliency_centroid(small_gray)
        return SceneMetrics(
            horizon=detect_horizon(small_gray),
            highlight_clip=round(float(np.mean(brightest >= 250)), 5),
            shadow_clip=round(float(np.mean(brightest <= 5)), 5),
            dynamic_range=round(float(high - low), 2),
            sharpness_focus=round(tiled_sharpness(ctx.gray), 2),
            saliency_x=cx,
            saliency_y=cy,
        )
