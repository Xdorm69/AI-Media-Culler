"""Image I/O: one orientation-correct decode path shared by every analyzer.

The v0.1 code decoded each image 5+ times through three libraries that disagreed
about EXIF rotation. Everything now goes through :func:`load_rgb`.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from media_culler.exceptions import ImageLoadError

EXIF_ORIENTATION = 0x0112
_SWAPPED_ORIENTATIONS = {5, 6, 7, 8}
_HASH_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class ImageHeader:
    width: int  # orientation-corrected
    height: int
    format: str | None


def configure_pillow(max_pixels: int) -> None:
    """Align Pillow's decompression-bomb guard with our configured limit."""
    Image.MAX_IMAGE_PIXELS = max_pixels


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_header(path: Path, max_pixels: int) -> ImageHeader:
    """Read dimensions/format without decoding pixels. Raises ImageLoadError."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            fmt = image.format
            orientation = image.getexif().get(EXIF_ORIENTATION)
    except Exception as error:  # PIL raises many types for corrupt input
        raise ImageLoadError(f"Cannot open image: {error}") from error

    if width * height > max_pixels:
        raise ImageLoadError(f"Image too large ({width}x{height} > {max_pixels} pixels)")
    if orientation in _SWAPPED_ORIENTATIONS:
        width, height = height, width
    return ImageHeader(width=width, height=height, format=fmt)


def load_rgb(path: Path, max_pixels: int) -> np.ndarray:
    """Decode to an orientation-corrected, C-contiguous uint8 RGB array (H, W, 3)."""
    try:
        with Image.open(path) as image:
            if image.width * image.height > max_pixels:
                raise ImageLoadError(
                    f"Image too large ({image.width}x{image.height} > {max_pixels} pixels)"
                )
            image.load()
            oriented: Image.Image = image
            with contextlib.suppress(Exception):  # malformed EXIF must not lose the photo
                oriented = ImageOps.exif_transpose(image)
            array = np.array(oriented.convert("RGB"), dtype=np.uint8)
    except ImageLoadError:
        raise
    except Exception as error:
        raise ImageLoadError(f"Cannot decode image: {error}") from error

    if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ImageLoadError("Decoded image has invalid shape")
    return np.ascontiguousarray(array)


def resize_max_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is <= max_side. Never upscales."""
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return rgb
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


@dataclass
class ImageContext:
    """A decoded image shared by all analyzers for one pass."""

    path: Path
    rgb: np.ndarray

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @cached_property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)

    @classmethod
    def from_path(cls, path: Path, max_pixels: int) -> ImageContext:
        return cls(path=path, rgb=load_rgb(path, max_pixels))
