"""Image I/O: one orientation-correct decode path shared by every analyzer.

The v0.1 code decoded each image 5+ times through three libraries that disagreed
about EXIF rotation. Everything now goes through :func:`load_rgb`.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from media_culler.exceptions import ImageLoadError

try:  # optional: pip install 'ai-media-culler[formats]'
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:  # core install: HEIC/HEIF files simply fail to open (reported per file)
    HEIF_AVAILABLE = False

try:  # optional, same extra as pillow-heif
    import rawpy as _rawpy

    rawpy: Any = _rawpy  # untyped on purpose: its stubs differ between releases
    RAW_AVAILABLE = True
except ImportError:
    rawpy = None
    RAW_AVAILABLE = False

# Camera RAW containers. Pillow can open some of these (TIFF-based ones) but only as a
# small embedded preview, which would be scored as if it were the photo: decode with rawpy.
RAW_EXTENSIONS = frozenset(
    {".arw", ".cr2", ".cr3", ".dng", ".nef", ".nrw", ".orf", ".pef", ".raf", ".rw2", ".srw"}
)
_RAW_FLIPS_SWAPPING = {5, 6}  # LibRaw flip codes that rotate by 90 degrees
_raw_half_size = False

EXIF_ORIENTATION = 0x0112
_EXIF_IFD = 0x8769
_TIME_TAGS = (0x9003, 0x9004)  # DateTimeOriginal, DateTimeDigitized
_SUBSEC_TAGS = {0x9003: 0x9291, 0x9004: 0x9292}
_TIME_FORMAT = "%Y:%m:%d %H:%M:%S"
_SWAPPED_ORIENTATIONS = {5, 6, 7, 8}
_HASH_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class ImageHeader:
    width: int  # orientation-corrected
    height: int
    format: str | None
    captured_at: str | None = None  # ISO 8601, naive local time as written by the camera


def configure_pillow(max_pixels: int) -> None:
    """Align Pillow's decompression-bomb guard with our configured limit."""
    Image.MAX_IMAGE_PIXELS = max_pixels


def configure_raw(half_size: bool) -> None:
    """Choose RAW decode resolution: half-size is ~4x faster but changes sharpness readings."""
    global _raw_half_size
    _raw_half_size = half_size


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_capture_time(exif: Image.Exif) -> str | None:
    """Capture time from EXIF as a naive ISO string, or None.

    Cameras write local time with no zone, so only *differences* between photos are
    meaningful. Sub-second digits are kept: bursts routinely fall inside one second.
    File modification time is deliberately NOT used as a fallback: copying or exporting
    resets it, and a wrong time is worse than none (grouping then falls back to
    appearance alone).
    """
    try:
        ifd = exif.get_ifd(_EXIF_IFD)
    except Exception:  # malformed EXIF
        return None
    for tag in _TIME_TAGS:
        raw = ifd.get(tag)
        if not isinstance(raw, str):
            continue
        try:
            moment = datetime.strptime(raw.strip().rstrip("\x00"), _TIME_FORMAT)
        except ValueError:
            continue  # includes the "0000:00:00 00:00:00" some cameras write
        subsec = ifd.get(_SUBSEC_TAGS[tag])
        if isinstance(subsec, str | bytes):
            digits = subsec.decode(errors="ignore") if isinstance(subsec, bytes) else subsec
            digits = digits.strip().rstrip("\x00")
            if digits.isdigit():
                moment = moment.replace(microsecond=int(float("0." + digits) * 1_000_000))
        return moment.isoformat()
    return None


def _require_rawpy() -> None:
    if not RAW_AVAILABLE:
        raise ImageLoadError("RAW decoding needs rawpy: pip install 'ai-media-culler[formats]'")


def _raw_capture_time(path: Path, raw: Any) -> str | None:
    """EXIF capture time of a RAW file: the container's own EXIF, else its embedded preview's."""
    try:
        with Image.open(path) as image:  # TIFF-based RAW (DNG, NEF, ARW, CR2...)
            moment = parse_capture_time(image.getexif())
        if moment:
            return moment
    except Exception:
        pass
    try:
        thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            with Image.open(io.BytesIO(thumb.data)) as preview:
                return parse_capture_time(preview.getexif())
    except Exception:
        pass
    return None


def _read_raw_header(path: Path, max_pixels: int) -> ImageHeader:
    _require_rawpy()
    try:
        with rawpy.imread(str(path)) as raw:
            width, height = int(raw.sizes.width), int(raw.sizes.height)
            flip = int(raw.sizes.flip)
            captured_at = _raw_capture_time(path, raw)
    except Exception as error:
        raise ImageLoadError(f"Cannot open RAW file: {error}") from error
    if width * height > max_pixels:
        raise ImageLoadError(f"Image too large ({width}x{height} > {max_pixels} pixels)")
    if flip in _RAW_FLIPS_SWAPPING:
        width, height = height, width
    return ImageHeader(width=width, height=height, format="RAW", captured_at=captured_at)


def _load_raw(path: Path, max_pixels: int) -> np.ndarray:
    _require_rawpy()
    try:
        with rawpy.imread(str(path)) as raw:
            if raw.sizes.width * raw.sizes.height > max_pixels:
                raise ImageLoadError(
                    f"Image too large ({raw.sizes.width}x{raw.sizes.height} > {max_pixels} pixels)"
                )
            # LibRaw applies the camera's orientation flag itself.
            rgb = raw.postprocess(use_camera_wb=True, half_size=_raw_half_size, output_bps=8)
    except ImageLoadError:
        raise
    except Exception as error:
        raise ImageLoadError(f"Cannot decode RAW file: {error}") from error
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def read_header(path: Path, max_pixels: int) -> ImageHeader:
    """Read dimensions/format without decoding pixels. Raises ImageLoadError."""
    if is_raw(path):
        return _read_raw_header(path, max_pixels)
    try:
        with Image.open(path) as image:
            width, height = image.size
            fmt = image.format
            exif = image.getexif()
            orientation = exif.get(EXIF_ORIENTATION)
            captured_at = parse_capture_time(exif)
    except Exception as error:  # PIL raises many types for corrupt input
        raise ImageLoadError(f"Cannot open image: {error}") from error

    if width * height > max_pixels:
        raise ImageLoadError(f"Image too large ({width}x{height} > {max_pixels} pixels)")
    if orientation in _SWAPPED_ORIENTATIONS:
        width, height = height, width
    return ImageHeader(width=width, height=height, format=fmt, captured_at=captured_at)


def load_rgb(path: Path, max_pixels: int) -> np.ndarray:
    """Decode to an orientation-corrected, C-contiguous uint8 RGB array (H, W, 3)."""
    if is_raw(path):
        array = _load_raw(path, max_pixels)
        if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ImageLoadError("Decoded image has invalid shape")
        return array
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
