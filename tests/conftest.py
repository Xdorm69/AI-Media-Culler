from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from media_culler.config import Settings, load_settings


def textured(seed: int, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[1] // 16, size[0] // 16, 3), dtype=np.uint8)
    return cv2.resize(base, size, interpolation=cv2.INTER_CUBIC)


def save(array: np.ndarray, path: Path, **kwargs: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, **kwargs)  # type: ignore[arg-type]
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with every writable path inside tmp_path and heavy analyzers off."""
    return load_settings(
        None,
        {
            "paths": {
                "input_directory": str(tmp_path / "in"),
                "output_directory": str(tmp_path / "out"),
                "reports_directory": str(tmp_path / "reports"),
                "cache_directory": str(tmp_path / "cache"),
                "models_directory": str(tmp_path / "models"),
            },
            "vision": {"enabled": False},
            "faces": {"enabled": False},
            "performance": {"workers": 2},
        },
    )


@pytest.fixture
def photo_dir(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    sharp = textured(1)
    save(sharp, root / "sharp.jpg", quality=95)
    save(cv2.GaussianBlur(sharp, (0, 0), 10), root / "blurry.jpg")
    save(textured(2), root / "sub" / "other.png")
    (root / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0 definitely not a jpeg")
    (root / "notes.txt").write_text("not an image")
    return root


def landscape_image(
    width: int = 1200,
    height: int = 800,
    horizon: float = 0.4,
    tilt: float = 0.0,
    seed: int = 0,
    detail: int = 8,
) -> np.ndarray:
    """Synthetic scenery: smooth gradient sky over textured ground, optionally rotated.

    ``detail`` is the ground texture cell size in pixels (2 = fine, genuinely sharp detail).
    ``tilt`` follows cv2.getRotationMatrix2D (positive = counter-clockwise), so a positive
    tilt here is detected as a *negative* horizon angle (right side higher).
    """
    rng = np.random.default_rng(seed)
    rows = np.linspace(0, 1, height)[:, None]
    sky = np.stack([120 + 80 * rows, 170 + 50 * rows, 235 - 10 * rows], -1) * np.ones((1, width, 1))
    ground = cv2.resize(
        rng.integers(20, 120, (height // detail, width // detail, 3), dtype=np.uint8),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    ground = (ground * np.array([0.5, 1.0, 0.4])).astype(np.uint8)
    image = sky.astype(np.uint8)
    split = int(height * horizon)
    image[split:] = ground[split:]
    if tilt:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), tilt, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return image
