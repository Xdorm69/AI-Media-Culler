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
