import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import save, textured
from media_culler.analyzers.technical import perceptual_hash
from media_culler.exceptions import ImageLoadError
from media_culler.imaging import ImageContext, load_rgb, read_header, resize_max_side, sha256_file


def _oriented(tmp_path: Path, orientation: int) -> Path:
    img = Image.fromarray(textured(3, (400, 300)))
    exif = img.getexif()
    exif[0x0112] = orientation
    path = tmp_path / f"o{orientation}.jpg"
    img.save(path, exif=exif, quality=95)
    return path


@pytest.mark.parametrize(
    "orientation,expected", [(1, (400, 300)), (3, (400, 300)), (6, (300, 400)), (8, (300, 400))]
)
def test_header_and_decode_agree_on_oriented_size(tmp_path, orientation, expected):
    """Regression: v0.1 scanner (PIL) and face analysis (cv2) disagreed on rotated photos."""
    path = _oriented(tmp_path, orientation)
    header = read_header(path, 10**9)
    rgb = load_rgb(path, 10**9)
    assert (header.width, header.height) == expected
    assert (rgb.shape[1], rgb.shape[0]) == expected


def test_decode_is_contiguous_uint8_rgb(tmp_path):
    p = save(textured(1), tmp_path / "a.png")
    rgb = load_rgb(p, 10**9)
    assert rgb.dtype == np.uint8 and rgb.flags["C_CONTIGUOUS"] and rgb.shape[2] == 3


def test_palette_and_alpha_images_decode(tmp_path):
    Image.new("RGBA", (50, 40), (255, 0, 0, 128)).save(tmp_path / "rgba.png")
    Image.new("P", (50, 40)).save(tmp_path / "pal.png")
    assert load_rgb(tmp_path / "rgba.png", 10**9).shape == (40, 50, 3)
    assert load_rgb(tmp_path / "pal.png", 10**9).shape == (40, 50, 3)


def test_corrupt_and_oversized_raise_imageloaderror(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"\xff\xd8garbage")
    with pytest.raises(ImageLoadError):
        load_rgb(bad, 10**9)
    big = save(textured(1), tmp_path / "big.png")
    with pytest.raises(ImageLoadError, match="too large"):
        load_rgb(big, 100)
    with pytest.raises(ImageLoadError, match="too large"):
        read_header(big, 100)


def test_truncated_file_is_rejected(tmp_path):
    buf = io.BytesIO()
    Image.fromarray(textured(1)).save(buf, "JPEG")
    path = tmp_path / "trunc.jpg"
    path.write_bytes(buf.getvalue()[: len(buf.getvalue()) // 2])
    with pytest.raises(ImageLoadError):
        load_rgb(path, 10**9)


def test_resize_never_upscales():
    a = np.zeros((100, 200, 3), np.uint8)
    assert resize_max_side(a, 500) is a
    assert resize_max_side(a, 100).shape[:2] == (50, 100)


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "f.bin"
    p.write_bytes(b"x" * 3_000_000)
    assert sha256_file(p) == hashlib.sha256(b"x" * 3_000_000).hexdigest()


def test_gray_is_cached():
    ctx = ImageContext(Path("x"), textured(1))
    assert ctx.gray is ctx.gray


def test_phash_matches_imagehash():
    """Own DCT pHash must be bit-compatible with the library it replaced."""
    imagehash = pytest.importorskip("imagehash")
    for seed in range(5):
        rgb = textured(seed, (333, 251))
        assert perceptual_hash(rgb) == str(imagehash.phash(Image.fromarray(rgb)))


def test_phash_is_stable_across_exif_rotation(tmp_path):
    upright = load_rgb(_oriented(tmp_path, 1), 10**9)
    rotated = load_rgb(_oriented(tmp_path, 1), 10**9)
    assert perceptual_hash(upright) == perceptual_hash(rotated)
