from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import textured
from media_culler.analyzers.face_metrics import (
    LEFT_EYE,
    RIGHT_EYE,
    build_face_analysis,
    eye_openness,
    face_pose,
)
from media_culler.analyzers.faces import FaceAnalyzer
from media_culler.analyzers.technical import TechnicalAnalyzer, laplacian_variance
from media_culler.analyzers.vision import ALL_LABELS, VisionAnalyzer, build_assessment
from media_culler.config import Settings, VisionConfig
from media_culler.exceptions import AnalyzerUnavailable
from media_culler.imaging import ImageContext


# ----------------------------------------------------------------------- technical
def test_technical_metrics_sharp_vs_blurred():
    sharp = np.random.default_rng(1).integers(
        0, 255, (480, 640, 3), dtype=np.uint8
    )  # pixel-level detail
    a = TechnicalAnalyzer()
    s = a.analyze(ImageContext(Path("s"), sharp))
    b = a.analyze(ImageContext(Path("b"), cv2.GaussianBlur(sharp, (0, 0), 8)))
    assert s.sharpness > 10 * b.sharpness
    assert s.megapixels == pytest.approx(640 * 480 / 1e6, abs=0.01)
    assert len(s.perceptual_hash) == 16


def test_black_image_metrics():
    m = TechnicalAnalyzer().analyze(ImageContext(Path("k"), np.zeros((100, 100, 3), np.uint8)))
    assert (m.sharpness, m.brightness, m.contrast) == (0, 0, 0)


def test_laplacian_variance_matches_float64_reference():
    gray = cv2.cvtColor(textured(4), cv2.COLOR_RGB2GRAY)
    ref = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    assert laplacian_variance(gray) == pytest.approx(ref, rel=1e-4)


# --------------------------------------------------------------------------- vision
class FakeVision:
    def __init__(self, favourites: dict[str, float] | None = None):
        self.favourites = favourites or {}
        self.calls = 0

    def score(self, rgb, labels):
        self.calls += 1
        return [self.favourites.get(lab, 0.1) for lab in labels]


def test_one_backend_call_per_image_and_labels_split_correctly():
    fake = FakeVision(
        {
            "a professional photograph": 0.9,
            "a poorly composed photograph": 0.8,
            "an ordinary photograph": 0.7,
            "a food photograph": 0.6,
        }
    )
    analyzer = VisionAnalyzer(VisionConfig(), backend=fake)
    analyzer.setup()
    out = analyzer.analyze(ImageContext(Path("x"), textured(1)))
    assert fake.calls == 1  # v0.1 needed four model passes
    assert out.photographic_quality.label == "a professional photograph"
    assert out.photographic_quality.confidence == 90.0
    assert out.composition.label == "a poorly composed photograph"
    assert out.aesthetic.label == "an ordinary photograph"
    assert out.content.label == "a food photograph"
    assert set(out.content.scores) <= set(ALL_LABELS)


def test_build_assessment_rejects_wrong_length():
    with pytest.raises(ValueError):
        build_assessment([0.5] * 3)


def test_vision_fingerprint_tracks_model_name():
    a = VisionAnalyzer(VisionConfig(model_name="m1"), backend=FakeVision())
    b = VisionAnalyzer(VisionConfig(model_name="m2"), backend=FakeVision())
    assert a.fingerprint() != b.fingerprint()


def test_vision_without_torch_is_unavailable_not_a_crash(monkeypatch):
    import builtins

    real = builtins.__import__

    def blocked(name, *a, **k):
        if name in ("torch", "transformers"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(AnalyzerUnavailable, match=r"ai-media-culler\[ai\]"):
        VisionAnalyzer(VisionConfig()).setup()


# ---------------------------------------------------------------------------- faces
def synthetic_face(cx=0.5, cy=0.5, size=0.2, eye_open=0.3, turn=0.0) -> np.ndarray:
    pts = np.zeros((478, 3))
    pts[:, 0] = cx + (np.arange(478) % 7 - 3) * size / 14
    pts[:, 1] = cy + (np.arange(478) % 5 - 2) * size / 10
    pts[234] = (cx - size / 2, cy, 0)
    pts[454] = (cx + size / 2, cy, 0)
    pts[1] = (cx + turn * size / 2, cy, 0)
    for idx in (LEFT_EYE, RIGHT_EYE):
        pts[idx[0]], pts[idx[3]] = (0.4, 0.4, 0), (0.5, 0.4, 0)  # width 0.1
        pts[idx[1]], pts[idx[5]] = (
            (0.42, 0.4 - eye_open * 0.05, 0),
            (0.42, 0.4 + eye_open * 0.05, 0),
        )
        pts[idx[2]], pts[idx[4]] = (
            (0.47, 0.4 - eye_open * 0.05, 0),
            (0.47, 0.4 + eye_open * 0.05, 0),
        )
    return pts


def test_eye_aspect_ratio_open_vs_closed():
    assert eye_openness(synthetic_face(eye_open=0.3)).average == pytest.approx(0.3, abs=0.01)
    assert eye_openness(synthetic_face(eye_open=0.0)).average == 0.0


def test_pose_classification():
    assert face_pose(synthetic_face(turn=0.0)).orientation == "frontal"
    assert (
        face_pose(synthetic_face(turn=0.25)).orientation == "slightly_turned"
    )  # symmetry = 1 - turn
    assert face_pose(synthetic_face(turn=0.95)).orientation == "strongly_turned"


def test_build_face_analysis_geometry_and_clamping():
    ctx = ImageContext(Path("f"), textured(1, (800, 600)))
    out = build_face_analysis([synthetic_face(cx=0.5, cy=0.5)], ctx)
    f = out.faces[0]
    assert out.face_count == 1 and 0 < f.area_ratio < 1
    assert 0 <= f.x < f.x + f.width <= 800 and 0 <= f.y < f.y + f.height <= 600
    edge = build_face_analysis([synthetic_face(cx=0.99, cy=0.99)], ctx).faces[0]
    assert edge.x + edge.width <= 800 and edge.y + edge.height <= 600


def test_partial_meshes_are_ignored():
    ctx = ImageContext(Path("f"), textured(1))
    assert build_face_analysis([np.zeros((10, 3))], ctx).face_count == 0


def test_face_analyzer_uses_injected_backend():
    class Backend:
        def detect(self, rgb):
            return [synthetic_face(), synthetic_face(cx=0.2)]

    analyzer = FaceAnalyzer(Settings(), backend=Backend())
    analyzer.setup()
    assert analyzer.analyze(ImageContext(Path("f"), textured(1))).face_count == 2


def test_missing_face_model_gives_actionable_error(tmp_path):
    s = Settings(paths={"models_directory": tmp_path})
    with pytest.raises(AnalyzerUnavailable, match="culler models download"):
        FaceAnalyzer(s).setup()
