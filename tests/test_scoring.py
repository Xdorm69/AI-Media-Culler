import pytest

from media_culler.config import Settings, WeightsConfig
from media_culler.models import (
    AIAssessment,
    EyeMetrics,
    FaceAnalysis,
    FaceInfo,
    ImageRecord,
    LabelPrediction,
    PoseMetrics,
    TechnicalMetrics,
)
from media_culler.scoring import composition as comp
from media_culler.scoring import score_record
from media_culler.scoring import technical as tech
from media_culler.scoring.final import combine
from media_culler.scoring.subject import score_subject


def test_sharpness_boundaries_match_v01():
    assert tech.sharpness_score(0) == 10
    assert tech.sharpness_score(50) == 10
    assert tech.sharpness_score(500) == 100
    assert tech.sharpness_score(275) == pytest.approx(55)


def test_exposure_peaks_at_125_and_is_clamped():
    assert tech.exposure_score(125) == 100
    assert tech.exposure_score(0) == 0
    assert tech.exposure_score(255) == 0


def test_resolution_and_contrast_ramps():
    assert tech.resolution_score(1) == 20 and tech.resolution_score(12) == 100
    assert tech.resolution_score(7) == pytest.approx(60)
    assert tech.contrast_score(20) == 15 and tech.contrast_score(80) == 100


def test_technical_weighting():
    m = TechnicalMetrics(
        sharpness=500, brightness=125, contrast=80, megapixels=12, perceptual_hash="0" * 16
    )
    assert tech.score_technical(m)[0] == 100.0


def _face(**kw) -> FaceInfo:
    base = dict(
        x=400,
        y=300,
        width=200,
        height=200,
        area_ratio=0.05,
        sharpness=500,
        eyes=EyeMetrics(left=0.3, right=0.3, average=0.3),
        pose=PoseMetrics(symmetry=1.0, orientation="frontal"),
    )
    base.update(kw)
    return FaceInfo(**base)


def test_perfect_face_scores_100():
    s, detail = score_subject(FaceAnalysis(face_count=1, faces=[_face()]))
    assert s == 100.0 and detail["reason"] == "Single subject"


def test_no_faces_is_neutral_not_penalised():
    assert score_subject(FaceAnalysis(face_count=0))[0] == 50.0


def test_two_faces_weight_the_weaker_one():
    good, bad = _face(), _face(eyes=EyeMetrics(left=0.1, right=0.1, average=0.1))
    s, _ = score_subject(FaceAnalysis(face_count=2, faces=[good, bad]))
    only_good = score_subject(FaceAnalysis(face_count=1, faces=[good]))[0]
    only_bad = score_subject(FaceAnalysis(face_count=1, faces=[bad]))[0]
    assert only_bad < s < only_good
    assert s == pytest.approx(only_good * 0.4 + only_bad * 0.6, abs=0.02)


def test_edge_safety_penalises_cropped_faces():
    centered = _face(x=450, y=350)
    clipped = _face(x=0, y=0)
    assert comp.edge_safety_score(centered, 1000, 800) == 100
    assert comp.edge_safety_score(clipped, 1000, 800) == 50


def test_occupancy_curve_peaks_mid_range():
    assert comp.occupancy_score(0.001) == 30
    assert comp.occupancy_score(0.15) == pytest.approx(100)
    assert comp.occupancy_score(0.5) == 60


def test_combine_renormalises_over_available_components():
    w = WeightsConfig()
    full, used = combine({"technical": 80, "ai": 60, "subject": 50, "composition": 50}, w)
    assert full == pytest.approx(80 * 0.4 + 60 * 0.2 + 50 * 0.2 + 50 * 0.2)
    assert sum(used.values()) == pytest.approx(1.0, abs=1e-3)

    # AI unavailable => excluded, NOT scored as 0 (the v0.1 behaviour)
    partial, used = combine({"technical": 80, "ai": None, "subject": 50, "composition": 50}, w)
    assert "ai" not in used
    assert partial == pytest.approx((80 * 0.4 + 50 * 0.2 + 50 * 0.2) / 0.8, abs=0.01)


def test_combine_requires_technical():
    assert combine({"technical": None, "ai": 90}, WeightsConfig()) == (None, {})


def test_score_record_end_to_end_with_all_sections():
    def pred(label):
        return LabelPrediction(label=label, confidence=90, scores={label: 0.9})

    record = ImageRecord(
        path="/x/a.jpg",
        relative_path="a.jpg",
        filename="a.jpg",
        extension=".jpg",
        width=1000,
        height=800,
        technical=TechnicalMetrics(
            sharpness=500, brightness=125, contrast=80, megapixels=12, perceptual_hash="f" * 16
        ),
        ai=AIAssessment(
            content=pred("a photograph of people"),
            photographic_quality=pred("a professional photograph"),
            composition=pred("a well composed photograph"),
            aesthetic=pred("a visually striking photograph"),
        ),
        faces=FaceAnalysis(face_count=0),
    )
    s = score_record(record, Settings())
    assert s.technical == 100 and s.ai == 100 and s.subject == 50 and s.composition == 50
    assert s.final == pytest.approx(100 * 0.4 + 100 * 0.2 + 50 * 0.4)
