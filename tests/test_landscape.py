import cv2
import numpy as np
import pytest

from conftest import landscape_image, save, textured
from media_culler.analyzers.scene import (
    SceneAnalyzer,
    detect_horizon,
    saliency_centroid,
    tiled_sharpness,
)
from media_culler.analyzers.technical import TechnicalAnalyzer
from media_culler.analyzers.vision import VisionAnalyzer
from media_culler.config import SceneConfig, Settings
from media_culler.imaging import ImageContext
from media_culler.labels import LANDSCAPE_LABEL
from media_culler.models import (
    AIAssessment,
    EyeMetrics,
    FaceAnalysis,
    FaceInfo,
    HorizonInfo,
    ImageRecord,
    LabelPrediction,
    PoseMetrics,
    SceneMetrics,
    SceneType,
    TechnicalMetrics,
)
from media_culler.pipeline import Pipeline
from media_culler.scoring import landscape as ls
from media_culler.scoring import score_record
from media_culler.scoring.scene import classify_scene
from media_culler.scoring.technical import score_technical
from test_analyzers import FakeVision


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# ------------------------------------------------------------------ horizon detection
@pytest.mark.parametrize("horizon", [0.33, 0.5, 0.67])
@pytest.mark.parametrize("tilt", [0.0, 1.0, -2.5, 6.0])
def test_horizon_tilt_and_position_are_recovered(horizon, tilt):
    found = detect_horizon(gray(landscape_image(horizon=horizon, tilt=tilt)))
    assert found is not None and found.confidence > 0.4
    assert found.angle_deg == pytest.approx(-tilt, abs=0.35)  # cv2 CCW-positive vs ours CW-positive
    assert found.y == pytest.approx(horizon, abs=0.03)


def test_tilt_sign_convention_right_side_lower_is_positive():
    found = detect_horizon(gray(landscape_image(tilt=-3.0)))  # clockwise: right side drops
    assert found is not None and found.angle_deg > 2.5


def smooth_noise(seed=3, size=(800, 1200)):
    noise = np.random.default_rng(seed).normal(128, 40, (*size, 3)).astype(np.float32)
    return np.clip(cv2.GaussianBlur(noise, (0, 0), 3), 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((800, 1200, 3), np.uint8),
        np.tile((np.arange(1200) // 100 % 2 * 200).astype(np.uint8)[None, :, None], (800, 1, 3)),
        cv2.circle(np.full((800, 1200, 3), 90, np.uint8), (600, 400), 150, (230, 200, 180), -1),
        smooth_noise(),
    ],
    ids=["black", "vertical-bars", "centered-blob", "texture"],
)
def test_non_landscapes_do_not_yield_a_confident_horizon(image):
    found = detect_horizon(gray(image))
    assert found is None or found.confidence < SceneConfig().heuristic_horizon_confidence


def test_steeply_slanted_lines_are_not_horizons():
    assert detect_horizon(gray(landscape_image(tilt=25.0))) is None


# --------------------------------------------------------------------- other metrics
def test_tiled_sharpness_ignores_smooth_regions_but_not_blur():
    rng = np.random.default_rng(0)
    half = np.full((800, 1200), 128, np.uint8)
    half[400:] = rng.integers(0, 255, (400, 1200), dtype=np.uint8)  # detail only in lower half
    global_variance = float(cv2.Laplacian(half, cv2.CV_64F).var())
    assert tiled_sharpness(half) > 1.6 * global_variance

    blurred = cv2.GaussianBlur(half, (0, 0), 6)
    assert tiled_sharpness(blurred) < 0.02 * tiled_sharpness(half)  # blur is soft in every tile


def test_tiled_sharpness_handles_tiny_images():
    assert tiled_sharpness(np.zeros((10, 10), np.uint8)) == 0.0


def test_clipping_and_dynamic_range():
    image = np.full((400, 600, 3), 100, np.uint8)
    image[:100] = 255  # 25% blown
    image[300:] = 0  # 25% crushed
    m = SceneAnalyzer().analyze(ImageContext(None, image))  # type: ignore[arg-type]
    assert m.highlight_clip == pytest.approx(0.25, abs=0.01)
    assert m.shadow_clip == pytest.approx(0.25, abs=0.01)
    assert m.dynamic_range > 240


def test_saliency_finds_an_off_centre_subject():
    image = np.full((600, 900, 3), 110, np.uint8)
    cv2.circle(image, (700, 200), 60, (250, 250, 250), -1)
    x, y = saliency_centroid(gray(image))
    assert x > 0.6 and y < 0.4


# ----------------------------------------------------------------------- scoring rules
def scene(**kw) -> SceneMetrics:
    base = dict(
        horizon=HorizonInfo(angle_deg=0.0, y=1 / 3, confidence=0.9),
        highlight_clip=0.0,
        shadow_clip=0.0,
        dynamic_range=200.0,
        sharpness_focus=400.0,
        saliency_x=1 / 3,
        saliency_y=1 / 3,
    )
    base.update(kw)
    return SceneMetrics(**base)


def test_level_horizon_beats_tilted_symmetrically():
    scores = [ls.horizon_level_score(t) for t in (0, 0.3, 1, 3, 6, 12)]
    assert scores[0] == scores[1] == 100
    assert scores == sorted(scores, reverse=True) and scores[-1] == 10
    assert ls.horizon_level_score(-2) == ls.horizon_level_score(2)


def test_horizon_placement_prefers_thirds_over_centre_over_edge():
    third, centre, edge = (ls.horizon_placement_score(y) for y in (1 / 3, 0.5, 0.03))
    assert third == 100 and 60 < centre < 80 and edge < 50 and edge < centre


def test_exposure_is_clipping_based_not_mean_brightness():
    sunset = ls.exposure_score(scene(), brightness=60)  # dark but nothing clipped
    assert sunset >= 95
    from media_culler.scoring.technical import exposure_score as portrait_rule

    assert portrait_rule(60) < 60  # what the generic rule would have said
    assert ls.exposure_score(scene(highlight_clip=0.2), 130) < 40  # blown sky
    assert ls.exposure_score(scene(shadow_clip=0.5), 130) < 60
    assert ls.exposure_score(scene(), 250) < 70  # absurdly bright overall


def test_dynamic_range_flat_vs_full():
    assert ls.dynamic_range_score(40) == 25 and ls.dynamic_range_score(200) == 100


def test_tiled_sharpness_rescues_a_crisp_landscape_with_empty_sky():
    technical = TechnicalMetrics(
        sharpness=40, brightness=125, contrast=60, megapixels=12, perceptual_hash="0" * 16
    )
    generic, _ = score_technical(technical)
    overrides, info = ls.technical_overrides(technical, scene(sharpness_focus=450))
    landscape, parts = score_technical(technical, overrides)
    assert landscape > generic + 15 and info["sharpness_source"] == "tiled"
    assert parts["sharpness"] > 85


def test_tiled_sharpness_never_lowers_a_score():
    technical = TechnicalMetrics(
        sharpness=300, brightness=125, contrast=60, megapixels=12, perceptual_hash="0" * 16
    )
    _, info = ls.technical_overrides(technical, scene(sharpness_focus=100))
    assert info["sharpness_source"] == "global" and info["sharpness_used"] == 300


def test_composition_without_a_confident_horizon_uses_focal_point_only():
    weak = scene(horizon=HorizonInfo(angle_deg=9, y=0.5, confidence=0.1))
    score, details = ls.score_landscape_composition(weak, SceneConfig())
    assert set(details["parts"]) == {"focal_point"} and details["horizon"] == "none detected"
    assert score == details["parts"]["focal_point"]


def test_composition_penalises_a_tilted_horizon():
    config = SceneConfig()
    level, _ = ls.score_landscape_composition(scene(), config)
    tilted, details = ls.score_landscape_composition(
        scene(horizon=HorizonInfo(angle_deg=4.0, y=1 / 3, confidence=0.9)), config
    )
    assert tilted < level - 15 and details["horizon_tilt_deg"] == 4.0


# ------------------------------------------------------------------ scene classification
def face(area: float) -> FaceInfo:
    return FaceInfo(
        x=10,
        y=10,
        width=50,
        height=50,
        area_ratio=area,
        sharpness=300,
        eyes=EyeMetrics(left=0.3, right=0.3, average=0.3),
        pose=PoseMetrics(symmetry=1.0, orientation="frontal"),
    )


def pred(label: str, confidence: float = 60.0) -> LabelPrediction:
    return LabelPrediction(label=label, confidence=confidence, scores={label: confidence / 100})


def ai(content: str, confidence: float = 60.0) -> AIAssessment:
    other = pred("a professional photograph")
    return AIAssessment(
        content=pred(content, confidence),
        photographic_quality=other,
        composition=pred("a well composed photograph"),
        aesthetic=pred("a visually striking photograph"),
    )


def record(**kw) -> ImageRecord:
    base = dict(
        path="/x/a.jpg",
        relative_path="a.jpg",
        filename="a.jpg",
        extension=".jpg",
        width=1200,
        height=800,
        technical=TechnicalMetrics(
            sharpness=300, brightness=125, contrast=60, megapixels=1, perceptual_hash="0" * 16
        ),
    )
    base.update(kw)
    return ImageRecord(**base)


def test_classification_rules():
    cfg = SceneConfig()
    landscape_ai = ai(LANDSCAPE_LABEL)
    assert classify_scene(record(ai=landscape_ai), cfg)[0] is SceneType.LANDSCAPE
    assert classify_scene(record(ai=ai("a food photograph")), cfg)[0] is SceneType.GENERAL
    # a distant hiker does not make a landscape a portrait...
    tiny = FaceAnalysis(face_count=1, faces=[face(0.0008)])
    assert classify_scene(record(ai=landscape_ai, faces=tiny), cfg)[0] is SceneType.LANDSCAPE
    # ...but a real face does, whatever the AI says
    big = FaceAnalysis(face_count=1, faces=[face(0.05)])
    assert classify_scene(record(ai=landscape_ai, faces=big), cfg)[0] is SceneType.PEOPLE


def test_landscape_min_confidence_is_respected():
    cfg = SceneConfig(landscape_min_confidence=40)
    assert classify_scene(record(ai=ai(LANDSCAPE_LABEL, 10)), cfg)[0] is SceneType.GENERAL


def test_heuristic_only_applies_without_ai_and_with_a_strong_horizon():
    strong = scene(horizon=HorizonInfo(angle_deg=0, y=0.4, confidence=0.9))
    weak = scene(horizon=HorizonInfo(angle_deg=0, y=0.4, confidence=0.4))
    cfg = SceneConfig()
    assert classify_scene(record(scene=strong), cfg)[0] is SceneType.LANDSCAPE
    assert classify_scene(record(scene=weak), cfg)[0] is SceneType.GENERAL
    # AI ran and said "food": a horizon line must not override it
    assert classify_scene(record(scene=strong, ai=ai("a food photograph")), cfg)[0] is (
        SceneType.GENERAL
    )
    assert classify_scene(record(scene=strong), SceneConfig(heuristic_without_ai=False))[0] is (
        SceneType.GENERAL
    )


# ------------------------------------------------------------------- record-level scoring
def test_landscape_records_exclude_subject_instead_of_scoring_a_neutral_50():
    tiny_faces = FaceAnalysis(face_count=1, faces=[face(0.0008)])
    rec = record(ai=ai(LANDSCAPE_LABEL), scene=scene(), faces=tiny_faces)
    s = score_record(rec, Settings())
    assert s.scene_type is SceneType.LANDSCAPE
    assert s.subject is None and "subject" not in s.weights_used
    assert set(s.weights_used) == {"technical", "ai", "composition"}
    assert s.composition is not None and s.composition > 90  # level, thirds, focal point


def test_non_landscape_scoring_is_unchanged():
    rec = record(ai=ai("a food photograph"), faces=FaceAnalysis(face_count=0))
    s = score_record(rec, Settings())
    assert s.scene_type is SceneType.GENERAL
    assert s.subject == 50.0 and s.composition == 50.0  # legacy neutral behaviour kept


def test_landscape_without_scene_metrics_degrades_gracefully():
    s = score_record(record(ai=ai(LANDSCAPE_LABEL)), Settings())
    assert s.scene_type is SceneType.LANDSCAPE and s.composition is None
    assert s.final is not None and set(s.weights_used) == {"technical", "ai"}


# ------------------------------------------------------------------------- end to end
def landscape_stack(settings, vision):
    return [
        TechnicalAnalyzer(),
        SceneAnalyzer(),
        VisionAnalyzer(settings.vision, backend=vision),
    ]


def test_level_landscape_outscores_tilted_one_end_to_end(settings, tmp_path):
    root = tmp_path / "in"
    save(landscape_image(horizon=0.4, seed=5), root / "level.jpg", quality=95)
    save(landscape_image(horizon=0.4, tilt=4.0, seed=5), root / "tilted.jpg", quality=95)
    result = Pipeline(
        settings, analyzers=landscape_stack(settings, FakeVision({LANDSCAPE_LABEL: 0.9}))
    ).run(root)
    by = {r.filename: r for r in result.records}
    assert {r.scores.scene_type for r in by.values()} == {SceneType.LANDSCAPE}
    assert by["level.jpg"].scores.composition > by["tilted.jpg"].scores.composition + 20
    assert by["level.jpg"].scores.final > by["tilted.jpg"].scores.final
    assert by["tilted.jpg"].scores.details["composition"]["horizon_tilt_deg"] == pytest.approx(
        -4.0, abs=0.5
    )


def test_scene_results_are_cached(settings, tmp_path):
    root = tmp_path / "in"
    save(landscape_image(seed=1), root / "a.jpg")
    first = Pipeline(settings, analyzers=landscape_stack(settings, FakeVision())).run(root)
    again = Pipeline(settings, analyzers=landscape_stack(settings, FakeVision())).run(root)
    assert first.cache_hits == 0 and again.cache_misses == 0
    assert again.records[0].scene == first.records[0].scene


def test_default_stack_includes_scene_and_can_be_disabled(settings, tmp_path):
    from media_culler.analyzers import build_analyzers

    assert "scene" in {a.name for a in build_analyzers(settings)}
    settings.scene.enabled = False
    assert "scene" not in {a.name for a in build_analyzers(settings)}


def test_plain_texture_photos_are_not_mistaken_for_landscapes(settings, tmp_path):
    root = tmp_path / "in"
    save(textured(1), root / "texture.jpg")
    result = Pipeline(settings).run(root)
    assert result.records[0].scores.scene_type is SceneType.GENERAL


# ------------------------------------------------------------------- soft-focus gate
def blurred_landscape_pair(tmp_path):
    root = tmp_path / "in"
    sharp = landscape_image(horizon=0.4, seed=4, detail=2)
    save(sharp, root / "sharp.jpg", quality=95)
    save(cv2.GaussianBlur(sharp, (0, 0), 6), root / "blurred.jpg", quality=95)
    return root


def test_blurred_landscape_cannot_be_rescued_by_good_composition(settings, tmp_path):
    """Regression guard: landscape composition rewards a level horizon, so without the
    gate a totally out-of-focus frame was promoted from review to usable."""
    from media_culler.models import Recommendation

    root = blurred_landscape_pair(tmp_path)
    result = Pipeline(
        settings, analyzers=landscape_stack(settings, FakeVision({LANDSCAPE_LABEL: 0.9}))
    ).run(root)
    by = {r.filename: r for r in result.records}
    assert by["sharp.jpg"].recommendation in {Recommendation.SELECT, Recommendation.USABLE}
    assert by["blurred.jpg"].recommendation is Recommendation.REVIEW
    assert "soft" in by["blurred.jpg"].recommendation_reason.lower()
    assert by["sharp.jpg"].scores.final > by["blurred.jpg"].scores.final + 10


def test_soft_focus_gate_is_landscape_only_and_configurable(settings, tmp_path):
    from media_culler.config import ThresholdsConfig
    from media_culler.curation import classify
    from media_culler.models import Recommendation

    root = blurred_landscape_pair(tmp_path)
    landscape = Pipeline(
        settings, analyzers=landscape_stack(settings, FakeVision({LANDSCAPE_LABEL: 0.9}))
    ).run(root)
    blurred = next(r for r in landscape.records if r.filename == "blurred.jpg")
    assert (
        classify(blurred, ThresholdsConfig(soft_focus_sharpness=0))[0]
        is not (Recommendation.REVIEW)
        or blurred.scores.final < 65
    )  # gate off: ordinary thresholds apply

    blurred.scores.scene_type = SceneType.GENERAL  # same pixels, not a landscape
    assert "soft" not in classify(blurred, ThresholdsConfig())[1].lower()
