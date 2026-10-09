from pathlib import Path

import pytest

from media_culler.analyzers import Analyzer
from media_culler.analyzers.faces import FaceAnalyzer
from media_culler.analyzers.technical import TechnicalAnalyzer
from media_culler.analyzers.vision import VisionAnalyzer
from media_culler.exceptions import AnalyzerUnavailable
from media_culler.models import GroupRole, Recommendation
from media_culler.pipeline import Pipeline
from test_analyzers import FakeVision, synthetic_face


class Recorder:
    def __init__(self):
        self.stages, self.advances, self.warnings = [], 0, []

    def stage(self, name, total=None):
        self.stages.append((name, total))

    def advance(self, detail=""):
        self.advances += 1

    def warn(self, message):
        self.warnings.append(message)


class FakeFaces:
    def detect(self, rgb):
        return [synthetic_face()]


def full_stack(settings, vision=None, faces=None):
    return [
        TechnicalAnalyzer(),
        VisionAnalyzer(settings.vision, backend=vision or FakeVision()),
        FaceAnalyzer(settings, backend=faces or FakeFaces()),
    ]


def test_end_to_end_technical_only(settings, photo_dir):
    result = Pipeline(settings).run(photo_dir)
    by = {r.filename: r for r in result.records}

    assert by["corrupt.jpg"].status == "error"
    ok = [r for r in result.records if r.status == "valid"]
    assert len(ok) == 3 and all(r.technical and r.scores and r.scores.final is not None for r in ok)
    assert by["sharp.jpg"].scores.final > by["blurry.jpg"].scores.final
    assert {r.collection_rank for r in ok} == {1, 2, 3}
    assert all(r.recommendation in set(Recommendation) for r in ok)
    # sharp + its blurred twin are one group; the sharper one represents it
    assert by["sharp.jpg"].duplicate_group == by["blurry.jpg"].duplicate_group is not None
    assert by["sharp.jpg"].group_role is GroupRole.REPRESENTATIVE
    assert by["blurry.jpg"].group_role is GroupRole.SIMILAR
    assert by["other.png"].group_role is GroupRole.UNIQUE


def test_full_stack_with_fakes_populates_every_section(settings, photo_dir):
    result = Pipeline(settings, analyzers=full_stack(settings)).run(photo_dir)
    r = next(r for r in result.records if r.filename == "sharp.jpg")
    assert r.ai and r.faces and r.faces.face_count == 1
    assert set(r.scores.weights_used) == {"technical", "ai", "subject", "composition"}


def test_cache_makes_second_run_free(settings, photo_dir):
    vision = FakeVision()
    first = Pipeline(settings, analyzers=full_stack(settings, vision)).run(photo_dir)
    assert first.cache_hits == 0 and vision.calls == 3

    vision2 = FakeVision()
    second = Pipeline(settings, analyzers=full_stack(settings, vision2)).run(photo_dir)
    assert vision2.calls == 0  # nothing re-analysed
    assert second.cache_hits == 3 * 3 and second.cache_misses == 0
    a = {r.filename: r.scores.final for r in first.records if r.scores}
    b = {r.filename: r.scores.final for r in second.records if r.scores}
    assert a == b  # identical results from cache


def test_no_cache_flag_forces_reanalysis(settings, photo_dir):
    settings.performance.use_cache = False
    v = FakeVision()
    Pipeline(settings, analyzers=full_stack(settings, v)).run(photo_dir)
    Pipeline(settings, analyzers=full_stack(settings, v)).run(photo_dir)
    assert v.calls == 6


def test_changing_weights_does_not_require_reanalysis(settings, photo_dir):
    v = FakeVision()
    Pipeline(settings, analyzers=full_stack(settings, v)).run(photo_dir)
    settings.weights.technical = 0.9
    v2 = FakeVision()
    result = Pipeline(settings, analyzers=full_stack(settings, v2)).run(photo_dir)
    assert v2.calls == 0 and result.cache_misses == 0


def test_analyzer_failure_is_isolated_per_image(settings, photo_dir):
    class Flaky(FakeVision):
        seen = 0

        def score(self, rgb, labels):
            Flaky.seen += 1
            if Flaky.seen == 2:
                raise RuntimeError("CUDA out of memory")
            return super().score(rgb, labels)

    result = Pipeline(settings, analyzers=full_stack(settings, vision=Flaky())).run(photo_dir)
    valid = [r for r in result.records if r.status == "valid"]
    failed = [r for r in valid if "vision" in r.analyzer_errors]
    assert len(failed) == 1 and "CUDA out of memory" in failed[0].analyzer_errors["vision"]
    assert failed[0].ai is None
    # the failed image is still scored from its other components, not zeroed
    assert failed[0].scores.final is not None and "ai" not in failed[0].scores.weights_used
    assert len(valid) == 3


def test_failed_analysis_is_not_cached(settings, photo_dir):
    class Broken(FakeVision):
        def score(self, rgb, labels):
            raise RuntimeError("boom")

    Pipeline(settings, analyzers=full_stack(settings, vision=Broken())).run(photo_dir)
    healthy = FakeVision()
    Pipeline(settings, analyzers=full_stack(settings, vision=healthy)).run(photo_dir)
    assert healthy.calls == 3


def test_unavailable_analyzer_is_skipped_with_warning(settings, photo_dir):
    class Missing(Analyzer):
        name, section = "faces", "faces"
        result_type = TechnicalAnalyzer.result_type

        def setup(self):
            raise AnalyzerUnavailable("model missing")

        def analyze(self, ctx):  # pragma: no cover
            raise AssertionError

    rec = Recorder()
    result = Pipeline(settings, rec, analyzers=[TechnicalAnalyzer(), Missing()]).run(photo_dir)
    assert result.skipped_analyzers == {"faces": "model missing"}
    assert any("model missing" in w for w in rec.warnings)
    assert all(r.scores.final is not None for r in result.records if r.status == "valid")


def test_strict_mode_raises_on_unavailable(settings, photo_dir):
    s = settings.model_copy(deep=True)
    s.faces.enabled = True  # no model file in tmp models dir
    with pytest.raises(AnalyzerUnavailable):
        Pipeline(s, strict=True).run(photo_dir)


def test_file_corrupted_after_scan_becomes_error_record(settings, photo_dir, monkeypatch):
    """Header reads fine at scan time, pixel decode fails later: must not abort the run."""
    from media_culler import imaging

    real = imaging.load_rgb

    def flaky(path, max_pixels):
        if Path(path).name == "other.png":
            raise imaging.ImageLoadError("Cannot decode image: simulated")
        return real(path, max_pixels)

    monkeypatch.setattr(imaging, "load_rgb", flaky)
    result = Pipeline(settings).run(photo_dir)
    bad = next(r for r in result.records if r.filename == "other.png")
    assert bad.status == "error" and "simulated" in bad.error
    assert sum(r.status == "valid" for r in result.records) == 2


def test_progress_events_are_emitted(settings, photo_dir):
    rec = Recorder()
    Pipeline(settings, rec).run(photo_dir)
    names = [n for n, _ in rec.stages]
    assert (
        names[:2] == ["Scanning", "Reading metadata"]
        and "Analyzing" in names
        and "Scoring" in names
    )
    assert ("Analyzing", 3) in rec.stages and rec.advances >= 3 + 4


def test_empty_directory(settings, tmp_path):
    (tmp_path / "in").mkdir()
    result = Pipeline(settings).run(tmp_path / "in")
    assert result.records == [] and result.duplicate_groups == 0


def test_duplicates_can_be_disabled(settings, photo_dir):
    settings.duplicates.enabled = False
    result = Pipeline(settings).run(photo_dir)
    assert all(r.duplicate_group is None for r in result.records)
    assert all(r.group_role is GroupRole.UNIQUE for r in result.records if r.status == "valid")


def test_identical_files_share_one_analysis(settings, tmp_path):
    from conftest import save, textured

    root = tmp_path / "in"
    img = textured(5)
    save(img, root / "a.png")
    save(img, root / "b.png")
    v = FakeVision()
    Pipeline(
        settings,
        analyzers=[TechnicalAnalyzer(), VisionAnalyzer(settings.vision, backend=v)],
    ).run(root)
    assert v.calls <= 2  # bounded; the cache can only help
