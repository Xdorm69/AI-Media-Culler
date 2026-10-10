from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import save_timed
from media_culler.analyzers.technical import TechnicalAnalyzer, perceptual_hash
from media_culler.config import BurstConfig, SelectionConfig
from media_culler.curation import select_best_shots
from media_culler.duplicates import assign_duplicate_groups, hamming_distance
from media_culler.imaging import parse_capture_time, read_header
from media_culler.models import (
    GroupAssessment,
    GroupKind,
    GroupRole,
    ImageRecord,
    Scores,
    TechnicalMetrics,
)
from media_culler.pipeline import Pipeline
from media_culler.reporting import build_report
from media_culler.scanner import scan_directory

T0 = datetime(2026, 6, 12, 7, 7, 26)
BURST = BurstConfig()


# ------------------------------------------------------------------- capture time
def exif_with(**tags) -> Image.Exif:
    exif = Image.Exif()
    ifd = exif.get_ifd(0x8769)
    for tag, value in tags.items():
        ifd[int(tag.removeprefix("t"), 16)] = value
    return exif


def test_capture_time_with_subseconds():
    exif = exif_with(t9003="2026:06:12 07:07:26", t9291="45")
    assert parse_capture_time(exif) == "2026-06-12T07:07:26.450000"
    assert parse_capture_time(exif_with(t9003="2026:06:12 07:07:26", t9291="123")) == (
        "2026-06-12T07:07:26.123000"
    )


def test_capture_time_fallbacks_and_garbage():
    assert parse_capture_time(exif_with(t9004="2026:06:12 07:07:26")) == "2026-06-12T07:07:26"
    original_wins = exif_with(t9003="2026:06:12 07:07:26", t9004="2020:01:01 00:00:00")
    assert parse_capture_time(original_wins) == "2026-06-12T07:07:26"
    for junk in ("0000:00:00 00:00:00", "not a date", "", "2026-06-12 07:07:26"):
        assert parse_capture_time(exif_with(t9003=junk)) is None
    assert parse_capture_time(Image.Exif()) is None
    assert parse_capture_time(exif_with(t9003="2026:06:12 07:07:26", t9291="xx")) == (
        "2026-06-12T07:07:26"
    )  # unreadable sub-seconds: keep the whole second


def test_header_and_scanner_expose_capture_time(settings, tmp_path):
    root = tmp_path / "in"
    pixels = np.zeros((40, 60, 3), np.uint8)
    save_timed(pixels, root / "timed.jpg", "2026:06:12 07:07:26", "5")
    save_timed(pixels, root / "untimed.jpg", None)
    assert read_header(root / "timed.jpg", 10**9).captured_at == "2026-06-12T07:07:26.500000"
    by = {r.filename: r for r in scan_directory(root, settings)}
    assert by["timed.jpg"].captured_at == "2026-06-12T07:07:26.500000"
    assert by["untimed.jpg"].captured_at is None  # never guessed from file mtime


# ---------------------------------------------------------------------- grouping
def shot(name, bits=0, at=0.0, sha=None, timed=True) -> ImageRecord:
    """A frame whose perceptual hash is ``bits`` low bits set, captured ``at`` seconds after T0."""
    r = ImageRecord(path=f"/p/{name}", relative_path=name, filename=name, extension=".jpg")
    r.sha256 = sha or f"sha-{name}"
    r.captured_at = (T0 + timedelta(seconds=at)).isoformat() if timed else None
    r.technical = TechnicalMetrics(
        sharpness=300,
        brightness=125,
        contrast=60,
        megapixels=12,
        perceptual_hash=f"{(1 << bits) - 1:016x}",
    )
    return r


def group_of(records):
    return {r.filename: r.duplicate_group for r in records}


def test_hash_helper_distances_are_what_the_tests_assume():
    assert (
        hamming_distance(
            shot("a", 0).technical.perceptual_hash, shot("b", 10).technical.perceptual_hash
        )
        == 10
    )


def test_consecutive_similar_frames_form_a_burst_a_plain_duplicate_pass_would_miss():
    frames = [shot("a", 0, 0.0), shot("b", 10, 0.4)]  # 10 bits apart: more than duplicates allow
    assert assign_duplicate_groups(frames, 8) == 0
    assert assign_duplicate_groups(frames, 8, BURST) == 1
    assert frames[0].group_kind is GroupKind.BURST and frames[0].group_span_seconds == 0.4


def test_the_time_gap_limit_is_respected():
    frames = [shot("a", 0, 0.0), shot("b", 10, 2.5)]
    assert assign_duplicate_groups(frames, 8, BURST) == 0
    frames = [shot("a", 0, 0.0), shot("b", 10, 2.0)]  # exactly at the limit still links
    assert assign_duplicate_groups(frames, 8, BURST) == 1


def test_unrelated_frames_at_the_same_instant_are_not_a_burst():
    frames = [shot("a", 0, 0.0), shot("b", 25, 0.0)]
    assert assign_duplicate_groups(frames, 8, BURST) == 0


def test_chaining_follows_a_pan_where_first_and_last_look_different():
    a, b, c = shot("a", 0, 0.0), shot("b", 10, 0.5), shot("c", 20, 1.0)
    c.technical.perceptual_hash = f"{(1 << 20) - 1:016x}"
    assert hamming_distance(a.technical.perceptual_hash, c.technical.perceptual_hash) == 20
    assert assign_duplicate_groups([a, b, c], 8, BURST) == 1
    assert len({a.duplicate_group, b.duplicate_group, c.duplicate_group}) == 1


def test_long_chains_are_cut_at_max_span():
    frames = [shot(f"f{i:02d}", 0, float(i)) for i in range(0, 31)]  # one a second for 30 s
    # identical hashes, so appearance alone would join all; isolate the time rule:
    assert assign_duplicate_groups(frames, None, BurstConfig(max_span_seconds=10)) >= 3
    sizes = {r.duplicate_group: 0 for r in frames}
    for r in frames:
        sizes[r.duplicate_group] += 1
    assert max(sizes.values()) <= 11  # 0..10 s inclusive


def test_another_photographers_frames_do_not_break_a_burst():
    a1, other, a2 = shot("a1", 0, 0.0), shot("x", 30, 0.5), shot("a2", 10, 1.0)
    assert assign_duplicate_groups([a1, other, a2], None, BURST) == 1
    assert a1.duplicate_group == a2.duplicate_group is not None and other.duplicate_group is None


def test_frames_without_capture_time_fall_back_to_appearance_only():
    near = [shot("a", 0, timed=False), shot("b", 5, timed=False)]
    far = [shot("c", 0, timed=False), shot("d", 10, timed=False)]
    assert assign_duplicate_groups(near, 8, BURST) == 1
    assert assign_duplicate_groups(far, 8, BURST) == 0  # no time to justify the looser rule
    assert near[0].group_kind is GroupKind.SIMILAR and near[0].group_span_seconds is None


def test_bursts_can_be_disabled_and_duplicates_can_be_skipped():
    frames = [shot("a", 0, 0.0), shot("b", 10, 0.4)]
    assert assign_duplicate_groups(frames, 8, BurstConfig(enabled=False)) == 0
    assert assign_duplicate_groups(frames, None, BURST) == 1  # bursts alone, no appearance pass
    same_look = [shot("c", 0, 0.0), shot("d", 3, 600.0)]
    assert assign_duplicate_groups(same_look, None, BURST) == 0


def test_group_kinds():
    twins = [shot("a", 0, 0.0, sha="same"), shot("b", 0, 0.0, sha="same")]
    assign_duplicate_groups(twins, 8, BURST)
    assert {r.group_kind for r in twins} == {GroupKind.DUPLICATE}

    burst = [shot("c", 0, 0.0), shot("d", 4, 1.0)]
    assign_duplicate_groups(burst, 8, BURST)
    assert {r.group_kind for r in burst} == {GroupKind.BURST}

    retake = [shot("e", 0, 0.0), shot("f", 4, 3600.0)]  # alike, an hour apart
    assign_duplicate_groups(retake, 8, BURST)
    assert {r.group_kind for r in retake} == {GroupKind.SIMILAR}
    assert retake[0].group_span_seconds == 3600.0


def test_group_ids_are_stable_and_ordered_by_path():
    frames = [shot("z2", 0, 0.0), shot("z1", 3, 0.2), shot("a1", 20, 9.0), shot("a2", 22, 9.3)]
    assign_duplicate_groups(frames, 8, BURST)
    ids = group_of(frames)
    assert ids["a1"] == ids["a2"] == 1 and ids["z1"] == ids["z2"] == 2


def test_errors_and_unhashed_images_are_ignored():
    bad = ImageRecord(
        path="/p/bad", relative_path="bad", filename="bad", extension=".jpg", status="error"
    )
    no_hash = ImageRecord(path="/p/nh", relative_path="nh", filename="nh", extension=".jpg")
    assert assign_duplicate_groups([bad, no_hash, shot("a")], 8, BURST) == 0


# --------------------------------------------------- sharpness-aware selection
def frame(name, selection, sharp, group=1) -> ImageRecord:
    r = shot(name)
    r.duplicate_group = group
    r.technical.sharpness = sharp
    r.scores = Scores(technical=80, final=selection, selection=selection, group=GroupAssessment())
    return r


def test_the_sharper_frame_wins_even_when_absolute_scores_saturate():
    """Frames of a soft scene all hit the sharpness floor and tie on score; only the
    relative measure can tell the motion-blurred one from the crisp one."""
    crisp, smeared = frame("a", 70.0, sharp=40), frame("b", 70.0, sharp=10)
    select_best_shots([smeared, crisp], SelectionConfig())
    assert crisp.group_role is GroupRole.REPRESENTATIVE
    assert smeared.group_role is GroupRole.SIMILAR
    assert smeared.selection_reason == "Softer than the sharpest frame (25% as sharp)"


def test_sharpness_weight_zero_restores_pure_score_ranking():
    crisp, smeared = frame("a", 70.0, sharp=40), frame("b", 71.0, sharp=10)
    select_best_shots([crisp, smeared], SelectionConfig(sharpness_weight=0))
    assert smeared.group_role is GroupRole.REPRESENTATIVE


def test_a_slightly_softer_frame_is_not_called_out_as_soft():
    best, near = frame("a", 80.0, sharp=100), frame("b", 79.0, sharp=90)
    select_best_shots([best, near], SelectionConfig())
    assert near.selection_reason == "Lower selection score than the group's best shot"


def test_landscape_tiled_sharpness_is_used_for_comparison():
    from media_culler.models import SceneMetrics

    def scene(focus):
        return SceneMetrics(
            highlight_clip=0,
            shadow_clip=0,
            dynamic_range=100,
            sharpness_focus=focus,
            saliency_x=0.5,
            saliency_y=0.5,
        )

    a, b = frame("a", 70.0, sharp=5), frame("b", 70.0, sharp=5)  # global sharpness identical
    a.scene, b.scene = scene(400), scene(100)
    select_best_shots([a, b], SelectionConfig())
    assert a.group_role is GroupRole.REPRESENTATIVE


# ---------------------------------------------------------------------- end to end
def panning_scene():
    """Wide scene = smooth structure (what the perceptual hash sees) + fine grain (the detail
    that motion blur destroys, and which Laplacian-variance sharpness measures)."""
    rng = np.random.default_rng(2)
    smooth = cv2.resize(
        rng.integers(0, 255, (13, 40, 3), dtype=np.uint8),
        (1920, 640),
        interpolation=cv2.INTER_CUBIC,
    ).astype(np.float32)
    grain = np.random.default_rng(102).normal(0, 15, smooth.shape)
    return np.clip(smooth + grain, 0, 255).astype(np.uint8)


def test_panning_burst_is_grouped_and_the_motion_blurred_frame_is_dropped(settings, tmp_path):
    wide = panning_scene()
    crops = [np.ascontiguousarray(wide[:, dx : dx + 960]) for dx in (0, 40, 80)]
    smear = np.full((1, 15), 1 / 15, np.float32)  # horizontal motion blur
    crops[1] = cv2.filter2D(crops[1], -1, smear)

    root = tmp_path / "in"
    for index, (crop, subsec) in enumerate(zip(crops, ("10", "35", "60"), strict=True)):
        save_timed(crop, root / f"burst_{index}.png", "2026:06:12 07:07:26", subsec)

    hashes = [perceptual_hash(c) for c in crops]
    assert hamming_distance(hashes[0], hashes[2]) > settings.duplicates.hamming_threshold, (
        "premise: first and last frames must be beyond what a plain duplicate pass would join"
    )

    off = settings.model_copy(deep=True)
    off.bursts.enabled = False
    plain = Pipeline(off, analyzers=[TechnicalAnalyzer()]).run(root)
    assert len({r.duplicate_group for r in plain.records if r.duplicate_group}) <= 1
    assert any(r.duplicate_group is None for r in plain.records)  # not all joined without bursts

    result = Pipeline(settings, analyzers=[TechnicalAnalyzer()]).run(root)
    by = {r.filename: r for r in result.records}
    assert {r.duplicate_group for r in by.values()} == {1}
    assert {r.group_kind for r in by.values()} == {GroupKind.BURST}
    assert by["burst_1.png"].group_span_seconds == pytest.approx(0.5, abs=0.01)
    assert by["burst_1.png"].group_role is GroupRole.SIMILAR
    assert by["burst_1.png"].selection_reason.startswith("Softer than the sharpest frame (")
    kept = [r for r in by.values() if r.group_role is GroupRole.REPRESENTATIVE]
    assert len(kept) == 1 and kept[0].filename in {"burst_0.png", "burst_2.png"}

    report = build_report(result)
    assert (report.summary.burst_groups, report.summary.burst_frames) == (1, 3)
    assert report.summary.redundant_images == 2
