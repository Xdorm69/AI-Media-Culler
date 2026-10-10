import cv2
import numpy as np

from conftest import save, textured
from media_culler.analyzers import Analyzer
from media_culler.analyzers.technical import TechnicalAnalyzer
from media_culler.config import SelectionConfig, Settings, ThresholdsConfig
from media_culler.curation import classify, select_best_shots
from media_culler.models import (
    EyeMetrics,
    FaceAnalysis,
    FaceInfo,
    GroupAssessment,
    GroupRole,
    ImageRecord,
    PoseMetrics,
    Recommendation,
    Scores,
    TechnicalMetrics,
)
from media_culler.pipeline import Pipeline
from media_culler.reporting import build_report
from media_culler.scoring import score_record
from media_culler.scoring.group import assess_faces, eyes_closed, subject_faces
from media_culler.scoring.subject import score_subject

CFG = SelectionConfig()
OPEN, CLOSED = 0.30, 0.08


def face(area=0.03, ear=OPEN, orientation="frontal", x=400, y=300) -> FaceInfo:
    return FaceInfo(
        x=x,
        y=y,
        width=120,
        height=120,
        area_ratio=area,
        sharpness=400,
        eyes=EyeMetrics(left=ear, right=ear, average=ear),
        pose=PoseMetrics(
            symmetry=1.0 if orientation == "frontal" else 0.3, orientation=orientation
        ),
    )


def faces(*items: FaceInfo) -> FaceAnalysis:
    return FaceAnalysis(face_count=len(items), faces=list(items))


# ------------------------------------------------------------------ who is a subject
def test_passer_by_is_not_a_subject():
    a = assess_faces(faces(face(0.03), face(0.03), face(0.0009)), CFG)
    assert a.significant == [0, 1] and a.ignored == 1


def test_relative_rule_ignores_much_smaller_faces_but_keeps_a_distant_group():
    mixed = assess_faces(faces(face(0.05), face(0.004)), CFG)  # 0.004 < 0.12 * 0.05
    assert mixed.significant == [0]
    far_group = assess_faces(faces(*[face(0.003) for _ in range(6)]), CFG)
    assert len(far_group.significant) == 6  # all small, but they are the subjects


def test_nobody_is_a_subject_when_every_face_is_tiny():
    a = assess_faces(faces(face(0.0005), face(0.0008)), CFG)
    assert a.significant == [] and a.ignored == 2


def test_no_faces():
    assert assess_faces(FaceAnalysis(face_count=0), CFG) == GroupAssessment()


# --------------------------------------------------------------------- closed eyes
def test_closed_eye_detection_and_its_limits():
    assert eyes_closed(face(ear=CLOSED), CFG)
    assert not eyes_closed(face(ear=OPEN), CFG)
    assert not eyes_closed(face(ear=0.16), CFG)  # squint/laugh is not a blink
    assert not eyes_closed(face(area=0.003, ear=CLOSED), CFG)  # too small to judge
    assert not eyes_closed(face(ear=CLOSED, orientation="strongly_turned"), CFG)  # profile


def test_looking_away_is_tracked_separately():
    a = assess_faces(faces(face(), face(orientation="strongly_turned")), CFG)
    assert a.looking_away == [1] and a.closed_eyes == []


# ------------------------------------------------------------------ subject scoring
def test_a_bystander_does_not_change_the_score():
    group = [face(0.03), face(0.03), face(0.03)]
    clean, _ = score_subject(faces(*group))
    noisy, details = score_subject(faces(*group, face(0.0008, ear=CLOSED)))
    assert noisy == clean and details["ignored_faces"] == 1


def test_only_background_faces_is_neutral_with_a_clear_reason():
    score, details = score_subject(faces(face(0.0005)))
    assert score == 50.0 and "background" in details["reason"]


def test_closed_eyes_are_capped_hard():
    """Regression: the weighted sum let a face with shut eyes still score about 80."""
    open_face, _ = score_subject(faces(face(0.05)))
    blink, _ = score_subject(faces(face(0.05, ear=CLOSED)))
    assert open_face > 90 and blink <= CFG.closed_face_cap


def test_one_blinker_caps_the_whole_group():
    four = [face(0.05) for _ in range(3)] + [face(0.05, ear=CLOSED)]
    score, details = score_subject(faces(*four))
    assert score <= CFG.closed_subject_cap and details["closed_eyes"] == 1
    assert score_subject(faces(*four[:3]))[0] > 90


def test_composition_ignores_a_cropped_bystander_on_the_frame_edge():
    from test_landscape import record

    centred = [face(0.03, x=450, y=350)]
    edge_bystander = face(0.0008, x=0, y=0)  # clipped by the corner
    base = score_record(record(faces=faces(*centred)), Settings())
    with_extra = score_record(record(faces=faces(*centred, edge_bystander)), Settings())
    assert with_extra.composition == base.composition
    assert subject_faces(faces(*centred, edge_bystander), with_extra.group).face_count == 1


# --------------------------------------------------------------------- best-shot rule
def frame(
    name, selection, *, subjects=3, closed=0, final=None, group=1, sharp=300.0
) -> ImageRecord:
    r = ImageRecord(
        path=f"/p/{name}",
        relative_path=name,
        filename=name,
        extension=".jpg",
        duplicate_group=group,
    )
    r.technical = TechnicalMetrics(
        sharpness=sharp, brightness=125, contrast=60, megapixels=12, perceptual_hash="0" * 16
    )
    r.scores = Scores(
        technical=80,
        final=selection if final is None else final,
        selection=selection,
        group=GroupAssessment(significant=list(range(subjects)), closed_eyes=list(range(closed))),
    )
    return r


def roles(records):
    return {r.filename: r.group_role for r in records}


def test_open_eyes_win_within_the_margin():
    blink, open_eyes = frame("a", 86, closed=1), frame("b", 80)
    select_best_shots([blink, open_eyes], CFG)
    assert roles([blink, open_eyes]) == {
        "a": GroupRole.SIMILAR,
        "b": GroupRole.REPRESENTATIVE,
    }
    assert "fewer problems" in open_eyes.selection_reason
    assert "Closed eyes on 1" in blink.selection_reason


def test_quality_wins_outside_the_margin():
    sharp_blink, blurry_open = frame("a", 86, closed=1), frame("b", 60)  # 26 points worse
    select_best_shots([sharp_blink, blurry_open], CFG)
    assert sharp_blink.group_role is GroupRole.REPRESENTATIVE
    assert blurry_open.selection_reason == "Lower selection score than the group's best shot"


def test_margin_is_configurable():
    a, b = frame("a", 86, closed=1), frame("b", 80)
    select_best_shots([a, b], SelectionConfig(margin=3))
    assert a.group_role is GroupRole.REPRESENTATIVE


def test_without_problems_the_highest_score_wins_and_ties_break_by_path():
    a, b, c = frame("b", 80), frame("a", 80), frame("c", 79)
    select_best_shots([a, b, c], CFG)
    assert b.group_role is GroupRole.REPRESENTATIVE  # path 'a' beats 'b' on a tie
    assert a.selection_reason == "Lower selection score than the group's best shot"


def test_a_missing_subject_counts_as_a_problem():
    full, missing = frame("a", 82, subjects=4), frame("b", 85, subjects=3)
    third, fourth = frame("c", 70, subjects=4), frame("d", 70, subjects=4)
    select_best_shots([full, missing, third, fourth], CFG)
    assert full.group_role is GroupRole.REPRESENTATIVE
    assert "missing" in missing.selection_reason


def test_one_false_detection_does_not_make_every_other_frame_look_short():
    ghost = frame("a", 85, subjects=5)  # a false positive adds a 5th face
    honest = [frame("b", 84, subjects=4), frame("c", 80, subjects=4), frame("d", 80, subjects=4)]
    select_best_shots([ghost, *honest], CFG)
    assert ghost.group_role is GroupRole.REPRESENTATIVE  # highest score; nobody penalised


def test_unscored_frames_never_win_and_all_unscored_still_picks_one():
    scored = frame("b", 50)
    unscored = ImageRecord(
        path="/p/a", relative_path="a", filename="a", extension=".jpg", duplicate_group=1
    )
    select_best_shots([unscored, scored], CFG)
    assert scored.group_role is GroupRole.REPRESENTATIVE
    u1 = ImageRecord(
        path="/p/a", relative_path="a", filename="a", extension=".jpg", duplicate_group=2
    )
    u2 = ImageRecord(
        path="/p/b", relative_path="b", filename="b", extension=".jpg", duplicate_group=2
    )
    select_best_shots([u1, u2], CFG)
    assert {u1.group_role, u2.group_role} == {GroupRole.REPRESENTATIVE, GroupRole.SIMILAR}


def test_singletons_are_unique():
    solo = frame("a", 70, group=None)
    select_best_shots([solo], CFG)
    assert solo.group_role is GroupRole.UNIQUE and solo.is_group_representative


# ------------------------------------------------------------------ verdict gate
def test_closed_eyes_cap_the_verdict_at_review():
    r = frame("a", 90, closed=1, subjects=4)
    assert classify(r, ThresholdsConfig()) == (
        Recommendation.REVIEW,
        "Closed eyes on 1 of 4 faces; check first",
    )
    assert classify(r, ThresholdsConfig(closed_eyes_gate=False))[0] is Recommendation.SELECT


def test_gate_never_promotes_a_weak_photo():
    assert classify(frame("a", 30, closed=1), ThresholdsConfig())[0] is Recommendation.REJECT


# -------------------------------------------------------------------- end to end
class MarkerFaces(Analyzer):
    """Fake face analyzer that reads which 'person state' a frame has from pixel (0, 0)."""

    name = "faces"
    section = "faces"
    result_type = FaceAnalysis

    def analyze(self, ctx):
        marker = int(ctx.rgb[0, 0, 0])
        ear = CLOSED if marker == 1 else OPEN
        people = [face(0.03, x=200), face(0.03, x=500), face(0.03, ear=ear, x=800)]
        return faces(*people)


def marked(base, marker, blur=0.0):
    image = cv2.GaussianBlur(base, (0, 0), blur) if blur else base.copy()
    image[0, 0] = marker  # one pixel: invisible to the perceptual hash
    return image


def test_group_shot_picks_the_frame_where_nobody_blinks(settings, tmp_path):
    """The blinking frame is technically sharper, so it still outscores the open-eyes frame
    even after the closed-eyes caps; selection must prefer the open-eyes one inside the margin."""
    settings.selection.margin = 20
    settings.selection.sharpness_weight = 0  # isolate the blink rule; sharpness is tested elsewhere
    root = tmp_path / "in"
    base = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    save(marked(base, 1), root / "a_blink_sharper.png")
    save(marked(base, 2, blur=1.2), root / "b_open_eyes.png")

    result = Pipeline(settings, analyzers=[TechnicalAnalyzer(), MarkerFaces()]).run(root)
    a, b = sorted(result.records, key=lambda r: r.filename)

    assert a.duplicate_group == b.duplicate_group is not None
    gap = a.scores.selection - b.scores.selection  # premise of this test
    assert 0 < gap < settings.selection.margin, f"scoring changed: gap is {gap:.1f}"
    assert a.scores.group.closed_eyes == [2] and b.scores.group.closed_eyes == []

    assert b.group_role is GroupRole.REPRESENTATIVE
    assert a.group_role is GroupRole.SIMILAR
    assert "fewer problems" in b.selection_reason
    assert "Closed eyes" in a.selection_reason
    assert a.recommendation is Recommendation.REVIEW  # also gated by the closed eyes


def test_a_much_sharper_blink_frame_still_wins_outside_the_margin(settings, tmp_path):
    settings.selection.margin = 5
    root = tmp_path / "in"
    base = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    save(marked(base, 1), root / "a_blink_sharp.png")
    save(marked(base, 2, blur=2.0), root / "b_open_blurry.png")
    result = Pipeline(settings, analyzers=[TechnicalAnalyzer(), MarkerFaces()]).run(root)
    a, b = sorted(result.records, key=lambda r: r.filename)
    assert a.scores.selection - b.scores.selection > settings.selection.margin
    assert a.group_role is GroupRole.REPRESENTATIVE


def test_report_lists_closed_eye_photos(settings, tmp_path):
    root = tmp_path / "in"
    base = textured(2)
    save(marked(base, 1), root / "blink.png")
    save(textured(9), root / "fine.png")
    report = build_report(
        Pipeline(settings, analyzers=[TechnicalAnalyzer(), MarkerFaces()]).run(root)
    )
    assert report.summary.closed_eyes_photos == 1
