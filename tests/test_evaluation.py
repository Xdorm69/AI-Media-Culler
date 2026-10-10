from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from media_culler.cli import app
from media_culler.evaluation import LabelError, evaluate, load_labels, write_template
from media_culler.models import (
    GroupAssessment,
    GroupRole,
    ImageRecord,
    Recommendation,
    SceneType,
    Scores,
)
from media_culler.reporting import Report, build_summary, write_reports


def rec(name: str, recommendation: Recommendation, **kw: object) -> ImageRecord:
    scene = kw.pop("scene", SceneType.GENERAL)
    closed = kw.pop("closed", [])
    scores = Scores(final=70, scene_type=scene, group=GroupAssessment(closed_eyes=closed))
    return ImageRecord(
        path=f"/p/{name}",
        relative_path=name,
        filename=Path(name).name,
        extension=".jpg",
        recommendation=recommendation,
        scores=scores,
        **kw,  # type: ignore[arg-type]
    )


def report(images: list[ImageRecord]) -> Report:
    return Report(
        generated_at="t",
        duration_seconds=0,
        input_directory="/p",
        skipped_analyzers={},
        summary=build_summary(images, 0),
        images=images,
    )


def labels_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "labels.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_verdict_metrics(tmp_path):
    images = [
        rec("k1.jpg", Recommendation.SELECT),
        rec("k2.jpg", Recommendation.USABLE),
        rec("k3.jpg", Recommendation.REJECT),  # false reject
        rec("k4.jpg", Recommendation.REVIEW),
        rec("r1.jpg", Recommendation.REJECT),
        rec("r2.jpg", Recommendation.SELECT),  # false keep
    ]
    text = "path,verdict\n" + "\n".join(
        f"{n},{'keep' if n[0] == 'k' else 'reject'}" for n in (i.relative_path for i in images)
    )
    result = evaluate(report(images), load_labels(labels_file(tmp_path, text)))
    v = result.verdict
    assert v.labelled == 6
    assert v.matrix == {
        "keep": {"keep": 2, "reject": 1, "review": 1},
        "reject": {"reject": 1, "keep": 1},
    }
    assert v.false_reject_rate == 0.25 and v.keep_recall == 0.5
    assert v.keep_precision == round(2 / 3, 4) and v.reject_recall == 0.5
    assert v.review_rate == round(1 / 6, 4)


def test_scene_and_closed_eye_metrics(tmp_path):
    images = [
        rec("a.jpg", Recommendation.SELECT, scene=SceneType.LANDSCAPE),
        rec("b.jpg", Recommendation.SELECT, scene=SceneType.GENERAL),
        rec("c.jpg", Recommendation.SELECT, closed=[0]),
        rec("d.jpg", Recommendation.SELECT, closed=[1]),
        rec("e.jpg", Recommendation.SELECT),
    ]
    text = (
        "path,scene,closed_eyes\n"
        "a.jpg,landscape,\nb.jpg,landscape,\nc.jpg,,yes\nd.jpg,,no\ne.jpg,,yes\n"
    )
    result = evaluate(report(images), load_labels(labels_file(tmp_path, text)))
    assert result.scene.accuracy == 0.5
    assert result.scene.confusion == {"landscape": {"landscape": 1, "general": 1}}
    c = result.closed_eyes
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 0)
    assert c.precision == 0.5 and c.recall == 0.5


def test_best_shot_metrics(tmp_path):
    images = [
        rec("g1a.jpg", Recommendation.SELECT, group_role=GroupRole.REPRESENTATIVE),
        rec("g1b.jpg", Recommendation.SELECT, group_role=GroupRole.SIMILAR),
        rec("g2a.jpg", Recommendation.SELECT, group_role=GroupRole.SIMILAR),  # your pick dropped
        rec("g2b.jpg", Recommendation.SELECT, group_role=GroupRole.REPRESENTATIVE),
        rec("g3a.jpg", Recommendation.SELECT, group_role=GroupRole.REPRESENTATIVE),
        rec("g3b.jpg", Recommendation.SELECT, group_role=GroupRole.REPRESENTATIVE),  # not culled
    ]
    text = (
        "path,group,best\n"
        "g1a.jpg,1,1\ng1b.jpg,1,\n"
        "g2a.jpg,2,1\ng2b.jpg,2,\n"
        "g3a.jpg,3,1\ng3b.jpg,3,\n"
    )
    b = evaluate(report(images), load_labels(labels_file(tmp_path, text))).best_shot
    assert (b.groups, b.best_kept, b.picked_exactly) == (3, 2, 1)
    assert b.misses == ["2"] and b.best_kept_rate == round(2 / 3, 4)


def test_groups_without_exactly_one_pick_are_ignored(tmp_path):
    images = [rec(f"{n}.jpg", Recommendation.SELECT) for n in "ab"]
    text = "path,group,best\na.jpg,1,1\nb.jpg,1,1\n"
    assert evaluate(report(images), load_labels(labels_file(tmp_path, text))).best_shot.groups == 0


def test_matching_by_filename_and_unmatched_reporting(tmp_path):
    images = [
        rec("trip/a.jpg", Recommendation.SELECT),
        rec("x/dup.jpg", Recommendation.SELECT),
        rec("y/dup.jpg", Recommendation.SELECT),
    ]
    text = "path,verdict\na.jpg,keep\ndup.jpg,keep\nmissing.jpg,keep\n"
    result = evaluate(report(images), load_labels(labels_file(tmp_path, text)))
    assert result.matched == 1 and result.unmatched == ["dup.jpg", "missing.jpg"]  # ambiguous name


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("name,verdict\na,keep\n", "'path' column"),
        ("path,verdict\na.jpg,maybe\n", "keep or reject"),
        ("path,scene\na.jpg,beach\n", "scene must be"),
        ("path,closed_eyes\na.jpg,perhaps\n", "1/0 or yes/no"),
        ("path,verdict\na.jpg,keep\na.jpg,reject\n", "listed twice"),
    ],
)
def test_bad_labels_are_rejected_with_the_row_in_the_message(tmp_path, text, message):
    with pytest.raises(LabelError, match=message):
        load_labels(labels_file(tmp_path, text))
    with pytest.raises(LabelError, match="not found"):
        load_labels(tmp_path / "nope.csv")


def test_template_lists_valid_images_only(tmp_path):
    bad = rec("bad.jpg", Recommendation.UNSCORED)
    bad.status = "error"
    rep = report([rec("b.jpg", Recommendation.SELECT), rec("a.jpg", Recommendation.SELECT), bad])
    out = tmp_path / "labels.csv"
    assert write_template(rep, out) == 2
    assert out.read_text().splitlines()[:3] == [
        "path,verdict,scene,closed_eyes,group,best",
        "a.jpg,,,,,",
        "b.jpg,,,,,",
    ]
    assert load_labels(out)[0].verdict is None  # blank template loads


def test_cli_eval_roundtrip(tmp_path):
    rep = report([rec("a.jpg", Recommendation.SELECT), rec("b.jpg", Recommendation.REJECT)])
    report_path = write_reports(rep, tmp_path / "r", {"json"})["json"]
    labels = tmp_path / "labels.csv"
    runner = CliRunner()
    init = runner.invoke(app, ["eval", str(report_path), str(labels), "--init"])
    assert init.exit_code == 0 and labels.exists()
    assert runner.invoke(app, ["eval", str(report_path), str(labels), "--init"]).exit_code != 0
    labels.write_text("path,verdict\na.jpg,keep\nb.jpg,keep\n")
    out = tmp_path / "m.json"
    done = runner.invoke(app, ["eval", str(report_path), str(labels), "--json", str(out)])
    assert done.exit_code == 0 and "False-reject rate" in done.output and "50%" in done.output
    assert json.loads(out.read_text())["verdict"]["false_reject_rate"] == 0.5
    bad = runner.invoke(app, ["eval", str(report_path), str(tmp_path / "nope.csv")])
    assert bad.exit_code == 1
