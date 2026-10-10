"""Verified copies, undo journal, sidecars and group folders."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from media_culler import organizer
from media_culler.cli import app
from media_culler.imaging import sha256_file
from media_culler.models import GroupKind, GroupRole, ImageRecord, Recommendation
from media_culler.organizer import (
    Mode,
    execute,
    find_sidecars,
    latest_journal,
    plan_operations,
    undo_run,
)
from media_culler.reporting import Report, build_summary


def make_report(root: Path, names: list[str], **extra: object) -> Report:
    images = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(f"content of {name}".encode())
        images.append(
            ImageRecord(
                path=str(path),
                relative_path=name,
                filename=path.name,
                extension=path.suffix,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                recommendation=Recommendation.SELECT,
            )
        )
    return Report(
        generated_at="now",
        duration_seconds=0,
        input_directory=str(root),
        skipped_analyzers={},
        summary=build_summary(images, 0),
        images=images,
    )


@pytest.fixture
def src(tmp_path: Path) -> Path:
    return tmp_path / "src"


def test_copy_is_verified_against_the_analysed_hash(src, tmp_path):
    report = make_report(src, ["a.jpg"])
    (src / "a.jpg").write_bytes(b"CONTENT OF a.jpg")  # same size, different bytes
    ops = plan_operations(report, tmp_path / "out")
    result = execute(ops, Mode.COPY, tmp_path / "out")
    assert not result.done and "does not match" in result.failed[0][1]
    assert not list((tmp_path / "out").rglob("a.jpg*"))  # no partial or final file left
    assert result.journal is None


def test_cross_device_move_deletes_the_source_only_after_verification(src, tmp_path, monkeypatch):
    report = make_report(src, ["a.jpg", "b.jpg"])
    real_rename = os.rename

    def exdev(a, b):  # pretend the destination is another filesystem
        raise OSError(errno.EXDEV, "cross-device")

    monkeypatch.setattr(organizer.os, "rename", exdev)
    ops = plan_operations(report, tmp_path / "out")
    (src / "b.jpg").write_bytes(b"CONTENT OF b.jpg")  # changed after analysis, same size
    result = execute(ops, Mode.MOVE, tmp_path / "out")
    monkeypatch.setattr(organizer.os, "rename", real_rename)
    assert [o.source.name for o in result.done] == ["a.jpg"]
    assert not (src / "a.jpg").exists() and (tmp_path / "out/select/a.jpg").exists()
    assert (src / "b.jpg").exists(), "unverified source must never be deleted"
    assert result.failed and not (tmp_path / "out/select/b.jpg").exists()


def test_undo_copy_removes_only_verified_copies(src, tmp_path):
    out = tmp_path / "out"
    report = make_report(src, ["a.jpg", "b.jpg", "c.jpg"])
    result = execute(plan_operations(report, out), Mode.COPY, out)
    (out / "select/b.jpg").write_bytes(b"edited after copy")
    (src / "c.jpg").unlink()  # original gone: the copy is now the only one

    dry = undo_run(result.journal, out)
    assert (out / "select/a.jpg").exists(), "dry run changes nothing"
    assert [p.name for p in dry.reverted] == ["a.jpg"]

    done = undo_run(result.journal, out, apply=True)
    reasons = {p.name: r for p, r in done.skipped}
    assert reasons["b.jpg"] == "copy was modified since it was made"
    assert reasons["c.jpg"].startswith("original is missing")
    assert not (out / "select/a.jpg").exists()
    assert (out / "select/b.jpg").exists() and (out / "select/c.jpg").exists()
    assert done.journal and done.journal.name.endswith(".undone")
    assert latest_journal(out) is None


def test_undo_move_restores_and_refuses_to_clobber(src, tmp_path):
    out = tmp_path / "out"
    report = make_report(src, ["a.jpg", "b.jpg"])
    result = execute(plan_operations(report, out), Mode.MOVE, out)
    assert not (src / "a.jpg").exists()
    (src / "b.jpg").write_bytes(b"someone put a new file here")
    done = undo_run(result.journal, out, apply=True)
    assert (src / "a.jpg").read_bytes() == b"content of a.jpg"
    assert (src / "b.jpg").read_bytes() == b"someone put a new file here"
    assert dict((p.name, r) for p, r in done.skipped) == {"b.jpg": "original location is occupied"}
    assert not (out / "select").exists() or any((out / "select").iterdir())


def test_undo_symlinks_and_empty_folder_cleanup(src, tmp_path):
    out = tmp_path / "out"
    report = make_report(src, ["a.jpg"])
    result = execute(plan_operations(report, out), Mode.SYMLINK, out)
    if result.failed:
        pytest.skip("symlinks unavailable")
    undo_run(result.journal, out, apply=True)
    assert not (out / "select").exists()


def test_skipped_operations_are_not_journalled_so_undo_never_touches_them(src, tmp_path):
    out = tmp_path / "out"
    report = make_report(src, ["a.jpg", "b.jpg"])
    ops = plan_operations(report, out)
    (out / "select").mkdir(parents=True)
    (out / "select/a.jpg").write_bytes(b"the user's own file")
    result = execute(ops, Mode.COPY, out)
    undo_run(result.journal, out, apply=True)
    assert (out / "select/a.jpg").read_bytes() == b"the user's own file"


def test_a_crash_after_a_move_is_still_undoable(src, tmp_path):
    out = tmp_path / "out"
    report = make_report(src, ["a.jpg"])
    [op] = plan_operations(report, out)
    journal = organizer._Journal(out, Mode.MOVE)
    journal.write(
        {
            "event": "begin",
            "source": str(op.source),
            "destination": str(op.destination),
            "mode": "move",
            "kind": "photo",
            "size": op.size_bytes,
        }
    )
    op.destination.parent.mkdir(parents=True)
    os.rename(op.source, op.destination)  # ...and the process dies before "done"
    journal.close()
    with journal.path.open("a") as handle:
        handle.write('{"event": "begin", "sour')  # torn final line
    undo_run(journal.path, out, apply=True)
    assert op.source.exists()


def test_sidecars_travel_with_their_photo(src, tmp_path):
    for name in ("IMG_1.CR2", "IMG_1.xmp", "IMG_1.CR2.xmp", "IMG_1.AAE", "IMG_1.MOV", "IMG_2.jpg"):
        (src).mkdir(exist_ok=True)
        (src / name).write_bytes(name.encode())
    assert [p.name for p in find_sidecars(src / "IMG_1.CR2")] == [
        "IMG_1.AAE",
        "IMG_1.CR2.xmp",
        "IMG_1.xmp",
    ]  # no Live Photo video for a RAW
    assert "IMG_1.MOV" not in {p.name for p in find_sidecars(src / "IMG_1.CR2")}
    assert "IMG_1.MOV" not in {p.name for p in find_sidecars(src / "IMG_2.jpg")}

    (src / "IMG_3.heic").write_bytes(b"h")
    (src / "IMG_3.mov").write_bytes(b"m")
    assert [p.name for p in find_sidecars(src / "IMG_3.heic")] == ["IMG_3.mov"]

    out = tmp_path / "out"
    report = make_report(src, ["IMG_1.CR2", "IMG_2.jpg"])
    ops = plan_operations(report, out)
    names = sorted(o.destination.name for o in ops)
    assert names == ["IMG_1.AAE", "IMG_1.CR2", "IMG_1.CR2.xmp", "IMG_1.xmp", "IMG_2.jpg"]
    assert plan_operations(report, out, include_sidecars=False)[0].kind == "photo"
    assert len(plan_operations(report, out, include_sidecars=False)) == 2
    result = execute(ops, Mode.COPY, out)
    assert len(result.done) == 5
    undo_run(result.journal, out, apply=True)
    assert not list(out.rglob("IMG_*"))


def test_a_sidecar_shared_by_a_raw_and_its_jpeg_is_taken_once(src, tmp_path):
    (src).mkdir()
    (src / "IMG_1.xmp").write_bytes(b"x")
    report = make_report(src, ["IMG_1.CR2", "IMG_1.JPG"])
    ops = plan_operations(report, tmp_path / "out")
    assert [o.kind for o in ops].count("sidecar") == 1


def test_sidecar_follows_a_renamed_photo(src, tmp_path):
    report = make_report(src, ["a/x.jpg", "b/x.jpg"])
    (src / "b/x.xmp").write_bytes(b"x")
    ops = plan_operations(report, tmp_path / "out")
    photo = next(o for o in ops if o.source == src / "b/x.jpg")
    sidecar = next(o for o in ops if o.kind == "sidecar")
    assert photo.destination.stem != "x"  # collision suffix applied
    assert sidecar.destination.stem == photo.destination.stem


def test_group_folders_include_a_raw_companions(src, tmp_path):
    report = make_report(src, ["a.CR2", "a.jpg", "solo.jpg"])
    raw, companion, _solo = report.images
    raw.duplicate_group, raw.group_kind = 7, GroupKind.BURST
    companion.paired_with = raw.relative_path
    raw.group_role = GroupRole.REPRESENTATIVE
    out = tmp_path / "out"
    flat = {o.destination.parent.name for o in plan_operations(report, out)}
    assert flat == {"select"}
    grouped = plan_operations(report, out, group_folders=True)
    folders = {o.source.name: o.destination.parent.name for o in grouped}
    assert folders == {"a.CR2": "burst_0007", "a.jpg": "burst_0007", "solo.jpg": "select"}


def test_cli_undo_roundtrip_and_dry_run_default(src, tmp_path):
    from media_culler.reporting import write_reports

    out = tmp_path / "out"
    report = make_report(src, ["a.jpg"])
    report_path = write_reports(report, tmp_path / "r", {"json"})["json"]
    runner = CliRunner()
    done = runner.invoke(app, ["organize", str(report_path), "--dest", str(out), "--apply"])
    assert done.exit_code == 0 and "culler undo" in done.output
    assert (out / "select/a.jpg").exists()
    dry = runner.invoke(app, ["undo", "--dest", str(out)])
    assert dry.exit_code == 0 and (out / "select/a.jpg").exists() and "Dry run" in dry.output
    real = runner.invoke(app, ["undo", "--dest", str(out), "--apply"])
    assert real.exit_code == 0 and not (out / "select/a.jpg").exists()
    assert (src / "a.jpg").exists()
    again = runner.invoke(app, ["undo", "--dest", str(out)])
    assert again.exit_code != 0 and "No organize run" in again.output
