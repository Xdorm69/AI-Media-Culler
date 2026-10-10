import csv
import json

import pytest

from media_culler.exceptions import ReportError
from media_culler.model_download import DownloadError, download_file
from media_culler.organizer import DUPLICATES_BUCKET, Mode, execute, plan_operations
from media_culler.pipeline import Pipeline
from media_culler.reporting import build_report, load_report, write_reports


@pytest.fixture
def report(settings, photo_dir):
    return build_report(Pipeline(settings).run(photo_dir))


def test_summary_counts_are_consistent(report):
    s = report.summary
    assert s.total_files == 4 and s.valid_images == 3 and s.errors == 1
    assert sum(s.recommendations.values()) == s.valid_images
    assert s.duplicate_groups == 1 and s.redundant_images == 1
    assert s.top and s.top[0].final_score >= s.top[-1].final_score
    assert s.mean_score is not None


def test_json_roundtrip(report, tmp_path):
    paths = write_reports(report, tmp_path / "r", {"json", "csv"})
    loaded = load_report(paths["json"])
    assert loaded.summary == report.summary
    assert [i.path for i in loaded.images] == [i.path for i in report.images]
    assert not list((tmp_path / "r").glob("*.tmp"))  # atomic write left no debris


def test_csv_is_sorted_by_rank_and_handles_errors(report, tmp_path):
    p = write_reports(report, tmp_path, {"csv"})["csv"]
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    ranks = [int(r["rank"]) for r in rows if r["rank"]]
    assert ranks == sorted(ranks)
    assert rows[-1]["status"] == "error" and rows[-1]["error"]


def test_load_report_rejects_bad_inputs(tmp_path):
    with pytest.raises(ReportError, match="not found"):
        load_report(tmp_path / "nope.json")
    legacy = tmp_path / "old.json"
    legacy.write_text("[]")
    with pytest.raises(ReportError, match=r"v0\.1"):
        load_report(legacy)
    wrong = tmp_path / "w.json"
    wrong.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ReportError, match="Unsupported"):
        load_report(wrong)
    broken = tmp_path / "b.json"
    broken.write_text("{not json")
    with pytest.raises(ReportError, match="Cannot read"):
        load_report(broken)


# --------------------------------------------------------------------- organizer
def test_plan_routes_weaker_duplicates_to_alternates(report, tmp_path):
    ops = plan_operations(report, tmp_path / "out")
    assert len(ops) == 3
    assert [o.bucket for o in ops].count(DUPLICATES_BUCKET) == 1
    kept = plan_operations(report, tmp_path / "out", separate_duplicates=False)
    assert DUPLICATES_BUCKET not in {o.bucket for o in kept}


def test_only_filter(report, tmp_path):
    assert plan_operations(report, tmp_path, only={"nonexistent"}) == []


def test_name_collisions_get_hash_suffix(report, tmp_path):
    for img in report.images:
        img.filename = "same.jpg"
        img.recommendation = report.images[0].recommendation or img.recommendation
        if img.status == "valid":
            img.group_role = None
    ops = plan_operations(report, tmp_path, separate_duplicates=False)
    dests = [o.destination for o in ops]
    assert len(set(dests)) == len(dests)


def test_copy_leaves_sources_and_is_idempotent(report, tmp_path):
    dest = tmp_path / "out"
    ops = plan_operations(report, dest)
    first = execute(ops, Mode.COPY, dest)
    assert len(first.done) == 3 and all(o.source.exists() and o.destination.exists() for o in ops)
    assert first.journal and first.journal.parent == dest / ".culler" / "undo"
    again = execute(ops, Mode.COPY, dest)
    assert not again.done and {r for _, r in again.skipped} == {"destination exists"}
    assert not list(dest.rglob("*.part"))


def test_move_relocates_files(report, tmp_path):
    ops = plan_operations(report, tmp_path / "out")
    result = execute(ops, Mode.MOVE)
    assert len(result.done) == 3 and not any(o.source.exists() for o in ops)


def test_missing_or_changed_sources_are_skipped_not_guessed(report, tmp_path):
    ops = plan_operations(report, tmp_path / "out")
    ops[0].source.unlink()
    ops[1].source.write_bytes(b"changed size")
    result = execute(ops, Mode.COPY)
    reasons = sorted(r for _, r in result.skipped)
    assert reasons == ["source changed since analysis", "source missing"]
    assert len(result.done) == 1


def test_never_overwrites_existing_destination(report, tmp_path):
    ops = plan_operations(report, tmp_path / "out")
    ops[0].destination.parent.mkdir(parents=True)
    ops[0].destination.write_bytes(b"precious")
    execute(ops, Mode.COPY)
    assert ops[0].destination.read_bytes() == b"precious"


def test_symlink_mode(report, tmp_path):
    ops = plan_operations(report, tmp_path / "out")
    result = execute(ops, Mode.SYMLINK)
    if result.failed:
        pytest.skip("symlinks unavailable on this platform")
    assert all(o.destination.is_symlink() for o in result.done)


# ---------------------------------------------------------------------- downloader
def test_download_atomic_and_checksum(tmp_path):
    import hashlib

    src = tmp_path / "src.bin"
    src.write_bytes(b"model" * 1000)
    out = download_file(
        src.as_uri(), tmp_path / "m" / "f.task", sha256=hashlib.sha256(src.read_bytes()).hexdigest()
    )
    assert out.read_bytes() == src.read_bytes() and not list(tmp_path.rglob("*.part"))


def test_download_rejects_checksum_mismatch_and_tiny_files(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"model" * 1000)
    dest = tmp_path / "f.task"
    with pytest.raises(DownloadError, match="SHA-256"):
        download_file(src.as_uri(), dest, sha256="0" * 64)
    assert not dest.exists() and not list(tmp_path.glob("*.part"))
    tiny = tmp_path / "tiny.bin"
    tiny.write_bytes(b"<html>404</html>")
    with pytest.raises(DownloadError, match="too small"):
        download_file(tiny.as_uri(), dest)


def test_download_failure_after_retries(tmp_path):
    with pytest.raises(DownloadError, match="Could not download"):
        download_file((tmp_path / "missing").as_uri(), tmp_path / "x", retries=1)
