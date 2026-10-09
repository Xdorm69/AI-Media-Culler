import json

import pytest
from typer.testing import CliRunner

from media_culler.cli import app

runner = CliRunner()


@pytest.fixture
def cfg(tmp_path, photo_dir):
    path = tmp_path / "c.yaml"
    path.write_text(
        f"paths:\n  input_directory: {photo_dir}\n  reports_directory: {tmp_path / 'rep'}\n"
        f"  cache_directory: {tmp_path / 'cache'}\n  output_directory: {tmp_path / 'out'}\n"
        f"  models_directory: {tmp_path / 'models'}\n"
    )
    return path


def run(*args):
    return runner.invoke(app, list(args), catch_exceptions=False)


def test_version_and_help():
    assert "culler" in run("--version").stdout
    assert "run" in run("--help").stdout


def test_run_prints_tldr_and_writes_reports(cfg, tmp_path):
    r = run("run", "-c", str(cfg), "--no-ai", "--no-faces")
    assert r.exit_code == 0, r.stdout
    assert "TL;DR" in r.stdout and "photos analysed" in r.stdout
    assert (tmp_path / "rep" / "inventory.json").exists() and (
        tmp_path / "rep" / "inventory.csv"
    ).exists()
    assert (tmp_path / "rep" / "culler.log").exists()
    assert "corrupt.jpg" in r.stdout  # surfaced in Notes, not lost


def test_run_json_mode_is_pure_json(cfg):
    r = run("run", "-c", str(cfg), "--no-ai", "--no-faces", "--json")
    data = json.loads(r.stdout)  # nothing else on stdout
    assert data["summary"]["valid_images"] == 3 and "json" in data["reports"]


def test_run_quiet_prints_only_paths(cfg):
    r = run("run", "-c", str(cfg), "--no-ai", "--no-faces", "-q")
    lines = r.stdout.strip().splitlines()
    assert lines and all(ln.endswith((".json", ".csv")) for ln in lines)


def test_single_format(cfg, tmp_path):
    run("run", "-c", str(cfg), "--no-ai", "--no-faces", "-f", "csv", "-q")
    assert (tmp_path / "rep" / "inventory.csv").exists() and not (
        tmp_path / "rep" / "inventory.json"
    ).exists()


def test_bad_format_is_usage_error(cfg):
    assert run("run", "-c", str(cfg), "-f", "xml").exit_code == 2


def test_missing_input_dir_is_clean_error(cfg, tmp_path):
    r = run("run", str(tmp_path / "nope"), "-c", str(cfg), "--no-ai", "--no-faces")
    assert r.exit_code == 1 and "does not exist" in r.output and "Traceback" not in r.output


def test_invalid_config_exits_2(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("thresholds:\n  select: 10\n  usable: 50\n")
    r = run("config", "validate", "-c", str(bad))
    assert r.exit_code == 2 and "select > usable > review" in r.output


def test_strict_fails_when_analyzer_missing(cfg):
    r = run("run", "-c", str(cfg), "--no-ai", "--strict")  # faces on, no model file
    assert r.exit_code == 1 and "culler models download" in r.output


def test_missing_face_model_degrades_with_fix_hint(cfg):
    r = run("run", "-c", str(cfg), "--no-ai")
    assert r.exit_code == 0
    assert "faces analysis skipped" in r.stdout and "culler models download" in r.stdout


def test_report_command_roundtrip(cfg):
    run("run", "-c", str(cfg), "--no-ai", "--no-faces", "-q")
    assert "TL;DR" in run("report", "-c", str(cfg)).stdout


def test_report_missing_file(cfg):
    assert run("report", "-c", str(cfg)).exit_code == 1


def test_organize_dry_run_then_apply(cfg, tmp_path):
    run("run", "-c", str(cfg), "--no-ai", "--no-faces", "-q")
    report = str(tmp_path / "rep" / "inventory.json")
    dry = run("organize", report, "-c", str(cfg))
    assert "Dry run" in dry.stdout and not (tmp_path / "out").exists()
    done = run("organize", report, "-c", str(cfg), "--apply")
    assert done.exit_code == 0 and "3 done" in done.stdout
    assert any((tmp_path / "out").rglob("*.jpg"))


def test_organize_move_requires_confirmation(cfg, tmp_path, photo_dir):
    run("run", "-c", str(cfg), "--no-ai", "--no-faces", "-q")
    report = str(tmp_path / "rep" / "inventory.json")
    r = runner.invoke(
        app, ["organize", report, "-c", str(cfg), "-m", "move", "--apply"], input="n\n"
    )
    assert r.exit_code == 1 and (photo_dir / "sharp.jpg").exists()


def test_scan_command(cfg):
    r = run("scan", "-c", str(cfg))
    assert "unreadable" in r.stdout and "corrupt.jpg" in r.stdout


def test_config_init_show_and_overwrite_guard(tmp_path):
    target = tmp_path / "culler.yaml"
    assert run("config", "init", str(target)).exit_code == 0 and target.exists()
    assert run("config", "init", str(target)).exit_code == 1
    assert run("config", "init", str(target), "--force").exit_code == 0
    assert run("config", "validate", "-c", str(target)).exit_code == 0
    assert '"thresholds"' in run("config", "show", "-c", str(target)).stdout


def test_doctor_and_tldr_and_models_status(cfg):
    assert "Environment" in run("doctor", "-c", str(cfg)).stdout
    assert "culler run" in run("tldr").stdout
    assert "missing" in run("models", "status", "-c", str(cfg)).stdout
