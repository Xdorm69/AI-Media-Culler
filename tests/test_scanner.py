from pathlib import Path

import pytest

from conftest import save, textured
from media_culler.scanner import discover_files, scan_directory


def test_discovers_supported_files_sorted_and_recursive(settings, photo_dir):
    names = [p.relative_to(photo_dir).as_posix() for p in discover_files(photo_dir, settings)]
    assert names == sorted(names)
    assert "notes.txt" not in names and "sub/other.png" in names


def test_extension_matching_is_case_insensitive(settings, tmp_path):
    save(textured(1), tmp_path / "in" / "UPPER.JPG")
    assert len(discover_files(tmp_path / "in", settings)) == 1


def test_non_recursive(settings, photo_dir):
    settings.scan.recursive = False
    assert all(p.parent == photo_dir for p in discover_files(photo_dir, settings))


def test_own_output_dirs_are_not_rescanned(settings, tmp_path):
    root = tmp_path / "in"
    save(textured(1), root / "keep.jpg")
    save(textured(2), root / "organized" / "select" / "copy.jpg")
    settings.paths.output_directory = root / "organized"
    assert [p.name for p in discover_files(root, settings)] == ["keep.jpg"]


def test_missing_and_non_directory_inputs(settings, tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "nope", settings)
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        discover_files(f, settings)


def test_records_isolate_failures(settings, photo_dir):
    records = scan_directory(photo_dir, settings)
    by_name = {r.filename: r for r in records}
    assert by_name["corrupt.jpg"].status == "error" and by_name["corrupt.jpg"].error
    good = by_name["sharp.jpg"]
    assert good.status == "valid" and good.sha256 and len(good.sha256) == 64
    assert (good.width, good.height) == (640, 480)
    assert good.relative_path == "sharp.jpg"
    assert Path(good.path).is_absolute()


def test_empty_file_is_an_error_record(settings, tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "empty.jpg").write_bytes(b"")
    (rec,) = scan_directory(tmp_path / "in", settings)
    assert rec.status == "error" and "empty" in rec.error.lower()


def test_scan_is_deterministic(settings, photo_dir):
    a = [r.relative_path for r in scan_directory(photo_dir, settings)]
    b = [r.relative_path for r in scan_directory(photo_dir, settings)]
    assert a == b
