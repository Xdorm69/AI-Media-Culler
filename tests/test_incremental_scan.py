from __future__ import annotations

import os

from tests.conftest import save, textured

from media_culler import scanner
from media_culler.cache import AnalysisCache
from media_culler.scanner import scan_directory


def test_unchanged_files_are_not_hashed_again(settings, tmp_path, monkeypatch):
    root = tmp_path / "in"
    for seed in range(3):
        save(textured(seed), root / f"{seed}.png")
    calls: list[str] = []
    real = scanner.sha256_file
    monkeypatch.setattr(scanner, "sha256_file", lambda p: calls.append(p.name) or real(p))

    db = tmp_path / "cache.db"
    with AnalysisCache(db) as cache:
        first = scan_directory(root, settings, cache=cache)
    assert sorted(calls) == ["0.png", "1.png", "2.png"]

    calls.clear()
    with AnalysisCache(db) as cache:
        second = scan_directory(root, settings, cache=cache)
    assert calls == [] and [r.sha256 for r in second] == [r.sha256 for r in first]


def test_a_changed_file_is_rehashed(settings, tmp_path, monkeypatch):
    root = tmp_path / "in"
    path = save(textured(1), root / "a.png")
    save(textured(2), root / "b.png")
    db = tmp_path / "cache.db"
    with AnalysisCache(db) as cache:
        before = {r.filename: r.sha256 for r in scan_directory(root, settings, cache=cache)}

    save(textured(9), path)  # different pixels, so size and mtime differ
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    with AnalysisCache(db) as cache:
        after = {r.filename: r.sha256 for r in scan_directory(root, settings, cache=cache)}
    assert after["a.png"] != before["a.png"] and after["b.png"] == before["b.png"]


def test_scanning_without_a_cache_still_works(settings, tmp_path):
    root = tmp_path / "in"
    save(textured(1), root / "a.png")
    assert scan_directory(root, settings)[0].sha256
    disabled = AnalysisCache(None)
    assert disabled.known_hashes() == {}
    disabled.remember_hashes([("x", 1, 1, "h")])  # no-op, no error
