from media_culler.cache import AnalysisCache


def test_roundtrip_and_persistence(tmp_path):
    db = tmp_path / "c" / "a.sqlite3"
    with AnalysisCache(db) as cache:
        assert cache.get("h", "technical", "1") is None
        cache.put("h", "technical", "1", '{"x":1}')
        assert cache.get("h", "technical", "1") == '{"x":1}'
    with AnalysisCache(db) as again:
        assert again.get("h", "technical", "1") == '{"x":1}'
        assert again.hits == 1


def test_fingerprint_change_invalidates(tmp_path):
    with AnalysisCache(tmp_path / "a.db") as cache:
        cache.put("h", "vision", "model-a", "A")
        assert cache.get("h", "vision", "model-b") is None


def test_disabled_cache_is_a_noop():
    cache = AnalysisCache(None)
    cache.put("h", "a", "1", "x")
    assert cache.get("h", "a", "1") is None and not cache.enabled


def test_corrupt_database_is_replaced_not_fatal(tmp_path):
    db = tmp_path / "a.db"
    db.write_bytes(b"this is not sqlite" * 100)
    with AnalysisCache(db) as cache:
        cache.put("h", "a", "1", "x")
        assert cache.get("h", "a", "1") == "x"
    assert db.with_suffix(".corrupt").exists()


def test_unwritable_location_degrades_gracefully(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cache = AnalysisCache(blocker / "sub" / "a.db")  # parent is a file
    assert not cache.enabled
    cache.put("h", "a", "1", "x")
