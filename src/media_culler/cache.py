"""SQLite cache of analyzer outputs keyed by (content hash, analyzer, fingerprint).

Analysis is the expensive part; scoring is cheap and re-run every time, so changing
weights or thresholds never requires re-analysis, and interrupted runs resume for free.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from types import TracebackType

log = logging.getLogger(__name__)
_SCHEMA_VERSION = 1
_COMMIT_EVERY = 50


class AnalysisCache:
    def __init__(self, path: Path | None) -> None:
        """``path=None`` gives a disabled cache (every lookup misses, writes are dropped)."""
        self.hits = 0
        self.misses = 0
        self._pending = 0
        self._db: sqlite3.Connection | None = None
        if path is not None:
            self._db = self._open(path)

    # -- lifecycle
    def _open(self, path: Path) -> sqlite3.Connection | None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return self._connect(path)
        except sqlite3.DatabaseError:
            backup = path.with_suffix(".corrupt")
            log.warning("Cache %s is corrupt; moving to %s and starting fresh.", path, backup)
            try:
                path.replace(backup)
                return self._connect(path)
            except (OSError, sqlite3.Error):
                log.warning("Cache unavailable; continuing without caching.")
                return None
        except OSError as error:
            log.warning("Cache unavailable (%s); continuing without caching.", error)
            return None

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        db = sqlite3.connect(path)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version != _SCHEMA_VERSION:
            db.execute("DROP TABLE IF EXISTS results")
            db.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        db.execute(
            "CREATE TABLE IF NOT EXISTS results ("
            "sha256 TEXT NOT NULL, analyzer TEXT NOT NULL, fingerprint TEXT NOT NULL, "
            "payload TEXT NOT NULL, PRIMARY KEY (sha256, analyzer, fingerprint))"
        )
        db.commit()
        return db

    def close(self) -> None:
        if self._db is not None:
            self._db.commit()
            self._db.close()
            self._db = None

    def __enter__(self) -> AnalysisCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- access
    @property
    def enabled(self) -> bool:
        return self._db is not None

    def get(self, sha256: str, analyzer: str, fingerprint: str) -> str | None:
        if self._db is None:
            self.misses += 1
            return None
        row = self._db.execute(
            "SELECT payload FROM results WHERE sha256=? AND analyzer=? AND fingerprint=?",
            (sha256, analyzer, fingerprint),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return str(row[0])

    def put(self, sha256: str, analyzer: str, fingerprint: str, payload: str) -> None:
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO results VALUES (?,?,?,?)",
                (sha256, analyzer, fingerprint, payload),
            )
            self._pending += 1
            if self._pending >= _COMMIT_EVERY:
                self._db.commit()
                self._pending = 0
        except sqlite3.Error as error:
            log.warning("Cache write failed: %s", error)
