"""Discover images and build base records (metadata + SHA-256) with per-file fault isolation."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from media_culler.config import Settings
from media_culler.exceptions import ImageLoadError
from media_culler.imaging import read_header, sha256_file
from media_culler.models import ImageRecord
from media_culler.progress import NullReporter, ProgressReporter

log = logging.getLogger(__name__)


def discover_files(root: Path, settings: Settings) -> list[Path]:
    """Sorted list of supported image files under ``root``.

    Our own output/report/cache/model directories are skipped if they live inside ``root``,
    so re-running never re-ingests previously organised copies.
    """
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    root = root.resolve()
    extensions = set(settings.scan.extensions)
    excluded_names = set(settings.scan.exclude_dirs)
    excluded_paths = {
        p.resolve()
        for p in (
            settings.paths.output_directory,
            settings.paths.reports_directory,
            settings.paths.cache_directory,
            settings.paths.models_directory,
        )
    }

    found: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=settings.scan.follow_symlinks):
        current_path = Path(current)
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in excluded_names and (current_path / d).resolve() not in excluded_paths
        )
        if not settings.scan.recursive:
            dirs[:] = []
        for name in sorted(files):
            path = current_path / name
            if path.suffix.lower() in extensions and (
                settings.scan.follow_symlinks or not path.is_symlink()
            ):
                found.append(path)
    return found


def build_record(path: Path, root: Path, max_pixels: int) -> ImageRecord:
    """Never raises: failures become ``status='error'`` records."""
    record = ImageRecord(
        path=str(path.resolve()),
        relative_path=path.relative_to(root.resolve()).as_posix(),
        filename=path.name,
        extension=path.suffix.lower(),
    )
    try:
        stat = path.stat()
        if stat.st_size == 0:
            raise ImageLoadError("File is empty")
        header = read_header(path, max_pixels)
        record.size_bytes = stat.st_size
        record.width, record.height, record.format = header.width, header.height, header.format
        record.modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        record.captured_at = header.captured_at
        record.sha256 = sha256_file(path)
    except (ImageLoadError, OSError) as error:
        record.status = "error"
        record.error = str(error)
        log.info("Skipping %s: %s", record.relative_path, error)
    return record


def scan_directory(
    root: Path, settings: Settings, reporter: ProgressReporter | None = None
) -> list[ImageRecord]:
    reporter = reporter or NullReporter()
    reporter.stage("Scanning", None)
    files = discover_files(root, settings)
    reporter.stage("Reading metadata", len(files))

    records: list[ImageRecord | None] = [None] * len(files)
    with ThreadPoolExecutor(max_workers=settings.performance.workers) as pool:
        futures = {
            pool.submit(build_record, path, root, settings.scan.max_pixels): index
            for index, path in enumerate(files)
        }
        for future in as_completed(futures):
            index = futures[future]
            records[index] = future.result()
            reporter.advance(files[index].name)

    return [r for r in records if r is not None]
