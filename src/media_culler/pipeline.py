"""Pipeline orchestration: scan -> analyze (cached, prefetched) -> group -> score -> curate."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ValidationError

from media_culler.analyzers import Analyzer, build_analyzers
from media_culler.cache import AnalysisCache
from media_culler.config import Settings
from media_culler.curation import apply_recommendations, rank_collection, select_best_shots
from media_culler.duplicates import assign_duplicate_groups
from media_culler.exceptions import AnalyzerUnavailable, ImageLoadError
from media_culler.imaging import ImageContext, configure_pillow
from media_culler.logging_setup import silence_third_party
from media_culler.models import ImageRecord
from media_culler.progress import NullReporter, ProgressReporter
from media_culler.scanner import scan_directory
from media_culler.scoring import score_record

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    records: list[ImageRecord]
    settings: Settings
    input_directory: Path
    started_at: str
    duration_seconds: float
    duplicate_groups: int = 0
    skipped_analyzers: dict[str, str] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0


_WorkerOutput = tuple[ImageContext, dict[str, BaseModel], dict[str, str]]


def _decode_and_run(path: Path, max_pixels: int, light: Sequence[Analyzer]) -> _WorkerOutput:
    """Runs in a worker thread: decode once, then the thread-safe analyzers."""
    ctx = ImageContext.from_path(path, max_pixels)
    outputs: dict[str, BaseModel] = {}
    errors: dict[str, str] = {}
    for analyzer in light:
        try:
            outputs[analyzer.name] = analyzer.analyze(ctx)
        except Exception as error:
            log.debug("%s failed on %s", analyzer.name, path, exc_info=True)
            errors[analyzer.name] = f"{type(error).__name__}: {error}"
    return ctx, outputs, errors


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        reporter: ProgressReporter | None = None,
        *,
        analyzers: list[Analyzer] | None = None,
        strict: bool = False,
    ) -> None:
        self.settings = settings
        self.reporter = reporter or NullReporter()
        self._analyzers = analyzers
        self.strict = strict

    # ------------------------------------------------------------------ public

    def run(self, input_directory: Path | None = None) -> RunResult:
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        root = input_directory or self.settings.paths.input_directory
        configure_pillow(self.settings.scan.max_pixels)

        records = scan_directory(root, self.settings, self.reporter)
        result = RunResult(
            records=records,
            settings=self.settings,
            input_directory=root.resolve(),
            started_at=started_at,
            duration_seconds=0.0,
        )

        valid = [r for r in records if r.status == "valid"]
        if valid:
            analyzers = (
                self._analyzers if self._analyzers is not None else build_analyzers(self.settings)
            )
            active = self._setup(analyzers, result)
            try:
                cache_path = (
                    self.settings.cache_db_path if self.settings.performance.use_cache else None
                )
                with AnalysisCache(cache_path) as cache:
                    self._analyze(valid, active, cache)
                    result.cache_hits, result.cache_misses = cache.hits, cache.misses
            finally:
                for analyzer in active:
                    analyzer.close()

        self._curate(result)
        result.duration_seconds = round(time.monotonic() - started, 2)
        return result

    # --------------------------------------------------------------- internals

    def _setup(self, analyzers: list[Analyzer], result: RunResult) -> list[Analyzer]:
        silence_third_party()
        self.reporter.stage("Loading models", None)
        active: list[Analyzer] = []
        for analyzer in analyzers:
            try:
                analyzer.setup()
                active.append(analyzer)
            except AnalyzerUnavailable as error:
                if self.strict:
                    raise
                result.skipped_analyzers[analyzer.name] = str(error)
                self.reporter.warn(f"{analyzer.name} analysis skipped: {error}")
                log.warning("%s skipped: %s", analyzer.name, error)
        return active

    def _hydrate(
        self,
        record: ImageRecord,
        analyzers: list[Analyzer],
        fingerprints: dict[str, str],
        cache: AnalysisCache,
    ) -> list[Analyzer]:
        """Fill record sections from cache; return analyzers that still need to run."""
        needed: list[Analyzer] = []
        for analyzer in analyzers:
            payload = cache.get(record.sha256 or "", analyzer.name, fingerprints[analyzer.name])
            if payload is not None:
                try:
                    setattr(
                        record, analyzer.section, analyzer.result_type.model_validate_json(payload)
                    )
                    continue
                except ValidationError:
                    log.debug("Stale cache entry for %s/%s", record.filename, analyzer.name)
            needed.append(analyzer)
        return needed

    def _analyze(
        self, valid: list[ImageRecord], analyzers: list[Analyzer], cache: AnalysisCache
    ) -> None:
        self.reporter.stage("Analyzing", len(valid))
        fingerprints = {a.name: a.fingerprint() for a in analyzers}
        workers = self.settings.performance.workers
        max_pixels = self.settings.scan.max_pixels

        pending: deque[tuple[ImageRecord, list[Analyzer], Future[_WorkerOutput] | None]] = deque()
        iterator = iter(valid)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="culler") as pool:
            try:

                def fill() -> None:
                    # Bounded look-ahead keeps memory flat: at most `workers` decoded images.
                    while len(pending) < workers:
                        record = next(iterator, None)
                        if record is None:
                            return
                        needed = self._hydrate(record, analyzers, fingerprints, cache)
                        future = None
                        if needed:
                            light = [a for a in needed if a.thread_safe]
                            future = pool.submit(
                                _decode_and_run, Path(record.path), max_pixels, light
                            )
                        pending.append((record, needed, future))

                fill()
                while pending:
                    record, needed, future = pending.popleft()
                    fill()
                    if future is not None:
                        self._finish(record, needed, future, fingerprints, cache)
                    self.reporter.advance(record.filename)
            finally:
                for _, _, future in pending:
                    if future is not None:
                        future.cancel()

    def _finish(
        self,
        record: ImageRecord,
        needed: list[Analyzer],
        future: Future[_WorkerOutput],
        fingerprints: dict[str, str],
        cache: AnalysisCache,
    ) -> None:
        try:
            ctx, outputs, errors = future.result()
        except Exception as error:  # ImageLoadError or anything unexpected: isolate this image
            record.status = "error"
            record.error = (
                str(error)
                if isinstance(error, ImageLoadError)
                else (f"{type(error).__name__}: {error}")
            )
            log.info("Cannot analyze %s: %s", record.relative_path, record.error)
            return

        for analyzer in needed:
            if analyzer.name in errors:
                record.analyzer_errors[analyzer.name] = errors[analyzer.name]
                continue
            result = outputs.get(analyzer.name)
            if result is None:
                try:
                    result = analyzer.analyze(ctx)
                except Exception as error:
                    log.debug("%s failed on %s", analyzer.name, record.path, exc_info=True)
                    record.analyzer_errors[analyzer.name] = f"{type(error).__name__}: {error}"
                    continue
            setattr(record, analyzer.section, result)
            if record.sha256:
                cache.put(
                    record.sha256,
                    analyzer.name,
                    fingerprints[analyzer.name],
                    result.model_dump_json(),
                )

    def _curate(self, result: RunResult) -> None:
        records = result.records
        if self.settings.duplicates.enabled:
            self.reporter.stage("Grouping similar images", None)
            result.duplicate_groups = assign_duplicate_groups(
                records, self.settings.duplicates.hamming_threshold
            )
        self.reporter.stage("Scoring", None)
        for record in records:
            if record.status == "valid":
                record.scores = score_record(record, self.settings)
        rank_collection(records)
        apply_recommendations(records, self.settings.thresholds)
        select_best_shots(records)
