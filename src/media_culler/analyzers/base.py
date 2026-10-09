"""Analyzer plugin contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from media_culler.imaging import ImageContext


class Analyzer(ABC):
    """One independent analysis step producing a typed section of an ImageRecord.

    ``thread_safe`` analyzers run in prefetch worker threads alongside image decoding;
    others (GPU models, MediaPipe) run serially on the main thread.
    """

    name: ClassVar[str]
    section: ClassVar[str]  # ImageRecord attribute this analyzer fills
    result_type: ClassVar[type[BaseModel]]
    version: ClassVar[str] = "1"
    thread_safe: ClassVar[bool] = False

    def setup(self) -> None:  # noqa: B027 - optional hook
        """Load models / validate environment. Raise AnalyzerUnavailable if impossible."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""

    def fingerprint(self) -> str:
        """Cache key: change whenever outputs would change (version, model, parameters)."""
        return self.version

    @abstractmethod
    def analyze(self, ctx: ImageContext) -> BaseModel:
        """Analyze one decoded image. May raise; the pipeline isolates failures per image."""
