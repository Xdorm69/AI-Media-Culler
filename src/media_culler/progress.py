"""UI-agnostic progress interface so the core pipeline never imports Rich."""

from __future__ import annotations

from typing import Protocol


class ProgressReporter(Protocol):
    def stage(self, name: str, total: int | None = None) -> None:
        """Begin a new stage (finishing the previous one)."""

    def advance(self, detail: str = "") -> None:
        """One unit of work finished in the current stage."""

    def warn(self, message: str) -> None:
        """Non-fatal notice to surface to the user."""


class NullReporter:
    def stage(self, name: str, total: int | None = None) -> None:
        pass

    def advance(self, detail: str = "") -> None:
        pass

    def warn(self, message: str) -> None:
        pass
