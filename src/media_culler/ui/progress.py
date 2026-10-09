from __future__ import annotations

from types import TracebackType

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class _CountColumn(ProgressColumn):
    """'n/N' for counted stages; blank for open-ended ones (scoring, model loading)."""

    def render(self, task):  # type: ignore[no-untyped-def]
        if task.fields.get("indeterminate"):
            return Text("")
        return Text(f"{int(task.completed)}/{int(task.total or 0)}", style="progress.download")


class RichReporter:
    """Implements ProgressReporter: one line per stage, finished stages keep a green tick."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._progress = Progress(
            SpinnerColumn(finished_text="[ok]✓[/]"),
            TextColumn("[bold]{task.description:<24}"),
            BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
            _CountColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[muted]{task.fields[detail]}[/]", justify="left"),
            console=console,
            transient=False,
            expand=True,
        )
        self._task: TaskID | None = None

    def __enter__(self) -> RichReporter:
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._finish_current()
        self._progress.stop()

    def _finish_current(self) -> None:
        if self._task is None:
            return
        task = self._progress.tasks[self._task]
        total = task.total if task.total is not None else 1
        self._progress.update(self._task, total=total, completed=total, detail="")
        self._task = None

    def stage(self, name: str, total: int | None = None) -> None:
        self._finish_current()
        self._task = self._progress.add_task(
            name, total=total, detail="", indeterminate=total is None
        )

    def advance(self, detail: str = "") -> None:
        if self._task is not None:
            self._progress.update(self._task, advance=1, detail=detail[:32])

    def warn(self, message: str) -> None:
        self._progress.console.print(f"[warn]![/] {message}")
