from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

THEME = Theme(
    {
        "select": "bold green",
        "usable": "cyan",
        "review": "yellow",
        "reject_candidate": "red",
        "unscored": "dim",
        "similar_alternates": "magenta",
        "ok": "green",
        "warn": "yellow",
        "fail": "bold red",
        "muted": "dim",
        "accent": "bold cyan",
    }
)


def make_console(*, quiet: bool = False, stderr: bool = False) -> Console:
    return Console(theme=THEME, quiet=quiet, stderr=stderr, highlight=False)
