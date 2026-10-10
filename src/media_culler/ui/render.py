"""Pure rendering helpers: data in, Rich renderables out."""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from media_culler.models import Recommendation
from media_culler.organizer import DUPLICATES_BUCKET, Operation
from media_culler.reporting import Report

_ORDER = [
    Recommendation.SELECT,
    Recommendation.USABLE,
    Recommendation.REVIEW,
    Recommendation.REJECT,
    Recommendation.UNSCORED,
]
_LABEL = {
    Recommendation.SELECT: "select",
    Recommendation.USABLE: "usable",
    Recommendation.REVIEW: "review",
    Recommendation.REJECT: "reject candidates",
    Recommendation.UNSCORED: "unscored",
}

ANALYZER_HINTS = {
    "vision": "pip install 'ai-media-culler[ai]'  (or disable with --no-ai)",
    "faces": "pip install 'ai-media-culler[faces]' && culler models download  (or --no-faces)",
}


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"


def tldr_panel(report: Report, report_path: Path | None = None) -> Panel:
    s = report.summary
    counts = s.recommendations
    parts = []
    for rec in _ORDER:
        n = counts.get(rec.value, 0)
        if n:
            parts.append(Text(f"{n} {_LABEL[rec]}", style=rec.value))
    headline = Text(
        f"{s.valid_images} photos analysed in {human_duration(report.duration_seconds)} → "
    )
    for index, part in enumerate(parts):
        if index:
            headline.append(" · ", style="muted")
        headline.append_text(part)
    if not parts:
        headline.append("nothing to score", style="muted")

    lines: list[RenderableType] = [headline]
    if s.duplicate_groups:
        plural = "s" if s.duplicate_groups != 1 else ""
        bursts = (
            f" ({s.burst_groups} burst{'s' if s.burst_groups != 1 else ''})"
            if s.burst_groups
            else ""
        )
        lines.append(
            Text(
                f"{s.duplicate_groups} group{plural} of similar shots{bursts}; "
                f"{s.redundant_images} weaker alternate{'s' if s.redundant_images != 1 else ''} "
                "you can skip"
            )
        )
    if s.errors:
        lines.append(
            Text(f"{s.errors} file{'s' if s.errors != 1 else ''} could not be read", style="warn")
        )
    if report_path:
        lines.append(Text.assemble(("Report: ", "muted"), str(report_path)))
    return Panel(
        Group(*lines),
        title="[accent]TL;DR[/]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def recommendation_table(report: Report) -> Table:
    s = report.summary
    total = max(1, s.valid_images)
    table = Table(title="Recommendations", box=box.SIMPLE_HEAVY, title_justify="left")
    table.add_column("Bucket")
    table.add_column("Count", justify="right")
    table.add_column("Share", justify="right")
    table.add_column("", min_width=20)
    for rec in _ORDER:
        n = s.recommendations.get(rec.value, 0)
        if not n and rec is Recommendation.UNSCORED:
            continue
        share = n / total
        table.add_row(
            Text(_LABEL[rec], style=rec.value),
            str(n),
            f"{share:.0%}",
            Text("█" * round(share * 20), style=rec.value),
        )
    return table


def top_table(report: Report) -> Table | None:
    if not report.summary.top:
        return None
    table = Table(title="Top images", box=box.SIMPLE_HEAVY, title_justify="left")
    table.add_column("#", justify="right", style="muted")
    table.add_column("File", overflow="ellipsis", no_wrap=True, max_width=44)
    table.add_column("Score", justify="right")
    table.add_column("Verdict")
    for index, item in enumerate(report.summary.top, 1):
        table.add_row(
            str(index),
            item.relative_path,
            f"{item.final_score:.1f}",
            Text(_LABEL[item.recommendation], style=item.recommendation.value),
        )
    return table


def notes_panel(report: Report) -> Panel | None:
    lines: list[Text] = []
    for name, reason in report.skipped_analyzers.items():
        lines.append(Text.assemble((f"{name} skipped: ", "warn"), reason))
        hint = ANALYZER_HINTS.get(name)
        if hint:
            lines.append(Text(f"  fix: {hint}", style="muted"))
    for name, count in report.summary.analyzer_failures.items():
        lines.append(Text.assemble((f"{name}: ", "warn"), f"failed on {count} image(s); see log"))
    broken = [i for i in report.images if i.status == "error"]
    for image in broken[:5]:
        lines.append(Text.assemble((f"{image.relative_path}: ", "fail"), image.error or "unknown"))
    if len(broken) > 5:
        lines.append(Text(f"…and {len(broken) - 5} more unreadable file(s)", style="muted"))
    if not lines:
        return None
    return Panel(Group(*lines), title="Notes", border_style="yellow", box=box.ROUNDED)


def print_results(
    console: Console, report: Report, report_path: Path | None, top: int = 10
) -> None:
    console.print()
    console.print(tldr_panel(report, report_path))
    if report.summary.valid_images:
        console.print(recommendation_table(report))
        if top > 0:
            table = top_table(report)
            if table:
                console.print(table)
    if report.summary.scene_counts:
        mix = " · ".join(f"{n} {name}" for name, n in report.summary.scene_counts.items())
        console.print(f"[muted]scenes: {mix}[/]")
    if report.summary.closed_eyes_photos:
        console.print(
            f"[muted]closed eyes flagged in {report.summary.closed_eyes_photos} photo(s): "
            "capped at review; the CSV 'closed_eyes' column lists them[/]"
        )
    notes = notes_panel(report)
    if notes:
        console.print(notes)
    if report.cache_hits:
        total = report.cache_hits + report.cache_misses
        console.print(f"[muted]cache: {report.cache_hits}/{total} analyses reused[/]")


def organize_plan_table(ops: list[Operation], limit: int = 15) -> Table:
    table = Table(title=f"Plan ({len(ops)} files)", box=box.SIMPLE_HEAVY, title_justify="left")
    table.add_column("Bucket")
    table.add_column("Count", justify="right")
    table.add_column("Size", justify="right")
    buckets: dict[str, list[Operation]] = {}
    for op in ops:
        buckets.setdefault(op.bucket, []).append(op)
    for bucket, items in sorted(buckets.items()):
        style = "similar_alternates" if bucket == DUPLICATES_BUCKET else bucket
        table.add_row(
            Text(bucket, style=style), str(len(items)), human_size(sum(o.size_bytes for o in items))
        )
    return table
