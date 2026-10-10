"""Command-line interface (Typer + Rich). Thin: parse -> call core -> render."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from rich.table import Table

from media_culler import __version__
from media_culler.config import DEFAULT_CONFIG_YAML, Settings, load_settings
from media_culler.doctor import run_checks
from media_culler.exceptions import ConfigError, CullerError
from media_culler.logging_setup import configure_logging
from media_culler.model_download import download_file
from media_culler.organizer import Mode, execute, latest_journal, plan_operations, undo_run
from media_culler.pipeline import Pipeline
from media_culler.reporting import build_report, load_report, write_reports
from media_culler.scanner import scan_directory
from media_culler.ui.console import make_console
from media_culler.ui.progress import RichReporter
from media_culler.ui.render import (
    human_size,
    organize_plan_table,
    print_results,
)

app = typer.Typer(
    name="culler",
    help="Score, group and cull photo collections.  Try [bold]culler tldr[/] for quick examples.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
models_app = typer.Typer(help="Manage downloadable model files.", no_args_is_help=True)
config_app = typer.Typer(help="Create and inspect configuration.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")

EXIT_ERROR, EXIT_USAGE, EXIT_INTERRUPTED = 1, 2, 130

ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        envvar="CULLER_CONFIG",
        help="YAML config (default: ./culler.yaml if present).",
    ),
]
VerboseOpt = Annotated[
    int,
    typer.Option("--verbose", "-v", count=True, help="-v info, -vv debug logging on the terminal."),
]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"culler {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    pass


def _load(config: Path | None, overrides: dict[str, Any] | None = None) -> Settings:
    try:
        return load_settings(config, overrides)
    except ConfigError as error:
        make_console(stderr=True).print(f"[fail]Config error[/]\n{error}")
        raise typer.Exit(EXIT_USAGE) from error


def _fail(error: Exception) -> typer.Exit:
    make_console(stderr=True).print(f"[fail]Error:[/] {error}")
    return typer.Exit(EXIT_ERROR)


# ----------------------------------------------------------------------------- run


@app.command()
def run(
    input_dir: Annotated[
        Path | None,
        typer.Argument(help="Folder of photos (default from config).", show_default=False),
    ] = None,
    config: ConfigOpt = None,
    reports_dir: Annotated[
        Path | None,
        typer.Option("--reports-dir", "-o", help="Where to write inventory.json/.csv."),
    ] = None,
    no_ai: Annotated[bool, typer.Option("--no-ai", help="Skip SigLIP AI assessment.")] = False,
    no_faces: Annotated[bool, typer.Option("--no-faces", help="Skip face analysis.")] = False,
    no_duplicates: Annotated[
        bool, typer.Option("--no-duplicates", help="Skip similar-image grouping.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Ignore and don't update the cache.")
    ] = False,
    workers: Annotated[
        int | None,
        typer.Option("--workers", "-j", min=1, max=64, help="Worker threads for I/O and OpenCV."),
    ] = None,
    formats: Annotated[
        list[str] | None,
        typer.Option(
            "--format", "-f", help="Report formats: json, csv (repeatable). Default: both."
        ),
    ] = None,
    top: Annotated[int, typer.Option("--top", min=0, help="Rows in the 'Top images' table.")] = 10,
    strict: Annotated[
        bool, typer.Option("--strict", help="Fail instead of skipping an unavailable analyzer.")
    ] = False,
    download_models: Annotated[
        bool, typer.Option("--download-models", help="Fetch the face model first if it is missing.")
    ] = False,
    json_out: Annotated[
        bool, typer.Option("--json", help="Print the summary as JSON on stdout; no other output.")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Print only report paths.")] = False,
    verbose: VerboseOpt = 0,
) -> None:
    """Analyse a folder, score every image and write a report."""
    overrides: dict[str, Any] = {}
    if no_ai:
        overrides.setdefault("vision", {})["enabled"] = False
    if no_faces:
        overrides.setdefault("faces", {})["enabled"] = False
    if no_duplicates:
        overrides.setdefault("duplicates", {})["enabled"] = False
    if no_cache:
        overrides.setdefault("performance", {})["use_cache"] = False
    if workers:
        overrides.setdefault("performance", {})["workers"] = workers
    if reports_dir:
        overrides.setdefault("paths", {})["reports_directory"] = str(reports_dir)
    settings = _load(config, overrides)

    chosen = {f.lower() for f in (formats or ["json", "csv"])}
    if not chosen <= {"json", "csv"}:
        make_console(stderr=True).print("[fail]--format must be json and/or csv[/]")
        raise typer.Exit(EXIT_USAGE)

    silent = quiet or json_out
    console = make_console(quiet=silent)
    configure_logging(console, verbose, settings.paths.reports_directory / "culler.log")
    root = input_dir or settings.paths.input_directory

    console.print(f"[accent]AI Media Culler[/] [muted]v{__version__}[/]  [muted]{root}[/]")
    try:
        if download_models and settings.faces.enabled and not settings.face_model_path.is_file():
            _download_face_model(settings, console)

        with RichReporter(console) as reporter:
            result = Pipeline(settings, reporter, strict=strict).run(root)

        report = build_report(result, top_n=top)
        paths = write_reports(report, settings.paths.reports_directory, chosen)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise _fail(error) from error
    except CullerError as error:
        raise _fail(error) from error
    except KeyboardInterrupt:
        make_console(stderr=True).print(
            "\n[warn]Interrupted.[/] Finished analyses are cached; re-run to resume."
        )
        raise typer.Exit(EXIT_INTERRUPTED) from None

    primary = paths.get("json") or next(iter(paths.values()), None)
    if json_out:
        typer.echo(
            json.dumps(
                {
                    "reports": {k: str(v) for k, v in paths.items()},
                    "summary": report.summary.model_dump(mode="json"),
                },
                indent=2,
            )
        )
    elif quiet:
        for path in paths.values():
            typer.echo(str(path))
    else:
        print_results(console, report, primary, top)
        if len(paths) > 1:
            console.print(
                "[muted]Also wrote: "
                + ", ".join(str(p) for p in paths.values() if p != primary)
                + "[/]"
            )


# ---------------------------------------------------------------------------- scan


@app.command()
def scan(
    input_dir: Annotated[Path | None, typer.Argument(show_default=False)] = None,
    config: ConfigOpt = None,
    verbose: VerboseOpt = 0,
) -> None:
    """Inventory a folder (formats, sizes, unreadable files) without analysing it."""
    settings = _load(config)
    console = make_console()
    configure_logging(console, verbose, None)
    root = input_dir or settings.paths.input_directory
    try:
        with RichReporter(console) as reporter:
            records = scan_directory(root, settings, reporter)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise _fail(error) from error

    valid = [r for r in records if r.status == "valid"]
    table = Table(title=f"{root}", box=box.SIMPLE_HEAVY, title_justify="left")
    table.add_column("Format")
    table.add_column("Files", justify="right")
    table.add_column("Size", justify="right")
    by_format: dict[str, list[int]] = {}
    for r in valid:
        by_format.setdefault(r.format or r.extension, []).append(r.size_bytes)
    for fmt, sizes in sorted(by_format.items()):
        table.add_row(fmt, str(len(sizes)), human_size(sum(sizes)))
    console.print(table)
    console.print(
        f"[ok]{len(valid)}[/] readable · [{'fail' if len(records) > len(valid) else 'muted'}]"
        f"{len(records) - len(valid)}[/] unreadable · "
        f"{human_size(sum(r.size_bytes for r in valid))}"
    )
    for r in records:
        if r.status == "error":
            console.print(f"  [fail]{r.relative_path}[/]: {r.error}")


# --------------------------------------------------------------------------- report


@app.command()
def report(
    report_file: Annotated[
        Path | None,
        typer.Argument(help="Default: <reports_directory>/inventory.json", show_default=False),
    ] = None,
    config: ConfigOpt = None,
    top: Annotated[int, typer.Option("--top", min=0)] = 10,
) -> None:
    """Re-display a previous report without re-analysing."""
    settings = _load(config)
    path = report_file or settings.paths.reports_directory / "inventory.json"
    try:
        loaded = load_report(path)
    except CullerError as error:
        raise _fail(error) from error
    print_results(make_console(), loaded, path, top)


# ------------------------------------------------------------------------ organize


@app.command()
def organize(
    report_file: Annotated[Path, typer.Argument(help="inventory.json from `culler run`.")],
    dest: Annotated[
        Path | None,
        typer.Option("--dest", "-d", help="Destination root (default: output_directory)."),
    ] = None,
    mode: Annotated[
        Mode, typer.Option("--mode", "-m", help="copy (safe default), move or symlink.")
    ] = Mode.COPY,
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Only these buckets, e.g. --only select --only usable."),
    ] = None,
    keep_duplicates_in_place: Annotated[
        bool,
        typer.Option(
            "--keep-duplicates-in-place",
            help="Don't route weaker duplicates to similar_alternates/.",
        ),
    ] = False,
    group_folders: Annotated[
        bool,
        typer.Option(
            "--group-folders",
            help="Put each burst/duplicate group in its own folder, e.g. select/burst_0007/.",
        ),
    ] = False,
    no_sidecars: Annotated[
        bool,
        typer.Option(
            "--no-sidecars", help="Don't take .xmp/.aae files and Live Photo videos along."
        ),
    ] = False,
    apply: Annotated[
        bool, typer.Option("--apply", help="Actually do it. Without this it is a dry run.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the confirmation for --mode move.")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Sort photos into select/ usable/ review/ reject_candidate/ folders. Dry run by default."""
    settings = _load(config)
    console = make_console()
    try:
        loaded = load_report(report_file)
    except CullerError as error:
        raise _fail(error) from error

    destination = dest or settings.paths.output_directory
    ops = plan_operations(
        loaded,
        destination,
        separate_duplicates=not keep_duplicates_in_place,
        only=set(only) if only else None,
        include_sidecars=not no_sidecars,
        group_folders=group_folders,
    )
    if not ops:
        console.print("[warn]Nothing to organize.[/]")
        return

    console.print(organize_plan_table(ops))
    console.print(f"[muted]{mode.value} → {destination}[/]")
    input_root = Path(loaded.input_directory).resolve()
    if (
        input_root in destination.resolve().parents
        and destination.resolve() != settings.paths.output_directory.resolve()
    ):
        console.print(
            "[warn]The destination is inside the scanned folder: a later `culler run` on it "
            "will pick up the copies too. Use paths.output_directory or a folder outside.[/]"
        )
    if not apply:
        console.print(
            Panel(
                "Dry run: nothing was changed. Add [bold]--apply[/] to proceed.",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        return
    if (
        mode is Mode.MOVE
        and not yes
        and not typer.confirm(
            f"Move {len(ops)} original files? This changes your source folder", default=False
        )
    ):
        raise typer.Exit(EXIT_ERROR)

    outcome = execute(ops, mode, destination)
    if outcome.journal:
        console.print(f"[muted]Undo with: culler undo --dest {destination} --apply[/]")
    console.print(
        f"[ok]{len(outcome.done)} done[/] · {len(outcome.skipped)} skipped · "
        f"[{'fail' if outcome.failed else 'muted'}]{len(outcome.failed)} failed[/]"
    )
    for op, reason in (outcome.skipped + outcome.failed)[:10]:
        console.print(f"  [muted]{op.source.name}: {reason}[/]")
    if outcome.failed:
        raise typer.Exit(EXIT_ERROR)


@app.command()
def undo(
    dest: Annotated[
        Path | None,
        typer.Option("--dest", "-d", help="Destination root used by organize."),
    ] = None,
    journal: Annotated[
        Path | None, typer.Option("--journal", help="A specific run (default: the latest).")
    ] = None,
    apply: Annotated[
        bool, typer.Option("--apply", help="Actually do it. Without this it is a dry run.")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Reverse the last `organize` run. Dry run by default; never deletes unverified files."""
    settings = _load(config)
    console = make_console()
    destination = dest or settings.paths.output_directory
    target = journal or latest_journal(destination)
    if target is None or not target.exists():
        console.print(f"[warn]No organize run to undo in {destination}.[/]")
        raise typer.Exit(EXIT_ERROR)

    outcome = undo_run(target, destination, apply=apply)
    verb = "Reverted" if apply else "Would revert"
    console.print(f"[ok]{verb} {len(outcome.reverted)}[/] · {len(outcome.skipped)} skipped")
    for path, reason in outcome.skipped[:10]:
        console.print(f"  [muted]{path.name}: {reason}[/]")
    if not apply:
        console.print(
            Panel(
                "Dry run: nothing was changed. Add [bold]--apply[/] to proceed.",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )


# ---------------------------------------------------------------------------- models


def _download_face_model(settings: Settings, console: Any) -> Path:
    destination = settings.face_model_path
    with Progress(
        TextColumn("[bold]Downloading face model"),
        BarColumn(bar_width=None),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("dl", total=None)

        def on_progress(received: int, total: int | None) -> None:
            progress.update(task, completed=received, total=total)

        return download_file(
            settings.faces.model_url,
            destination,
            sha256=settings.faces.model_sha256,
            on_progress=on_progress,
        )


@models_app.command("download")
def models_download(config: ConfigOpt = None) -> None:
    """Download the MediaPipe face-landmarker model."""
    settings = _load(config)
    console = make_console()
    try:
        path = _download_face_model(settings, console)
    except CullerError as error:
        raise _fail(error) from error
    console.print(f"[ok]✓[/] Saved {path} ({human_size(path.stat().st_size)})")
    console.print(
        "[muted]The SigLIP vision model downloads itself from Hugging Face on first use.[/]"
    )


@models_app.command("status")
def models_status(config: ConfigOpt = None) -> None:
    """Show which model files are present."""
    settings = _load(config)
    console = make_console()
    path = settings.face_model_path
    if path.is_file():
        console.print(f"[ok]✓[/] face model: {path} ({human_size(path.stat().st_size)})")
    else:
        console.print(f"[warn]✗[/] face model missing: {path}  →  [bold]culler models download[/]")


# ---------------------------------------------------------------------------- config


@config_app.command("init")
def config_init(
    path: Annotated[Path, typer.Argument()] = Path("culler.yaml"),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a commented starter config."""
    console = make_console()
    if path.exists() and not force:
        console.print(f"[warn]{path} already exists.[/] Use --force to overwrite.")
        raise typer.Exit(EXIT_ERROR)
    path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
    console.print(f"[ok]✓[/] Wrote {path}")


@config_app.command("show")
def config_show(config: ConfigOpt = None) -> None:
    """Print the effective configuration (defaults + file)."""
    settings = _load(config)
    make_console().print_json(settings.model_dump_json())


@config_app.command("validate")
def config_validate(config: ConfigOpt = None) -> None:
    """Check a config file and exit non-zero if it is invalid."""
    _load(config)
    make_console().print("[ok]✓[/] Configuration is valid.")


# --------------------------------------------------------------------- doctor / tldr


@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """Check your environment and say exactly how to fix anything missing."""
    settings = _load(config)
    console = make_console()
    table = Table(box=box.SIMPLE_HEAVY, title="Environment", title_justify="left")
    table.add_column("")
    table.add_column("Check")
    table.add_column("Detail")
    symbols = {"ok": "[ok]✓[/]", "warn": "[warn]![/]", "fail": "[fail]✗[/]"}
    worst = "ok"
    for check in run_checks(settings):
        detail = check.detail + (f"\n[muted]fix: {check.fix}[/]" if check.fix else "")
        table.add_row(symbols[check.status], check.name, detail)
        if check.status == "fail" or (check.status == "warn" and worst == "ok"):
            worst = check.status
    console.print(table)
    if worst == "fail":
        raise typer.Exit(EXIT_ERROR)


@app.command()
def tldr() -> None:
    """Quick, copy-pasteable examples (tldr-pages style)."""
    text = (
        resources.files("media_culler.data").joinpath("culler.tldr.md").read_text(encoding="utf-8")
    )
    make_console().print(Markdown(text))


if __name__ == "__main__":  # pragma: no cover
    app()
