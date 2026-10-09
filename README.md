# AI Media Culler

Score, group and cull photo collections. For every image it measures technical quality
(sharpness, exposure, contrast, resolution), asks a vision model how it looks
(SigLIP, zero-shot), analyses faces (eyes open, pose, size, framing), finds near-duplicates,
picks the best shot of each group and recommends **select / usable / review / reject**.

```
$ culler run ~/Photos/trip        # illustrative output
✓ Scanning ✓ Reading metadata ✓ Loading models ✓ Analyzing 412/412 ✓ Grouping ✓ Scoring
╭──────────────────────────────── TL;DR ────────────────────────────────╮
│ 412 photos analysed in 2m 31s → 38 select · 120 usable · 90 review …  │
│ 57 groups of similar shots; 86 weaker alternates you can skip         │
│ Report: reports/inventory.json                                        │
╰───────────────────────────────────────────────────────────────────────╯
```

## Install

```bash
pip install -e .                 # core: technical analysis + duplicates (no ML needed)
pip install -e ".[ai]"           # + SigLIP vision (torch, transformers)
pip install -e ".[faces]"        # + MediaPipe face analysis
pip install -e ".[all,dev]"      # everything + test tooling
culler models download           # fetch the face-landmarker model (one-time)
culler doctor                    # verify the environment, with exact fixes
```

Python 3.10+. `opencv-python-headless` replaces the old `opencv-python` + `opencv-contrib-python`
pair (installing both is a known source of import conflicts, and no GUI is used).

## Usage

| Command | What it does |
|---|---|
| `culler run [DIR]` | Analyse, score, write `reports/inventory.json` + `.csv` |
| `culler scan [DIR]` | Inventory only: formats, sizes, unreadable files |
| `culler report` | Re-display the last report |
| `culler organize REPORT --dest OUT` | Dry-run plan; add `--apply` to copy (or `--mode move/symlink`) |
| `culler models download\|status` | Manage the face model |
| `culler config init\|show\|validate` | Starter config / effective config / validation |
| `culler doctor` | Environment check |
| `culler tldr` | Copy-pasteable examples (also in `docs/tldr/culler.md`, tldr-pages format) |

Useful `run` flags: `--no-ai`, `--no-faces`, `--no-duplicates`, `--no-cache`, `-j N`, `-f csv`,
`--json` (pure JSON on stdout for scripts), `-q` (paths only), `-v/-vv` (logging),
`--strict` (fail instead of skipping an unavailable analyzer), `--download-models`.

Exit codes: `0` ok · `1` runtime error · `2` bad usage/config · `130` interrupted
(finished analyses are cached, so re-running resumes).

## Architecture

```
cli.py ─────────► pipeline.py ──► scanner.py        discover files, metadata, sha256 (threaded)
 │  (Typer)         │            analyzers/          technical · vision (SigLIP) · faces (MediaPipe)
 │                  │            cache.py            sqlite, keyed by (content hash, analyzer, fingerprint)
 ▼                  ├──────────► duplicates.py       pHash Hamming grouping
ui/ (Rich)          ├──────────► scoring/            PURE functions: metrics → 0-100 scores
 render, progress   └──────────► curation.py         rank · recommend · best shot
                    reporting.py   organizer.py   model_download.py   doctor.py
models.py (typed records)   config.py (validated settings)   imaging.py (single decode path)
```

Design rules that keep it modular:

* **One decode per image.** `imaging.load_rgb` is the only decoder and applies EXIF orientation.
* **Analyzers are plugins** (`Analyzer` base class): typed output, own cache fingerprint, failure
  isolated per image. Heavy backends (SigLIP, MediaPipe) are injected behind small protocols, so
  everything around them is tested with fakes.
* **Scoring is pure and re-run every time**, only analysis is cached. Changing weights or thresholds
  re-scores in milliseconds with zero re-analysis.
* **The core never imports Rich.** It reports through a `ProgressReporter` protocol.
* **Prefetch pipeline:** worker threads decode + run OpenCV analyzers for the next images while the
  main thread runs the model, with a bounded look-ahead so memory stays flat.

## Configuration

`culler config init` writes a commented `culler.yaml` (auto-discovered in the working directory).
Unknown keys are rejected so typos fail loudly. Weights are normalised over the components that
actually exist for an image, so disabling or losing the AI model never silently scores it as 0.

## Behaviour changes from v0.1 (deliberate)

| Issue in v0.1 | Now |
|---|---|
| Scanner (PIL) and faces (OpenCV) disagreed on EXIF rotation, so composition used the wrong width/height on portrait phone photos | One orientation-correct decode; header dims match pixels |
| `config.yaml` was never read; thresholds were hard-coded, and *different* (90/70/40 in the file vs 80/65/45 in code) | Config drives everything. Defaults are the **code's** values (80/65/45) so results don't shift |
| SigLIP ran 4× per image (once per label group) | One pass for all 18 labels. SigLIP scores labels independently, so per-label probabilities are identical |
| AI failure scored the AI component as **0** (silent −20 pts) | Failed/disabled components are excluded and weights renormalised |
| pHash computed twice; pairwise grouping re-parsed hex on every comparison | Computed once; vectorised NumPy Hamming distance |
| Re-running re-analysed everything | SQLite cache by content hash |
| Face model path/`num_faces` hard-coded; unused blendshape/matrix outputs computed | Configurable; unused outputs off |
| `print()` progress, no error summary, no log | Rich progress, TL;DR panel, Notes panel, rotating log file |
| Ad-hoc `test_*.py` scripts with hard-coded paths | Real pytest suite |
| `output_directory` configured but unused | `culler organize` |
| Report was a bare JSON list | Versioned report with summary and run metadata (`schema_version: 2`) |

## Known limitations

* Scoring formulas and thresholds are the original heuristics, **not calibrated** on a labelled set.
* Composition is face-based: images with no faces get a neutral 50 for subject and composition.
* MediaPipe landmarks carry no visibility, so that sub-score is a constant 1.0.
* Duplicate grouping is O(n²) (vectorised; fine for ~10⁴ images, use an LSH/BK-tree beyond that).
* Hashes detect *similar framing*, so a heavily blurred twin of a photo groups with it; that is
  intentional (the sharper one is picked as representative).
* HEIC/RAW are not supported (would need `pillow-heif` / `rawpy`).

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && ruff format --check src tests && mypy src && pytest --cov=media_culler
```
