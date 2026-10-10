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
| `culler organize REPORT --dest OUT` | Dry-run plan; add `--apply` to copy (or `--mode move/symlink`), `--group-folders` for a folder per burst |
| `culler undo --dest OUT` | Reverse the last organize run (dry run; add `--apply`) |
| `culler eval REPORT LABELS.csv` | Score the tool's decisions against your own labels (`--init` writes a template) |
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
cli.py ─────────► pipeline.py ──► scanner.py        discover files, metadata, EXIF time, sha256 (threaded)
 │  (Typer)         │            analyzers/          technical · scene · vision (SigLIP) · faces (MediaPipe)
 │                  │            cache.py            sqlite, keyed by (content hash, analyzer, fingerprint)
 ▼                  ├──────────► duplicates.py       appearance clusters + burst links → groups
ui/ (Rich)          ├──────────► scoring/            PURE functions: metrics → 0-100 scores
 render, progress   │                                (technical · landscape · group · subject · composition)
                    └──────────► curation.py         rank · recommend · best shot per group
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

## Scenes, group photos and bursts

**Landscapes.** A photo is judged as a landscape when the vision model's top content label is
"landscape" (or, with AI off, when a strong horizon is found) *and* no face is big enough to be
the subject (`scene.people_min_face_area`). Landscapes then use scenery rules instead of portrait
rules: exposure from highlight/shadow **clipping** (not mean brightness, which punished sunsets and
snow), **dynamic range** instead of global contrast, **tiled sharpness** (75th-percentile tile, so
a crisp scene with empty sky is not "blurry"), and composition from **horizon level** (signed tilt),
**horizon placement** and the **focal point** (spectral-residual saliency). Subject is excluded from
the weighted score rather than scored as a neutral 50. A landscape that is out of focus is capped at
`review` (`thresholds.soft_focus_sharpness`), so good composition cannot rescue it.

**Group photos.** Only *subjects* count: a face must cover `selection.min_face_area` of the frame and
`selection.relative_face_area` of the largest face, so passers-by are ignored but a distant group is
still a group. A subject whose eyes are clearly closed (`selection.closed_ear_threshold`; judged only on
faces big enough to read and not in profile) caps that face and the whole subject score, and caps the
photo at `review` (`thresholds.closed_eyes_gate`).

**Bursts and best-shot choice.** Capture time comes from EXIF (never file mtime). A frame joins any frame
within `bursts.max_gap_seconds` that looks continuous with it (`bursts.hamming_threshold`, looser than
the duplicate threshold), so a panning or moving sequence groups even when its first and last frames
differ. Groups are labelled `duplicate` (byte-identical), `burst` or `similar`. Within a group frames
are compared on a *selection score* (final score without the AI component, because AI labels flip
between near-identical frames) minus a **relative-sharpness** penalty
(`selection.sharpness_weight`). Within `selection.margin` points of the best, the frame with the fewest
problems wins (closed eyes, or a subject missing compared with the group's median); outside the margin
quality wins. Every frame's `selection_reason` says why it was kept or dropped.

All of these thresholds are heuristics chosen without a labelled photo set; see *Known limitations*
and tune them in `culler.yaml` (`scene`, `selection`, `bursts`, `thresholds`).

Report schema is now **3** (additive). Schema-2 reports still load.

## RAW, organizing safely, and measuring

**RAW.** Camera RAW (`.cr2 .cr3 .nef .arw .dng .orf .raf .rw2 .pef .srw .nrw`) is decoded with `rawpy`
(extra `formats`) at full resolution, with the camera's orientation applied. `scan.raw_half_size: true`
is ~4x faster but reads sharpness at half resolution, so RAW and JPEG scores are less comparable.
Without `rawpy`, a RAW file fails with a message saying what to install instead of being scored on a
tiny embedded preview. With RAW+JPEG shooting, a JPEG beside a same-named RAW (same folder, capture
times within 2 s) is that RAW's *companion*: it joins no group and inherits the RAW's decision, so the
pair is never "culled" as a two-frame burst (`scan.pair_raw_jpeg`).

**Organizing.** `culler organize` is a dry run until `--apply`. Copies are written to a temp file and
checked against the SHA-256 recorded at analysis before being renamed into place; a cross-device move
deletes its source only after that check. Nothing is overwritten. `.xmp`/`.aae`/`.pp3`/`.dop`/`.on1`/
`.thm` sidecars and Live Photo `.mov` files travel with their photo (`--no-sidecars` to skip).
Every run is journalled to `<dest>/.culler/undo/<time>.jsonl`; `culler undo --dest <dest> --apply`
reverses the latest one. Undo removes a copy only if it still matches what was made and the original
still exists, restores a moved file only if its old location is free, and skips (and tells you) anything
else. Report schema is **4** (adds `paired_with`); older reports still load.

**Measuring.** `culler eval REPORT labels.csv --init` writes a template with one row per photo. Fill in
only what you care about: `verdict` (keep/reject), `scene`, `closed_eyes`, and `group` + `best` for the
frame you would keep from a burst. `culler eval REPORT labels.csv` then prints false-reject rate (your
keepers it rejected, the costly error), keep/reject precision and recall, scene accuracy, closed-eye
precision and recall, and how often your best frame survives. Change a threshold, re-run `culler run`
(analysis is cached), and re-run `eval` to see whether it helped. Re-scanning an unchanged library does
not re-hash it: file hashes are remembered by path, size and modification time.

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
* Composition is face-based except for landscapes: a photo that is neither (food, objects, architecture)
  still gets a neutral 50 for subject and composition.
* Eye state uses an eye-aspect-ratio threshold on MediaPipe landmarks. It is conservative, but it has
  not been validated across faces, glasses or lighting; a wink is not detected and glare can hide eyes.
  `closed_ear_threshold` and `closed_eyes_gate` are there to tune or turn it off.
* Horizon detection finds one dominant, near-level line. Mountains, shorelines at an angle, or
  reflections can fool it; weak detections (`scene.horizon_min_confidence`) are ignored.
* Burst grouping needs EXIF capture time. Without it, frames group by appearance alone. A RAW+JPEG
  pair is one shot (see above), but a JPEG exported *later* under another name is not recognised as
  a companion and will group with its RAW as a duplicate. Bursts are keyed to capture time as written, with no
  time zone, so two cameras with different clock settings will not link.
* MediaPipe landmarks carry no visibility, so that sub-score is a constant 1.0.
* Hashes detect *similar framing*, so a heavily blurred twin of a photo groups with it; that is
  intentional (the sharper one is picked as representative).
* HEIC/HEIF is decoded through `pillow-heif` and camera RAW through `rawpy` (both in extra `formats`).
  The RAW path is tested against a fake `rawpy` plus a CI job that installs the real one, but not yet
  against a corpus of real RAW files from every camera maker; try a few of yours first. CR3 and recent
  bodies depend on your installed LibRaw version.
* Duplicate grouping is vectorised but still compares every leader with every image: ~1.4 s for 20,000
  random hashes, growing quadratically. Real libraries cluster, so it is usually faster.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && ruff format --check src tests && mypy src && pytest --cov=media_culler
```
