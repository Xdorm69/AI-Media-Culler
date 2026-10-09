"""Typed, validated configuration.

Precedence (lowest to highest): built-in defaults < YAML file < CLI overrides.
Relative paths are resolved against the current working directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from media_culler.exceptions import ConfigError

log = logging.getLogger(__name__)

DEFAULT_CONFIG_FILENAMES = ("culler.yaml", "culler.yml")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


class _Section(BaseModel):
    # Unknown keys are almost always typos; fail loudly instead of silently ignoring them.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsConfig(_Section):
    input_directory: Path = Path("media/input")
    output_directory: Path = Path("media/output")
    reports_directory: Path = Path("reports")
    cache_directory: Path = Path(".culler_cache")
    models_directory: Path = Path("models")


class ScanConfig(_Section):
    extensions: list[str] = [
        ".arw",
        ".bmp",
        ".cr2",
        ".cr3",
        ".dng",
        ".heic",
        ".heif",
        ".jpeg",
        ".jpg",
        ".nef",
        ".orf",
        ".pef",
        ".png",
        ".raf",
        ".rw2",
        ".srw",
        ".tif",
        ".tiff",
        ".webp",
    ]
    recursive: bool = True
    follow_symlinks: bool = False
    exclude_dirs: list[str] = [".git", "__pycache__", ".culler_cache"]
    # Safety limit against decompression bombs / accidental gigapixel panoramas.
    max_pixels: int = Field(250_000_000, gt=0)

    @field_validator("extensions")
    @classmethod
    def _normalise_extensions(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one extension is required")
        return sorted({(e if e.startswith(".") else f".{e}").lower() for e in value})


class ThresholdsConfig(_Section):
    """Final-score cut-offs (0-100) for each recommendation bucket."""

    select: float = Field(80, ge=0, le=100)
    usable: float = Field(65, ge=0, le=100)
    review: float = Field(45, ge=0, le=100)
    # Images whose technical score is below this are rejected regardless of other scores.
    technical_floor: float = Field(25, ge=0, le=100)

    @model_validator(mode="after")
    def _ordered(self) -> ThresholdsConfig:
        if not (self.select > self.usable > self.review):
            raise ValueError("thresholds must satisfy select > usable > review")
        return self


class WeightsConfig(_Section):
    """Relative weight of each component in the final score.

    Weights need not sum to 1: they are normalised over the components that are
    actually available for an image (e.g. when AI analysis is disabled).
    """

    technical: float = Field(0.40, ge=0)
    ai: float = Field(0.20, ge=0)
    subject: float = Field(0.20, ge=0)
    composition: float = Field(0.20, ge=0)

    @model_validator(mode="after")
    def _positive_total(self) -> WeightsConfig:
        if self.technical <= 0:
            raise ValueError("technical weight must be > 0 (it is the only mandatory component)")
        return self


class DuplicatesConfig(_Section):
    enabled: bool = True
    # Max Hamming distance between 64-bit perceptual hashes to count as "similar".
    hamming_threshold: int = Field(8, ge=0, le=32)


class VisionConfig(_Section):
    enabled: bool = True
    model_name: str = "google/siglip-base-patch16-224"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    # Images are shrunk to this longest side before reaching the model processor.
    max_side: int = Field(896, ge=224)


class FacesConfig(_Section):
    enabled: bool = True
    max_faces: int = Field(10, ge=1, le=50)
    min_detection_confidence: float = Field(0.5, ge=0, le=1)
    min_presence_confidence: float = Field(0.5, ge=0, le=1)
    model_filename: str = "face_landmarker.task"
    model_url: str = FACE_MODEL_URL
    model_sha256: str | None = None  # set to enforce integrity on download
    # Detection runs on a downscaled copy; sharpness is still measured at full resolution.
    max_side: int = Field(1920, ge=256)


class PerformanceConfig(_Section):
    # Worker threads for scanning/hashing and for decode + OpenCV analysis prefetch.
    workers: int = Field(4, ge=1, le=64)
    use_cache: bool = True


class Settings(_Section):
    paths: PathsConfig = PathsConfig()
    scan: ScanConfig = ScanConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    weights: WeightsConfig = WeightsConfig()
    duplicates: DuplicatesConfig = DuplicatesConfig()
    vision: VisionConfig = VisionConfig()
    faces: FacesConfig = FacesConfig()
    performance: PerformanceConfig = PerformanceConfig()

    @property
    def face_model_path(self) -> Path:
        return self.paths.models_directory / self.faces.model_filename

    @property
    def cache_db_path(self) -> Path:
        return self.paths.cache_directory / "analysis.sqlite3"


# --------------------------------------------------------------------------- loading


def _format_validation_error(error: ValidationError, source: str) -> str:
    lines = [f"Invalid configuration ({source}):"]
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"  - {location}: {item['msg']}")
    return "\n".join(lines)


def find_config_file(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"Config file not found: {explicit}")
        return explicit
    for name in DEFAULT_CONFIG_FILENAMES:
        candidate = Path(name)
        if candidate.is_file():
            return candidate
    return None


def load_settings(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    """Load settings from ``path`` (or auto-discovered ``culler.yaml``) plus overrides.

    ``overrides`` is a nested dict, e.g. ``{"vision": {"enabled": False}}``.
    """
    config_file = find_config_file(path)
    data: dict[str, Any] = {}

    if config_file is not None:
        try:
            loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f"Cannot read config {config_file}: {error}") from error
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config {config_file} must be a YAML mapping at the top level.")
        data = loaded
        if "analysis" in data:  # legacy v0.1 section that was never actually honoured
            log.warning(
                "Ignoring legacy 'analysis' section in %s (use vision/faces/duplicates).",
                config_file,
            )
            data.pop("analysis")

    data = _deep_merge(data, overrides or {})

    try:
        return Settings.model_validate(data)
    except ValidationError as error:
        source = str(config_file) if config_file else "defaults + overrides"
        raise ConfigError(_format_validation_error(error, source)) from error


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


DEFAULT_CONFIG_YAML = """\
# AI Media Culler configuration. Every key is optional; defaults are shown.
# Relative paths are resolved from the directory you run `culler` in.

paths:
  input_directory: media/input
  output_directory: media/output    # used by `culler organize`
  reports_directory: reports
  cache_directory: .culler_cache    # analysis results are cached by file hash
  models_directory: models

scan:
  extensions:
    - .jpg
    - .jpeg
    - .png
    - .webp
    - .bmp
    - .tiff
    - .tif
    - .heic
    - .heif
    - .dng
    - .cr2
    - .cr3
    - .nef
    - .arw
    - .orf
    - .rw2
    - .raf
    - .pef
    - .srw
  recursive: true
  follow_symlinks: false
  exclude_dirs: [.git, __pycache__, .culler_cache]
  max_pixels: 250000000             # refuse images larger than this (decompression-bomb guard)

thresholds:                         # final score (0-100) cut-offs
  select: 80
  usable: 65
  review: 45
  technical_floor: 25               # below this technical score => always reject_candidate

weights:                            # relative; normalised over available components
  technical: 0.40
  ai: 0.20
  subject: 0.20
  composition: 0.20

duplicates:
  enabled: true
  hamming_threshold: 8              # 0 = identical hashes only, higher = looser grouping

vision:                             # needs: pip install 'ai-media-culler[ai]'
  enabled: true
  model_name: google/siglip-base-patch16-224
  device: auto                      # auto | cpu | cuda | mps
  max_side: 896

faces:                              # needs: pip install 'ai-media-culler[faces]'
  enabled: true
  max_faces: 10
  min_detection_confidence: 0.5
  min_presence_confidence: 0.5
  model_filename: face_landmarker.task
  model_sha256: null                # optional integrity check for `culler models download`
  max_side: 1920

performance:
  workers: 4
  use_cache: true
"""
