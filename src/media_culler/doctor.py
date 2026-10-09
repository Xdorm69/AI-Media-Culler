"""Environment diagnostics: what will work, and exactly how to fix what won't."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from importlib import metadata

from media_culler.config import Settings


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str
    fix: str | None = None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def run_checks(settings: Settings, *, probe_gpu: bool = True) -> list[Check]:
    checks: list[Check] = []

    py = sys.version_info
    checks.append(
        Check(
            "Python",
            "ok" if py >= (3, 10) else "fail",
            f"{py.major}.{py.minor}.{py.micro}",
            None if py >= (3, 10) else "Python 3.10+ required",
        )
    )

    for label, package in (
        ("OpenCV", "opencv-python-headless"),
        ("NumPy", "numpy"),
        ("Pillow", "pillow"),
    ):
        version = _version(package) or (_version("opencv-python") if label == "OpenCV" else None)
        checks.append(
            Check(
                label,
                "ok" if version else "fail",
                version or "not installed",
                None if version else f"pip install {package}",
            )
        )

    # AI stack
    if _has("torch") and _has("transformers"):
        detail = f"torch {_version('torch')}, transformers {_version('transformers')}"
        status = "ok"
        if probe_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    detail += f" · CUDA: {torch.cuda.get_device_name(0)}"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    detail += " · Apple MPS"
                else:
                    detail += " · CPU only (slower)"
                    status = "warn"
            except Exception as error:
                detail += f" · torch import failed: {error}"
                status = "warn"
        checks.append(Check("AI vision (SigLIP)", status, detail))
    else:
        checks.append(
            Check(
                "AI vision (SigLIP)",
                "warn",
                "torch/transformers not installed",
                "pip install 'ai-media-culler[ai]'  (or run with --no-ai)",
            )
        )

    # Faces
    if _has("mediapipe"):
        checks.append(Check("MediaPipe", "ok", _version("mediapipe") or "installed"))
    else:
        checks.append(
            Check(
                "MediaPipe",
                "warn",
                "not installed",
                "pip install 'ai-media-culler[faces]'  (or run with --no-faces)",
            )
        )
    model = settings.face_model_path
    checks.append(
        Check(
            "Face model",
            "ok" if model.is_file() else "warn",
            str(model) if model.is_file() else f"missing: {model}",
            None if model.is_file() else "culler models download",
        )
    )

    inp = settings.paths.input_directory
    checks.append(
        Check(
            "Input directory",
            "ok" if inp.is_dir() else "warn",
            str(inp) if inp.is_dir() else f"not found: {inp}",
            None if inp.is_dir() else "pass a path: culler run <folder>",
        )
    )
    return checks
