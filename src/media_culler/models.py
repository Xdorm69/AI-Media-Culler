"""Typed domain models. Analyzer outputs, scores and the per-image record."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Recommendation(str, Enum):
    SELECT = "select"
    USABLE = "usable"
    REVIEW = "review"
    REJECT = "reject_candidate"
    UNSCORED = "unscored"


class GroupRole(str, Enum):
    UNIQUE = "unique"
    REPRESENTATIVE = "representative"
    SIMILAR = "similar_non_representative"


# ----------------------------------------------------------------- analyzer outputs


class TechnicalMetrics(BaseModel):
    sharpness: float
    brightness: float
    contrast: float
    megapixels: float
    perceptual_hash: str  # 16 hex chars (64-bit pHash)


class LabelPrediction(BaseModel):
    label: str
    confidence: float  # 0-100
    scores: dict[str, float]


class AIAssessment(BaseModel):
    content: LabelPrediction
    photographic_quality: LabelPrediction
    composition: LabelPrediction
    aesthetic: LabelPrediction


class EyeMetrics(BaseModel):
    left: float
    right: float
    average: float


class PoseMetrics(BaseModel):
    symmetry: float
    orientation: Literal["frontal", "slightly_turned", "strongly_turned", "unknown"]


class FaceInfo(BaseModel):
    x: int
    y: int
    width: int
    height: int
    area_ratio: float
    sharpness: float
    eyes: EyeMetrics
    # MediaPipe face landmarks carry no visibility, so this is a constant 1.0 today.
    visibility: float = 1.0
    pose: PoseMetrics
    landmark_count: int = 0


class FaceAnalysis(BaseModel):
    face_count: int
    faces: list[FaceInfo] = Field(default_factory=list)


# ------------------------------------------------------------------------- scores


class Scores(BaseModel):
    technical: float | None = None
    ai: float | None = None
    subject: float | None = None
    composition: float | None = None
    final: float | None = None
    # Normalised weights actually applied (components that were unavailable are absent).
    weights_used: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------------ record


class ImageRecord(BaseModel):
    path: str
    relative_path: str
    filename: str
    extension: str
    status: Literal["valid", "error"] = "valid"
    error: str | None = None

    size_bytes: int = 0
    width: int = 0  # orientation-corrected
    height: int = 0
    format: str | None = None
    modified_at: str | None = None
    sha256: str | None = None

    technical: TechnicalMetrics | None = None
    ai: AIAssessment | None = None
    faces: FaceAnalysis | None = None
    analyzer_errors: dict[str, str] = Field(default_factory=dict)

    scores: Scores | None = None
    recommendation: Recommendation | None = None
    recommendation_reason: str | None = None

    collection_rank: int | None = None
    collection_size: int | None = None
    collection_percentile: float | None = None

    duplicate_group: int | None = None
    duplicate_group_size: int = 1
    group_role: GroupRole | None = None
    is_group_representative: bool | None = None
    best_shot_score: float | None = None
    selection_reason: str | None = None

    @property
    def final_score(self) -> float | None:
        return self.scores.final if self.scores else None
