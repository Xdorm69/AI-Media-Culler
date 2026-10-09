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


class SceneType(str, Enum):
    PEOPLE = "people"
    LANDSCAPE = "landscape"
    GENERAL = "general"


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


class HorizonInfo(BaseModel):
    """A dominant near-horizontal line (sea/sky/ground boundary)."""

    angle_deg: float  # signed tilt; 0 = level. Positive = right side lower (clockwise)
    y: float  # vertical position, 0 = top edge, 1 = bottom edge
    confidence: float  # 0-1: how much of the frame width the line spans, and how consistent it is


class SceneMetrics(BaseModel):
    """Scene-level measurements used by landscape-aware scoring."""

    horizon: HorizonInfo | None = None
    highlight_clip: float  # fraction of pixels with any channel >= 250
    shadow_clip: float  # fraction of pixels with every channel <= 5
    dynamic_range: float  # 99th - 1st luminance percentile (0-255)
    sharpness_focus: float  # 75th-percentile tile Laplacian variance (ignores smooth sky/water)
    saliency_x: float  # centroid of the most salient region, 0-1 from the left
    saliency_y: float


# ------------------------------------------------------------------------- scores


class Scores(BaseModel):
    technical: float | None = None
    ai: float | None = None
    subject: float | None = None
    composition: float | None = None
    final: float | None = None
    scene_type: SceneType | None = None
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
    scene: SceneMetrics | None = None
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
