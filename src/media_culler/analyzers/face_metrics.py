"""Pure geometry on MediaPipe face-mesh landmarks (arrays of shape (N, 3), normalised)."""

from __future__ import annotations

import numpy as np

from media_culler.imaging import ImageContext
from media_culler.models import EyeMetrics, FaceAnalysis, FaceInfo, PoseMetrics

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
_NOSE, _LEFT_CHEEK, _RIGHT_CHEEK = 1, 234, 454


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(landmarks: np.ndarray, idx: tuple[int, ...]) -> float:
    horizontal = _dist(landmarks[idx[0]], landmarks[idx[3]])
    if horizontal == 0:
        return 0.0
    v1 = _dist(landmarks[idx[1]], landmarks[idx[5]])
    v2 = _dist(landmarks[idx[2]], landmarks[idx[4]])
    return (v1 + v2) / (2 * horizontal)


def eye_openness(landmarks: np.ndarray) -> EyeMetrics:
    left = eye_aspect_ratio(landmarks, LEFT_EYE)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE)
    return EyeMetrics(
        left=round(left, 4), right=round(right, 4), average=round((left + right) / 2, 4)
    )


def face_pose(landmarks: np.ndarray) -> PoseMetrics:
    """Rough yaw heuristic from nose-to-cheek symmetry (not a true head-pose estimate)."""
    nose = landmarks[_NOSE]
    left = _dist(nose, landmarks[_LEFT_CHEEK])
    right = _dist(nose, landmarks[_RIGHT_CHEEK])
    total = left + right
    if total == 0:
        return PoseMetrics(symmetry=0.0, orientation="unknown")
    symmetry = 1 - abs(left - right) / total
    if symmetry >= 0.85:
        orientation = "frontal"
    elif symmetry >= 0.65:
        orientation = "slightly_turned"
    else:
        orientation = "strongly_turned"
    return PoseMetrics(symmetry=round(symmetry, 4), orientation=orientation)


def _laplacian_variance(gray_crop: np.ndarray) -> float:
    import cv2

    _, std = cv2.meanStdDev(cv2.Laplacian(gray_crop, cv2.CV_32F))
    return float(std[0][0] ** 2)


def build_face_analysis(faces: list[np.ndarray], ctx: ImageContext) -> FaceAnalysis:
    """Turn raw landmark arrays into FaceInfo, measuring sharpness on the full-res image."""
    width, height = ctx.width, ctx.height
    infos: list[FaceInfo] = []

    for landmarks in faces:
        if landmarks.shape[0] <= max(_RIGHT_CHEEK, max(RIGHT_EYE)):
            continue  # not a full face mesh
        min_x, max_x = (
            float(max(0.0, landmarks[:, 0].min())),
            float(min(1.0, landmarks[:, 0].max())),
        )
        min_y, max_y = (
            float(max(0.0, landmarks[:, 1].min())),
            float(min(1.0, landmarks[:, 1].max())),
        )

        x1 = max(0, min(int(min_x * width), width - 1))
        y1 = max(0, min(int(min_y * height), height - 1))
        x2 = max(0, min(int(max_x * width), width))
        y2 = max(0, min(int(max_y * height), height))

        sharpness = 0.0
        if x2 > x1 and y2 > y1:
            sharpness = _laplacian_variance(ctx.gray[y1:y2, x1:x2])

        infos.append(
            FaceInfo(
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1,
                area_ratio=round((max_x - min_x) * (max_y - min_y), 6),
                sharpness=round(sharpness, 2),
                eyes=eye_openness(landmarks),
                pose=face_pose(landmarks),
                landmark_count=int(landmarks.shape[0]),
            )
        )
    return FaceAnalysis(face_count=len(infos), faces=infos)
