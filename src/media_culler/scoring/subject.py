from __future__ import annotations

from media_culler.models import FaceAnalysis, FaceInfo
from media_culler.scoring._math import clamp, ramp

NO_FACE_SCORE = 50.0  # neutral: absence of faces is not a defect

_POSE_SCORES = {"frontal": 100.0, "slightly_turned": 80.0, "strongly_turned": 60.0, "unknown": 40.0}


def score_face(face: FaceInfo) -> dict[str, float]:
    size = ramp(face.area_ratio, 0.002, 0.05, 20, 100)
    sharp = ramp(face.sharpness, 30, 500, 10, 100)
    eyes = ramp(face.eyes.average, 0.12, 0.30, 15, 100)
    visibility = clamp(face.visibility * 100)
    pose = _POSE_SCORES.get(face.pose.orientation, 40.0)
    total = sharp * 0.25 + size * 0.20 + eyes * 0.25 + visibility * 0.15 + pose * 0.15
    return {
        "size": round(size, 2),
        "sharpness": round(sharp, 2),
        "eyes": round(eyes, 2),
        "visibility": round(visibility, 2),
        "pose": round(pose, 2),
        "face_score": round(total, 2),
    }


def score_subject(analysis: FaceAnalysis) -> tuple[float, dict[str, object]]:
    if not analysis.faces:
        return NO_FACE_SCORE, {"face_scores": [], "reason": "No detectable faces"}

    face_scores = [score_face(face)["face_score"] for face in analysis.faces]
    ordered = sorted(face_scores, reverse=True)

    if len(ordered) == 1:
        score, reason = ordered[0], "Single subject"
    elif len(ordered) == 2:
        # Weight the weaker face more so one great face can't hide a closed-eyes friend.
        score, reason = ordered[0] * 0.4 + ordered[1] * 0.6, "Two-subject weighted score"
    else:
        score = sum(ordered) / len(ordered) * 0.70 + ordered[-1] * 0.30
        reason = "Group average with weakest-subject penalty"

    return round(score, 2), {"face_scores": face_scores, "reason": reason}
