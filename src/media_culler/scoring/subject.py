from __future__ import annotations

from media_culler.config import SelectionConfig
from media_culler.models import FaceAnalysis, FaceInfo, GroupAssessment
from media_culler.scoring._math import clamp, ramp
from media_culler.scoring.group import assess_faces

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


def score_subject(
    analysis: FaceAnalysis,
    config: SelectionConfig | None = None,
    assessment: GroupAssessment | None = None,
) -> tuple[float, dict[str, object]]:
    config = config or SelectionConfig()
    assessment = assessment or assess_faces(analysis, config)

    if not assessment.significant:
        reason = "Only distant/background faces" if analysis.faces else "No detectable faces"
        return NO_FACE_SCORE, {
            "face_scores": [],
            "reason": reason,
            "ignored_faces": len(analysis.faces),
        }

    face_scores: list[float] = []
    for index in assessment.significant:
        score = score_face(analysis.faces[index])["face_score"]
        if index in assessment.closed_eyes:
            # The weighted sum lets a face with shut eyes still score ~80: cap it explicitly.
            score = min(score, config.closed_face_cap)
        face_scores.append(score)
    ordered = sorted(face_scores, reverse=True)

    if len(ordered) == 1:
        score, reason = ordered[0], "Single subject"
    elif len(ordered) == 2:
        # Weight the weaker face more so one great face can't hide a closed-eyes friend.
        score, reason = ordered[0] * 0.4 + ordered[1] * 0.6, "Two-subject weighted score"
    else:
        score = sum(ordered) / len(ordered) * 0.70 + ordered[-1] * 0.30
        reason = "Group average with weakest-subject penalty"

    if assessment.closed_eyes:
        score = min(score, config.closed_subject_cap)

    return round(score, 2), {
        "face_scores": face_scores,
        "reason": reason,
        "ignored_faces": assessment.ignored,
        "closed_eyes": len(assessment.closed_eyes),
    }
