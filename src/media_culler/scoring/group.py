"""Group-photo analysis: who is a subject, who is blinking, who is looking away.

A group photo is only as good as its worst *subject*, but a passer-by 30 m behind the
group must not count as one. Everything here is pure and driven by ``SelectionConfig``.
"""

from __future__ import annotations

from media_culler.config import SelectionConfig
from media_culler.models import FaceAnalysis, FaceInfo, GroupAssessment


def eyes_closed(face: FaceInfo, config: SelectionConfig) -> bool:
    """Clearly closed eyes. Unknown (-> False) when the face is too small or in profile,
    where eye landmarks are unreliable: better to miss a blink than invent one."""
    if face.area_ratio < config.blink_min_face_area:
        return False
    if face.pose.orientation == "strongly_turned":
        return False
    return face.eyes.average < config.closed_ear_threshold


def assess_faces(analysis: FaceAnalysis, config: SelectionConfig) -> GroupAssessment:
    faces = analysis.faces
    if not faces:
        return GroupAssessment()

    largest = max(face.area_ratio for face in faces)
    floor = max(config.min_face_area, config.relative_face_area * largest)
    significant = [i for i, face in enumerate(faces) if face.area_ratio >= floor]
    return GroupAssessment(
        significant=significant,
        closed_eyes=[i for i in significant if eyes_closed(faces[i], config)],
        looking_away=[i for i in significant if faces[i].pose.orientation == "strongly_turned"],
        ignored=len(faces) - len(significant),
    )


def subject_faces(analysis: FaceAnalysis, assessment: GroupAssessment) -> FaceAnalysis:
    """The analysis restricted to subjects (for composition and subject scoring)."""
    kept = [analysis.faces[i] for i in assessment.significant]
    return FaceAnalysis(face_count=len(kept), faces=kept)
