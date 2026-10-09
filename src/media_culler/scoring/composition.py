from __future__ import annotations

from math import hypot

from media_culler.models import FaceAnalysis, FaceInfo
from media_culler.scoring._math import clamp

NO_FACE_SCORE = 50.0
_MAX_DISTANCE = hypot(0.5, 0.5)
_THIRDS = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]


def rule_of_thirds_score(cx: float, cy: float) -> float:
    nearest = min(hypot(cx - px, cy - py) for px, py in _THIRDS)
    return round(clamp(100 - nearest / _MAX_DISTANCE * 100), 2)


def centering_score(cx: float, cy: float) -> float:
    return round(clamp(100 - hypot(cx - 0.5, cy - 0.5) / _MAX_DISTANCE * 100), 2)


def edge_safety_score(face: FaceInfo, width: int, height: int, margin: float = 0.03) -> float:
    mx, my = width * margin, height * margin
    penalties = sum(
        (
            face.x < mx,
            face.y < my,
            face.x + face.width > width - mx,
            face.y + face.height > height - my,
        )
    )
    return {0: 100.0, 1: 75.0, 2: 50.0, 3: 25.0}.get(penalties, 10.0)


def occupancy_score(area_ratio: float) -> float:
    if area_ratio <= 0.005:
        return 30.0
    if area_ratio < 0.02:
        return 30 + (area_ratio - 0.005) / 0.015 * 50
    if area_ratio <= 0.15:
        return 80 + (area_ratio - 0.02) / 0.13 * 20
    if area_ratio <= 0.30:
        return 100 - (area_ratio - 0.15) / 0.15 * 40
    return 60.0


def score_face_composition(face: FaceInfo, width: int, height: int) -> dict[str, float]:
    cx = (face.x + face.width / 2) / width
    cy = (face.y + face.height / 2) / height
    thirds = rule_of_thirds_score(cx, cy)
    centering = centering_score(cx, cy)
    edge = edge_safety_score(face, width, height)
    occupancy = occupancy_score(face.area_ratio)
    total = thirds * 0.25 + centering * 0.15 + edge * 0.25 + occupancy * 0.35
    return {
        "rule_of_thirds": thirds,
        "centering": centering,
        "edge_safety": edge,
        "occupancy": round(occupancy, 2),
        "composition_score": round(total, 2),
    }


def score_composition(
    analysis: FaceAnalysis, width: int, height: int
) -> tuple[float, dict[str, object]]:
    if not analysis.faces or width <= 0 or height <= 0:
        return NO_FACE_SCORE, {"reason": "No detectable face for composition analysis"}

    per_face = [score_face_composition(f, width, height) for f in analysis.faces]
    values = [item["composition_score"] for item in per_face]
    if len(values) == 1:
        score, reason = values[0], "Single-subject composition"
    else:
        score = sum(values) / len(values) * 0.70 + min(values) * 0.30
        reason = "Group composition"
    return round(score, 2), {"faces": per_face, "reason": reason}
