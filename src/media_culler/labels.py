"""Zero-shot label sets and their score mappings (shared by vision analyzer and scorer)."""

from __future__ import annotations

CONTENT_LABELS = [
    "a photograph of people",
    "a landscape photograph",
    "an architecture photograph",
    "a food photograph",
    "a photograph of an animal",
    "a photograph of a vehicle",
    "a photograph of an object",
    "an event photograph",
    "another type of photograph",
]
QUALITY_LABELS = ["a professional photograph", "a casual snapshot", "a poor quality photograph"]
COMPOSITION_LABELS = [
    "a well composed photograph",
    "an average composition",
    "a poorly composed photograph",
]
AESTHETIC_LABELS = [
    "a visually striking photograph",
    "a pleasant photograph",
    "an ordinary photograph",
]

# Order matters: it defines how a flat score vector is split back into groups.
LABEL_GROUPS: dict[str, list[str]] = {
    "content": CONTENT_LABELS,
    "photographic_quality": QUALITY_LABELS,
    "composition": COMPOSITION_LABELS,
    "aesthetic": AESTHETIC_LABELS,
}

QUALITY_SCORES = dict(zip(QUALITY_LABELS, (100.0, 55.0, 15.0), strict=True))
COMPOSITION_SCORES = dict(zip(COMPOSITION_LABELS, (100.0, 60.0, 20.0), strict=True))
AESTHETIC_SCORES = dict(zip(AESTHETIC_LABELS, (100.0, 70.0, 45.0), strict=True))
