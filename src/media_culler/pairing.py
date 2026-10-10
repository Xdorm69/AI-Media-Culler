"""RAW+JPEG pairs: two files, one shot.

Cameras set to RAW+JPEG write ``IMG_0001.CR2`` and ``IMG_0001.JPG`` for the same frame.
Left alone, the pair looks like a two-frame burst and the tool would "cull" one of them.
Instead the JPEG becomes the RAW's *companion*: it takes part in no group and simply
inherits the RAW's decisions, so both files always land in the same bucket.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from media_culler.imaging import is_raw
from media_culler.models import ImageRecord

# Same-named files this far apart in capture time are different shots (a counter rolled over).
_MAX_PAIR_GAP_SECONDS = 2.0
_INHERITED = (
    "recommendation",
    "recommendation_reason",
    "group_role",
    "is_group_representative",
    "best_shot_score",
    "selection_reason",
)


def _key(record: ImageRecord) -> tuple[Path, str]:
    path = Path(record.relative_path)
    return path.parent, path.stem.lower()


def _same_moment(a: ImageRecord, b: ImageRecord) -> bool:
    if not a.captured_at or not b.captured_at:
        return True  # nothing to contradict the matching name
    try:
        gap = abs(datetime.fromisoformat(a.captured_at) - datetime.fromisoformat(b.captured_at))
    except ValueError:
        return True
    return gap.total_seconds() <= _MAX_PAIR_GAP_SECONDS


def link_raw_companions(records: list[ImageRecord]) -> int:
    """Mark every JPEG/HEIC/... that sits beside a same-named RAW; returns the pair count.

    Only pairs where *both* files decoded are linked: if the RAW failed, the other file is
    the only usable copy of the shot and is treated as an ordinary photo.
    """
    raws: dict[tuple[Path, str], ImageRecord] = {}
    for record in sorted(records, key=lambda r: r.path):
        if record.status == "valid" and is_raw(Path(record.path)):
            raws.setdefault(_key(record), record)

    pairs = 0
    for record in records:
        record.paired_with = None
        if record.status != "valid" or is_raw(Path(record.path)):
            continue
        raw = raws.get(_key(record))
        if raw is not None and _same_moment(raw, record):
            record.paired_with = raw.relative_path
            pairs += 1
    return pairs


def inherit_companion_decisions(records: list[ImageRecord]) -> None:
    """Copy the RAW's recommendation and group decisions onto its companion file."""
    by_path = {r.relative_path: r for r in records}
    for record in records:
        if record.paired_with is None:
            continue
        raw = by_path.get(record.paired_with)
        if raw is None:
            continue
        for name in _INHERITED:
            setattr(record, name, getattr(raw, name))
        record.selection_reason = f"Companion of {raw.filename}; follows its decision" + (
            f" ({raw.selection_reason})" if raw.selection_reason else ""
        )
