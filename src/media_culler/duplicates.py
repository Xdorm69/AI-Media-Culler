"""Group visually similar images by Hamming distance between 64-bit perceptual hashes."""

from __future__ import annotations

from collections import deque
from datetime import datetime

import numpy as np

from media_culler.config import BurstConfig
from media_culler.models import GroupKind, ImageRecord


def _popcount(values: np.ndarray) -> np.ndarray:
    counts: np.ndarray
    if hasattr(np, "bitwise_count"):  # NumPy >= 2.0
        counts = np.bitwise_count(values)
    else:
        counts = np.unpackbits(values.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
    return counts.astype(np.int64)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return int(_popcount(np.array([int(hash_a, 16) ^ int(hash_b, 16)], dtype=np.uint64))[0])


def group_hashes(hashes: list[str], threshold: int) -> list[list[int]]:
    """Leader clustering, deterministic in input order.

    Each image joins the first leader within ``threshold`` bits. This is deliberately
    non-transitive (unlike union-find) so a chain of slowly-changing frames can't merge
    unrelated shots. Vectorised: O(n^2) bit-ops but fast in NumPy (~10^4 images in seconds).
    """
    if not hashes:
        return []
    values = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    assigned = np.zeros(len(values), dtype=bool)
    groups: list[list[int]] = []

    for i in range(len(values)):
        if assigned[i]:
            continue
        distances = _popcount(values ^ values[i])
        members = np.flatnonzero(~assigned & (distances <= threshold))
        members = np.union1d(members, [i])
        assigned[members] = True
        groups.append([int(m) for m in members])
    return groups


class _Components:
    """Union-find that remembers the earliest capture time of each component."""

    def __init__(self, size: int, times: list[datetime | None]) -> None:
        self._parent = list(range(size))
        self._start: dict[int, datetime] = {i: t for i, t in enumerate(times) if t is not None}

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def start(self, item: int) -> datetime | None:
        return self._start.get(self.find(item))

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        self._parent[root_b] = root_a
        earliest = [t for t in (self._start.pop(root_a, None), self._start.pop(root_b, None)) if t]
        if earliest:
            self._start[root_a] = min(earliest)


def _link_bursts(
    components: _Components,
    hashes: list[int],
    times: list[datetime | None],
    config: BurstConfig,
) -> None:
    """Link each frame to every recent frame it looks continuous with.

    Looking back over a time window (not just at the previous frame) keeps a burst intact
    when another photographer's frames interleave with it. Chains are cut once a group
    spans ``max_span_seconds`` so a slow walk can't chain a whole afternoon together.
    """
    order = sorted((i for i, t in enumerate(times) if t is not None), key=lambda i: (times[i], i))
    window: deque[int] = deque()
    for index in order:
        moment = times[index]
        assert moment is not None
        while window and (moment - times[window[0]]).total_seconds() > config.max_gap_seconds:  # type: ignore[operator]
            window.popleft()
        for other in window:
            if (hashes[index] ^ hashes[other]).bit_count() > config.hamming_threshold:
                continue
            begun = components.start(other)
            if begun is not None and (moment - begun).total_seconds() > config.max_span_seconds:
                continue
            components.union(other, index)
        window.append(index)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _classify_group(members: list[ImageRecord], max_span: float) -> tuple[GroupKind, float | None]:
    shas = {r.sha256 for r in members}
    times = [_parse_time(r.captured_at) for r in members]
    span: float | None = None
    if all(t is not None for t in times):
        stamps = [t for t in times if t is not None]
        span = round((max(stamps) - min(stamps)).total_seconds(), 3)
    if len(shas) == 1 and None not in shas:
        return GroupKind.DUPLICATE, span
    if span is not None and span <= max_span:
        return GroupKind.BURST, span
    return GroupKind.SIMILAR, span


def assign_duplicate_groups(
    records: list[ImageRecord], threshold: int | None, bursts: BurstConfig | None = None
) -> int:
    """Group similar images and bursts; annotate records in place.

    ``threshold`` is the perceptual-hash distance for "looks the same" (None skips that
    pass). ``bursts`` adds time-aware links. Returns the number of groups with 2+ images.
    """
    hashable = [r for r in records if r.status == "valid" and r.technical is not None]
    hashable.sort(key=lambda r: r.path)  # deterministic group ids
    hashes = [int(r.technical.perceptual_hash, 16) for r in hashable]  # type: ignore[union-attr]
    times = [_parse_time(r.captured_at) for r in hashable]
    components = _Components(len(hashable), times)

    if threshold is not None:
        clusters = group_hashes([r.technical.perceptual_hash for r in hashable], threshold)  # type: ignore[union-attr]
        for cluster in clusters:
            for other in cluster[1:]:
                components.union(cluster[0], other)
    if bursts is not None and bursts.enabled:
        _link_bursts(components, hashes, times, bursts)

    grouped: dict[int, list[int]] = {}
    for index in range(len(hashable)):
        grouped.setdefault(components.find(index), []).append(index)

    max_span = (bursts or BurstConfig()).max_span_seconds
    group_id = 0
    for member_indexes in sorted(grouped.values(), key=lambda m: m[0]):
        members = [hashable[i] for i in member_indexes]
        size = len(members)
        if size == 1:
            members[0].duplicate_group_size = 1
            members[0].duplicate_group = None
            continue
        group_id += 1
        kind, span = _classify_group(members, max_span)
        for record in members:
            record.duplicate_group = group_id
            record.duplicate_group_size = size
            record.group_kind = kind
            record.group_span_seconds = span
    return group_id
