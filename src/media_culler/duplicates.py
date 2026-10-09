"""Group visually similar images by Hamming distance between 64-bit perceptual hashes."""

from __future__ import annotations

import numpy as np

from media_culler.models import ImageRecord


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


def assign_duplicate_groups(records: list[ImageRecord], threshold: int) -> int:
    """Annotate records in place. Returns the number of groups with 2+ images."""
    hashable = [r for r in records if r.status == "valid" and r.technical is not None]
    hashable.sort(key=lambda r: r.path)  # deterministic group ids

    group_id = 0
    for members in group_hashes([r.technical.perceptual_hash for r in hashable], threshold):  # type: ignore[union-attr]
        size = len(members)
        group_id_for_members = None
        if size > 1:
            group_id += 1
            group_id_for_members = group_id
        for index in members:
            hashable[index].duplicate_group_size = size
            hashable[index].duplicate_group = group_id_for_members
    return group_id
