import numpy as np

from media_culler.config import ThresholdsConfig
from media_culler.curation import (
    apply_recommendations,
    classify,
    rank_collection,
    select_best_shots,
)
from media_culler.duplicates import (
    _popcount,
    assign_duplicate_groups,
    group_hashes,
    hamming_distance,
)
from media_culler.models import GroupRole, ImageRecord, Recommendation, Scores, TechnicalMetrics


def rec(name, final=None, technical=80.0, phash="0" * 16, status="valid") -> ImageRecord:
    r = ImageRecord(
        path=f"/p/{name}", relative_path=name, filename=name, extension=".jpg", status=status
    )
    if status == "valid":
        r.technical = TechnicalMetrics(
            sharpness=1, brightness=1, contrast=1, megapixels=1, perceptual_hash=phash
        )
        if final is not None:
            r.scores = Scores(technical=technical, final=final)
    return r


def test_popcount_fallback_matches_fast_path():
    values = np.array([0, 1, 0xFFFFFFFFFFFFFFFF, 0x8000000000000001], dtype=np.uint64)
    assert _popcount(values).tolist() == [0, 1, 64, 2]
    bytes_view = values.view(np.uint8).reshape(-1, 8)
    assert np.unpackbits(bytes_view, axis=1).sum(axis=1).tolist() == [0, 1, 64, 2]


def test_hamming_distance():
    assert hamming_distance("0" * 16, "0" * 15 + "f") == 4
    assert hamming_distance("a" * 16, "a" * 16) == 0


def test_group_hashes_threshold_and_order():
    h = ["0" * 16, "0" * 15 + "3", "f" * 16, "f" * 15 + "e"]
    assert group_hashes(h, 2) == [[0, 1], [2, 3]]
    assert group_hashes(h, 0) == [[0], [1], [2], [3]]
    assert group_hashes([], 8) == []


def test_leader_clustering_is_not_transitive():
    a, b, c = "0" * 16, "00000000000000ff", "000000000000ffff"  # a-b=8, b-c=8, a-c=16
    assert group_hashes([a, b, c], 8) == [[0, 1], [2]]


def test_assign_groups_skips_errors_and_sets_sizes():
    rs = [
        rec("a", phash="0" * 16),
        rec("b", phash="0" * 15 + "1"),
        rec("c", phash="f" * 16),
        rec("bad", status="error"),
    ]
    assert assign_duplicate_groups(rs, 8) == 1
    by = {r.filename: r for r in rs}
    assert by["a"].duplicate_group == by["b"].duplicate_group == 1
    assert by["a"].duplicate_group_size == 2
    assert by["c"].duplicate_group is None and by["c"].duplicate_group_size == 1
    assert by["bad"].duplicate_group is None


def test_ranking_is_deterministic_with_ties():
    rs = [rec("b", 70), rec("a", 70), rec("c", 90), rec("u")]
    rank_collection(rs)
    order = sorted((r for r in rs if r.collection_rank), key=lambda r: r.collection_rank)
    assert [r.filename for r in order] == ["c", "a", "b"]  # tie broken by path
    assert order[0].collection_percentile == round(2 / 3 * 100, 2)
    assert next(r for r in rs if r.filename == "u").collection_rank is None


def test_classification_boundaries_and_floor():
    t = ThresholdsConfig()
    cases = [
        (80, 80, Recommendation.SELECT),
        (79.99, 80, Recommendation.USABLE),
        (65, 80, Recommendation.USABLE),
        (45, 80, Recommendation.REVIEW),
        (44.99, 80, Recommendation.REJECT),
        (95, 24.9, Recommendation.REJECT),
    ]
    for final, technical, expected in cases:
        assert classify(rec("x", final, technical), t)[0] is expected, (final, technical)


def test_unscored_when_analysis_incomplete():
    r = rec("x")
    assert classify(r, ThresholdsConfig())[0] is Recommendation.UNSCORED


def test_custom_thresholds_are_honoured():
    t = ThresholdsConfig(select=90, usable=70, review=40)
    assert classify(rec("x", 85), t)[0] is Recommendation.USABLE


def test_best_shot_selection():
    group = [rec("a", 60), rec("b", 85), rec("c", 85)]
    for r in group:
        r.duplicate_group = 1
    lone = rec("z", 50)
    rs = [*group, lone]
    select_best_shots(rs)
    roles = {r.filename: r.group_role for r in rs}
    assert roles == {
        "a": GroupRole.SIMILAR,
        "b": GroupRole.REPRESENTATIVE,
        "c": GroupRole.SIMILAR,
        "z": GroupRole.UNIQUE,
    }
    assert all(r.best_shot_score == 85 for r in group)


def test_unscored_never_wins_a_group():
    group = [rec("a"), rec("b", 10)]
    for r in group:
        r.duplicate_group = 1
    select_best_shots(group)
    assert group[1].group_role is GroupRole.REPRESENTATIVE


def test_apply_recommendations_skips_error_records():
    bad = rec("bad", status="error")
    apply_recommendations([bad], ThresholdsConfig())
    assert bad.recommendation is None
