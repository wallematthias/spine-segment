from __future__ import annotations

import numpy as np
import pytest

from spine_segment.sequence_solver import (
    SpineSequenceConfig,
    greedy_local_maxima,
    solve_spine_sequence,
)
from spine_segment.landmarks import landmark_from_iterable
from spine_segment.postprocess import add_landmarks_from_neighbors


def test_greedy_local_maxima_finds_distinct_peaks() -> None:
    heatmap = np.zeros((7, 7, 7), dtype=np.float32)
    heatmap[1, 2, 3] = 1.0
    heatmap[5, 5, 5] = 0.8
    maxima = greedy_local_maxima(heatmap, max_candidates=2, threshold=0.1, suppression_radius_vox=1)
    assert len(maxima) == 2


def test_solve_spine_sequence_returns_valid_length() -> None:
    candidates = [[] for _ in range(26)]
    candidates[0] = [landmark_from_iterable((0.0, 0.0, 0.0), value=1.0)]
    candidates[1] = [landmark_from_iterable((0.0, 0.0, 10.0), value=1.0)]
    solved = solve_spine_sequence(candidates)
    assert len(solved) == 26
    assert solved[0].is_valid


def test_sequence_preserves_detected_levels_in_a_curved_spine() -> None:
    # Recorded localization peaks, translated into an anonymous physical frame.
    # The old shape-heavy weighting drops T3 and shifts the remaining thoracic levels.
    peaks = [
        (6, (2, -20, 20), .256284),
        (7, (2, -14, 12), .589498),
        (8, (0, 0, 0), .575402),
        (9, (-2, 14, -16), .613788),
        (10, (-2, 30, -30), .615367),
        (11, (-2, 46, -44), .579591),
        (12, (-4, 60, -60), .539445),
        (13, (-6, 72, -80), .468812),
        (14, (-10, 82, -104), .438447),
        (15, (-12, 86, -128), .523153),
        (16, (-10, 90, -154), .524519),
        (17, (-8, 90, -180), .472650),
        (18, (-4, 88, -204), .460688),
        (19, (10, 84, -226), .289706),
        (20, (26, 84, -248), .554616),
        (21, (42, 82, -272), .460945),
        (22, (50, 74, -298), .100662),
    ]
    candidates = [[] for _ in range(26)]
    for label, xyz, score in peaks:
        candidates[label - 1] = [landmark_from_iterable(xyz, value=score)]

    solved = solve_spine_sequence(add_landmarks_from_neighbors(candidates))

    for label, xyz, _score in peaks:
        assert solved[label - 1].is_valid, f"discarded detected level {label}"
        assert solved[label - 1].coords == xyz, f"shifted detected level {label}"


@pytest.mark.parametrize("xyz", [(0, 0, 0), (0, 0, 30), (100, 100, -20)])
def test_sequence_still_rejects_duplicate_reversed_or_distant_candidates(xyz) -> None:
    candidates = [[] for _ in range(26)]
    candidates[8] = [landmark_from_iterable((0, 0, 0), value=.9)]
    candidates[9] = [landmark_from_iterable(xyz, value=.8)]

    solved = solve_spine_sequence(candidates)

    assert solved[8].is_valid
    assert not solved[9].is_valid


@pytest.mark.parametrize("weight", [-.1, 0, 1, 1.1, float("nan"), float("inf")])
def test_sequence_config_rejects_invalid_confidence_weight(weight) -> None:
    with pytest.raises(ValueError, match="confidence weight"):
        SpineSequenceConfig(lambda_weight=weight)
