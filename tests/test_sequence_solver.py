from __future__ import annotations

import numpy as np

from spine_segment.sequence_solver import greedy_local_maxima, solve_spine_sequence
from spine_segment.landmarks import landmark_from_iterable


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
