from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from spine_segment.landmarks import Landmark
from spine_segment.graph_resources import load_spine_graph_resources


@dataclass(frozen=True, slots=True)
class SpineSequenceConfig:
    bias: float = 2.0
    lambda_weight: float = 0.2


def _distance_value(
    landmark_from: Landmark,
    landmark_to: Landmark,
    landmark_from_index: int,
    landmark_to_index: int,
    offsets_mean,
    distances_mean,
) -> float:
    mean_dir = np.asarray(offsets_mean[landmark_from_index][landmark_to_index], dtype=np.float32)
    mean_dist = float(distances_mean[landmark_from_index][landmark_to_index])
    offset = np.asarray(landmark_from.coords, dtype=np.float32) - np.asarray(landmark_to.coords, dtype=np.float32)
    diff_single = (mean_dir * mean_dist - offset) / mean_dist
    diff_single[2] = diff_single[2] * 3 if diff_single[2] > 0 else diff_single[2]
    diff = diff_single * 2
    dist = float(np.sum(np.square(diff)))
    return 1.0 - dist


def _unary_term(landmark: Landmark, config: SpineSequenceConfig) -> float:
    return config.lambda_weight * float(landmark.value) + config.bias


def _pairwise_term(
    landmark_from: Landmark,
    landmark_to: Landmark,
    landmark_from_index: int,
    landmark_to_index: int,
    config: SpineSequenceConfig,
    offsets_mean,
    distances_mean,
) -> float:
    distance_value = _distance_value(
        landmark_from,
        landmark_to,
        landmark_from_index,
        landmark_to_index,
        offsets_mean,
        distances_mean,
    )
    return (1.0 - config.lambda_weight) * distance_value


def solve_spine_sequence(
    local_heatmap_maxima: Sequence[Sequence[Landmark]],
    *,
    config: SpineSequenceConfig | None = None,
) -> list[Landmark]:
    cfg = config or SpineSequenceConfig()
    possible_successors, offsets_mean, distances_mean, _distances_std = load_spine_graph_resources()
    num_landmarks = len(local_heatmap_maxima)

    scores: dict[tuple[int, int], float] = {}
    previous: dict[tuple[int, int], tuple[int, int] | None] = {}

    for landmark_index, candidates in enumerate(local_heatmap_maxima):
        for maxima_index, candidate in enumerate(candidates):
            if not candidate.is_valid:
                continue
            base_score = _unary_term(candidate, cfg)
            best_score = base_score
            best_prev: tuple[int, int] | None = None
            for prev_index in range(landmark_index):
                if landmark_index not in possible_successors[prev_index]:
                    continue
                for prev_maxima_index, prev_candidate in enumerate(local_heatmap_maxima[prev_index]):
                    if not prev_candidate.is_valid:
                        continue
                    prev_key = (prev_index, prev_maxima_index)
                    if prev_key not in scores:
                        continue
                    score = scores[prev_key] + base_score + _pairwise_term(
                        prev_candidate,
                        candidate,
                        prev_index,
                        landmark_index,
                        cfg,
                        offsets_mean,
                        distances_mean,
                    )
                    if score > best_score:
                        best_score = score
                        best_prev = prev_key
            scores[(landmark_index, maxima_index)] = best_score
            previous[(landmark_index, maxima_index)] = best_prev

    if not scores:
        return [Landmark.invalid() for _ in range(num_landmarks)]

    best_key = max(scores.items(), key=lambda item: item[1])[0]
    chosen = [Landmark.invalid() for _ in range(num_landmarks)]
    current: tuple[int, int] | None = best_key
    while current is not None:
        landmark_index, maxima_index = current
        chosen[landmark_index] = local_heatmap_maxima[landmark_index][maxima_index]
        current = previous[current]
    return chosen


def greedy_local_maxima(
    heatmap: np.ndarray,
    *,
    max_candidates: int = 4,
    threshold: float = 0.05,
    suppression_radius_vox: int = 3,
) -> list[Landmark]:
    working = heatmap.astype(np.float32, copy=True)
    maxima: list[Landmark] = []
    if working.size == 0:
        return maxima
    absolute_max = float(np.max(working))
    if absolute_max <= 0:
        return maxima
    threshold_value = absolute_max * float(threshold)

    for _ in range(max_candidates):
        flat_index = int(np.argmax(working))
        value = float(working.reshape(-1)[flat_index])
        if not math.isfinite(value) or value < threshold_value:
            break
        z, y, x = np.unravel_index(flat_index, working.shape)
        maxima.append(Landmark(float(x), float(y), float(z), is_valid=True, value=value))

        z0 = max(0, z - suppression_radius_vox)
        z1 = min(working.shape[0], z + suppression_radius_vox + 1)
        y0 = max(0, y - suppression_radius_vox)
        y1 = min(working.shape[1], y + suppression_radius_vox + 1)
        x0 = max(0, x - suppression_radius_vox)
        x1 = min(working.shape[2], x + suppression_radius_vox + 1)
        working[z0:z1, y0:y1, x0:x1] = 0.0
    return maxima
