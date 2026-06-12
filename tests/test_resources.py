from __future__ import annotations

from spine_segment.graph_resources import load_spine_graph_resources


def test_load_spine_graph_resources() -> None:
    possible_successors, offsets_mean, distances_mean, distances_std = load_spine_graph_resources()
    assert len(possible_successors) == 26
    assert len(offsets_mean) == 26
    assert len(distances_mean) == 26
    assert len(distances_std) == 26
