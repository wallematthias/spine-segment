from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import pickle


@lru_cache(maxsize=1)
def load_spine_graph_resources():
    root = files("spine_segment.resources")
    with (root / "possible_successors.pickle").open("rb") as handle:
        possible_successors = pickle.load(handle)
    with (root / "units_distances.pickle").open("rb") as handle:
        offsets_mean, distances_mean, distances_std = pickle.load(handle)
    return possible_successors, offsets_mean, distances_mean, distances_std
