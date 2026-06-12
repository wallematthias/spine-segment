from __future__ import annotations

import numpy as np

from spine_segment.tiling import extract_tile, iter_sliding_window_requests


def test_iter_sliding_window_requests_covers_tail() -> None:
    requests = list(iter_sliding_window_requests((10, 10, 10), (6, 6, 6), (4, 4, 4)))
    assert requests[0].start_zyx == (0, 0, 0)
    assert requests[-1].stop_zyx == (10, 10, 10)


def test_extract_tile_returns_expected_slice() -> None:
    array = np.arange(5 * 5 * 5).reshape((5, 5, 5))
    request = next(iter(iter_sliding_window_requests((5, 5, 5), (3, 3, 3), (3, 3, 3))))
    tile = extract_tile(array, request)
    assert tile.shape == (3, 3, 3)
