from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True, slots=True)
class TileRequest:
    start_zyx: tuple[int, int, int]
    stop_zyx: tuple[int, int, int]


def iter_sliding_window_requests(
    shape_zyx: tuple[int, int, int],
    tile_shape_zyx: tuple[int, int, int],
    step_zyx: tuple[int, int, int],
) -> Iterator[TileRequest]:
    z_max, y_max, x_max = shape_zyx
    tz, ty, tx = tile_shape_zyx
    sz, sy, sx = step_zyx

    def axis_positions(size: int, tile: int, step: int) -> list[int]:
        if size <= tile:
            return [0]
        positions = list(range(0, max(1, size - tile + 1), max(1, step)))
        last = size - tile
        if positions[-1] != last:
            positions.append(last)
        return positions

    for z0 in axis_positions(z_max, tz, sz):
        for y0 in axis_positions(y_max, ty, sy):
            for x0 in axis_positions(x_max, tx, sx):
                yield TileRequest(
                    start_zyx=(z0, y0, x0),
                    stop_zyx=(min(z0 + tz, z_max), min(y0 + ty, y_max), min(x0 + tx, x_max)),
                )


def extract_tile(array: np.ndarray, request: TileRequest) -> np.ndarray:
    z0, y0, x0 = request.start_zyx
    z1, y1, x1 = request.stop_zyx
    return array[z0:z1, y0:y1, x0:x1]
