from __future__ import annotations

import SimpleITK as sitk

from spine_segment.native_torch_backend import (
    _landmark_label_from_index,
    _spine_localization_tiles,
)


def test_landmark_label_from_index_uses_verse_labels() -> None:
    assert _landmark_label_from_index(0) == 1
    assert _landmark_label_from_index(24) == 25
    assert _landmark_label_from_index(25) == 28


def test_spine_localization_tiles_cover_long_whole_body_scan() -> None:
    image = sitk.Image([512, 512, 372], sitk.sitkFloat32)
    image.SetSpacing((0.87890625, 0.87890625, 5.0))
    image.SetOrigin((-225.0, -155.0, -1570.5))

    tiles = _spine_localization_tiles(
        image,
        spacing=8.0,
        valid_sizes_xyz=((32, 64, 96, 128), (32, 64, 96, 128), (32, 64, 96, 128)),
    )

    tile_names = [tile.name for tile in tiles]
    tile_z_starts = [tile.transform.TransformPoint((0.0, 0.0, 0.0))[2] for tile in tiles]

    assert tile_names == ["inferior", "tile1", "superior", "center"]
    assert all(tile.size_xyz == (64, 64, 128) for tile in tiles)
    assert all(tile.image_array.shape == (128, 64, 64) for tile in tiles)
    assert min(tile_z_starts) == -1570.5
    assert max(tile_z_starts) == -734.5
    assert any(abs(z_start - -1151.0) < 1.0 for z_start in tile_z_starts)
