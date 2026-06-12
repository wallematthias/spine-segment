from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from spine_segment.landmarks import Landmark
from spine_segment.postprocess import bounding_box_from_heatmap, filter_landmarks_top_bottom


def test_bounding_box_from_heatmap_uses_percentile_peak_not_single_hot_voxel() -> None:
    heatmap = np.zeros((12, 12, 12), dtype=np.float32)
    heatmap[4:8, 4:8, 4:8] = 10.0
    heatmap[0, 0, 0] = 1000.0

    bbox = bounding_box_from_heatmap(
        heatmap,
        transformation=sitk.Transform(3, sitk.sitkIdentity),
        image_spacing_xyz=(1.0, 1.0, 1.0),
        threshold_percentile=98.0,
        threshold_fraction_of_peak_percentile=0.5,
    )

    assert bbox.start == (4.0, 4.0, 4.0)
    assert bbox.end == (8.0, 8.0, 8.0)


def test_filter_landmarks_top_bottom_uses_physical_z_bounds() -> None:
    image = sitk.Image((8, 8, 20), sitk.sitkUInt8)
    image.SetSpacing((1.0, 1.0, 2.0))
    image.SetOrigin((0.0, 0.0, -30.0))

    landmarks = [
        Landmark(0.0, 0.0, -25.0),
        Landmark(0.0, 0.0, -5.0),
        Landmark(0.0, 0.0, 5.0),
    ]

    filtered = filter_landmarks_top_bottom(landmarks, image=image, margin_mm=5.0)

    assert not filtered[0].is_valid
    assert filtered[1].is_valid
    assert not filtered[2].is_valid
