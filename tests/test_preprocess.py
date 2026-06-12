from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from spine_segment.preprocess import PreprocessConfig, clamp_shift_scale, preprocess_ct_image


def test_clamp_shift_scale_clips_and_normalizes() -> None:
    arr = np.array([-2000.0, -1024.0, 0.0, 1024.0], dtype=np.float32)
    out = clamp_shift_scale(
        arr,
        clamp_min=-1024.0,
        clamp_max=1024.0,
        shift=1024.0,
        scale=1.0 / 2048.0,
    )
    assert np.allclose(out, np.array([0.0, 0.0, 0.5, 1.0], dtype=np.float32))


def test_preprocess_ct_image_preserves_geometry() -> None:
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
    image.SetSpacing((1.2, 1.3, 1.4))
    image.SetOrigin((5.0, 6.0, 7.0))
    out = preprocess_ct_image(
        image,
        config=PreprocessConfig(spacing_xyz=(1.0, 1.0, 1.0), smooth_sigma=0.0),
    )
    assert out.GetSpacing() == image.GetSpacing()
    assert out.GetOrigin() == image.GetOrigin()
