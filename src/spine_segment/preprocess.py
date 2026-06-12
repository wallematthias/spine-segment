from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    spacing_xyz: tuple[float, float, float]
    smooth_sigma: float = 0.75
    clamp_min_hu: float = -1024.0
    clamp_max_hu: float = 8192.0
    shift: float = 1024.0
    scale: float = 1.0 / 2048.0


def clamp_shift_scale(array: np.ndarray, *, clamp_min: float, clamp_max: float, shift: float, scale: float) -> np.ndarray:
    clipped = np.clip(array.astype(np.float32, copy=False), clamp_min, clamp_max)
    return (clipped + float(shift)) * float(scale)


def preprocess_ct_image(
    image: sitk.Image,
    *,
    config: PreprocessConfig,
) -> sitk.Image:
    working = sitk.Cast(image, sitk.sitkFloat32)
    if config.smooth_sigma > 0:
        working = sitk.SmoothingRecursiveGaussian(working, float(config.smooth_sigma))

    arr = sitk.GetArrayFromImage(working).astype(np.float32, copy=False)
    arr = clamp_shift_scale(
        arr,
        clamp_min=config.clamp_min_hu,
        clamp_max=config.clamp_max_hu,
        shift=config.shift,
        scale=config.scale,
    )
    processed = sitk.GetImageFromArray(arr)
    processed.CopyInformation(working)
    return processed


def resample_image(
    image: sitk.Image,
    *,
    spacing_xyz: tuple[float, float, float],
    interpolator: int = sitk.sitkLinear,
    default_value: float = -1024.0,
) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        max(1, int(round(original_size[i] * original_spacing[i] / spacing_xyz[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(tuple(float(v) for v in spacing_xyz))
    resampler.SetSize([int(v) for v in new_size])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(default_value))
    return resampler.Execute(image)


def save_preprocessed_image(image: sitk.Image, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path), useCompression=True)
    return output_path
