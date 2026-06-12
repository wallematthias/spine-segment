from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from spine_segment.landmarks import Landmark


@dataclass(frozen=True, slots=True)
class PhysicalBoundingBox:
    start: tuple[float, float, float]
    end: tuple[float, float, float]


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    binary = sitk.GetImageFromArray(mask.astype(np.uint8))
    connected = sitk.ConnectedComponent(binary)
    relabeled = sitk.RelabelComponent(connected, sortByObjectSize=True)
    arr = sitk.GetArrayFromImage(relabeled)
    return arr == 1


def bounding_box_from_heatmap(
    prediction: np.ndarray,
    *,
    transformation: sitk.Transform,
    image_spacing_xyz: Sequence[float],
    threshold: float | None = None,
    threshold_percentile: float = 98.0,
    threshold_fraction_of_peak_percentile: float = 0.5,
    min_threshold: float = 0.0,
) -> PhysicalBoundingBox:
    squeezed = np.squeeze(prediction.astype(np.float32, copy=False))
    finite_positive = squeezed[np.isfinite(squeezed) & (squeezed > 0)]
    if finite_positive.size == 0:
        raise ValueError("Cannot compute bounding box from an empty heatmap.")
    fraction = (
        float(threshold_fraction_of_peak_percentile)
        if threshold is None
        else float(threshold)
    )
    peak_percentile = float(np.percentile(finite_positive, float(threshold_percentile)))
    threshold_value = max(1e-6, float(min_threshold), peak_percentile * fraction)
    mask = squeezed >= threshold_value
    mask = largest_connected_component(mask)
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("No foreground voxels after thresholding heatmap.")
    min_zyx = coords.min(axis=0)
    max_zyx = coords.max(axis=0) + 1
    spacing = np.asarray(tuple(float(v) for v in image_spacing_xyz), dtype=np.float64)
    start_xyz = np.flip(min_zyx.astype(np.float64)) * spacing
    end_xyz = np.flip(max_zyx.astype(np.float64)) * spacing
    start = transformation.TransformPoint(tuple(float(v) for v in start_xyz))
    end = transformation.TransformPoint(tuple(float(v) for v in end_xyz))
    return PhysicalBoundingBox(
        start=tuple(float(v) for v in start),
        end=tuple(float(v) for v in end),
    )


def filter_landmarks_top_bottom(
    landmarks: Sequence[Landmark],
    *,
    image: sitk.Image,
    reference_image: sitk.Image | None = None,
    margin_mm: float = 10.0,
) -> list[Landmark]:
    image_for_extent = reference_image or image
    size = image_for_extent.GetSize()
    z_values = [
        image_for_extent.TransformIndexToPhysicalPoint((x, y, z))[2]
        for x in (0, max(0, size[0] - 1))
        for y in (0, max(0, size[1] - 1))
        for z in (0, max(0, size[2] - 1))
    ]
    z_min = min(z_values) + float(margin_mm)
    z_max = max(z_values) - float(margin_mm)
    filtered: list[Landmark] = []
    for landmark in landmarks:
        if landmark.is_valid and z_min < landmark.z < z_max:
            filtered.append(landmark)
        else:
            filtered.append(Landmark.invalid())
    return filtered


def reshift_landmarks(landmarks: Sequence[Landmark]) -> list[Landmark]:
    current = list(landmarks)
    invalid = Landmark.invalid()
    if (not current[0].is_valid) and current[7].is_valid:
        if (not current[6].is_valid) and current[5].is_valid:
            current = [invalid] + current[0:5] + current[6:26]
    if (not current[7].is_valid) and current[19].is_valid:
        if (not current[18].is_valid) and current[17].is_valid:
            current = current[0:7] + [invalid] + current[7:18] + current[19:26]
        elif current[25].is_valid:
            current = current[0:7] + current[8:19] + [current[25]] + current[19:25] + [invalid]
    return current


def add_landmarks_from_neighbors(local_maxima_landmarks: Sequence[Sequence[Landmark]]) -> list[list[Landmark]]:
    source = [list(group) for group in local_maxima_landmarks]
    copied = [list(group) for group in local_maxima_landmarks]
    duplicate_penalty = 0.1
    for i in range(2, 6):
        copied[i + 1].extend(
            [Landmark(l.x, l.y, l.z, is_valid=l.is_valid, value=l.value * duplicate_penalty) for l in source[i]]
        )
        copied[i].extend(
            [Landmark(l.x, l.y, l.z, is_valid=l.is_valid, value=l.value * duplicate_penalty) for l in source[i + 1]]
        )
    for i in range(8, 18):
        copied[i + 1].extend(
            [Landmark(l.x, l.y, l.z, is_valid=l.is_valid, value=l.value * duplicate_penalty) for l in source[i]]
        )
        copied[i].extend(
            [Landmark(l.x, l.y, l.z, is_valid=l.is_valid, value=l.value * duplicate_penalty) for l in source[i + 1]]
        )
    copied[25].extend([Landmark(l.x, l.y, l.z, is_valid=l.is_valid, value=l.value) for l in source[18]])
    copied[18].extend([Landmark(l.x, l.y, l.z, is_valid=l.is_valid, value=l.value) for l in source[25]])
    for i in range(20, 24):
        copied[i + 1].extend(
            [Landmark(l.x, l.y, l.z, is_valid=l.is_valid, value=l.value * duplicate_penalty) for l in source[i]]
        )
        copied[i].extend(
            [Landmark(l.x, l.y, l.z, is_valid=l.is_valid, value=l.value * duplicate_penalty) for l in source[i + 1]]
        )
    return copied
