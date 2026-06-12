from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from spine_segment.labels import BODY_LABEL, CORTICAL_LABEL, TRABECULAR_LABEL


@dataclass(frozen=True, slots=True)
class CortTrabConfig:
    cortical_threshold_hu: float = 500.0
    isotropic_spacing_mm: float = 1.0
    cortical_max_thickness_mm: float = 6.0
    force_outline_voxels: int = 1


def derive_cort_trab_labels(
    *,
    image: sitk.Image,
    process_body: sitk.Image,
    vertebral_level: sitk.Image | None = None,
    config: CortTrabConfig | None = None,
) -> sitk.Image:
    cfg = config or CortTrabConfig()
    roi_mask = sitk.Cast((vertebral_level > 0) if vertebral_level is not None else (process_body > 0), sitk.sitkUInt8)
    roi_mask = _largest_component_mask(roi_mask)
    roi_arr = sitk.GetArrayFromImage(roi_mask).astype(bool)
    output = np.zeros(roi_arr.shape, dtype=np.uint8)
    if not np.any(roi_arr):
        out_img = sitk.GetImageFromArray(output)
        out_img.CopyInformation(process_body)
        return out_img

    crop_bbox = _expanded_mask_bbox(
        roi_mask,
        margin_mm=max(
            float(cfg.cortical_max_thickness_mm),
            float(cfg.isotropic_spacing_mm),
        ),
    )
    image_crop = _crop_image_to_bbox(image, crop_bbox)
    roi_crop = _crop_image_to_bbox(roi_mask, crop_bbox)
    roi_crop_arr = sitk.GetArrayFromImage(roi_crop).astype(bool)

    density_iso = _resample_image_to_isotropic(
        image_crop,
        iso_spacing=float(cfg.isotropic_spacing_mm),
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
        pixel_id=sitk.sitkFloat32,
    )
    roi_iso = _resample_image_to_isotropic(
        sitk.Cast(roi_crop > 0, sitk.sitkUInt8),
        iso_spacing=float(cfg.isotropic_spacing_mm),
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0.0,
        pixel_id=sitk.sitkUInt8,
    )
    density_arr = sitk.GetArrayFromImage(sitk.Cast(density_iso, sitk.sitkFloat32)).astype(np.float32)
    roi = sitk.GetArrayFromImage(roi_iso).astype(bool)
    if not np.any(roi):
        out_img = sitk.GetImageFromArray(output)
        out_img.CopyInformation(process_body)
        return out_img

    distance_image = sitk.SignedMaurerDistanceMap(
        sitk.Cast(roi_iso > 0, sitk.sitkUInt8),
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    distance_arr = sitk.GetArrayFromImage(distance_image).astype(np.float32)
    outer_shell = roi & (distance_arr <= max(float(cfg.isotropic_spacing_mm), 1e-3))
    annulus_shell = roi & (distance_arr <= float(cfg.cortical_max_thickness_mm))
    high_density_shell = annulus_shell & (density_arr >= float(cfg.cortical_threshold_hu))
    cortical_iso_arr = _keep_components_touching_seed(
        candidate=(outer_shell | high_density_shell) & roi,
        seed=outer_shell,
    )

    cortical_iso = sitk.GetImageFromArray(cortical_iso_arr.astype(np.uint8))
    cortical_iso.CopyInformation(roi_iso)
    cortical_native = _resample_mask_to_reference(cortical_iso, roi_crop)
    cortical_native_arr = sitk.GetArrayFromImage(cortical_native).astype(bool)
    outline_arr = _native_outline(roi_crop, radius_voxels=max(1, int(cfg.force_outline_voxels)))

    crop_output = np.zeros(roi_crop_arr.shape, dtype=np.uint8)
    cortical = roi_crop_arr & (cortical_native_arr | outline_arr)
    trabecular = roi_crop_arr & ~cortical
    crop_output[cortical] = CORTICAL_LABEL
    crop_output[trabecular] = TRABECULAR_LABEL

    z_slice, y_slice, x_slice = _bbox_to_slices_zyx(crop_bbox)
    output[z_slice, y_slice, x_slice] = crop_output
    out_img = sitk.GetImageFromArray(output)
    out_img.CopyInformation(process_body)
    return out_img


def _largest_component_mask(mask: sitk.Image) -> sitk.Image:
    connected = sitk.ConnectedComponent(sitk.Cast(mask > 0, sitk.sitkUInt8))
    relabeled = sitk.RelabelComponent(connected, sortByObjectSize=True)
    return sitk.Cast(relabeled == 1, sitk.sitkUInt8)


def _keep_components_touching_seed(*, candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    if not np.any(candidate) or not np.any(seed):
        return candidate & seed
    connected = sitk.ConnectedComponent(sitk.GetImageFromArray(candidate.astype(np.uint8)))
    connected_arr = sitk.GetArrayFromImage(connected).astype(np.int32)
    touching = np.unique(connected_arr[seed & (connected_arr > 0)])
    if touching.size == 0:
        return seed & candidate
    return np.isin(connected_arr, touching)


def _native_outline(mask: sitk.Image, *, radius_voxels: int) -> np.ndarray:
    binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
    eroded = sitk.BinaryErode(binary, [int(radius_voxels)] * 3)
    return sitk.GetArrayFromImage(binary & sitk.Not(eroded)).astype(bool)


def _expanded_mask_bbox(mask: sitk.Image, *, margin_mm: float) -> tuple[int, int, int, int, int, int]:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    if not stats.HasLabel(1):
        return (0, 0, 0, 1, 1, 1)
    x, y, z, sx, sy, sz = (int(v) for v in stats.GetBoundingBox(1))
    spacing = mask.GetSpacing()
    margin = [
        int(np.ceil(float(margin_mm) / float(spacing[axis])))
        for axis in range(3)
    ]
    size = mask.GetSize()
    x0 = max(0, x - margin[0])
    y0 = max(0, y - margin[1])
    z0 = max(0, z - margin[2])
    x1 = min(size[0], x + sx + margin[0])
    y1 = min(size[1], y + sy + margin[1])
    z1 = min(size[2], z + sz + margin[2])
    return (x0, y0, z0, max(1, x1 - x0), max(1, y1 - y0), max(1, z1 - z0))


def _crop_image_to_bbox(
    image: sitk.Image,
    bbox_xyz_size: tuple[int, int, int, int, int, int],
) -> sitk.Image:
    x, y, z, sx, sy, sz = bbox_xyz_size
    return sitk.RegionOfInterest(image, size=[sx, sy, sz], index=[x, y, z])


def _bbox_to_slices_zyx(
    bbox_xyz_size: tuple[int, int, int, int, int, int],
) -> tuple[slice, slice, slice]:
    x, y, z, sx, sy, sz = bbox_xyz_size
    return (slice(z, z + sz), slice(y, y + sy), slice(x, x + sx))


def _resample_image_to_isotropic(
    image: sitk.Image,
    *,
    iso_spacing: float,
    interpolator: int,
    default_value: float,
    pixel_id: int,
) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_spacing = (float(iso_spacing),) * 3
    new_size = [
        max(1, int(round(original_size[axis] * original_spacing[axis] / new_spacing[axis])))
        for axis in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetOutputPixelType(pixel_id)
    return resampler.Execute(image)


def _resample_mask_to_reference(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return sitk.Cast(resampler.Execute(mask) > 0, sitk.sitkUInt8)
