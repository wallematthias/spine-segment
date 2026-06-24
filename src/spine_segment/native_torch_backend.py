from __future__ import annotations

import os
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from spine_segment.backend import LocalizationResult, SegmentationResult, SpineSegmentBackend, SpineSegmentBackendError
from spine_segment.landmarks import Landmark
from spine_segment.model_bundle import (
    NativeTorchBundlePaths,
    resolve_native_torch_bundle,
    validate_native_torch_bundle,
)
from spine_segment.postprocess import bounding_box_from_heatmap
from spine_segment.postprocess import add_landmarks_from_neighbors, filter_landmarks_top_bottom, reshift_landmarks
from spine_segment.process_body_relabel import ProcessBodyRelabeler
from spine_segment.pytorch_models import MdatArchitectureSpec, build_model
from spine_segment.sequence_solver import solve_spine_sequence


@dataclass(frozen=True, slots=True)
class _LocalizationStageResult:
    landmarks: list[tuple[str, Landmark]]
    spine_bbox_start: tuple[float, float, float]
    spine_bbox_end: tuple[float, float, float]
    spine_tile_count: int = 1
    selected_spine_tile: str = "center"


@dataclass(frozen=True, slots=True)
class _SpineLocalizationTile:
    name: str
    image_array: np.ndarray
    transform: sitk.Transform
    size_xyz: tuple[int, int, int]


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise SpineSegmentBackendError(
            "PyTorch is required for the native torch backend."
        ) from exc
    return torch


def _copy_array_to_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def _normalize_ct(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(array.astype(np.float32, copy=False), -2048.0, 2048.0)
    return np.clip(clipped / 2048.0, -1.0, 1.0)


def _print_progress(message: str) -> None:
    print(f"[spine-segment] {message}", flush=True)


def _valid_output_size_for_extent(
    extent_xyz: tuple[float, float, float],
    *,
    spacing: float,
    valid_sizes_xyz: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> tuple[int, int, int]:
    sizes: list[int] = []
    for axis, extent in enumerate(extent_xyz):
        needed = int(math.ceil(float(extent) / float(spacing)))
        chosen = valid_sizes_xyz[axis][-1]
        for valid_size in sorted(valid_sizes_xyz[axis]):
            if needed < valid_size:
                chosen = int(valid_size)
                break
        sizes.append(chosen)
    return tuple(sizes)  # type: ignore[return-value]


def _image_extent_xyz(image: sitk.Image) -> tuple[float, float, float]:
    return tuple(float(size) * float(spacing) for size, spacing in zip(image.GetSize(), image.GetSpacing()))


def _image_center_xyz(image: sitk.Image) -> np.ndarray:
    half_index = [(size - 1) * 0.5 for size in image.GetSize()]
    return np.asarray(image.TransformContinuousIndexToPhysicalPoint(half_index), dtype=np.float64)


def _output_center_xyz(size_xyz: tuple[int, int, int], spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    return np.asarray([(size_xyz[i] - 1) * spacing_xyz[i] * 0.5 for i in range(3)], dtype=np.float64)


def _translation_transform(offset_xyz: np.ndarray | tuple[float, float, float]) -> sitk.Transform:
    transform = sitk.AffineTransform(3)
    transform.Translate(tuple(float(v) for v in offset_xyz))
    return transform


def _resample_with_offset(
    image: sitk.Image,
    *,
    size_xyz: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    offset_xyz: np.ndarray,
    interpolator: int,
    default_value: float,
    pixel_id: int = sitk.sitkFloat32,
) -> tuple[sitk.Image, sitk.Transform]:
    transform = _translation_transform(offset_xyz)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(v) for v in size_xyz])
    resampler.SetOutputSpacing(tuple(float(v) for v in spacing_xyz))
    resampler.SetOutputOrigin((0.0, 0.0, 0.0))
    resampler.SetOutputDirection(np.eye(3).reshape(-1).tolist())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetOutputPixelType(pixel_id)
    resampler.SetTransform(transform)
    return resampler.Execute(image), transform


def _orientation_code(image: sitk.Image) -> str:
    orienter = sitk.DICOMOrientImageFilter()
    return str(orienter.GetOrientationFromDirectionCosines(image.GetDirection()))


def _orient_image(image: sitk.Image, orientation: str) -> sitk.Image:
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(str(orientation))
    return orienter.Execute(image)


def _preprocess_source_image(image: sitk.Image) -> sitk.Image:
    working = sitk.Cast(image, sitk.sitkFloat32)
    working = sitk.Clamp(working, sitk.sitkFloat32, -1024.0, 32767.0)
    return sitk.SmoothingRecursiveGaussian(working, 0.75)


def _resample_centered(
    image: sitk.Image,
    *,
    spacing: float,
    valid_sizes_xyz: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, sitk.Transform, tuple[int, int, int]]:
    size_xyz = _valid_output_size_for_extent(
        _image_extent_xyz(image),
        spacing=spacing,
        valid_sizes_xyz=valid_sizes_xyz,
    )
    spacing_xyz = (float(spacing),) * 3
    offset = _image_center_xyz(image) - _output_center_xyz(size_xyz, spacing_xyz)
    resampled, transform = _resample_with_offset(
        image,
        size_xyz=size_xyz,
        spacing_xyz=spacing_xyz,
        offset_xyz=offset,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
    )
    return _normalize_ct(sitk.GetArrayFromImage(resampled)), transform, size_xyz


def _spine_localization_tiles(
    image: sitk.Image,
    *,
    spacing: float,
    valid_sizes_xyz: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> list[_SpineLocalizationTile]:
    """
    Build one or more spine-localizer inputs.

    The MDAT spine-localizer supports at most the largest configured z size.
    Whole-body scans can exceed that physical extent, so a single centered
    resample can crop away the superior or inferior spine. For long scans,
    evaluate overlapping z windows and let vertebra localization choose the
    most plausible downstream crop.
    """
    extent_xyz = _image_extent_xyz(image)
    centered_array, centered_transform, centered_size = _resample_centered(
        image,
        spacing=spacing,
        valid_sizes_xyz=valid_sizes_xyz,
    )
    max_size_z = int(max(valid_sizes_xyz[2]))
    max_extent_z = float(max_size_z) * float(spacing)
    if extent_xyz[2] <= max_extent_z:
        return [
            _SpineLocalizationTile(
                name="center",
                image_array=centered_array,
                transform=centered_transform,
                size_xyz=centered_size,
            )
        ]

    size_x = _valid_output_size_for_extent(
        extent_xyz,
        spacing=spacing,
        valid_sizes_xyz=valid_sizes_xyz,
    )[0]
    size_y = _valid_output_size_for_extent(
        extent_xyz,
        spacing=spacing,
        valid_sizes_xyz=valid_sizes_xyz,
    )[1]
    size_xyz = (size_x, size_y, max_size_z)
    spacing_xyz = (float(spacing),) * 3
    image_center = _image_center_xyz(image)
    output_center = _output_center_xyz(size_xyz, spacing_xyz)
    base_offset = image_center - output_center

    z_min = float(image.GetOrigin()[2])
    z_max = z_min + float(image.GetSize()[2]) * float(image.GetSpacing()[2])
    z_last = z_max - max_extent_z
    stride = max_extent_z * 0.75
    starts: list[tuple[str, float]] = [("inferior", z_min)]
    current = z_min + stride
    tile_index = 1
    while current < z_last:
        starts.append((f"tile{tile_index}", current))
        tile_index += 1
        current += stride
    starts.append(("superior", z_last))
    starts.append(("center", float(base_offset[2])))

    unique_starts: list[tuple[str, float]] = []
    for name, z_start in starts:
        z_start = float(max(z_min, min(z_start, z_last)))
        if any(abs(z_start - existing) < 1e-3 for _n, existing in unique_starts):
            continue
        unique_starts.append((name, z_start))

    tiles: list[_SpineLocalizationTile] = []
    for name, z_start in unique_starts:
        offset = np.asarray(base_offset, dtype=np.float64).copy()
        offset[2] = z_start
        resampled, transform = _resample_with_offset(
            image,
            size_xyz=size_xyz,
            spacing_xyz=spacing_xyz,
            offset_xyz=offset,
            interpolator=sitk.sitkLinear,
            default_value=-1024.0,
        )
        tiles.append(
            _SpineLocalizationTile(
                name=name,
                image_array=_normalize_ct(sitk.GetArrayFromImage(resampled)),
                transform=transform,
                size_xyz=size_xyz,
            )
        )
    return tiles


def _resample_bbox_centered(
    image: sitk.Image,
    *,
    bbox_start_xyz: tuple[float, float, float],
    bbox_end_xyz: tuple[float, float, float],
    spacing: float,
    valid_sizes_xyz: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, sitk.Transform, tuple[int, int, int]]:
    image_min = np.asarray(image.GetOrigin(), dtype=np.float64)
    image_max = np.asarray(image.GetOrigin(), dtype=np.float64) + np.asarray(_image_extent_xyz(image), dtype=np.float64)
    start = np.maximum(np.asarray(bbox_start_xyz, dtype=np.float64) - 64.0, image_min)
    end = np.minimum(np.asarray(bbox_end_xyz, dtype=np.float64) + 64.0, image_max)
    extent = np.maximum(end - start, 1.0)
    size_xyz = _valid_output_size_for_extent(
        tuple(float(v) for v in extent),
        spacing=spacing,
        valid_sizes_xyz=valid_sizes_xyz,
    )
    spacing_xyz = (float(spacing),) * 3
    bbox_center = start + (extent - 1.0) * 0.5
    offset = bbox_center - _output_center_xyz(size_xyz, spacing_xyz)
    resampled, transform = _resample_with_offset(
        image,
        size_xyz=size_xyz,
        spacing_xyz=spacing_xyz,
        offset_xyz=offset,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
    )
    return _normalize_ct(sitk.GetArrayFromImage(resampled)), transform, size_xyz


def _tensor_from_zyx(array: np.ndarray, *, device: str):
    torch = _require_torch()
    return torch.from_numpy(array.astype(np.float32, copy=False))[None, None].to(
        device=device,
        dtype=torch.float32,
    )


def _to_numpy_zyx(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _landmark_from_heatmap_voxel(
    landmark: Landmark,
    *,
    transform: sitk.Transform,
    spacing: float,
) -> Landmark:
    if not landmark.is_valid:
        return Landmark.invalid()
    point = transform.TransformPoint(
        (
            float(landmark.x) * float(spacing),
            float(landmark.y) * float(spacing),
            float(landmark.z) * float(spacing),
        )
    )
    return Landmark(float(point[0]), float(point[1]), float(point[2]), True, landmark.value)


def _landmark_label_from_index(index: int) -> int:
    labels = [i + 1 for i in range(25)] + [28]
    return int(labels[int(index)])


def _landmarks_to_centroids(
    landmarks: list[tuple[str, Landmark]],
    image: sitk.Image,
) -> dict[str, dict[str, Any]]:
    centroids: dict[str, dict[str, Any]] = {}
    for landmark_id, landmark in landmarks:
        label = _landmark_label_from_index(int(landmark_id))
        physical_xyz = tuple(float(v) for v in landmark.coords)
        voxel_xyz = image.TransformPhysicalPointToContinuousIndex(physical_xyz)
        centroids[str(label)] = {
            "label": label,
            "index": int(landmark_id),
            "voxel_xyz": [float(v) for v in voxel_xyz],
            "physical_xyz": [float(v) for v in physical_xyz],
            "score": float(landmark.value),
        }
    return centroids


def _smooth_array(image: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.astype(np.float32, copy=False)
    sitk_image = sitk.GetImageFromArray(image.astype(np.float32, copy=False))
    return sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(sitk_image, float(sigma)))


def _local_maxima_landmarks(
    heatmap: np.ndarray,
    *,
    transform: sitk.Transform,
    spacing: float,
    min_value: float = 0.05,
    smoothing_sigma: float = 2.0,
) -> list[Landmark]:
    smoothed = _smooth_array(heatmap, sigma=smoothing_sigma)
    if smoothed.size == 0:
        return []
    padded = np.pad(smoothed, 1, mode="edge")
    local_max = np.ones(smoothed.shape, dtype=bool)
    center = padded[1:-1, 1:-1, 1:-1]
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                neighbor = padded[
                    1 + dz : 1 + dz + smoothed.shape[0],
                    1 + dy : 1 + dy + smoothed.shape[1],
                    1 + dx : 1 + dx + smoothed.shape[2],
                ]
                local_max &= center >= neighbor
    coords = np.argwhere(local_max & (smoothed > float(min_value)))
    if coords.size == 0:
        return []
    values = smoothed[tuple(coords.T)]
    order = np.argsort(values)[::-1]
    landmarks: list[Landmark] = []
    for idx in order:
        z, y, x = coords[idx]
        value = float(values[idx])
        point = transform.TransformPoint(
            (float(x) * float(spacing), float(y) * float(spacing), float(z) * float(spacing))
        )
        landmarks.append(Landmark(float(point[0]), float(point[1]), float(point[2]), True, value))
    return landmarks


def _gaussian_heatmap(
    shape_zyx: tuple[int, int, int],
    *,
    center_xyz: tuple[float, float, float],
    sigma: float,
) -> np.ndarray:
    z_size, y_size, x_size = shape_zyx
    z, y, x = np.ogrid[:z_size, :y_size, :x_size]
    cx, cy, cz = center_xyz
    squared = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return np.exp(-squared / (2.0 * float(sigma) ** 2)).astype(np.float32)


def _resample_patch_to_scan(
    patch_zyx: np.ndarray,
    *,
    patch_transform: sitk.Transform,
    reference: sitk.Image,
    spacing: float,
) -> np.ndarray:
    patch_image = sitk.GetImageFromArray(patch_zyx.astype(np.float32, copy=False))
    patch_image.SetSpacing((float(spacing),) * 3)
    patch_image.SetOrigin((0.0, 0.0, 0.0))
    patch_image.SetDirection(np.eye(3).reshape(-1).tolist())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetTransform(patch_transform.GetInverse())
    return sitk.GetArrayFromImage(resampler.Execute(patch_image)).astype(np.float32, copy=False)


@dataclass(slots=True)
class NativeTorchBackend(SpineSegmentBackend):
    bundle_root: Path
    strict_bundle: bool = True
    max_segmentation_landmarks: int | None = None
    vertebra_segmentation_batch_size: int = 1
    process_body_batch_size: int = 4
    progress: bool = True
    _bundle_paths: NativeTorchBundlePaths | None = None
    _models_loaded: bool = False
    _localization_models_loaded: bool = False
    _spine_localization: Any = None
    _vertebrae_localization: Any = None
    _vertebrae_segmentation: Any = None
    _process_body_relabeler: Any = None

    def __post_init__(self) -> None:
        self.bundle_root = Path(self.bundle_root).expanduser().resolve()
        self._bundle_paths = resolve_native_torch_bundle(self.bundle_root)
        if self.strict_bundle:
            validate_native_torch_bundle(self._bundle_paths)

    def _load_models(self, device: str, *, require_segmentation: bool = True) -> None:
        if require_segmentation and self._models_loaded:
            return
        if not require_segmentation and (self._localization_models_loaded or self._models_loaded):
            return
        if self.progress:
            _print_progress(f"loading native PyTorch checkpoints from {self.bundle_root}")
        torch = _require_torch()
        assert self._bundle_paths is not None

        specs = {
            "spine_localization": MdatArchitectureSpec(
                name="spine_localization",
                in_channels=1,
                num_labels=1,
                num_filters_base=96,
                num_levels=5,
                activation="lrelu",
                model_kind="unet",
            ),
            "vertebrae_localization": MdatArchitectureSpec(
                name="vertebrae_localization",
                in_channels=1,
                num_labels=26,
                num_filters_base=96,
                num_levels=4,
                activation="lrelu",
                local_activation="tanh",
                spatial_activation="tanh",
                spatial_downsample=4,
                model_kind="scn",
            ),
            "vertebrae_segmentation": MdatArchitectureSpec(
                name="vertebrae_segmentation",
                in_channels=2,
                num_labels=1,
                num_filters_base=96,
                num_levels=5,
                activation="lrelu",
                model_kind="unet",
            ),
        }

        target_device = torch.device(device)
        if not self._localization_models_loaded:
            spine_localization = build_model(specs["spine_localization"])
            vertebrae_localization = build_model(specs["vertebrae_localization"])
            spine_localization.load_state_dict(
                torch.load(self._bundle_paths.spine_localization, map_location=device)
            )
            vertebrae_localization.load_state_dict(
                torch.load(self._bundle_paths.vertebrae_localization, map_location=device)
            )
            self._spine_localization = spine_localization.to(target_device).eval()
            self._vertebrae_localization = vertebrae_localization.to(target_device).eval()
            self._localization_models_loaded = True
        if require_segmentation and not self._models_loaded:
            vertebrae_segmentation = build_model(specs["vertebrae_segmentation"])
            vertebrae_segmentation.load_state_dict(
                torch.load(self._bundle_paths.vertebrae_segmentation, map_location=device)
            )
            self._vertebrae_segmentation = vertebrae_segmentation.to(target_device).eval()
            self._process_body_relabeler = ProcessBodyRelabeler(
                checkpoint_path=self._bundle_paths.process_body_checkpoint,
                batch_size=max(1, int(self.process_body_batch_size)),
            )
            self._models_loaded = True
        if self.progress:
            _print_progress("checkpoints loaded")

    def _localize_exact(
        self,
        image: sitk.Image,
        *,
        source_path: Path,
        device: str,
    ) -> _LocalizationStageResult:
        torch = _require_torch()
        assert self._spine_localization is not None
        assert self._vertebrae_localization is not None

        preprocessed = _preprocess_source_image(image)

        with torch.inference_mode():
            if self.progress:
                _print_progress("running spine localization model")
            spine_tiles = _spine_localization_tiles(
                preprocessed,
                spacing=8.0,
                valid_sizes_xyz=((32, 64, 96, 128), (32, 64, 96, 128), (32, 64, 96, 128)),
            )

            candidates: list[
                tuple[
                    tuple[int, float],
                    str,
                    Any,
                    list[tuple[str, Landmark]],
                ]
            ] = []
            last_error: Exception | None = None
            for tile in spine_tiles:
                spine_logits = self._spine_localization(
                    _tensor_from_zyx(tile.image_array, device=device)
                )
                spine_heatmap = _to_numpy_zyx(spine_logits[0, 0])

                try:
                    bbox = bounding_box_from_heatmap(
                        spine_heatmap,
                        transformation=tile.transform,
                        image_spacing_xyz=(8.0, 8.0, 8.0),
                        threshold=0.5,
                    )
                except ValueError as exc:
                    last_error = exc
                    continue

                loc_array, loc_transform, _loc_size = _resample_bbox_centered(
                    preprocessed,
                    bbox_start_xyz=bbox.start,
                    bbox_end_xyz=bbox.end,
                    spacing=2.0,
                    valid_sizes_xyz=(
                        (32, 64, 96, 128),
                        (32, 64, 96, 128),
                        tuple(32 + i * 32 for i in range(20)),
                    ),
                )
                vertebrae_heatmaps, _local, _spatial = self._vertebrae_localization(
                    _tensor_from_zyx(loc_array, device=device)
                )
                vertebrae_np = _to_numpy_zyx(vertebrae_heatmaps[0])
                local_maxima = [
                    _local_maxima_landmarks(
                        vertebrae_np[index],
                        transform=loc_transform,
                        spacing=2.0,
                    )
                    for index in range(vertebrae_np.shape[0])
                ]
                no_post_landmarks = [
                    candidates[0] if candidates else Landmark.invalid()
                    for candidates in local_maxima
                ]
                valid_candidate_count = sum(
                    1
                    for candidates in local_maxima
                    for candidate in candidates
                    if candidate.is_valid
                )
                if valid_candidate_count == 0:
                    landmarks = no_post_landmarks
                else:
                    landmarks = solve_spine_sequence(
                        add_landmarks_from_neighbors(local_maxima)
                    )
                    landmarks = reshift_landmarks(landmarks)
                    landmarks = filter_landmarks_top_bottom(landmarks, image=image)
                valid_landmarks = [
                    (str(index), landmark)
                    for index, landmark in enumerate(landmarks)
                    if landmark.is_valid and np.all(np.isfinite(landmark.coords))
                ]
                if self.max_segmentation_landmarks is not None:
                    valid_landmarks = valid_landmarks[
                        : int(self.max_segmentation_landmarks)
                    ]
                score = (
                    len(valid_landmarks),
                    float(sum(landmark.value for _index, landmark in valid_landmarks)),
                )
                candidates.append((score, tile.name, bbox, valid_landmarks))

            if not candidates:
                if last_error is not None:
                    raise SpineSegmentBackendError(
                        "MDAT spine localization found no valid bounding boxes."
                    ) from last_error
                raise SpineSegmentBackendError(
                    "MDAT vertebra localization found no valid landmarks."
                )

            _score, selected_tile, bbox, valid_landmarks = max(
                candidates,
                key=lambda item: item[0],
            )
            if not valid_landmarks:
                raise SpineSegmentBackendError("MDAT vertebra localization found no valid landmarks.")

        return _LocalizationStageResult(
            landmarks=valid_landmarks,
            spine_bbox_start=bbox.start,
            spine_bbox_end=bbox.end,
            spine_tile_count=len(spine_tiles),
            selected_spine_tile=selected_tile,
        )

    def _segment_exact(
        self,
        image: sitk.Image,
        *,
        source_path: Path,
        device: str,
        level_only: bool = False,
    ) -> SegmentationResult:
        torch = _require_torch()
        assert self._vertebrae_segmentation is not None

        preprocessed = _preprocess_source_image(image)
        localization = self._localize_exact(image, source_path=source_path, device=device)
        valid_landmarks = localization.landmarks

        with torch.inference_mode():
            if self.progress:
                _print_progress(f"running vertebra segmentation model for {len(valid_landmarks)} landmarks")

            prediction_labels_np = np.zeros(list(reversed(image.GetSize())), dtype=np.uint8)
            prediction_max_value_np = np.ones(prediction_labels_np.shape, dtype=np.float32) * 0.5
            patch_size_xyz = (128, 128, 96)
            patch_spacing = 1.0
            output_center = _output_center_xyz(patch_size_xyz, (patch_spacing,) * 3)

            batch_size = max(1, int(self.vertebra_segmentation_batch_size))
            for batch_start in range(0, len(valid_landmarks), batch_size):
                batch_landmarks = valid_landmarks[batch_start : batch_start + batch_size]
                patch_records = []
                for batch_offset, (landmark_id, landmark) in enumerate(batch_landmarks, start=1):
                    landmark_number = batch_start + batch_offset
                    if self.progress:
                        _print_progress(
                            "preparing vertebra segmentation patch "
                            f"{landmark_number}/{len(valid_landmarks)} "
                            f"label={_landmark_label_from_index(int(landmark_id))}"
                        )
                    center = np.asarray(landmark.coords, dtype=np.float64)
                    offset = center + np.asarray((0.0, 20.0, 0.0), dtype=np.float64) - output_center
                    patch_image, patch_transform = _resample_with_offset(
                        preprocessed,
                        size_xyz=patch_size_xyz,
                        spacing_xyz=(patch_spacing,) * 3,
                        offset_xyz=offset,
                        interpolator=sitk.sitkLinear,
                        default_value=-1024.0,
                    )
                    patch_array = _normalize_ct(sitk.GetArrayFromImage(patch_image))
                    center_in_patch_xyz = tuple(float(v) for v in (center - offset))
                    single_heatmap = _gaussian_heatmap(
                        patch_array.shape,
                        center_xyz=center_in_patch_xyz,
                        sigma=6.0,
                    )
                    patch_records.append(
                        {
                            "landmark_id": int(landmark_id),
                            "patch_transform": patch_transform,
                            "tensor": np.stack([patch_array, single_heatmap], axis=0).astype(
                                np.float32,
                                copy=False,
                            ),
                        }
                    )
                if self.progress:
                    batch_labels = ", ".join(
                        str(_landmark_label_from_index(record["landmark_id"]))
                        for record in patch_records
                    )
                    _print_progress(f"running vertebra segmentation batch labels=[{batch_labels}]")
                patch_tensor = torch.from_numpy(
                    np.stack([record["tensor"] for record in patch_records], axis=0)
                ).to(device=device, dtype=torch.float32)
                patch_probs = torch.sigmoid(self._vertebrae_segmentation(patch_tensor))[:, 0]
                for patch_prob, record in zip(patch_probs, patch_records):
                    patch_np = _to_numpy_zyx(patch_prob)
                    scan_prob = _resample_patch_to_scan(
                        patch_np,
                        patch_transform=record["patch_transform"],
                        reference=image,
                        spacing=patch_spacing,
                    )
                    update = scan_prob > prediction_max_value_np
                    prediction_max_value_np[update] = scan_prob[update]
                    prediction_labels_np[update] = _landmark_label_from_index(record["landmark_id"])

        vertebral_level = _copy_array_to_image(prediction_labels_np, image)
        if level_only:
            return SegmentationResult(
                vertebral_level=vertebral_level,
                metadata={
                    "backend": "native_torch_mdat",
                    "device": device,
                    "num_landmarks": len(valid_landmarks),
                    "level_only": True,
                    "spine_bbox_start": localization.spine_bbox_start,
                    "spine_bbox_end": localization.spine_bbox_end,
                    "spine_tile_count": localization.spine_tile_count,
                    "selected_spine_tile": localization.selected_spine_tile,
                },
            )
        assert self._process_body_relabeler is not None
        if self.progress:
            _print_progress("running process/body relabel model from vertebral segmentation")
        process_body = self._process_body_relabeler.relabel(vertebral_level, device=device)
        return SegmentationResult(
            vertebral_level=vertebral_level,
            process_body=process_body,
            metadata={
                "backend": "native_torch_mdat",
                "device": device,
                "num_landmarks": len(valid_landmarks),
                "process_body_source": "native_torch_segmentation_relabel",
                "spine_bbox_start": localization.spine_bbox_start,
                "spine_bbox_end": localization.spine_bbox_end,
                "spine_tile_count": localization.spine_tile_count,
                "selected_spine_tile": localization.selected_spine_tile,
            },
        )

    def segment(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
        level_only: bool = False,
    ) -> SegmentationResult:
        self._load_models(device, require_segmentation=True)
        input_orientation = _orientation_code(image)
        working_image = _orient_image(image, "LPS")
        result = self._segment_exact(
            working_image,
            source_path=source_path,
            device=device,
            level_only=level_only,
        )
        result.vertebral_level = _orient_image(result.vertebral_level, input_orientation)
        if result.process_body is not None:
            result.process_body = _orient_image(result.process_body, input_orientation)
        if result.cort_trab is not None:
            result.cort_trab = _orient_image(result.cort_trab, input_orientation)
        result.metadata = {
            **(result.metadata or {}),
            "input_orientation": input_orientation,
            "inference_orientation": "LPS",
        }
        return result

    def localize(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
    ) -> LocalizationResult:
        self._load_models(device, require_segmentation=False)
        input_orientation = _orientation_code(image)
        working_image = _orient_image(image, "LPS")
        localization = self._localize_exact(working_image, source_path=source_path, device=device)
        centroids = _landmarks_to_centroids(localization.landmarks, image)
        return LocalizationResult(
            centroids=centroids,
            metadata={
                "backend": "native_torch_mdat",
                "device": device,
                "num_landmarks": len(localization.landmarks),
                "spine_bbox_start": localization.spine_bbox_start,
                "spine_bbox_end": localization.spine_bbox_end,
                "spine_tile_count": localization.spine_tile_count,
                "selected_spine_tile": localization.selected_spine_tile,
                "input_orientation": input_orientation,
                "inference_orientation": "LPS",
            },
        )


def create_backend(bundle_root: str | Path = "./build/model-bundle") -> NativeTorchBackend:
    if str(bundle_root) != "./build/model-bundle":
        return NativeTorchBackend(bundle_root=Path(bundle_root))

    package_root = Path(__file__).resolve().parents[2]
    candidates = []
    env_bundle = os.environ.get("SPINE_SEGMENT_MODEL_BUNDLE")
    if env_bundle:
        candidates.append(Path(env_bundle))
    candidates.extend(
        [
            Path("./model-bundle"),
            Path("./build/model-bundle-pytorch"),
            Path("./build/model-bundle"),
            package_root / "model-bundle",
            package_root / "build" / "model-bundle-pytorch",
            package_root / "build" / "model-bundle",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return NativeTorchBackend(bundle_root=candidate)
    return NativeTorchBackend(bundle_root=Path(bundle_root))
