from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from spine_segment.backend import SpineSegmentBackend, SpineSegmentBackendError
from spine_segment.compartments import CortTrabConfig, derive_cort_trab_labels
from spine_segment.device import resolve_device
from spine_segment.io import read_image, write_image
from spine_segment.outputs import OutputPaths, build_output_paths


@dataclass(frozen=True, slots=True)
class SegmentRunResult:
    input_path: Path
    output_paths: OutputPaths
    device: str
    metadata: dict[str, Any]


def segment_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    backend: SpineSegmentBackend,
    device: str = "auto",
    overwrite: bool = False,
    cort_trab_config: CortTrabConfig | None = None,
    level_only: bool = False,
    localization_only: bool = False,
) -> SegmentRunResult:
    source_path = Path(input_path)
    image = read_image(source_path, pixel_type=sitk.sitkFloat32)
    selected_device = resolve_device(device)
    output_paths = build_output_paths(source_path, output_dir)
    if localization_only:
        localization = backend.localize(
            image=image,
            source_path=source_path,
            device=selected_device,
        )
        centroids = localization.centroids
        metadata = dict(localization.metadata or {})
        metadata["device"] = selected_device
        metadata["localization_only"] = True
        write_json(
            {
                "input": str(source_path),
                "coordinate_system": "voxel_xyz",
                "centroids": centroids,
                "metadata": metadata,
            },
            output_paths.centroids,
            overwrite=overwrite,
        )
        return SegmentRunResult(
            input_path=source_path,
            output_paths=output_paths,
            device=selected_device,
            metadata=metadata,
        )

    result = backend.segment(
        image=image,
        source_path=source_path,
        device=selected_device,
        level_only=level_only,
    )

    write_image(result.vertebral_level, output_paths.vertebral_level, overwrite=overwrite)
    centroids = centroids_from_labelmap(result.vertebral_level)
    if level_only:
        metadata = dict(result.metadata or {})
        metadata["device"] = selected_device
        metadata["level_only"] = True
        write_centroids_json(
            input_path=source_path,
            centroids=centroids,
            metadata=metadata,
            path=output_paths.centroids,
            overwrite=overwrite,
        )
        return SegmentRunResult(
            input_path=source_path,
            output_paths=output_paths,
            device=selected_device,
            metadata=metadata,
        )

    if result.process_body is None:
        raise SpineSegmentBackendError(
            "Standalone spine backend did not return a process/body labelmap."
        )

    cort_trab = result.cort_trab or derive_cort_trab_labels(
        image=image,
        process_body=result.process_body,
        vertebral_level=result.vertebral_level,
        config=cort_trab_config,
    )

    write_image(result.process_body, output_paths.process_body, overwrite=overwrite)
    write_image(cort_trab, output_paths.cort_trab, overwrite=overwrite)

    metadata = dict(result.metadata or {})
    metadata["device"] = selected_device
    write_centroids_json(
        input_path=source_path,
        centroids=centroids,
        metadata=metadata,
        path=output_paths.centroids,
        overwrite=overwrite,
    )
    return SegmentRunResult(
        input_path=source_path,
        output_paths=output_paths,
        device=selected_device,
        metadata=metadata,
    )


def segment_files(
    input_paths: list[str | Path],
    output_dir: str | Path,
    *,
    backend: SpineSegmentBackend,
    device: str = "auto",
    overwrite: bool = False,
    cort_trab_config: CortTrabConfig | None = None,
    level_only: bool = False,
    localization_only: bool = False,
) -> list[SegmentRunResult]:
    return [
        segment_file(
            path,
            output_dir,
            backend=backend,
            device=device,
            overwrite=overwrite,
            cort_trab_config=cort_trab_config,
            level_only=level_only,
            localization_only=localization_only,
        )
        for path in input_paths
    ]


def write_json(payload: dict[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output without --overwrite: {output_path}"
        )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_centroids_json(
    *,
    input_path: Path,
    centroids: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    path: str | Path,
    overwrite: bool,
) -> Path:
    return write_json(
        {
            "input": str(input_path),
            "coordinate_system": "voxel_xyz",
            "centroids": centroids,
            "metadata": metadata,
        },
        path,
        overwrite=overwrite,
    )


def centroids_from_labelmap(labelmap: sitk.Image) -> dict[str, dict[str, Any]]:
    labels = sitk.GetArrayFromImage(labelmap)
    centroids: dict[str, dict[str, Any]] = {}
    for raw_label in sorted(int(value) for value in np.unique(labels) if int(value) != 0):
        coords_zyx = np.argwhere(labels == raw_label)
        if coords_zyx.size == 0:
            continue
        centroid_zyx = coords_zyx.mean(axis=0)
        voxel_xyz = (
            float(centroid_zyx[2]),
            float(centroid_zyx[1]),
            float(centroid_zyx[0]),
        )
        physical_xyz = labelmap.TransformContinuousIndexToPhysicalPoint(voxel_xyz)
        centroids[str(raw_label)] = {
            "label": raw_label,
            "index": _verse_index_from_label(raw_label),
            "voxel_xyz": [float(value) for value in voxel_xyz],
            "physical_xyz": [float(value) for value in physical_xyz],
            "voxel_count": int(coords_zyx.shape[0]),
            "source": "segmentation",
        }
    return centroids


def _verse_index_from_label(label: int) -> int | None:
    if 1 <= int(label) <= 25:
        return int(label) - 1
    if int(label) == 28:
        return 25
    return None
