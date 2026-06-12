from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from spine_segment.api import segment_file
from spine_segment.backend import LocalizationResult, SegmentationResult
from spine_segment.labels import BODY_LABEL, PROCESS_LABEL


class _StubBackend:
    def localize(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
    ) -> LocalizationResult:
        return LocalizationResult(
            centroids={
                "20": {
                    "label": 20,
                    "index": 19,
                    "voxel_xyz": [2.0, 3.0, 4.0],
                    "physical_xyz": list(image.TransformContinuousIndexToPhysicalPoint((2.0, 3.0, 4.0))),
                    "score": 0.9,
                }
            },
            metadata={"backend": "stub", "device_seen": device},
        )

    def segment(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
        level_only: bool = False,
    ) -> SegmentationResult:
        shape = sitk.GetArrayFromImage(image).shape
        metadata = {
            "backend": "stub",
            "device_seen": device,
            "spacing_seen": image.GetSpacing(),
            "size_seen": image.GetSize(),
        }

        vertebral_arr = np.zeros(shape, dtype=np.uint8)
        vertebral_arr[1:4, 1:4, 1:4] = 20
        vertebral = sitk.GetImageFromArray(vertebral_arr)
        vertebral.CopyInformation(image)

        if level_only:
            return SegmentationResult(
                vertebral_level=vertebral,
                metadata={**metadata, "level_only_seen": True},
            )

        process_body_arr = np.zeros(shape, dtype=np.uint8)
        process_body_arr[1:4, 1:4, 1:4] = BODY_LABEL
        process_body_arr[1, 1:4, 1:4] = PROCESS_LABEL
        process_body = sitk.GetImageFromArray(process_body_arr)
        process_body.CopyInformation(image)

        return SegmentationResult(
            vertebral_level=vertebral,
            process_body=process_body,
            metadata=metadata,
        )


def test_segment_file_writes_default_outputs(tmp_path: Path) -> None:
    image = sitk.GetImageFromArray(np.full((6, 6, 6), 200.0, dtype=np.float32))
    input_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(image, str(input_path), useCompression=True)

    result = segment_file(
        input_path=input_path,
        output_dir=tmp_path / "out",
        backend=_StubBackend(),
        device="cpu",
        overwrite=True,
    )

    assert result.output_paths.vertebral_level.exists()
    assert result.output_paths.process_body.exists()
    assert result.output_paths.cort_trab.exists()
    assert result.output_paths.centroids.exists()
    payload = json.loads(result.output_paths.centroids.read_text(encoding="utf-8"))
    assert payload["centroids"]["20"]["voxel_xyz"] == [2.0, 2.0, 2.0]
    assert payload["centroids"]["20"]["voxel_count"] == 27
    assert payload["centroids"]["20"]["source"] == "segmentation"
    assert not list((tmp_path / "out").glob(".spine-segment-tmp-*"))


def test_segment_file_level_only_writes_only_vertebral_level(tmp_path: Path) -> None:
    image = sitk.GetImageFromArray(np.full((6, 6, 6), 200.0, dtype=np.float32))
    input_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(image, str(input_path), useCompression=True)

    result = segment_file(
        input_path=input_path,
        output_dir=tmp_path / "out",
        backend=_StubBackend(),
        device="cpu",
        overwrite=True,
        level_only=True,
    )

    assert result.output_paths.vertebral_level.exists()
    assert result.output_paths.centroids.exists()
    assert not result.output_paths.process_body.exists()
    assert not result.output_paths.cort_trab.exists()
    assert result.metadata["level_only_seen"] is True
    payload = json.loads(result.output_paths.centroids.read_text(encoding="utf-8"))
    assert payload["centroids"]["20"]["source"] == "segmentation"


def test_segment_file_localization_only_writes_centroid_json(tmp_path: Path) -> None:
    image = sitk.GetImageFromArray(np.full((8, 8, 8), 200.0, dtype=np.float32))
    image.SetSpacing((0.5, 0.5, 2.0))
    input_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(image, str(input_path), useCompression=True)

    result = segment_file(
        input_path=input_path,
        output_dir=tmp_path / "out",
        backend=_StubBackend(),
        device="cpu",
        overwrite=True,
        localization_only=True,
    )
    payload = json.loads(result.output_paths.centroids.read_text(encoding="utf-8"))

    assert result.output_paths.centroids.exists()
    assert not result.output_paths.vertebral_level.exists()
    assert payload["centroids"]["20"]["voxel_xyz"] == [2.0, 3.0, 4.0]
    assert payload["metadata"]["localization_only"] is True
