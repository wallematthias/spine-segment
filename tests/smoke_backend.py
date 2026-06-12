from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from spine_segment.backend import SegmentationResult
from spine_segment.labels import BODY_LABEL, PROCESS_LABEL


class SmokeBackend:
    def segment(self, *, image: sitk.Image, source_path: Path, device: str) -> SegmentationResult:
        shape = sitk.GetArrayFromImage(image).shape

        vertebral_arr = np.zeros(shape, dtype=np.uint8)
        vertebral_arr[1:-1, 1:-1, 1:-1] = 20
        vertebral = sitk.GetImageFromArray(vertebral_arr)
        vertebral.CopyInformation(image)

        process_body_arr = np.zeros(shape, dtype=np.uint8)
        process_body_arr[1:-1, 1:-1, 1:-1] = BODY_LABEL
        process_body_arr[1, 1:-1, 1:-1] = PROCESS_LABEL
        process_body = sitk.GetImageFromArray(process_body_arr)
        process_body.CopyInformation(image)

        return SegmentationResult(
            vertebral_level=vertebral,
            process_body=process_body,
            metadata={"backend": "smoke", "device_seen": device},
        )


def create_backend() -> SmokeBackend:
    return SmokeBackend()
