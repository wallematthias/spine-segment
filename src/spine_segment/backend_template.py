from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from spine_segment.backend import SegmentationResult, SpineSegmentBackend


class ExampleBackend(SpineSegmentBackend):
    """
    Replace this class with the real PyTorch runtime.

    The backend must return:
    - vertebral_level: integer vertebral labelmap in scan space
    - process_body: integer process/body labelmap in scan space
    - cort_trab: optional integer cort/trab labelmap in scan space
    """

    def segment(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
    ) -> SegmentationResult:
        raise NotImplementedError(
            "Replace spine_segment.backend_template.ExampleBackend with the "
            "PyTorch vertebral segmentation runtime."
        )


def create_backend() -> ExampleBackend:
    return ExampleBackend()
