from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from spine_segment.compartments import (
    CortTrabConfig,
    _expanded_mask_bbox,
    derive_cort_trab_labels,
)
from spine_segment.labels import BODY_LABEL, CORTICAL_LABEL, PROCESS_LABEL, TRABECULAR_LABEL


def test_derive_cort_trab_labels_marks_shell_and_core() -> None:
    image_arr = np.full((7, 7, 7), 120.0, dtype=np.float32)
    image_arr[1:6, 1:6, 1:6] = 150.0
    image_arr[2:5, 2:5, 2:5] = 80.0

    process_body_arr = np.zeros((7, 7, 7), dtype=np.uint8)
    process_body_arr[1:6, 1:6, 1:6] = BODY_LABEL

    image = sitk.GetImageFromArray(image_arr)
    process_body = sitk.GetImageFromArray(process_body_arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    process_body.CopyInformation(image)

    out = derive_cort_trab_labels(
        image=image,
        process_body=process_body,
        config=CortTrabConfig(
            cortical_threshold_hu=140.0,
            cortical_max_thickness_mm=2.0,
        ),
    )
    out_arr = sitk.GetArrayFromImage(out)

    assert out_arr[1, 1, 1] == CORTICAL_LABEL
    assert out_arr[3, 3, 3] == TRABECULAR_LABEL


def test_derive_cort_trab_labels_uses_full_vertebral_level_mask() -> None:
    image_arr = np.full((9, 9, 9), 100.0, dtype=np.float32)
    process_body_arr = np.zeros((9, 9, 9), dtype=np.uint8)
    process_body_arr[2:7, 2:7, 2:7] = BODY_LABEL
    process_body_arr[1, 3:6, 3:6] = PROCESS_LABEL

    vertebral_arr = np.zeros((9, 9, 9), dtype=np.uint8)
    vertebral_arr[2:7, 2:7, 2:7] = 20
    vertebral_arr[1, 3:6, 3:6] = 20

    image = sitk.GetImageFromArray(image_arr)
    process_body = sitk.GetImageFromArray(process_body_arr)
    vertebral_level = sitk.GetImageFromArray(vertebral_arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    process_body.CopyInformation(image)
    vertebral_level.CopyInformation(image)

    out = derive_cort_trab_labels(
        image=image,
        process_body=process_body,
        vertebral_level=vertebral_level,
        config=CortTrabConfig(cortical_threshold_hu=500.0),
    )
    out_arr = sitk.GetArrayFromImage(out)

    assert out_arr[1, 4, 4] == CORTICAL_LABEL


def test_derive_cort_trab_labels_covers_every_vertebral_level() -> None:
    image_arr = np.full((15, 9, 9), 100.0, dtype=np.float32)
    process_body_arr = np.zeros((15, 9, 9), dtype=np.uint8)
    process_body_arr[1:6, 2:7, 2:7] = BODY_LABEL
    process_body_arr[9:14, 2:7, 2:7] = BODY_LABEL

    vertebral_arr = np.zeros((15, 9, 9), dtype=np.uint8)
    vertebral_arr[1:6, 2:7, 2:7] = 20
    vertebral_arr[9:14, 2:7, 2:7] = 21

    image = sitk.GetImageFromArray(image_arr)
    process_body = sitk.GetImageFromArray(process_body_arr)
    vertebral_level = sitk.GetImageFromArray(vertebral_arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    process_body.CopyInformation(image)
    vertebral_level.CopyInformation(image)

    out = derive_cort_trab_labels(
        image=image,
        process_body=process_body,
        vertebral_level=vertebral_level,
        config=CortTrabConfig(
            cortical_threshold_hu=500.0,
            cortical_max_thickness_mm=2.0,
        ),
    )
    out_arr = sitk.GetArrayFromImage(out)

    for level in (20, 21):
        selected = out_arr[vertebral_arr == level]
        assert set(np.unique(selected)) <= {CORTICAL_LABEL, TRABECULAR_LABEL}
        assert np.count_nonzero(selected == TRABECULAR_LABEL) > 0


def test_derive_cort_trab_labels_uses_connected_high_density_annulus() -> None:
    image_arr = np.full((17, 17, 17), 100.0, dtype=np.float32)
    process_body_arr = np.zeros((17, 17, 17), dtype=np.uint8)
    process_body_arr[3:14, 3:14, 3:14] = BODY_LABEL
    image_arr[4:7, 8, 8] = 600.0
    image_arr[8, 8, 8] = 700.0

    image = sitk.GetImageFromArray(image_arr)
    process_body = sitk.GetImageFromArray(process_body_arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    process_body.CopyInformation(image)

    out = derive_cort_trab_labels(
        image=image,
        process_body=process_body,
        config=CortTrabConfig(
            cortical_threshold_hu=500.0,
            isotropic_spacing_mm=1.0,
            cortical_max_thickness_mm=4.0,
        ),
    )
    out_arr = sitk.GetArrayFromImage(out)

    assert out_arr[3, 8, 8] == CORTICAL_LABEL
    assert out_arr[6, 8, 8] == CORTICAL_LABEL
    assert out_arr[8, 8, 8] == TRABECULAR_LABEL


def test_expanded_mask_bbox_adds_physical_margin() -> None:
    arr = np.zeros((10, 11, 12), dtype=np.uint8)
    arr[3:5, 4:6, 5:7] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing((0.5, 1.0, 2.0))

    bbox = _expanded_mask_bbox(mask, margin_mm=2.0)

    assert bbox == (1, 2, 2, 10, 6, 4)
