from __future__ import annotations

from pathlib import Path

import pytest

from spine_segment.model_bundle import resolve_native_torch_bundle, validate_native_torch_bundle


def test_resolve_native_torch_bundle_paths(tmp_path: Path) -> None:
    bundle = resolve_native_torch_bundle(tmp_path)
    assert bundle.spine_localization == tmp_path / "weights" / "spine-locator.pt"
    assert bundle.vertebrae_localization == tmp_path / "weights" / "vertebra-locator.pt"
    assert bundle.vertebrae_segmentation == tmp_path / "weights" / "vertebra-segmenter.pt"
    assert bundle.process_body_checkpoint == tmp_path / "weights" / "process-body-segmenter.pt"


def test_resolve_native_torch_bundle_supports_legacy_paths(tmp_path: Path) -> None:
    native = tmp_path / "native_torch"
    native.mkdir(parents=True)
    (native / "spine_localization.pt").touch()
    legacy_process = (
        tmp_path
        / "nnunet"
        / "Dataset005_VerseRelabel"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "fold_0"
    )
    legacy_process.mkdir(parents=True)
    (legacy_process / "checkpoint_final.pth").touch()

    bundle = resolve_native_torch_bundle(tmp_path)

    assert bundle.spine_localization == native / "spine_localization.pt"
    assert bundle.process_body_checkpoint == legacy_process / "checkpoint_final.pth"


def test_validate_native_torch_bundle_requires_process_body_checkpoint(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("spine-locator.pt", "vertebra-locator.pt", "vertebra-segmenter.pt"):
        (weights / name).touch()

    with pytest.raises(FileNotFoundError, match="process-body-segmenter.pt"):
        validate_native_torch_bundle(resolve_native_torch_bundle(tmp_path))
