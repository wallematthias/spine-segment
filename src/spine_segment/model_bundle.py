from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NativeTorchBundlePaths:
    root: Path
    spine_localization: Path
    vertebrae_localization: Path
    vertebrae_segmentation: Path
    process_body_checkpoint: Path


def resolve_native_torch_bundle(root: str | Path) -> NativeTorchBundlePaths:
    bundle_root = Path(root).expanduser().resolve()
    weights_root = bundle_root / "weights"
    legacy_native_root = bundle_root / "native_torch"
    legacy_process_ckpt = (
        bundle_root
        / "nnunet"
        / "Dataset005_VerseRelabel"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "fold_0"
        / "checkpoint_final.pth"
    )
    return NativeTorchBundlePaths(
        root=bundle_root,
        spine_localization=_preferred_existing(
            weights_root / "spine-locator.pt",
            legacy_native_root / "spine_localization.pt",
        ),
        vertebrae_localization=_preferred_existing(
            weights_root / "vertebra-locator.pt",
            legacy_native_root / "vertebrae_localization.pt",
        ),
        vertebrae_segmentation=_preferred_existing(
            weights_root / "vertebra-segmenter.pt",
            legacy_native_root / "vertebrae_segmentation.pt",
        ),
        process_body_checkpoint=_preferred_existing(
            weights_root / "process-body-segmenter.pt",
            legacy_process_ckpt,
        ),
    )


def _preferred_existing(preferred: Path, legacy: Path) -> Path:
    return preferred if preferred.exists() or not legacy.exists() else legacy


def validate_native_torch_bundle(paths: NativeTorchBundlePaths) -> None:
    missing = [
        path
        for path in (
            paths.spine_localization,
            paths.vertebrae_localization,
            paths.vertebrae_segmentation,
            paths.process_body_checkpoint,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing converted native PyTorch checkpoint(s): "
            + ", ".join(str(path) for path in missing)
        )
