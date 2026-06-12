from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


MDAT_FILES = [
    "spine/model.yaml",
    "spine_localization.index",
    "spine_localization.data-00000-of-00002",
    "spine_localization.data-00001-of-00002",
    "vertebrae_localization.index",
    "vertebrae_localization.data-00000-of-00002",
    "vertebrae_localization.data-00001-of-00002",
    "vertebrae_segmentation.index",
    "vertebrae_segmentation.data-00000-of-00002",
    "vertebrae_segmentation.data-00001-of-00002",
]

WEIGHT_FILES = {
    "spine-locator.pt": ("native_torch", "spine_localization.pt"),
    "vertebra-locator.pt": ("native_torch", "vertebrae_localization.pt"),
    "vertebra-segmenter.pt": ("native_torch", "vertebrae_segmentation.pt"),
    "process-body-segmenter.pt": (
        "verse_relabel",
        "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
    ),
}

OPTIONAL_NATIVE_TORCH_FILES = ["conversion_manifest.json"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _copy_required_files(source_root: Path, target_root: Path, relative_paths: list[str]) -> list[dict[str, object]]:
    copied: list[dict[str, object]] = []
    for relative in relative_paths:
        src = source_root / relative
        if not src.exists():
            raise FileNotFoundError(f"Missing required model asset: {src}")
        dst = target_root / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(
            {
                "path": relative,
                "bytes": dst.stat().st_size,
                "sha256": _sha256(dst),
            }
        )
    return copied


def _copy_weight_file(source: Path, target_root: Path, output_name: str) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(f"Missing required model asset: {source}")
    dst = target_root / output_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst)
    return {
        "path": str(dst.relative_to(target_root.parent)),
        "bytes": dst.stat().st_size,
        "sha256": _sha256(dst),
    }


def _copy_optional_files(source_root: Path, target_root: Path, relative_paths: list[str]) -> list[dict[str, object]]:
    copied: list[dict[str, object]] = []
    for relative in relative_paths:
        src = source_root / relative
        if not src.exists():
            continue
        dst = target_root / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(
            {
                "path": str(dst.relative_to(target_root.parent)),
                "bytes": dst.stat().st_size,
                "sha256": _sha256(dst),
            }
        )
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage a clean inference-only model bundle for spine-segment.",
    )
    parser.add_argument("--mdat", help="Source mdat model folder.")
    parser.add_argument(
        "--native-torch",
        help="Source native_torch folder containing converted MDAT .pt files.",
    )
    parser.add_argument(
        "--omit-mdat-source",
        action="store_true",
        help="Do not include the original TensorFlow MDAT checkpoint files.",
    )
    parser.add_argument(
        "--verse-relabel",
        required=True,
        help="Source Dataset005_VerseRelabel folder.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination folder for the staged model bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output folder.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.omit_mdat_source and not args.mdat:
        parser.error("--mdat is required unless --omit-mdat-source is used")
    if args.omit_mdat_source and not args.native_torch:
        parser.error("--native-torch is required when --omit-mdat-source is used")

    mdat_root = Path(args.mdat).expanduser().resolve() if args.mdat else None
    native_torch_root = Path(args.native_torch).expanduser().resolve() if args.native_torch else None
    relabel_root = Path(args.verse_relabel).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists. Use --overwrite to replace it: {output_root}"
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    mdat_out = output_root / "mdat"
    weights_out = output_root / "weights"

    mdat_files = []
    if not args.omit_mdat_source:
        assert mdat_root is not None
        mdat_files = _copy_required_files(mdat_root, mdat_out, MDAT_FILES)
    weight_files = []
    if native_torch_root is not None:
        for output_name, (source_kind, relative) in WEIGHT_FILES.items():
            source_root = native_torch_root if source_kind == "native_torch" else relabel_root
            weight_files.append(_copy_weight_file(source_root / relative, weights_out, output_name))
        weight_files.extend(_copy_optional_files(native_torch_root, weights_out, OPTIONAL_NATIVE_TORCH_FILES))

    contents = []
    if mdat_files:
        contents.append(
            {
                "id": "spine_mdat_source",
                "source_root": str(mdat_root),
                "staged_root": "mdat",
                "files": mdat_files,
            }
        )
    if weight_files:
        contents.append(
            {
                "id": "spine_segment_weights",
                "source_root": {
                    "native_torch": str(native_torch_root),
                    "verse_relabel": str(relabel_root),
                },
                "staged_root": "weights",
                "files": weight_files,
            }
        )

    manifest = {
        "bundle_name": "spine-segment-model-bundle",
        "version": 1,
        "contents": contents,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Staged model bundle at {output_root}")
    print(f"MDAT files: {len(mdat_files)}")
    print(f"Weight files: {len(weight_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
