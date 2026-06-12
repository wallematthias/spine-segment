from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spine_segment.checkpoint_mapping import (
    check_conversion_coverage,
    map_tf_variable_to_torch_key,
    needs_conv_kernel_transpose,
    should_ignore_tf_variable,
)
from spine_segment.pytorch_models import MdatArchitectureSpec, build_model


def _require_tf_and_torch():
    tf = _require_tf()
    torch = _require_torch()
    return tf, torch


def _require_tf():
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for checkpoint conversion.") from exc
    return tf


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for checkpoint conversion.") from exc
    return torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MDAT TensorFlow checkpoints into native PyTorch .pt state_dict files.",
    )
    parser.add_argument("--mdat", help="Source MDAT folder.")
    parser.add_argument("--output", required=True, help="Output native_torch folder.")
    parser.add_argument(
        "--export-npz",
        help="Optional path for exported TensorFlow tensors as a compressed .npz cache.",
    )
    parser.add_argument(
        "--export-npz-only",
        action="store_true",
        help="Only export TensorFlow checkpoint tensors to --export-npz; do not require PyTorch.",
    )
    parser.add_argument(
        "--from-npz",
        help="Convert from a TensorFlow tensor .npz cache instead of reading TF checkpoints.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Write checkpoints even when variables are missing or mismatched. "
            "Use only for debugging conversion mappings."
        ),
    )
    return parser


def _architecture_specs() -> dict[str, MdatArchitectureSpec]:
    return {
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


def _checkpoint_roots(mdat_root: Path) -> dict[str, Path]:
    return {
        "spine_localization": mdat_root / "spine_localization",
        "vertebrae_localization": mdat_root / "vertebrae_localization",
        "vertebrae_segmentation": mdat_root / "vertebrae_segmentation",
    }


def _export_tf_tensors(*, mdat_root: Path, output_path: Path) -> None:
    tf = _require_tf()
    import numpy as np

    tensors: dict[str, object] = {}
    metadata: list[dict[str, str]] = []
    for model_name, checkpoint_root in _checkpoint_roots(mdat_root).items():
        for index, (variable_name, _shape) in enumerate(tf.train.list_variables(str(checkpoint_root))):
            array_key = f"{model_name}:{index}"
            tensors[array_key] = tf.train.load_variable(str(checkpoint_root), variable_name)
            metadata.append(
                {
                    "key": array_key,
                    "model": model_name,
                    "variable": variable_name,
                    "checkpoint_root": str(checkpoint_root),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensors["__metadata__"] = np.array(json.dumps(metadata), dtype=np.str_)
    np.savez_compressed(output_path, **tensors)


def _iter_npz_tensors(npz_path: Path):
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"]))
        for item in metadata:
            yield item["model"], item["checkpoint_root"], item["variable"], archive[item["key"]]


def _iter_tf_tensors(mdat_root: Path):
    tf = _require_tf()
    for model_name, checkpoint_root in _checkpoint_roots(mdat_root).items():
        for variable_name, _shape in tf.train.list_variables(str(checkpoint_root)):
            yield (
                model_name,
                str(checkpoint_root),
                variable_name,
                tf.train.load_variable(str(checkpoint_root), variable_name),
            )


def _convert_tensors(
    *,
    tensor_records,
    output_root: Path,
    source: str,
    allow_partial: bool,
) -> None:
    torch = _require_torch()
    specs = _architecture_specs()
    manifest: dict[str, object] = {
        "status": "converted",
        "source": source,
        "outputs": {},
        "models": [],
    }
    converted_models: list[tuple[str, object, dict[str, object], Path]] = []

    for model_name, spec in specs.items():
        model = build_model(spec)
        target_state = model.state_dict()
        converted_state = {}
        ignored_source_names = []
        unmapped_source_names = []
        shape_mismatches = []

        checkpoint_root = ""
        for record_model_name, record_checkpoint_root, variable_name, value in tensor_records:
            if record_model_name != model_name:
                continue
            checkpoint_root = record_checkpoint_root
            if should_ignore_tf_variable(variable_name):
                ignored_source_names.append(variable_name)
                continue
            torch_key = map_tf_variable_to_torch_key(variable_name)
            if torch_key is None:
                unmapped_source_names.append(variable_name)
                continue
            if needs_conv_kernel_transpose(variable_name):
                value = value.transpose(4, 3, 0, 1, 2)
            tensor = torch.as_tensor(value)
            if torch_key not in target_state:
                unmapped_source_names.append(variable_name)
                continue
            if tuple(target_state[torch_key].shape) != tuple(tensor.shape):
                shape_mismatches.append(
                    {
                        "tf": variable_name,
                        "torch": torch_key,
                        "tf_shape": tuple(tensor.shape),
                        "torch_shape": tuple(target_state[torch_key].shape),
                    }
                )
                continue
            converted_state[torch_key] = tensor

        coverage = check_conversion_coverage(
            converted_keys=set(converted_state),
            target_keys=set(target_state),
            ignored_source_names=ignored_source_names,
            unmapped_source_names=unmapped_source_names,
            shape_mismatches=shape_mismatches,
        )
        model_report = {
            "name": model_name,
            "checkpoint_root": checkpoint_root,
            "converted_keys": coverage.converted_count,
            "ignored_source_count": coverage.ignored_source_count,
            "missing_target_keys": list(coverage.missing_target_keys),
            "unmapped_source_names": list(coverage.unmapped_source_names),
            "shape_mismatches": list(coverage.shape_mismatches),
        }
        manifest["models"].append(model_report)

        has_errors = bool(
            coverage.missing_target_keys
            or coverage.unmapped_source_names
            or coverage.shape_mismatches
        )
        if has_errors and not allow_partial:
            manifest["status"] = "failed"
            continue

        for torch_key, target_tensor in target_state.items():
            if torch_key not in converted_state:
                converted_state[torch_key] = target_tensor

        model.load_state_dict(converted_state)
        output_path = output_root / f"{model_name}.pt"
        converted_models.append((model_name, model, converted_state, output_path))

    failed_models = [
        model
        for model in manifest["models"]
        if model["missing_target_keys"]
        or model["unmapped_source_names"]
        or model["shape_mismatches"]
    ]
    manifest_path = output_root / "conversion_manifest.json"
    if failed_models and not allow_partial:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        details = ", ".join(model["name"] for model in failed_models)
        raise RuntimeError(
            "MDAT checkpoint conversion was incomplete for: "
            f"{details}. See {manifest_path} for missing keys and shape mismatches."
        )

    for model_name, model, _converted_state, output_path in converted_models:
        torch.save(model.state_dict(), output_path)
        manifest["outputs"][model_name] = str(output_path)

    (output_root / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Converted MDAT checkpoints into {output_root}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.export_npz_only and not args.export_npz:
        parser.error("--export-npz-only requires --export-npz")
    if args.from_npz and args.mdat:
        parser.error("Use either --from-npz or --mdat, not both")
    if not args.from_npz and not args.mdat:
        parser.error("Either --mdat or --from-npz is required")

    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.export_npz_only:
        assert args.mdat is not None
        _export_tf_tensors(
            mdat_root=Path(args.mdat).expanduser().resolve(),
            output_path=Path(args.export_npz).expanduser().resolve(),
        )
        print(f"Exported TensorFlow tensors to {Path(args.export_npz).expanduser().resolve()}")
        return 0

    if args.from_npz:
        npz_path = Path(args.from_npz).expanduser().resolve()
        tensor_records = list(_iter_npz_tensors(npz_path))
        source = str(npz_path)
    else:
        assert args.mdat is not None
        mdat_root = Path(args.mdat).expanduser().resolve()
        if args.export_npz:
            npz_path = Path(args.export_npz).expanduser().resolve()
            _export_tf_tensors(mdat_root=mdat_root, output_path=npz_path)
            tensor_records = list(_iter_npz_tensors(npz_path))
            source = str(npz_path)
        else:
            tf, torch = _require_tf_and_torch()
            del tf, torch
            tensor_records = list(_iter_tf_tensors(mdat_root))
            source = str(mdat_root)

    _convert_tensors(
        tensor_records=tensor_records,
        output_root=output_root,
        source=source,
        allow_partial=args.allow_partial,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
