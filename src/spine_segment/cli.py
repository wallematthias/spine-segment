from __future__ import annotations

import argparse
import os
from pathlib import Path

from spine_segment.api import segment_files
from spine_segment.backend import SpineSegmentBackendError, load_backend
from spine_segment.compartments import CortTrabConfig
from spine_segment.io import expand_input_paths
from spine_segment.native_torch_backend import create_backend as create_native_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spine-segment",
        description="Standalone vertebral segmentation CLI.",
    )
    parser.add_argument("inputs", nargs="+", help="Input CT volumes (.nii or .nii.gz).")
    parser.add_argument(
        "--output",
        required=True,
        help="Directory that will receive outputs. With --level-only this is only *_vertebral-level.nii.gz.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="Runtime device preference.",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("SPINE_SEGMENT_BACKEND", ""),
        help="Backend factory in the form module_path:factory. "
        "Defaults to the native PyTorch backend.",
    )
    parser.add_argument(
        "--model-bundle",
        default=os.environ.get("SPINE_SEGMENT_MODEL_BUNDLE", ""),
        help="Path to a spine-segment model bundle. Defaults to "
        "SPINE_SEGMENT_MODEL_BUNDLE, ./model-bundle, ./build/model-bundle-pytorch, "
        "or ./build/model-bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--level-only",
        action="store_true",
        help="Write only *_vertebral-level.nii.gz and skip process/body plus cort/trab stages.",
    )
    parser.add_argument(
        "--localization-only",
        action="store_true",
        help="Write only *_centroids.json with detected vertebral centroids in original scan voxel coordinates.",
    )
    parser.add_argument(
        "--vertebra-batch-size",
        type=int,
        default=1,
        help="Number of MDAT vertebra segmentation patches to run per PyTorch batch.",
    )
    parser.add_argument(
        "--process-body-batch-size",
        type=int,
        default=4,
        help="Number of process/body relabel crops to run per PyTorch batch.",
    )
    parser.add_argument(
        "--cortical-threshold-hu",
        type=float,
        default=500.0,
        help="HU threshold used for connected cortical assignment within the outer shell.",
    )
    parser.add_argument(
        "--cort-trab-isotropic-spacing-mm",
        type=float,
        default=1.0,
        help="Isotropic spacing used internally for cort/trab shell generation.",
    )
    parser.add_argument(
        "--cortical-max-thickness-mm",
        type=float,
        default=6.0,
        help="Maximum distance from the vertebral surface considered for threshold-based cortical assignment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = expand_input_paths(list(args.inputs))
    if not inputs:
        parser.error("No input images were resolved.")

    missing = [path for path in inputs if not path.exists()]
    if missing:
        parser.error(f"Input does not exist: {missing[0]}")

    if args.backend:
        backend = load_backend(args.backend)
    else:
        backend = create_native_backend(args.model_bundle or "./build/model-bundle")
    if hasattr(backend, "vertebra_segmentation_batch_size"):
        backend.vertebra_segmentation_batch_size = max(1, int(args.vertebra_batch_size))
    if hasattr(backend, "process_body_batch_size"):
        backend.process_body_batch_size = max(1, int(args.process_body_batch_size))
    cort_trab_config = CortTrabConfig(
        cortical_threshold_hu=float(args.cortical_threshold_hu),
        isotropic_spacing_mm=float(args.cort_trab_isotropic_spacing_mm),
        cortical_max_thickness_mm=float(args.cortical_max_thickness_mm),
    )
    if args.localization_only and args.level_only:
        parser.error("--localization-only and --level-only are mutually exclusive.")

    try:
        results = segment_files(
            input_paths=inputs,
            output_dir=Path(args.output),
            backend=backend,
            device=args.device,
            overwrite=bool(args.overwrite),
            cort_trab_config=cort_trab_config,
            level_only=bool(args.level_only),
            localization_only=bool(args.localization_only),
        )
    except SpineSegmentBackendError as exc:
        print(f"[spine-segment] backend error: {exc}")
        return 2
    except FileExistsError as exc:
        print(f"[spine-segment] output exists: {exc}")
        return 2

    for result in results:
        fields = [
            f"input={result.input_path}",
            f"device={result.device}",
        ]
        if args.localization_only:
            fields.append(f"centroids={result.output_paths.centroids}")
        else:
            fields.append(f"vertebral_level={result.output_paths.vertebral_level}")
            if not args.level_only:
                fields.append(f"process_body={result.output_paths.process_body}")
                fields.append(f"cort_trab={result.output_paths.cort_trab}")
        print("[spine-segment] " + " ".join(fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
