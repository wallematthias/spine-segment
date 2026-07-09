# Model Assets

The standalone package expects one flattened model bundle for runtime:

1. `weights/`
   Purpose: all PyTorch inference weights used by the CLI.

The original `mdat` and `Dataset005_VerseRelabel` source folders may contain
extra training or validation artifacts. For publishing, we only want the minimal
inference subset.

## Lean Published Layout

```text
model-bundle/
  manifest.json
  weights/
    spine-locator.pt
    vertebra-locator.pt
    vertebra-segmenter.pt
    process-body-segmenter.pt
```

`weights/` is the runtime layout for the pure-PyTorch backend. The original
TensorFlow `mdat/` checkpoint files and the original nnU-Net training folder are
useful for conversion and auditing, but should be omitted from the normal
distribution bundle once the `.pt` files have been generated/staged.

## GitHub Release Asset

PyPI wheels do not include the model bundle. The default runtime downloader
expects this GitHub Release asset:

```text
tag: v0.1.0
asset: spine-segment-model-bundle-v0.1.0.zip
```

The zip may contain `manifest.json` and `weights/` either at the archive root or
inside a single top-level folder. At install/runtime, `spine-segment` verifies
the extracted checkpoint SHA256 hashes before using the cached bundle.

## Excluded From Release Bundle

These are useful for development, but should not be shipped in the inference
bundle by default:

- `.DS_Store`
- `.ipynb_checkpoints`
- `validation/`
- training logs
- `progress.png`
- `network_architecture.pdf`
- `debug.json`
- duplicate training checkpoints
- original nested training framework folders

## Staging Script

Use:

```bash
python3 scripts/stage_model_bundle.py \
  --native-torch /path/to/native_torch \
  --omit-mdat-source \
  --verse-relabel /path/to/Dataset005_VerseRelabel \
  --output /path/to/model-bundle
```

This creates a clean inference-only bundle and writes a `manifest.json`.

## MDAT Conversion

The package also has a checkpoint-conversion entrypoint:

```bash
python3 scripts/convert_mdat_tf_checkpoints.py \
  --mdat /path/to/mdat \
  --output /path/to/converted/native_torch
```

If TensorFlow and PyTorch are installed in separate environments, use the
two-stage flow:

```bash
python3 scripts/convert_mdat_tf_checkpoints.py \
  --mdat /path/to/mdat \
  --output /path/to/converted/native_torch \
  --export-npz /path/to/mdat_tf_tensors.npz \
  --export-npz-only

python3 scripts/convert_mdat_tf_checkpoints.py \
  --from-npz /path/to/mdat_tf_tensors.npz \
  --output /path/to/converted/native_torch
```

The converter maps TensorFlow checkpoint tensors into PyTorch `state_dict`
files and writes `conversion_manifest.json`. Conversion is strict by default:
the script stops before writing `.pt` files if any model weight is missing,
unmapped, or shape-mismatched. Use `--allow-partial` only while debugging a
mapping issue.
