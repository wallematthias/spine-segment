# spine-segment

`spine-segment` is a standalone CLI-first package for vertebral CT segmentation.

The intended output contract for each input volume is:

- `*_vertebral-level.nii.gz`
- `*_process-body.nii.gz`
- `*_cort-trab.nii.gz`
- `*_centroids.json` when using `--localization-only`

The package is designed to stay lightweight:

- `torch`
- `numpy`
- `SimpleITK`

For now, the package includes:

- a publishable package layout
- a stable Python API
- a CLI for one or more NIfTI inputs
- automatic device selection (`cuda`, `mps`, then `cpu`)
- SITK-based read/write helpers
- built-in cortical/trabecular derivation from the vertebral labelmap
- a pure-PyTorch MDAT vertebral segmentation backend
- a lightweight native PyTorch process/body relabel backend
- a staging script for building a clean inference-only model bundle
- native PyTorch MDAT architecture definitions and checkpoint conversion tools

## Install

Clone the repository with Git LFS enabled so the model weights are downloaded:

```bash
git lfs install
git clone <repo-url> spine-segment
cd spine-segment
git lfs pull
```

Then install from the package directory:

```bash
python3 -m pip install -e .
```

This installs the Python package and its minimal runtime dependencies:
`torch`, `numpy`, and `SimpleITK`.

For CUDA systems, install the PyTorch build that matches the machine first,
then run `pip install -e .`. For example, follow the selector at
<https://pytorch.org/get-started/locally/> for the target CUDA version. On
macOS, the default PyPI PyTorch wheel supports CPU and Apple MPS where
available. If no GPU backend is available, `spine-segment --device auto` falls
back to CPU.

The default model bundle lives at `./build/model-bundle-pytorch` and its
`.pt` weights are stored with Git LFS. If LFS did not fetch correctly, the
files in `build/model-bundle-pytorch/weights/` will be tiny pointer files
instead of large PyTorch checkpoint files; run `git lfs pull` to fetch them.

You can also point to another model bundle explicitly:

```bash
export SPINE_SEGMENT_MODEL_BUNDLE=/path/to/model-bundle
```

## CLI

```bash
spine-segment image.nii.gz --output /path/to/output
spine-segment *.nii.gz --output /path/to/output
```

Useful modes:

```bash
spine-segment image.nii.gz --output /path/to/output --level-only
spine-segment image.nii.gz --output /path/to/output --localization-only
```

`--level-only` writes only `*_vertebral-level.nii.gz`.
`--localization-only` writes only `*_centroids.json` and skips vertebra
segmentation.

Optional backend wiring is still available through `--backend
module_path:factory` or `SPINE_SEGMENT_BACKEND`.

## Backend Contract

Backends return:

- a vertebral-level labelmap
- a process/body labelmap
- optionally a cort/trab labelmap

If `cort_trab` is omitted, the package derives it locally from the CT image and
the vertebral-level segmentation.

## Model Bundle Staging

The repository includes a helper to stage publishable inference assets from raw
source folders:

```bash
python3 scripts/stage_model_bundle.py \
  --mdat /path/to/mdat \
  --verse-relabel /path/to/Dataset005_VerseRelabel \
  --native-torch /path/to/converted/checkpoints \
  --output ./build/model-bundle
```

This keeps only the inference-relevant files and excludes validation outputs,
plots, logs, notebook checkpoints, and duplicate training artifacts.

## Native PyTorch Backend

The package now includes:

- [pytorch_models.py](src/spine_segment/pytorch_models.py)
- [native_torch_backend.py](src/spine_segment/native_torch_backend.py)
- [scripts/convert_mdat_tf_checkpoints.py](scripts/convert_mdat_tf_checkpoints.py)
- pure-Python runtime helpers in `preprocess.py`, `postprocess.py`, `landmarks.py`, and `tiling.py`
- lightweight sequence solving and bundled vertebral graph resources without a runtime `networkx` dependency

The TensorFlow-to-PyTorch checkpoint conversion is implemented in strict mode:
all three MDAT checkpoints must map without missing keys, unexpected source
tensors, or shape mismatches before `.pt` files are written.
