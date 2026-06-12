from __future__ import annotations

from glob import glob
from pathlib import Path

import SimpleITK as sitk


def expand_input_paths(inputs: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw in inputs:
        matches = sorted(Path(path) for path in glob(raw))
        if matches:
            expanded.extend(matches)
            continue
        expanded.append(Path(raw))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def read_image(path: str | Path, pixel_type: int = sitk.sitkFloat32) -> sitk.Image:
    return sitk.ReadImage(str(path), pixel_type)


def write_image(
    image: sitk.Image,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output without --overwrite: {output_path}"
        )
    sitk.WriteImage(image, str(output_path), useCompression=True)
    return output_path
