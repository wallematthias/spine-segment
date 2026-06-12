from __future__ import annotations

from pathlib import Path

from spine_segment.outputs import build_output_paths


def test_build_output_paths_handles_nii_gz_suffix() -> None:
    outputs = build_output_paths("case001.nii.gz", Path("/tmp/out"))
    assert outputs.vertebral_level == Path("/tmp/out/case001_vertebral-level.nii.gz")
    assert outputs.process_body == Path("/tmp/out/case001_process-body.nii.gz")
    assert outputs.cort_trab == Path("/tmp/out/case001_cort-trab.nii.gz")
    assert outputs.centroids == Path("/tmp/out/case001_centroids.json")
