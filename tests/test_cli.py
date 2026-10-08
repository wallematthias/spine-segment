from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import SimpleITK as sitk

from spine_segment import cli
from spine_segment.sequence_solver import SpineSequenceConfig


def test_spine_segment_help_smoke() -> None:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    src_dir = Path(__file__).resolve().parents[1] / "src"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(src_dir)
    )
    result = subprocess.run(
        [sys.executable, "-m", "spine_segment.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0
    assert "spine-segment" in result.stdout
    assert "--output" in result.stdout
    assert "--no-model-download" in result.stdout
    assert "--centroids" in result.stdout
    assert "--sequence-confidence-weight" in result.stdout


@pytest.mark.parametrize("weight", ["0", "1", "nan", "inf", "-0.1"])
def test_sequence_weight_is_validated_before_loading_models(weight, tmp_path, monkeypatch):
    source = tmp_path / "case.nii.gz"
    source.touch()

    def fail_backend_loading(*args, **kwargs):
        raise AssertionError("invalid arguments must not load models")

    monkeypatch.setattr(cli, "create_native_backend", fail_backend_loading)
    with pytest.raises(SystemExit) as exc:
        cli.main([str(source), "--output", str(tmp_path / "out"),
                  "--sequence-confidence-weight", weight])
    assert exc.value.code == 2


def test_sequence_weight_reaches_native_backend(tmp_path, monkeypatch):
    source = tmp_path / "case.nii.gz"
    source.touch()
    backend = SimpleNamespace(sequence_config=SpineSequenceConfig(lambda_weight=.2))
    monkeypatch.setattr(cli, "create_native_backend", lambda *a, **kw: backend)

    def execute(**kwargs):
        assert kwargs["backend"].sequence_config.lambda_weight == .7
        return []

    monkeypatch.setattr(cli, "segment_files", execute)
    assert cli.main([str(source), "--output", str(tmp_path / "out"),
                     "--sequence-confidence-weight", ".7"]) == 0


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--centroids", "centroids.json"],
        ["--level-only", "--localization-only", "--centroids", "centroids.json"],
    ],
)
def test_centroids_reject_invalid_mode_combinations_before_backend_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
) -> None:
    input_path = tmp_path / "case.nii.gz"
    sitk.WriteImage(sitk.Image([2, 2, 2], sitk.sitkFloat32), str(input_path))

    def fail_backend_loading(*args, **kwargs):
        raise AssertionError("backend loading must happen after argument validation")

    monkeypatch.setattr(cli, "create_native_backend", fail_backend_loading)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(input_path), "--output", str(tmp_path / "out"), *extra_args])

    assert exc_info.value.code == 2


def test_centroids_reject_multiple_inputs_before_backend_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_paths = [tmp_path / "one.nii.gz", tmp_path / "two.nii.gz"]
    for input_path in input_paths:
        sitk.WriteImage(sitk.Image([2, 2, 2], sitk.sitkFloat32), str(input_path))

    monkeypatch.setattr(
        cli,
        "create_native_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("backend loading must happen after argument validation")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                *(str(path) for path in input_paths),
                "--output",
                str(tmp_path / "out"),
                "--level-only",
                "--centroids",
                str(tmp_path / "centroids.json"),
            ]
        )

    assert exc_info.value.code == 2
