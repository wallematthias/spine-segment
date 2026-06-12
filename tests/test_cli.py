from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


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
