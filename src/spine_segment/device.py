from __future__ import annotations

from typing import Any


def _mps_supports_conv3d(torch_ref: Any) -> bool:
    try:
        layer = torch_ref.nn.Conv3d(1, 1, kernel_size=1).to("mps")
        sample = torch_ref.zeros((1, 1, 3, 3, 3), device="mps")
        layer(sample)
        return True
    except Exception:
        return False


def resolve_device(
    requested: str = "auto",
    *,
    torch_module: Any | None = None,
) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized in {"cpu", "cuda", "mps"}:
        return normalized
    if normalized != "auto":
        raise ValueError(
            f"Unsupported device '{requested}'. Expected auto, cuda, mps, or cpu."
        )

    torch_ref = torch_module
    if torch_ref is None:
        try:
            import torch as torch_ref  # type: ignore[no-redef]
        except Exception:
            return "cpu"

    try:
        if bool(torch_ref.cuda.is_available()):
            return "cuda"
    except Exception:
        pass

    try:
        if bool(torch_ref.backends.mps.is_available()) and _mps_supports_conv3d(torch_ref):
            return "mps"
    except Exception:
        pass

    return "cpu"
