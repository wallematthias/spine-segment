from __future__ import annotations

from typing import Any


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
        if bool(torch_ref.backends.mps.is_available()):
            return "mps"
    except Exception:
        pass

    return "cpu"
