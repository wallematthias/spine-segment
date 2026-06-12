from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import SimpleITK as sitk


class SpineSegmentBackendError(RuntimeError):
    """Raised when the standalone spine backend cannot run."""


@dataclass(slots=True)
class SegmentationResult:
    vertebral_level: sitk.Image
    process_body: sitk.Image | None = None
    cort_trab: sitk.Image | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class LocalizationResult:
    centroids: dict[str, dict[str, Any]]
    metadata: dict[str, Any] | None = None


@runtime_checkable
class SpineSegmentBackend(Protocol):
    def segment(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
        level_only: bool = False,
    ) -> SegmentationResult: ...

    def localize(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
    ) -> LocalizationResult: ...


class UnavailableBackend:
    def segment(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
        level_only: bool = False,
    ) -> SegmentationResult:
        raise SpineSegmentBackendError(
            "No standalone spine backend is wired yet. "
            "Provide --backend module_path:factory or set SPINE_SEGMENT_BACKEND "
            "to the PyTorch runtime entrypoint for the vertebral-level and "
            "process/body models."
        )

    def localize(
        self,
        *,
        image: sitk.Image,
        source_path: Path,
        device: str,
    ) -> LocalizationResult:
        raise SpineSegmentBackendError(
            "No standalone spine backend is wired yet. "
            "Provide --backend module_path:factory or set SPINE_SEGMENT_BACKEND."
        )


def load_backend(spec: str | None) -> SpineSegmentBackend:
    if spec is None or not str(spec).strip():
        return UnavailableBackend()

    module_name, separator, attr_name = str(spec).partition(":")
    module = import_module(module_name)
    target_name = attr_name if separator else "create_backend"
    target = getattr(module, target_name)
    instance = target() if callable(target) else target

    if not isinstance(instance, SpineSegmentBackend) and not hasattr(instance, "segment"):
        raise SpineSegmentBackendError(
            f"Backend '{spec}' does not expose a segment(...) method."
        )
    return instance
