from __future__ import annotations

from types import SimpleNamespace

from spine_segment.device import resolve_device


class _FakeTorch:
    def __init__(self, *, cuda: bool, mps: bool) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: cuda)
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps)
        )


def test_resolve_device_prefers_cuda_then_mps_then_cpu() -> None:
    assert resolve_device("auto", torch_module=_FakeTorch(cuda=True, mps=True)) == "cuda"
    assert resolve_device("auto", torch_module=_FakeTorch(cuda=False, mps=True)) == "mps"
    assert resolve_device("auto", torch_module=_FakeTorch(cuda=False, mps=False)) == "cpu"
