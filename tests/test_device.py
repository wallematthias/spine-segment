from __future__ import annotations

from types import SimpleNamespace

from spine_segment.device import resolve_device


class _FakeConv3d:
    def __init__(self, supported: bool) -> None:
        self._supported = bool(supported)

    def to(self, _device: str):
        return self

    def __call__(self, _input):
        if not self._supported:
            raise RuntimeError("Conv3D is not supported on MPS")
        return object()


class _FakeTorch:
    def __init__(self, *, cuda: bool, mps: bool, mps_conv3d: bool = True) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: cuda)
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps)
        )
        self.nn = SimpleNamespace(Conv3d=lambda *_args, **_kwargs: _FakeConv3d(mps_conv3d))
        self.zeros = lambda *_args, **_kwargs: object()


def test_resolve_device_prefers_cuda_then_mps_then_cpu() -> None:
    assert resolve_device("auto", torch_module=_FakeTorch(cuda=True, mps=True)) == "cuda"
    assert resolve_device("auto", torch_module=_FakeTorch(cuda=False, mps=True)) == "mps"
    assert resolve_device("auto", torch_module=_FakeTorch(cuda=False, mps=False)) == "cpu"


def test_resolve_device_skips_mps_when_conv3d_is_unavailable() -> None:
    torch = _FakeTorch(cuda=False, mps=True, mps_conv3d=False)

    assert resolve_device("auto", torch_module=torch) == "cpu"
