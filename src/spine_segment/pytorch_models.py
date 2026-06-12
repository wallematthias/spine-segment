from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - import guard for non-runtime envs
        raise RuntimeError(
            "PyTorch is required for spine_segment.pytorch_models."
        ) from exc
    return torch, nn, F


def _activation_factory(name: str):
    torch, nn, _ = _require_torch()
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU(inplace=False)
    if normalized == "lrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    if normalized == "selu":
        return nn.SELU(inplace=False)
    raise ValueError(f"Unsupported activation '{name}'.")


class _ConcatChannels:
    def __call__(self, tensors):
        torch, _, _ = _require_torch()
        return torch.cat(tensors, dim=1)


def _repeat_pad_1d(x, axis: int, shift: int):
    torch, _, _ = _require_torch()
    if shift == 0:
        return x
    size = x.shape[axis]
    index = torch.arange(size, device=x.device)
    shifted = (index - shift).clamp(0, size - 1)
    return torch.index_select(x, axis, shifted)


def _f_linear(x: float) -> float:
    return 1.0 - abs(x) if abs(x) <= 1.0 else 0.0


def _f_cubic(x: float) -> float:
    a = -0.5
    ax = abs(x)
    if ax <= 1.0:
        return (a + 2.0) * ax**3 - (a + 3.0) * ax**2 + 1.0
    if ax < 2.0:
        return a * ax**3 - 5.0 * a * ax**2 + 8.0 * a * ax - 4.0 * a
    return 0.0


def _upsample_axis_with_kernel(x, factor: int, axis: int, kernel_fn: Callable[[float], float], support: int):
    torch, _, _ = _require_torch()
    pieces = []
    for i in range(factor):
        position = (1 - factor + i * 2) / (factor * 2)
        accum = None
        for offset in range(-support, support + 1):
            weight = kernel_fn(position + offset)
            if weight == 0:
                continue
            shifted = _repeat_pad_1d(x, axis, offset)
            current = shifted * weight
            accum = current if accum is None else accum + current
        pieces.append(accum if accum is not None else torch.zeros_like(x))
    stacked = torch.stack(pieces, dim=axis + 1)
    shape = list(x.shape)
    shape[axis] *= factor
    return stacked.reshape(shape)


def upsample_linear_3d(x, scale_factor: tuple[int, int, int]):
    out = x
    for factor, axis in zip(scale_factor, (2, 3, 4)):
        out = _upsample_axis_with_kernel(out, factor, axis, _f_linear, 1)
    return out


def upsample_cubic_3d(x, scale_factor: tuple[int, int, int]):
    out = x
    for factor, axis in zip(scale_factor, (2, 3, 4)):
        out = _upsample_axis_with_kernel(out, factor, axis, _f_cubic, 2)
    return out


class ConvBlock3d:
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        repeats: int,
        activation: str,
        dropout_ratio: float,
    ) -> None:
        _, nn, _ = _require_torch()
        layers = []
        current_in = in_channels
        for _index in range(repeats):
            layers.append(nn.Conv3d(current_in, out_channels, kernel_size=3, padding=1))
            layers.append(_activation_factory(activation))
            if dropout_ratio > 0:
                if activation == "selu":
                    layers.append(nn.AlphaDropout(dropout_ratio))
                else:
                    layers.append(nn.Dropout(dropout_ratio))
            current_in = out_channels
        self.module = nn.Sequential(*layers)


class UnetAvgLinear3D:
    def __init__(
        self,
        *,
        in_channels: int,
        num_filters_base: int,
        num_levels: int,
        repeats: int = 2,
        activation: str = "relu",
        dropout_ratio: float = 0.0,
    ) -> None:
        _, nn, _ = _require_torch()
        self.num_levels = int(num_levels)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.contracting = nn.ModuleList()
        self.expanding = nn.ModuleList()

        channels_per_level = [num_filters_base] * self.num_levels
        current_in = in_channels
        for out_channels in channels_per_level:
            self.contracting.append(
                ConvBlock3d(
                    in_channels=current_in,
                    out_channels=out_channels,
                    repeats=repeats,
                    activation=activation,
                    dropout_ratio=dropout_ratio,
                ).module
            )
            current_in = out_channels

        for level in reversed(range(self.num_levels)):
            skip_channels = channels_per_level[level]
            if level == self.num_levels - 1:
                in_channels_level = skip_channels
            else:
                in_channels_level = skip_channels * 2
            self.expanding.insert(
                0,
                ConvBlock3d(
                    in_channels=in_channels_level,
                    out_channels=skip_channels,
                    repeats=repeats,
                    activation=activation,
                    dropout_ratio=dropout_ratio,
                ).module,
            )

    def as_module(self):
        _, nn, _ = _require_torch()

        parent = self

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.contracting = parent.contracting
                self.expanding = parent.expanding
                self.pool = parent.pool
                self.concat = _ConcatChannels()

            def forward(self, x):
                skips = []
                node = x
                for level, block in enumerate(self.contracting):
                    node = block(node)
                    skips.append(node)
                    if level < len(self.contracting) - 1:
                        node = self.pool(node)
                for level in reversed(range(len(self.expanding))):
                    if level < len(self.expanding) - 1:
                        node = upsample_linear_3d(node, (2, 2, 2))
                        node = self.concat([skips[level], node])
                    else:
                        node = skips[level]
                    node = self.expanding[level](node)
                return node

        return _Module()


class PredictionHead3d:
    def __init__(self, *, in_channels: int, out_channels: int) -> None:
        _, nn, _ = _require_torch()
        self.module = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)


class Unet:
    def __init__(
        self,
        *,
        in_channels: int,
        num_labels: int,
        num_filters_base: int = 64,
        num_levels: int = 4,
        activation: str = "relu",
        dropout_ratio: float = 0.0,
    ) -> None:
        _, nn, _ = _require_torch()
        filters_out = num_filters_base * (2 ** (num_levels - 1))
        self.single_output = num_labels == 1
        self.unet = UnetAvgLinear3D(
            in_channels=in_channels,
            num_filters_base=num_filters_base,
            num_levels=num_levels,
            activation=activation,
            dropout_ratio=dropout_ratio,
        ).as_module()
        self.prediction = PredictionHead3d(
            in_channels=num_filters_base,
            out_channels=num_labels,
        ).module

        class _Module(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.unet = parent.unet
                self.prediction = parent.prediction
                self.single_output = parent.single_output

            def forward(self, x):
                node = self.unet(x)
                prediction = self.prediction(node)
                if self.single_output:
                    return prediction
                return prediction, prediction, prediction

        self.module = _Module(self)

    def as_module(self):
        return self.module


class SpatialConfigurationNet:
    def __init__(
        self,
        *,
        in_channels: int,
        num_labels: int,
        num_filters_base: int = 64,
        num_levels: int = 4,
        activation: str = "relu",
        local_activation: str = "none",
        spatial_activation: str = "none",
        spatial_downsample: int = 8,
        dropout_ratio: float = 0.0,
    ) -> None:
        torch, nn, _ = _require_torch()

        def output_activation(name: str):
            normalized = str(name).strip().lower()
            if normalized == "none":
                return nn.Identity()
            if normalized == "tanh":
                return nn.Tanh()
            if normalized == "abs_tanh":
                class _AbsTanh(nn.Module):
                    def forward(self, x):
                        return torch.abs(torch.tanh(x))
                return _AbsTanh()
            if normalized == "sigmoid":
                class _ShiftedSigmoid(nn.Module):
                    def forward(self, x):
                        return torch.sigmoid(x - 5.0)
                return _ShiftedSigmoid()
            raise ValueError(f"Unsupported output activation '{name}'.")

        self.downsampling_factor = int(spatial_downsample)
        self.scnet_local = UnetAvgLinear3D(
            in_channels=in_channels,
            num_filters_base=num_filters_base,
            num_levels=num_levels,
            activation=activation,
            dropout_ratio=dropout_ratio,
        ).as_module()
        self.local_heatmaps = nn.Sequential(
            nn.Conv3d(num_filters_base, num_labels, kernel_size=1, padding=0),
            output_activation(local_activation),
        )
        self.downsampling = nn.AvgPool3d(
            kernel_size=self.downsampling_factor,
            stride=self.downsampling_factor,
        )
        self.scnet_spatial = UnetAvgLinear3D(
            in_channels=num_labels,
            num_filters_base=num_filters_base,
            num_levels=num_levels,
            repeats=1,
            activation=activation,
            dropout_ratio=dropout_ratio,
        ).as_module()
        self.spatial_heatmaps = nn.Conv3d(num_filters_base, num_labels, kernel_size=1, padding=0)
        self.spatial_activation = output_activation(spatial_activation)

        class _Module(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.scnet_local = parent.scnet_local
                self.local_heatmaps = parent.local_heatmaps
                self.downsampling = parent.downsampling
                self.scnet_spatial = parent.scnet_spatial
                self.spatial_heatmaps = parent.spatial_heatmaps
                self.spatial_activation = parent.spatial_activation
                self.downsampling_factor = parent.downsampling_factor

            def forward(self, x):
                node = self.scnet_local(x)
                local_heatmaps = self.local_heatmaps(node)
                node = self.downsampling(local_heatmaps)
                node = self.scnet_spatial(node)
                node = self.spatial_heatmaps(node)
                spatial_heatmaps = upsample_cubic_3d(
                    node,
                    (self.downsampling_factor,) * 3,
                )
                spatial_heatmaps = self.spatial_activation(spatial_heatmaps)
                heatmaps = local_heatmaps * spatial_heatmaps
                return heatmaps, local_heatmaps, spatial_heatmaps

        self.module = _Module(self)

    def as_module(self):
        return self.module


@dataclass(frozen=True, slots=True)
class MdatArchitectureSpec:
    name: str
    in_channels: int
    num_labels: int
    num_filters_base: int
    num_levels: int
    activation: str
    local_activation: str = "none"
    spatial_activation: str = "none"
    spatial_downsample: int = 8
    dropout_ratio: float = 0.0
    model_kind: str = "unet"


def build_model(spec: MdatArchitectureSpec):
    if spec.model_kind == "scn":
        return SpatialConfigurationNet(
            in_channels=spec.in_channels,
            num_labels=spec.num_labels,
            num_filters_base=spec.num_filters_base,
            num_levels=spec.num_levels,
            activation=spec.activation,
            local_activation=spec.local_activation,
            spatial_activation=spec.spatial_activation,
            spatial_downsample=spec.spatial_downsample,
            dropout_ratio=spec.dropout_ratio,
        ).as_module()
    return Unet(
        in_channels=spec.in_channels,
        num_labels=spec.num_labels,
        num_filters_base=spec.num_filters_base,
        num_levels=spec.num_levels,
        activation=spec.activation,
        dropout_ratio=spec.dropout_ratio,
    ).as_module()
