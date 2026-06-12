from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

import numpy as np
import SimpleITK as sitk

from spine_segment.labels import BODY_LABEL, PROCESS_LABEL


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyTorch is required for process/body relabeling.") from exc
    return torch, nn, F


class _ConvNormAct3d:
    def __init__(self, in_channels: int, out_channels: int, *, stride: tuple[int, int, int]) -> None:
        _, nn, _ = _require_torch()
        self.module = nn.Sequential()
        self.module.add_module(
            "conv",
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        )
        self.module.add_module("norm", nn.InstanceNorm3d(out_channels, affine=True))
        self.module.add_module("act", nn.LeakyReLU(negative_slope=0.01, inplace=True))


class _StackedConvBlocks3d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        first_stride: tuple[int, int, int],
        convs_per_stage: int = 2,
    ) -> None:
        _, nn, _ = _require_torch()
        self.module = nn.Module()
        self.module.convs = nn.ModuleList()
        current_in = in_channels
        for conv_index in range(convs_per_stage):
            stride = first_stride if conv_index == 0 else (1, 1, 1)
            block = _ConvNormAct3d(current_in, out_channels, stride=stride).module
            self.module.convs.append(block)
            current_in = out_channels

        def forward(module_self, x):
            for block in module_self.convs:
                x = block(x)
            return x

        self.module.forward = forward.__get__(self.module, type(self.module))


class PlainConvUNet3D:
    def __init__(self) -> None:
        torch, nn, F = _require_torch()

        features = [32, 64, 128, 256, 320]
        encoder_strides = [
            (1, 1, 1),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 1, 2),
        ]

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.stages = nn.ModuleList()
                in_channels = 1
                for out_channels, stride in zip(features, encoder_strides):
                    stage = nn.Sequential(
                        _StackedConvBlocks3d(
                            in_channels,
                            out_channels,
                            first_stride=stride,
                        ).module
                    )
                    self.encoder.stages.append(stage)
                    in_channels = out_channels

                self.decoder = nn.Module()
                self.decoder.transpconvs = nn.ModuleList(
                    [
                        nn.ConvTranspose3d(320, 256, kernel_size=(2, 1, 2), stride=(2, 1, 2)),
                        nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
                        nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
                        nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
                    ]
                )
                self.decoder.stages = nn.ModuleList(
                    [
                        _StackedConvBlocks3d(512, 256, first_stride=(1, 1, 1)).module,
                        _StackedConvBlocks3d(256, 128, first_stride=(1, 1, 1)).module,
                        _StackedConvBlocks3d(128, 64, first_stride=(1, 1, 1)).module,
                        _StackedConvBlocks3d(64, 32, first_stride=(1, 1, 1)).module,
                    ]
                )
                self.decoder.seg_layers = nn.ModuleList(
                    [
                        nn.Conv3d(256, 3, kernel_size=1),
                        nn.Conv3d(128, 3, kernel_size=1),
                        nn.Conv3d(64, 3, kernel_size=1),
                        nn.Conv3d(32, 3, kernel_size=1),
                    ]
                )

            def forward(self, x):
                skips = []
                for stage in self.encoder.stages:
                    x = stage(x)
                    skips.append(x)
                for decoder_index, (transpose, stage) in enumerate(
                    zip(self.decoder.transpconvs, self.decoder.stages)
                ):
                    x = transpose(x)
                    skip = skips[-(decoder_index + 2)]
                    if x.shape[2:] != skip.shape[2:]:
                        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
                    x = torch.cat((x, skip), dim=1)
                    x = stage(x)
                return self.decoder.seg_layers[-1](x)

        self.module = _Module()

    def as_module(self):
        return self.module


@dataclass(slots=True)
class ProcessBodyRelabeler:
    checkpoint_path: Path
    crop_margin_voxels: int = 4
    batch_size: int = 4
    target_spacing_xyz: tuple[float, float, float] = (0.865234375, 0.8999999761581421, 0.9765625)
    min_shape_zyx: tuple[int, int, int] = (80, 56, 96)
    shape_multiple_zyx: tuple[int, int, int] = (16, 8, 16)
    _model: Any = None
    _device: str | None = None

    def load(self, device: str) -> None:
        if self._model is not None and self._device == device:
            return
        torch, _, _ = _require_torch()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        source_state = checkpoint["network_weights"]
        model = PlainConvUNet3D().as_module()
        target_state = model.state_dict()
        converted = {
            key: value
            for key, value in source_state.items()
            if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
        }
        missing = sorted(set(target_state) - set(converted))
        if missing:
            raise RuntimeError(
                "Process/body relabel checkpoint is missing expected tensors: "
                + ", ".join(missing[:8])
            )
        model.load_state_dict(converted, strict=True)
        self._model = model.to(device).eval()
        self._device = device

    def relabel(self, vertebral_level: sitk.Image, *, device: str) -> sitk.Image:
        self.load(device)
        torch, _, F = _require_torch()
        assert self._model is not None

        source_arr = sitk.GetArrayFromImage(vertebral_level).astype(np.uint8, copy=False)
        output = np.zeros(source_arr.shape, dtype=np.uint8)
        labels = [int(label) for label in np.unique(source_arr) if int(label) != 0]
        if not labels:
            out = sitk.GetImageFromArray(output)
            out.CopyInformation(vertebral_level)
            return out

        records = []
        for label in labels:
            mask_arr = source_arr == label
            crop_slices = _expanded_crop_slices(mask_arr, margin=int(self.crop_margin_voxels))
            if crop_slices is None:
                continue
            crop_mask_arr = mask_arr[crop_slices]
            crop_mask = sitk.GetImageFromArray(crop_mask_arr.astype(np.float32))
            _copy_crop_information(crop_mask, vertebral_level, crop_slices)
            resampled = _resample_to_spacing(crop_mask, self.target_spacing_xyz)
            input_arr = sitk.GetArrayFromImage(resampled).astype(np.float32, copy=False)
            padded, original_shape = _pad_for_network(
                input_arr,
                min_shape=self.min_shape_zyx,
                multiple=self.shape_multiple_zyx,
            )
            records.append(
                {
                    "crop_slices": crop_slices,
                    "crop_mask_arr": crop_mask_arr,
                    "crop_mask": crop_mask,
                    "resampled": resampled,
                    "input_arr": input_arr,
                    "padded": padded,
                    "original_shape": original_shape,
                }
            )

        batch_size = max(1, int(self.batch_size))
        for batch_start in range(0, len(records), batch_size):
            batch_records = records[batch_start : batch_start + batch_size]
            batch_shape = tuple(
                max(record["padded"].shape[axis] for record in batch_records)
                for axis in range(3)
            )
            batch_np = np.stack(
                [_pad_to_shape(record["padded"], batch_shape) for record in batch_records],
                axis=0,
            )
            tensor = torch.from_numpy(batch_np[:, None].astype(np.float32, copy=False)).to(
                device=device,
                dtype=torch.float32,
            )
            with torch.inference_mode():
                batch_logits = self._model(tensor)

            for batch_index, record in enumerate(batch_records):
                original_shape = record["original_shape"]
                input_arr = record["input_arr"]
                logits = batch_logits[
                    batch_index,
                    :,
                    : original_shape[0],
                    : original_shape[1],
                    : original_shape[2],
                ]
                if logits.shape[1:] != input_arr.shape:
                    logits = F.interpolate(
                        logits[None],
                        size=input_arr.shape,
                        mode="trilinear",
                        align_corners=False,
                    )[0]
                pred = torch.where(logits[PROCESS_LABEL] > logits[BODY_LABEL], PROCESS_LABEL, BODY_LABEL)
                pred_arr = pred.detach().cpu().numpy().astype(np.uint8, copy=False)
                pred_arr[input_arr <= 0.05] = 0
                pred_image = sitk.GetImageFromArray(pred_arr)
                pred_image.CopyInformation(record["resampled"])
                restored = _resample_like(pred_image, record["crop_mask"], interpolator=sitk.sitkNearestNeighbor)
                restored_arr = sitk.GetArrayFromImage(restored).astype(np.uint8, copy=False)
                crop_output = output[record["crop_slices"]]
                crop_mask_arr = record["crop_mask_arr"]
                crop_output[crop_mask_arr] = np.where(
                    restored_arr[crop_mask_arr] == PROCESS_LABEL,
                    PROCESS_LABEL,
                    BODY_LABEL,
                )
        out = sitk.GetImageFromArray(output)
        out.CopyInformation(vertebral_level)
        return out


def _expanded_crop_slices(mask: np.ndarray, *, margin: int) -> tuple[slice, slice, slice] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    start = np.maximum(coords.min(axis=0) - int(margin), 0)
    end = np.minimum(coords.max(axis=0) + int(margin) + 1, mask.shape)
    return tuple(slice(int(s), int(e)) for s, e in zip(start, end))


def _copy_crop_information(
    crop_image: sitk.Image,
    reference: sitk.Image,
    crop_slices_zyx: tuple[slice, slice, slice],
) -> None:
    z_slice, y_slice, x_slice = crop_slices_zyx
    origin = reference.TransformIndexToPhysicalPoint(
        (int(x_slice.start), int(y_slice.start), int(z_slice.start))
    )
    crop_image.SetOrigin(origin)
    crop_image.SetSpacing(reference.GetSpacing())
    crop_image.SetDirection(reference.GetDirection())


def _resample_to_spacing(image: sitk.Image, spacing_xyz: tuple[float, float, float]) -> sitk.Image:
    old_size = np.asarray(image.GetSize(), dtype=np.float64)
    old_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    new_spacing = np.asarray(spacing_xyz, dtype=np.float64)
    new_size = np.maximum(1, np.round(old_size * old_spacing / new_spacing)).astype(int)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(v) for v in new_size])
    resampler.SetOutputSpacing(tuple(float(v) for v in new_spacing))
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(image)


def _resample_like(image: sitk.Image, reference: sitk.Image, *, interpolator: int) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return resampler.Execute(image)


def _pad_for_network(
    array: np.ndarray,
    *,
    min_shape: tuple[int, int, int],
    multiple: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    shape = tuple(int(v) for v in array.shape)
    target = []
    for size, min_size, divisor in zip(shape, min_shape, multiple):
        padded_size = max(int(size), int(min_size))
        padded_size = int(math.ceil(padded_size / int(divisor)) * int(divisor))
        target.append(padded_size)
    output = np.zeros(tuple(target), dtype=np.float32)
    output[: shape[0], : shape[1], : shape[2]] = array
    return output, shape


def _pad_to_shape(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(array.shape) == tuple(shape):
        return array.astype(np.float32, copy=False)
    output = np.zeros(tuple(int(v) for v in shape), dtype=np.float32)
    output[: array.shape[0], : array.shape[1], : array.shape[2]] = array
    return output
