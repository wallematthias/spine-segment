from __future__ import annotations

import SimpleITK as sitk
import numpy as np

from spine_segment import native_torch_backend
from spine_segment.native_torch_backend import (
    NativeTorchBackend,
    _landmark_label_from_index,
    _spine_localization_tiles,
    _vertebrae_localization_tiles,
)
from spine_segment.backend import SegmentationResult


def test_landmark_label_from_index_uses_verse_labels() -> None:
    assert _landmark_label_from_index(0) == 1
    assert _landmark_label_from_index(24) == 25
    assert _landmark_label_from_index(25) == 28


def test_spine_localization_tiles_cover_long_whole_body_scan() -> None:
    image = sitk.Image([512, 512, 372], sitk.sitkFloat32)
    image.SetSpacing((0.87890625, 0.87890625, 5.0))
    image.SetOrigin((-225.0, -155.0, -1570.5))

    tiles = _spine_localization_tiles(
        image,
        spacing=8.0,
        valid_sizes_xyz=((32, 64, 96, 128), (32, 64, 96, 128), (32, 64, 96, 128)),
    )

    tile_names = [tile.name for tile in tiles]
    tile_z_starts = [tile.transform.TransformPoint((0.0, 0.0, 0.0))[2] for tile in tiles]

    assert tile_names == ["inferior", "tile1", "superior", "center"]
    assert all(tile.size_xyz == (64, 64, 128) for tile in tiles)
    assert all(tile.image_array.shape == (128, 64, 64) for tile in tiles)
    assert min(tile_z_starts) == -1570.5
    assert max(tile_z_starts) == -734.5
    assert any(abs(z_start - -1151.0) < 1.0 for z_start in tile_z_starts)


def test_vertebrae_localization_tiles_cover_large_crop_with_overlap() -> None:
    image = np.zeros((256, 128, 128), dtype=np.float32)
    transform = sitk.TranslationTransform(3, (10.0, 20.0, 30.0))

    tiles = _vertebrae_localization_tiles(
        image,
        transform=transform,
        spacing=2.0,
        max_depth=128,
    )

    assert [tile.name for tile in tiles] == ["slab0", "slab1", "slab2"]
    assert [tile.image_array.shape for tile in tiles] == [
        (128, 128, 128),
        (128, 128, 128),
        (128, 128, 128),
    ]
    assert [
        tile.transform.TransformPoint((0.0, 0.0, 0.0))[2] for tile in tiles
    ] == [30.0, 158.0, 286.0]


def test_segment_levels_uses_supplied_centroids_without_localization(
    tmp_path, monkeypatch
) -> None:
    image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
    captured_landmarks = []

    def skip_model_loading(
        self, device, *, require_segmentation=True, require_localization=True
    ):
        assert require_segmentation is True
        assert require_localization is False

    def fail_localization(*args, **kwargs):
        raise AssertionError("localization must not run")

    def capture_segmentation(
        self, working_image, *, source_path, device, level_only=False, landmarks=None
    ):
        captured_landmarks.extend(landmarks)
        return SegmentationResult(vertebral_level=sitk.Image(working_image))

    monkeypatch.setattr(NativeTorchBackend, "_load_models", skip_model_loading)
    monkeypatch.setattr(NativeTorchBackend, "_localize_exact", fail_localization)
    monkeypatch.setattr(NativeTorchBackend, "_segment_exact", capture_segmentation)
    backend = NativeTorchBackend(tmp_path, strict_bundle=False, progress=False)

    result = backend.segment_levels(
        image=image,
        source_path=tmp_path / "case.nii.gz",
        device="cpu",
        centroids={
            "20": {
                "label": 20,
                "index": 19,
                "voxel_xyz": [2.0, 3.0, 4.0],
                "physical_xyz": [2.0, 3.0, 4.0],
                "score": 0.9,
            }
        },
    )

    assert result.vertebral_level.GetSize() == image.GetSize()
    assert len(captured_landmarks) == 1
    assert captured_landmarks[0][0] == "19"
    assert captured_landmarks[0][1].coords == (2.0, 3.0, 4.0)
    assert captured_landmarks[0][1].value == 0.9


def test_model_loader_adds_localization_after_centroid_only_loading(
    tmp_path, monkeypatch
) -> None:
    built_models = []

    class _FakeModel:
        def load_state_dict(self, state_dict):
            return None

        def to(self, device):
            return self

        def eval(self):
            return self

    class _FakeTorch:
        @staticmethod
        def device(device):
            return device

        @staticmethod
        def load(path, map_location):
            return {}

    def build_fake_model(spec):
        built_models.append(spec.name)
        return _FakeModel()

    monkeypatch.setattr(native_torch_backend, "_require_torch", lambda: _FakeTorch())
    monkeypatch.setattr(native_torch_backend, "build_model", build_fake_model)
    backend = NativeTorchBackend(tmp_path, strict_bundle=False, progress=False)
    backend._models_loaded = True

    backend._load_models("cpu", require_segmentation=True)

    assert built_models == ["spine_localization", "vertebrae_localization"]
    assert backend._localization_models_loaded is True
