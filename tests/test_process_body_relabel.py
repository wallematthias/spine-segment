from __future__ import annotations

import numpy as np

from spine_segment.process_body_relabel import _expanded_crop_slices, _pad_for_network, _pad_to_shape


def test_expanded_crop_slices_returns_margin_clamped_to_image() -> None:
    mask = np.zeros((5, 6, 7), dtype=bool)
    mask[1:3, 2:4, 3:5] = True

    slices = _expanded_crop_slices(mask, margin=2)

    assert slices == (slice(0, 5), slice(0, 6), slice(1, 7))


def test_pad_for_network_uses_min_shape_and_multiples() -> None:
    array = np.ones((17, 9, 33), dtype=np.float32)

    padded, original_shape = _pad_for_network(
        array,
        min_shape=(20, 10, 40),
        multiple=(8, 4, 16),
    )

    assert original_shape == (17, 9, 33)
    assert padded.shape == (24, 12, 48)
    assert np.all(padded[:17, :9, :33] == 1)
    assert np.all(padded[17:, :, :] == 0)


def test_pad_to_shape_preserves_values_and_adds_zero_tail() -> None:
    array = np.ones((2, 3, 4), dtype=np.float32)

    padded = _pad_to_shape(array, (4, 5, 6))

    assert padded.shape == (4, 5, 6)
    assert np.all(padded[:2, :3, :4] == 1)
    assert np.all(padded[2:, :, :] == 0)
