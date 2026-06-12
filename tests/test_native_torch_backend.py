from __future__ import annotations

from spine_segment.native_torch_backend import (
    _landmark_label_from_index,
)


def test_landmark_label_from_index_uses_verse_labels() -> None:
    assert _landmark_label_from_index(0) == 1
    assert _landmark_label_from_index(24) == 25
    assert _landmark_label_from_index(25) == 28
