# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Tests for label-safe Dent hard-positive oversampling."""

from pathlib import Path

import numpy as np
import pytest

from ultralytics.data.dataset import YOLODataset


def make_dataset(tmp_path: Path) -> YOLODataset:
    """Create the minimal dataset state needed by the oversampling helper."""
    dataset = YOLODataset.__new__(YOLODataset)
    dataset.augment = True
    dataset.use_segments = True
    dataset.prefix = "train: "
    dataset.data = {
        "path": tmp_path,
        "names": {0: "Dent", 1: "Scratch"},
        "hard_positive": {
            "class": "Dent",
            "images": ["shadow_dent"],
            "repeat": 3,
            "require_class": True,
        },
    }
    return dataset


def make_label(path: Path, class_id: int, with_mask: bool = True) -> dict:
    """Return a small cached-label-like dictionary."""
    return {
        "im_file": str(path),
        "shape": (32, 32),
        "cls": np.array([[class_id]], dtype=np.float32),
        "bboxes": np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
        "segments": [np.array([[0.4, 0.4], [0.6, 0.4], [0.5, 0.6]], dtype=np.float32)] if with_mask else [],
        "keypoints": None,
        "normalized": True,
        "bbox_format": "xywh",
    }


def test_dent_hard_positive_is_repeated_without_aliasing(tmp_path: Path):
    """Repeat reviewed Dent masks while leaving ordinary samples at their original frequency."""
    dataset = make_dataset(tmp_path)
    labels = [
        make_label(tmp_path / "shadow_dent.jpg", 0),
        make_label(tmp_path / "ordinary.jpg", 1),
    ]
    dataset.im_files = [label["im_file"] for label in labels]

    repeated = dataset._oversample_hard_positives(labels)

    assert len(repeated) == 4
    assert [Path(label["im_file"]).stem for label in repeated].count("shadow_dent") == 3
    assert repeated[0] is not repeated[2]
    assert len(dataset.im_files) == len(dataset.label_files) == len(repeated)


@pytest.mark.parametrize((class_id, with_mask), [(1, True), (0, False)])
def test_listed_hard_positive_requires_complete_dent_annotation(tmp_path: Path, class_id: int, with_mask: bool):
    """Reject listed images that are missing either the Dent class or its segmentation polygon."""
    dataset = make_dataset(tmp_path)
    labels = [make_label(tmp_path / "shadow_dent.jpg", class_id, with_mask)]
    dataset.im_files = [labels[0]["im_file"]]

    with pytest.raises(ValueError, match="complete Dent annotation"):
        dataset._oversample_hard_positives(labels)
