# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Regression tests for the YOLA IIM front-end on the two-stage CSAR segmentation architecture."""

from pathlib import Path

import torch

from ultralytics.nn.modules import CSAR, IIMStem, IlluminationInvariantConv, Segment26
from ultralytics.nn.tasks import SegmentationModel


MODEL_CFG = Path("ultralytics/cfg/models/11_myself/yolo11-6csar-iim.yaml")


def test_iim_zero_mean_constraint_and_illumination_invariance():
    """Verify the projected kernels and the channel-pair response to achromatic illumination changes."""
    iim = IlluminationInvariantConv(kernel_nums=4, kernel_size=3).eval()
    image = torch.rand(2, 3, 24, 32) * 0.6 + 0.2
    illumination = torch.rand(2, 1, 24, 32) * 0.4 + 0.5

    with torch.no_grad():
        original = iim(image)
        relit = iim(image * illumination)

    assert torch.allclose(iim.zero_mean_weight().sum((2, 3)), torch.zeros(4, 1), atol=1e-6)
    assert torch.allclose(original, relit, atol=1e-5, rtol=1e-4)


def test_yolo11_6csar_iim_model_graph_and_forward():
    """Verify the residual IIM stem, context-aware P2 head, and prototype resolution."""
    model = SegmentationModel(MODEL_CFG, ch=3, nc=3, verbose=False)

    assert isinstance(model.model[0], IIMStem)
    assert model.model[0].iim.kernel_nums == 8
    assert model.model[0].iim.kernel_size == 3
    assert model.model[0].rgb_stem.conv.out_channels == model.model[0].iim_stem.conv.out_channels
    assert model.model[0].iim_gain.item() == 0.0
    assert isinstance(model.model[-2], CSAR)
    assert model.model[-2].f == [3, 18, 19]
    assert model.model[-2].target == 0
    assert isinstance(model.model[-1], Segment26)
    assert model.model[-1].f == [21, 18, 19, 20]
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]

    model.eval()
    image = torch.rand(1, 3, 128, 192)
    with torch.no_grad():
        appearance = model.model[0].rgb_stem(image)
        stem_output = model.model[0](image)
        output = model(image)
    assert torch.equal(stem_output, appearance)
    assert output[0][1].shape == (1, 32, image.shape[-2] // 2, image.shape[-1] // 2)
