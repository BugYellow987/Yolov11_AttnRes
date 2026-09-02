# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Regression tests for delayed RGB/IIM fusion, MSAT, and the class-query segmentation head."""

from pathlib import Path

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import (
    AdaptiveGatedFusion,
    DualIIMStem,
    MSAT,
    MSATMultiLabel,
    Segment26,
    Segment26ClassQuery,
)
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import v8MultiLabelSegmentationLoss, v8SegmentationLoss


MODEL_ROOT = Path("ultralytics/cfg/models/11_myself")
V1_CFG = MODEL_ROOT / "yolo11-dual-iim-6csar.yaml"
V2_CFG = MODEL_ROOT / "yolo11-dual-iim-msat.yaml"
V3_CFG = MODEL_ROOT / "yolo11-dual-iim-msat-classquery.yaml"


def test_dual_stem_and_gate_preserve_explicit_branches():
    """Verify separate full-width branches and a neutral, differentiable initial fusion."""
    stem = DualIIMStem(3, 16).eval()
    image = torch.rand(2, 3, 64, 96)
    appearance, structure = stem(image)
    assert appearance.shape == structure.shape == (2, 16, 32, 48)

    fusion = AdaptiveGatedFusion([16, 16], 16).eval()
    gate = fusion.fusion_gate(appearance, structure)
    output = fusion([appearance, structure])
    assert output.shape == appearance.shape
    assert torch.allclose(gate, torch.full_like(gate, 0.5))

    output.mean().backward()
    assert stem.rgb_stem.conv.weight.grad is not None
    assert stem.iim.weight.grad is not None
    assert fusion.channel_gate[-1].weight.grad is not None
    assert fusion.spatial_gate.weight.grad is not None


def test_ablation_graphs_change_only_the_intended_stages():
    """Verify V1/V2/V3 select CSAR, MSAT, and class-query heads as intended."""
    v1 = SegmentationModel(V1_CFG, ch=3, nc=4, verbose=False)
    v2 = SegmentationModel(V2_CFG, ch=3, nc=4, verbose=False)
    v3 = SegmentationModel(V3_CFG, ch=3, nc=4, verbose=False)

    for model in (v1, v2, v3):
        assert isinstance(model.model[0], DualIIMStem)
        assert sum(isinstance(module, AdaptiveGatedFusion) for module in model.modules()) == 2
        assert model.stride.tolist() == [8.0, 4.0, 16.0, 32.0]

    assert isinstance(v1.model[-1], Segment26)
    assert not any(isinstance(module, MSAT) for module in v1.modules())
    assert type(v2.model[-1]) is Segment26
    assert sum(type(module) is MSAT for module in v2.modules()) == 4
    assert isinstance(v3.model[-1], Segment26ClassQuery)
    assert sum(isinstance(module, MSATMultiLabel) for module in v3.modules()) == 4

    v1.args = v2.args = v3.args = get_cfg()
    v3.args.overlap_mask = False
    assert isinstance(v1.init_criterion(), v8SegmentationLoss)
    assert isinstance(v2.init_criterion(), v8SegmentationLoss)
    assert isinstance(v3.init_criterion(), v8MultiLabelSegmentationLoss)


def test_class_queries_condition_main_segmentation_features():
    """Verify aligned class-query maps and gradient flow into both query and conditioning paths."""
    model = SegmentationModel(V3_CFG, ch=3, nc=4, verbose=False).train()
    predictions = model(torch.rand(1, 3, 128, 192))
    assert [tuple(logits.shape) for logits in predictions["multilabel_logits"]] == [
        (1, 4, 16, 24),
        (1, 4, 32, 48),
        (1, 4, 8, 12),
        (1, 4, 4, 6),
    ]
    proto = predictions["proto"][0] if isinstance(predictions["proto"], tuple) else predictions["proto"]
    assert proto.shape == (1, 32, 32, 48)

    scalar = (
        predictions["boxes"].mean()
        + predictions["scores"].mean()
        + predictions["mask_coefficient"].mean()
        + sum(logits.mean() for logits in predictions["multilabel_logits"])
    )
    scalar.backward()
    query_modules = [module for module in model.modules() if isinstance(module, MSATMultiLabel)]
    assert all(module.multi_label_head.class_queries.grad is not None for module in query_modules)
    assert all(adapter.weight.grad is not None for adapter in model.model[-1].class_adapters)
