# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Training and inference checks for the six-CSAR damage semantic-token model."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import CSAR, DamageSemanticAttention, Segment26MultiLabel
from ultralytics.nn.tasks import SegmentationModel, yaml_model_load
from ultralytics.utils.loss import v8MultiLabelSegmentationLoss
from ultralytics.utils.torch_utils import ModelEMA

ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG = ROOT / "ultralytics/cfg/models/11_myself/yolo11-6csar-semantic-token.yaml"
BASE_CFG = ROOT / "ultralytics/cfg/models/11_myself/yolo11-6csar.yaml"


def overlapping_batch(empty=False):
    """Keep Rust and Dent masks independently positive in the same image region."""
    masks = torch.zeros(2, 32, 32)
    masks[0, 6:23, 5:21] = 1
    masks[1, 12:29, 12:27] = 1
    count = 0 if empty else 2
    return {
        "img": torch.randn(1, 3, 128, 128),
        "batch_idx": torch.zeros(count),
        "cls": torch.tensor([[0.0], [1.0]])[:count],
        "bboxes": torch.tensor([[0.40625, 0.453125, 0.5, 0.53125], [0.609375, 0.640625, 0.46875, 0.53125]])[:count],
        "masks": masks[:count],
        "heatmaps": torch.zeros(1, 3, 32, 32),
        "seedmaps": torch.zeros(1, 3, 32, 32),
    }


def test_damage_queries_train_through_the_visual_residual():
    """Main-head features alone must backpropagate through token queries, keys, and values."""
    module = DamageSemanticAttention(24, 3, num_heads=4, embed_channels=32)
    x = torch.randn(2, 24, 11, 15, requires_grad=True)
    enhanced, logits = module(x)
    assert enhanced.shape == x.shape
    assert logits.shape == (2, 3, 11, 15)
    enhanced.square().mean().backward()
    for parameter in (module.damage_tokens, module.query.weight, module.key.weight, module.value.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_damage_responses_allow_two_classes_at_the_same_pixel():
    """Identical class evidence can give two high responses without class softmax competition."""
    module = DamageSemanticAttention(16, 2, embed_channels=32).eval()
    with torch.no_grad():
        module.damage_tokens[1].copy_(module.damage_tokens[0])
        module.class_bias.fill_(8.0)
        x = torch.randn(1, 16, 7, 9)
        enhanced, logits = module(x)
        assert torch.allclose(logits[:, 0], logits[:, 1])
        assert (logits.sigmoid() > 0.95).all()
        module.residual_scale.zero_()
        identity, _ = module(x)
    assert torch.equal(identity, x)
    assert not torch.equal(enhanced, x)


@pytest.mark.parametrize("scale,nc", [("n", 3), ("s", 5)])
def test_yaml_preserves_backbone_and_six_csar_with_dataset_class_override(scale, nc):
    """Keep the actual base graph, including its P2 head input, across width and class-count changes."""
    base = yaml_model_load(BASE_CFG)
    cfg = yaml_model_load(MODEL_CFG)
    assert cfg["backbone"] == base["backbone"]
    assert cfg["head"][:6] == base["head"][:6]
    cfg["scale"] = scale
    model = SegmentationModel(cfg, nc=nc, verbose=False)
    assert sum(type(m) is CSAR for m in model.model) == 6
    modules = [m for m in model.model if isinstance(m, DamageSemanticAttention)]
    assert [m.f for m in modules] == base["head"][-1][0]
    assert all(m.damage_tokens.shape == (nc, 128) for m in modules)
    assert model.stride.tolist() == [8.0, 4.0, 16.0, 32.0]
    assert isinstance(model.model[-1], Segment26MultiLabel)


@pytest.mark.parametrize("empty", [False, True])
def test_multidamage_loss_backpropagates_with_overlap_and_background(empty):
    """Train all semantic branches using the real segmentation loss, including empty-label batches."""
    model = SegmentationModel(MODEL_CFG, nc=3, verbose=False)
    model.args = get_cfg(overrides={"overlap_mask": False})
    model.train()
    batch = overlapping_batch(empty)
    predictions = model(batch["img"])
    assert [tuple(t.shape[-2:]) for t in predictions["multilabel_logits"]] == [(16, 16), (32, 32), (8, 8), (4, 4)]
    criterion = model.init_criterion()
    assert isinstance(criterion, v8MultiLabelSegmentationLoss)
    target = criterion.build_multilabel_target(batch, 1, (32, 32), torch.float32)
    assert target[0, :, 15, 15].tolist() == ([0.0, 0.0, 0.0] if empty else [1.0, 1.0, 0.0])
    loss, items = criterion(predictions, batch)
    assert torch.isfinite(loss).all() and items[4] > 0
    loss.sum().backward()
    for module in model.modules():
        if isinstance(module, DamageSemanticAttention):
            assert torch.isfinite(module.damage_tokens.grad).all()
            assert module.damage_tokens.grad.abs().sum() > 0
    ModelEMA(model)  # training tensors must not remain in module attributes
    deepcopy(model)


def test_yolo_inference_validation_and_checkpoint_roundtrip(tmp_path):
    """Use ordinary YOLO task discovery and retain its standard inference/validation contract."""
    yolo = YOLO(MODEL_CFG, verbose=False)
    assert yolo.task == "segment"
    model = SegmentationModel(MODEL_CFG, nc=3, verbose=False).eval()
    model.args = get_cfg(overrides={"overlap_mask": False})
    batch = overlapping_batch()
    with torch.no_grad():
        predictions = model(batch["img"])
        loss, _ = model.loss(batch, predictions)
        rectangular = model(torch.randn(1, 3, 128, 192))
    assert torch.isfinite(loss).all()
    assert rectangular[0][1].shape == (1, 32, 32, 48)
    assert rectangular[0][0].shape[1] == 4 + 3 + 32
    assert len(rectangular[1]["multilabel_logits"]) == 4
    # Check the tensors used by exporters without requiring an optional ONNX installation.
    model.model[-1].export = True
    with torch.no_grad():
        exported = model(batch["img"])
    assert exported[0].shape == predictions[0][0].shape
    assert exported[1].shape == predictions[0][1].shape
    model.model[-1].export = False
    path = tmp_path / "semantic.pt"
    torch.save({"model": model, "train_args": {"task": "segment"}}, path)
    restored = YOLO(path, task="segment", verbose=False).model.eval()
    with torch.no_grad():
        actual = restored(batch["img"])
    torch.testing.assert_close(actual[0][0], predictions[0][0])


def test_merged_masks_are_rejected():
    """Fail clearly when training would discard overlapping labels."""
    model = SegmentationModel(MODEL_CFG, nc=3, verbose=False)
    model.args = get_cfg(overrides={"overlap_mask": True})
    with pytest.raises(ValueError, match="overlap_mask=False"):
        model.init_criterion()
