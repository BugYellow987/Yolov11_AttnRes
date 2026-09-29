# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Architecture paths must reach real mask predictions and survive checkpointing and fusion."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import CSAR, DamageSemanticAttention
from ultralytics.nn.modules.block import Proto26MultiLabel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.damage_architecture import (
    DualPathP2Proto,
    LuminanceResidualStem,
    Segment26MultiLabelExtent,
)
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import YAML
from ultralytics.utils.loss import v8MultiLabelSegmentationLoss
from ultralytics.utils.torch_utils import ModelEMA, fuse_conv_and_bn

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "ultralytics/cfg/models/11_myself"
BASE = CFG / "yolo11-6csar-semantic-token.yaml"
STAGES = ("00-baseline", "01-luma", "02-shallow", "03-dualproto")


def configured(stage, nc=8, scale="n"):
    cfg = YAML.load(CFG / f"yolo11-6csar-arch-0929-{stage}.yaml")
    cfg["scale"] = scale
    model = SegmentationModel(cfg, nc=nc, verbose=False)
    model.args = get_cfg(overrides={"overlap_mask": False})
    model.names = dict(enumerate(("B", "C", "D", "H", "O", "R", "X", "U")[:nc]))
    return model


def overlapping_batch(empty=False, nc=8):
    masks = torch.zeros(2, 32, 32)
    masks[0, 6:23, 5:21] = 1
    masks[1, 12:29, 12:27] = 1
    n = 0 if empty else 2
    classes = [[2.0], [5.0]] if nc == 8 else [[0.0], [1.0]]
    return {
        "img": torch.rand(1, 3, 128, 128) * 0.08,
        "masks": masks[:n],
        "batch_idx": torch.zeros(n),
        "cls": torch.tensor(classes)[:n],
        "bboxes": torch.tensor([[0.40625, 0.453125, 0.5, 0.53125], [0.609375, 0.640625, 0.46875, 0.53125]])[:n],
        "heatmaps": torch.zeros(1, nc, 32, 32),
        "seedmaps": torch.zeros(1, nc, 32, 32),
    }


def assert_gradients(module):
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_luminance_stem_preserves_conv_and_fused_path():
    base = Conv(3, 16, 3, 2).eval()
    model = LuminanceResidualStem(3, 16, detail_scale=0).eval()
    result = model.load_state_dict(base.state_dict(), strict=False)
    assert not result.unexpected_keys
    x = torch.rand(2, 3, 31, 45) * 0.03
    torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)
    with torch.no_grad():
        model.detail_scale.fill_(0.1)
    model(x).square().mean().backward()
    assert_gradients(model.detail_stem)
    before = model(x).detach()
    model.conv = fuse_conv_and_bn(model.conv, model.bn)
    del model.bn
    model.forward = model.forward_fuse
    torch.testing.assert_close(model(x), before, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        model.detail_scale.zero_()
    assert not torch.equal(model(x), before)


@pytest.mark.parametrize("kwargs", [{"c1": 1}, {"detail_scale": float("nan")}, {"detail_scale": True}])
def test_invalid_stem_config(kwargs):
    args = {"c1": 3, "c2": 16, **kwargs}
    with pytest.raises(ValueError):
        LuminanceResidualStem(**args)


def test_proto_preserves_old_path_and_learns_at_native_p2():
    ch = (16, 8, 24, 32)
    base = Proto26MultiLabel(ch, 16, 4, 3).eval()
    proto = DualPathP2Proto(ch, 16, 4, 3, detail_scale=0).eval()
    result = proto.load_state_dict(base.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith("native_") for k in result.missing_keys)
    features = [torch.randn(2, c, h, w, requires_grad=True) for c, h, w in zip(ch, (8, 16, 4, 2), (12, 24, 6, 3))]
    torch.testing.assert_close(proto(features), base(features), rtol=0, atol=0)
    with torch.no_grad():
        proto.native_scale.fill_(0.1)
    seen = []
    handle = proto.native_refine.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[-2:]))
    proto(features).square().mean().backward()
    handle.remove()
    assert seen == [torch.Size([16, 24])]
    for branch in (proto.native_p2, proto.native_context, proto.native_gate, proto.native_refine, proto.native_output):
        assert_gradients(branch)
    assert all(torch.isfinite(x.grad).all() for x in features)
    train_output = proto.train()(features)
    assert train_output[0].shape == (2, 4, 16, 24)
    assert set(train_output[1]) == {"heatmap", "seedmap"}


def test_proto_and_head_reject_wrong_feature_layout():
    with pytest.raises(ValueError):
        DualPathP2Proto((8, 16, 32))
    with pytest.raises(ValueError):
        Segment26MultiLabelExtent(ch=(8,) * 12)


@pytest.mark.parametrize("stage", STAGES)
def test_architecture_warmstart_training_and_common_recipes(stage):
    base_cfg = YAML.load(BASE)
    cfg = YAML.load(CFG / f"yolo11-6csar-arch-0929-{stage}.yaml")
    assert cfg["backbone"][1:] == base_cfg["backbone"][1:]
    assert cfg["head"][:-1] == base_cfg["head"][:-1]
    assert "training_aux" not in cfg
    recipe_dir = ROOT / "ultralytics/cfg/experiments/damage-arch-0929"
    recipe = get_cfg(recipe_dir / f"{stage}.yaml")
    assert recipe.imgsz == 640 and recipe.overlap_mask is False and recipe.mask_ratio == 4
    common = YAML.load(recipe_dir / "00-baseline.yaml")
    actual = YAML.load(recipe_dir / f"{stage}.yaml")
    for key in ("model", "name"):
        common.pop(key)
        actual.pop(key)
    assert common == actual

    base = SegmentationModel(BASE, nc=8, verbose=False)
    model = configured(stage)
    state = model.state_dict()
    assert all(k in state and v.shape == state[k].shape for k, v in base.state_dict().items())
    result = model.load_state_dict(base.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert bool(result.missing_keys) == (stage != "00-baseline")
    assert len(model.model) == 34
    assert sum(type(m) is CSAR for m in model.model) == 6
    assert sum(isinstance(m, DamageSemanticAttention) for m in model.model) == 4
    assert model.stride.tolist() == [8.0, 4.0, 16.0, 32.0]
    loss, _ = model.train()(overlapping_batch())
    assert isinstance(model.criterion, v8MultiLabelSegmentationLoss)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    if stage != "00-baseline":
        assert_gradients(model.model[0].detail_stem)
    if stage in ("02-shallow", "03-dualproto"):
        for module in model.model[-1].detail_fusion:
            assert_gradients(module)
    if stage == "03-dualproto":
        assert_gradients(model.model[-1].proto.native_output)
        assert_gradients(model.model[-1].proto.native_context)
    ModelEMA(model)
    with torch.no_grad():
        output = deepcopy(model).eval()(torch.rand(1, 3, 128, 192) * 0.03)
    assert output[0][1].shape == (1, 32, 32, 48)
    assert torch.isfinite(output[0][0]).all() and torch.isfinite(output[0][1]).all()


def test_all_zero_residuals_recover_base_predictions():
    base = SegmentationModel(BASE, nc=8, verbose=False).eval()
    model = configured("03-dualproto").eval()
    model.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        model.model[0].detail_scale.zero_()
        for module in model.model[-1].detail_fusion:
            module.detail_scale.zero_()
        model.model[-1].proto.native_scale.zero_()
        x = torch.rand(1, 3, 128, 192)
        expected, actual = base(x), model(x)
    torch.testing.assert_close(actual[0][0], expected[0][0], rtol=0, atol=0)
    torch.testing.assert_close(actual[0][1], expected[0][1], rtol=0, atol=0)


@pytest.mark.parametrize("empty", [False, True])
def test_class_and_scale_override_and_empty_training(empty):
    model = configured("03-dualproto", nc=3, scale="s").train()
    loss, _ = model(overlapping_batch(empty=empty, nc=3))
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_validation_export_checkpoint_fusion_and_half(tmp_path):
    model = configured("03-dualproto").eval()
    batch = overlapping_batch()
    with torch.no_grad():
        expected = model(batch["img"])
        loss, _ = model.loss(batch, expected)
        assert torch.isfinite(loss).all()
        model.model[-1].export = True
        exported = model(batch["img"])
        torch.testing.assert_close(exported[1], expected[0][1])
        model.model[-1].export = False
    path = tmp_path / "architecture.pt"
    torch.save({"model": model, "train_args": {"task": "segment"}}, path)
    restored = YOLO(path, task="segment", verbose=False).model.eval()
    with torch.no_grad():
        actual = restored(batch["img"])
        torch.testing.assert_close(actual[0][0], expected[0][0])
        torch.testing.assert_close(actual[0][1], expected[0][1])
        restored.fuse(verbose=False)
        fused = restored(batch["img"])
        torch.testing.assert_close(fused[0][0], expected[0][0], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(fused[0][1], expected[0][1], rtol=2e-4, atol=2e-4)
        half_output = restored.half()(batch["img"].half())
        assert torch.isfinite(half_output[0][0]).all() and torch.isfinite(half_output[0][1]).all()
