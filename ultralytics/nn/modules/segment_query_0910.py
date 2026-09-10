# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Class-query-conditioned Segment26 head used by the 0910 experiment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import Segment26MultiLabel

__all__ = ("Segment26ClassQuery0910",)


class Segment26ClassQuery0910(Segment26MultiLabel):
    """Feed MMS class evidence back into every Segment26 feature level."""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        multilabel_gain: float = 0.15,
        cooccurrence_weight: float = 2.0,
        condition_scale: float = 0.1,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Create one lightweight class-to-feature adapter per pyramid level."""
        super().__init__(nc, nm, npr, multilabel_gain, cooccurrence_weight, reg_max, end2end, ch)
        feature_channels = ch[: self.num_feature_levels]
        self.class_adapters = nn.ModuleList(nn.Conv2d(nc, channels, 1) for channels in feature_channels)
        self.condition_scale = nn.Parameter(torch.full((self.num_feature_levels,), float(condition_scale)))
        for adapter in self.class_adapters:
            nn.init.normal_(adapter.weight, std=1e-3)
            nn.init.zeros_(adapter.bias)
        self.class_query_conditioned = True

    def condition_features(
        self, features: list[torch.Tensor], auxiliary_logits: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Apply bounded residual class-query modulation to each feature level."""
        conditioned = []
        for index, (feature, logits, adapter) in enumerate(zip(features, auxiliary_logits, self.class_adapters)):
            if logits.shape[-2:] != feature.shape[-2:]:
                logits = F.interpolate(logits, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            modulation = torch.tanh(adapter(logits))
            scale = torch.tanh(self.condition_scale[index])
            conditioned.append(feature * (1.0 + scale * modulation))
        return conditioned

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Condition P2-P5 features and then run the standard multi-label Segment26 path."""
        if len(x) != 2 * self.num_feature_levels:
            raise ValueError(
                f"Segment26ClassQuery0910 expected {2 * self.num_feature_levels} inputs, but received {len(x)}."
            )
        features = x[: self.num_feature_levels]
        auxiliary_logits = x[self.num_feature_levels :]
        conditioned = self.condition_features(features, auxiliary_logits)
        return super().forward([*conditioned, *auxiliary_logits])
