# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Independent instance-boundary supervision for the boundary-guided multi-label head."""

import torch
import torch.nn.functional as F

from .loss import v8MultiLabelSegmentationLoss


class v8BoundaryMultiLabelSegmentationLoss(v8MultiLabelSegmentationLoss):
    """Keep the seven existing loss slots, adding the boundary objective to sem_loss."""

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.boundary_gain = float(model.model[-1].boundary_gain)

    def build_boundary_target(self, batch, batch_size, size, dtype):
        """Find each instance's 3x3 morphological boundary BEFORE merging by class.

        D/R overlap and interfaces between touching instances of the same class remain available.
        This uses annotated damage masks, not image gradients or a threshold on dark pixels.
        """
        masks = batch["masks"].to(self.device).float()
        if masks.ndim != 3 or masks.shape[0] != batch["cls"].numel():
            raise ValueError("Boundary targets require independent instance masks and overlap_mask=False.")
        if masks.shape[0]:
            padded = F.pad((masks > 0.5).float()[:, None], (1, 1, 1, 1), mode="replicate")
            dilated = F.max_pool2d(padded, 3, stride=1)
            eroded = -F.max_pool2d(-padded, 3, stride=1)
            masks = (dilated - eroded).squeeze(1)
        # Resize only AFTER finding boundaries, using the existing max-preserving target builder.
        return self.build_multilabel_target({**batch, "masks": masks}, batch_size, size, dtype)

    def _boundary_loss(self, logits, target):
        """FP32, normalized positive-weighted BCE plus class-wise Dice for thin boundaries."""
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits, target = logits.float(), target.float()
            positives = target.sum(dim=(0, 2, 3), keepdim=True)
            pixels = target.shape[0] * target.shape[2] * target.shape[3]
            positive_weight = ((pixels - positives) / positives.clamp_min(1)).clamp(1, 20)
            weights = 1 + (positive_weight - 1) * target
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            bce = (bce * weights).sum() / weights.sum().clamp_min(1)
            return 0.5 * bce + 0.5 * self.multilabel_dice(logits, target)

    def loss(self, preds, batch):
        scaled_loss, detached_loss = super().loss(preds, batch)
        boundaries = preds.get("boundary_logits")
        if not boundaries or len(boundaries) != 2:
            raise RuntimeError("Boundary head must supply explicit P3/P2 boundary logits for supervision.")
        batch_size = preds["boxes"].shape[0]
        boundary_loss = torch.zeros((), device=self.device)
        for logits in boundaries:
            target = self.build_boundary_target(batch, batch_size, logits.shape[-2:], torch.float32)
            boundary_loss = boundary_loss + self._boundary_loss(logits, target)
        boundary_loss = boundary_loss * (self.boundary_gain / len(boundaries))
        scaled_loss[4] += boundary_loss * batch_size
        detached_loss[4] += boundary_loss.detach()
        return scaled_loss, detached_loss
