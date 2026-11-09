"""Weighted cross-entropy + Dice loss (Section 3.3 of the Skadi document)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import IGNORE, MYI_INDEX


def class_weights(pixel_counts: torch.Tensor, power: float = 0.5, myi_boost: float = 1.5,
                  max_weight: float = 4.0) -> torch.Tensor:
    """Inverse-frequency weights, softened by `power`, with an extra factor on old ice.

    power=0 gives uniform weights, power=1 full inverse-frequency. Missing a
    multi-year-ice pixel is the most dangerous error for navigation, hence the
    additional `myi_boost` on that class.
    """
    freq = pixel_counts / pixel_counts.sum().clamp_min(1)
    w = (1.0 / freq.clamp_min(1e-6)) ** power
    w = w / w.mean()
    w = w.clamp(max=max_weight)
    w[MYI_INDEX] = w[MYI_INDEX] * myi_boost
    return w / w.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, dice_weight: float = 0.5,
                 label_smoothing: float = 0.0, ignore_index: int = IGNORE):
        super().__init__()
        self.register_buffer("weight", weight)
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        valid = (target != self.ignore_index)
        prob = logits.float().softmax(dim=1) * valid.unsqueeze(1)
        onehot = F.one_hot(torch.where(valid, target, torch.zeros_like(target)), n_classes)
        onehot = onehot.permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
        dims = (0, 2, 3)  # batch-level Dice: stable for classes that are rare in a single scene
        inter = (prob * onehot).sum(dims)
        denom = prob.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + 1.0) / (denom + 1.0)
        present = onehot.sum(dims) > 0
        w = torch.ones(n_classes, device=logits.device) if self.weight is None else self.weight
        w = w * present
        return ((1 - dice) * w).sum() / w.sum().clamp_min(1e-6)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not (target != self.ignore_index).any():
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits.float(), target, weight=self.weight,
                             ignore_index=self.ignore_index, label_smoothing=self.label_smoothing)
        if self.dice_weight == 0:
            return ce
        return ce + self.dice_weight * self.dice(logits, target)
