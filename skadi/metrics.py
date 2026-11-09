"""Confusion-matrix based metrics.

The headline number is the F1 score used by the AutoICE challenge and by the
SAR-only works we compare with: scikit-learn's `f1_score(average="weighted")`
over all labelled pixels of the test scenes taken together.
"""
from __future__ import annotations

import torch

from .constants import IGNORE, N_SOD, SOD_TO_SKADI


def confusion(pred: torch.Tensor, target: torch.Tensor, n_classes: int = N_SOD) -> torch.Tensor:
    """Rows = reference class, columns = predicted class."""
    valid = target != IGNORE
    idx = target[valid].long() * n_classes + pred[valid].long()
    return torch.bincount(idx, minlength=n_classes * n_classes).view(n_classes, n_classes).cpu()


def collapse(cm: torch.Tensor, mapping=SOD_TO_SKADI) -> torch.Tensor:
    k = max(mapping) + 1
    m = torch.zeros(k, cm.shape[0], dtype=cm.dtype)
    for src, dst in enumerate(mapping):
        m[dst, src] = 1
    return m @ cm @ m.T


def scores(cm: torch.Tensor) -> dict:
    cm = cm.double()
    tp = cm.diag()
    support = cm.sum(1)
    predicted = cm.sum(0)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    iou = tp / (support + predicted - tp).clamp_min(1)
    total = support.sum().clamp_min(1)
    present = support > 0
    return {
        "f1_weighted": 100 * float((f1 * support).sum() / total),
        "f1_macro": 100 * float(f1[present].mean()) if present.any() else 0.0,
        "accuracy": 100 * float(tp.sum() / total),
        "miou": 100 * float(iou[present].mean()) if present.any() else 0.0,
        "f1_per_class": [100 * float(v) for v in f1],
        "precision_per_class": [100 * float(v) for v in precision],
        "recall_per_class": [100 * float(v) for v in recall],
        "support": [int(v) for v in support],
    }
