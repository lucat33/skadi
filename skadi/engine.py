"""Shared inference / evaluation helpers and checkpoint handling."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from .constants import N_SOD
from .data import INPUT_MODES, Scenes
from .metrics import collapse, confusion, scores
from .models import build_model


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


class EMA:
    """Exponential moving average of the weights (BatchNorm statistics are copied)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))  # fast start
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor, tta: bool = False) -> torch.Tensor:
    """Class probabilities-ready logits; with `tta` the four flip variants are averaged."""
    if not tta:
        return model(x).float()
    out = 0
    for dims in ((), (2,), (3,), (2, 3)):
        xi = torch.flip(x, dims) if dims else x
        yi = model(xi).float().softmax(1)
        out = out + (torch.flip(yi, dims) if dims else yi)
    return torch.log(out / 4)


@torch.no_grad()
def evaluate(model: nn.Module, scenes: Scenes, mode: str, device, batch: int = 8,
             tta: bool = False, per_scene: bool = False) -> dict:
    model.eval()
    cm = torch.zeros(N_SOD, N_SOD, dtype=torch.long)
    scene_f1 = {}
    for start in range(0, len(scenes), batch):
        idx = torch.arange(start, min(start + batch, len(scenes)))
        x = scenes.inputs(idx, mode, device)
        y = scenes.labels(idx, device)
        pred = predict_logits(model, x, tta).argmax(1)
        for i in range(len(idx)):
            c = confusion(pred[i], y[i])
            cm += c
            if per_scene and c.sum() > 0:
                scene_f1[scenes.names[int(idx[i])]] = scores(c)["f1_weighted"]
    out = scores(cm)
    out["confusion"] = cm.tolist()
    out["skadi3"] = scores(collapse(cm))
    out["skadi3"]["confusion"] = collapse(cm).tolist()
    if per_scene:
        out["per_scene_f1"] = scene_f1
    return out


def save_checkpoint(path: Path, model: nn.Module, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **meta}, path)


def load_checkpoint(path: str | Path, device="cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], INPUT_MODES[ckpt["inputs"]], ckpt.get("n_classes", N_SOD),
                        pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
