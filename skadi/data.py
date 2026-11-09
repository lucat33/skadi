"""In-memory dataset and GPU-side augmentation.

The 512x512 cache of the whole training set is about 1 GB, so we keep it in RAM
as tensors and draw random batches by indexing. This avoids DataLoader worker
processes entirely (which are slow to spawn on Windows) and lets every
augmentation run batched on the GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import IGNORE, N_SOD

# channels of every input configuration
INPUT_MODES = {
    "sar": 2,  # HH, HV                                  (the configuration of the Skadi document)
    "sar+ia": 3,  # + incidence angle
    "sar+geo": 8,  # + incidence angle, latitude, lon sin/cos, month sin/cos
}


@dataclass
class Scenes:
    names: list[str]
    sar: torch.Tensor  # (N, 2, H, W) float16
    sod: torch.Tensor  # (N, H, W) uint8, 255 = ignore
    aux: torch.Tensor | None  # (N, 6, H, W) float16: ia, lat, lon_sin, lon_cos, month_sin, month_cos
    orig_shape: np.ndarray  # (N, 2)

    def __len__(self):
        return len(self.names)

    def subset(self, idx) -> "Scenes":
        idx = list(idx)
        t = torch.as_tensor(idx, dtype=torch.long)
        return Scenes([self.names[i] for i in idx], self.sar[t], self.sod[t],
                      None if self.aux is None else self.aux[t], self.orig_shape[idx])

    def inputs(self, idx: torch.Tensor, mode: str, device) -> torch.Tensor:
        x = self.sar[idx].to(device, non_blocking=True).float()
        if mode == "sar":
            return x
        if self.aux is None:
            raise RuntimeError(f"input mode '{mode}' needs incidence angle / lat-lon in the cache, "
                               "but the scenes do not provide them")
        aux = self.aux[idx].to(device, non_blocking=True).float()
        return torch.cat([x, aux[:, :1] if mode == "sar+ia" else aux], dim=1)

    def labels(self, idx: torch.Tensor, device) -> torch.Tensor:
        return self.sod[idx].to(device, non_blocking=True).long()


def load_cache(folder: str | Path, need_labels: bool = True, verbose: bool = True) -> Scenes:
    files = sorted(Path(folder).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz scenes in {folder}: run `python -m skadi.preprocess` first")
    names, sar, sod, aux, shapes = [], [], [], [], []
    have_aux = True
    for f in files:
        with np.load(f) as z:
            if "sod" not in z:
                if need_labels:
                    continue
                lab = np.full(z["sar"].shape[1:], IGNORE, np.uint8)
            else:
                lab = z["sod"]
            names.append(f.stem)
            sar.append(z["sar"])
            sod.append(lab)
            shapes.append(z["orig_shape"])
            if have_aux and "ia" in z and "lat" in z:
                h, w = lab.shape
                ang = 2 * math.pi * (int(z["month"]) - 1) / 12.0
                aux.append(np.stack([
                    z["ia"].astype(np.float32),
                    (z["lat"].astype(np.float32) - 70.0) / 10.0,
                    z["lon_sin"].astype(np.float32),
                    z["lon_cos"].astype(np.float32),
                    np.full((h, w), math.sin(ang), np.float32),
                    np.full((h, w), math.cos(ang), np.float32),
                ]).astype(np.float16))
            else:
                have_aux = False
    scenes = Scenes(names, torch.from_numpy(np.stack(sar)), torch.from_numpy(np.stack(sod)),
                    torch.from_numpy(np.stack(aux)) if have_aux else None, np.stack(shapes))
    if verbose:
        print(f"loaded {len(scenes)} scenes from {folder} (aux channels: {have_aux})")
    return scenes


def split_train_val(n: int, n_val: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """Deterministic hold-out of `n_val` training scenes used for model selection."""
    perm = np.random.RandomState(seed).permutation(n)
    return sorted(perm[n_val:].tolist()), sorted(perm[:n_val].tolist())


def class_pixel_counts(sod: torch.Tensor) -> torch.Tensor:
    flat = sod.reshape(-1).long()
    return torch.bincount(flat[flat != IGNORE], minlength=N_SOD).float()


class Augment(torch.nn.Module):
    """Batched augmentation of (inputs, labels) on the training device.

    Radiometric part (only on the HH/HV channels, expressed in the normalised dB
    units of the ready-to-train dataset):
      * incidence-angle-dependent intensity scaling: a random linear ramp along
        range, stronger on HH than on HV as observed for sea ice backscatter;
      * global gain offset (calibration differences between sensors/modes);
      * speckle injection: multiplicative Gamma noise with a random number of looks.
    Geometric part: flips, free rotation, anisotropic zoom and translation. The
    anisotropic zoom matters because resizing every scene to 512x512 stretches
    each scene by a different amount.
    """

    def __init__(self, ramp_std=0.15, gain_std=0.10, speckle_p=0.5, looks=(8.0, 64.0),
                 db_per_unit=5.0, scale=(0.7, 1.4), aniso=0.15, shift=0.15):
        super().__init__()
        self.ramp_std, self.gain_std = ramp_std, gain_std
        self.speckle_p, self.looks, self.db_per_unit = speckle_p, looks, db_per_unit
        self.scale, self.aniso, self.shift = scale, aniso, shift

    def radiometric(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        dev = x.device
        sar = x[:, :2]
        ramp = torch.linspace(-1, 1, w, device=dev).view(1, 1, 1, w)
        slope = torch.randn(b, 1, 1, 1, device=dev) * self.ramp_std
        pol = torch.tensor([1.0, 0.5], device=dev).view(1, 2, 1, 1)
        sar = sar + slope * pol * ramp + torch.randn(b, 2, 1, 1, device=dev) * self.gain_std
        if self.speckle_p > 0:
            lo, hi = self.looks
            looks = torch.exp(torch.empty(b, device=dev).uniform_(math.log(lo), math.log(hi)))
            gamma = torch.distributions.Gamma(looks, looks)  # mean 1, variance 1/L
            noise = gamma.sample((2, h, w)).permute(3, 0, 1, 2).clamp_min(1e-3)
            noise_db = 10.0 * torch.log10(noise) / self.db_per_unit
            on = (torch.rand(b, 1, 1, 1, device=dev) < self.speckle_p).float()
            sar = sar + on * noise_db
        return torch.cat([sar, x[:, 2:]], dim=1)

    def geometric(self, x: torch.Tensor, y: torch.Tensor):
        b = x.shape[0]
        dev = x.device
        theta = torch.empty(b, device=dev).uniform_(0, 2 * math.pi)
        lo, hi = self.scale
        zoom = torch.exp(torch.empty(b, device=dev).uniform_(math.log(lo), math.log(hi)))
        an = torch.exp(torch.empty(b, device=dev).uniform_(-self.aniso, self.aniso))
        sx, sy = zoom * an, zoom / an
        flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
        cos, sin = torch.cos(theta), torch.sin(theta)
        # output -> input mapping: a zoom factor > 1 magnifies the scene
        mat = torch.zeros(b, 2, 3, device=dev)
        mat[:, 0, 0] = cos / sx * flip
        mat[:, 0, 1] = -sin / sy
        mat[:, 1, 0] = sin / sx * flip
        mat[:, 1, 1] = cos / sy
        mat[:, :, 2] = torch.empty(b, 2, device=dev).uniform_(-self.shift, self.shift)
        grid = F.affine_grid(mat, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        # labels: nearest neighbour; shifting by one makes "outside the scene" (0) recognisable
        yy = F.grid_sample((y.float() + 1).unsqueeze(1), grid, mode="nearest",
                           padding_mode="zeros", align_corners=False).squeeze(1)
        y = torch.where(yy < 0.5, torch.full_like(yy, IGNORE + 1), yy).long() - 1
        return x, y

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.radiometric(x)
        return self.geometric(x, y)
