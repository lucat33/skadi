"""Run the trained network on one scene and produce the Skadi ice product.

    python -m skadi.infer --checkpoint checkpoints/skadi_sar/best.pt \
        --scene data/20180808T075536_dmi_prep.nc --out products/

For every scene it writes
  * <scene>.geojson  class-labelled polygons (OW / FYI / MYI) with a confidence score,
  * <scene>.npz      raster class map, 6-class SOD map and per-pixel confidence,
  * <scene>.png      quick-look (HH, HV, ice map, confidence).

This is the software counterpart of steps 3 and 5 of the on-board pipeline
(tiling/inference and vectorisation) plus the confidence and out-of-distribution
flags of Sec. 3.5 of the Skadi document.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import SKADI_CLASSES, SOD_TO_SKADI
from .data import INPUT_MODES
from .engine import load_checkpoint, pick_device
from .preprocess import HH, HV, open_scene, resize_area


def blend_window(tile: int, device) -> torch.Tensor:
    """Separable raised-cosine weights: tile borders count less than tile centres."""
    w = torch.hann_window(tile + 2, periodic=False, device=device)[1:-1]
    return (w[:, None] * w[None, :]).clamp_min(1e-3)


@torch.no_grad()
def tiled_predict(model, x: torch.Tensor, tile: int = 512, overlap: int = 128) -> torch.Tensor:
    """Class probabilities for an image of any size, from overlapping tiles blended together."""
    _, _, h, w = x.shape
    ph, pw = max(tile, h), max(tile, w)
    x = F.pad(x, (0, pw - w, 0, ph - h))
    stride = tile - overlap

    def starts(n):
        s = list(range(0, n - tile + 1, stride))
        if s[-1] != n - tile:
            s.append(n - tile)
        return s

    win = blend_window(tile, x.device)
    acc, norm = None, torch.zeros(1, 1, ph, pw, device=x.device)
    for top in starts(ph):
        for left in starts(pw):
            p = model(x[:, :, top:top + tile, left:left + tile]).float().softmax(1)
            if acc is None:
                acc = torch.zeros(1, p.shape[1], ph, pw, device=x.device)
            acc[:, :, top:top + tile, left:left + tile] += p * win
            norm[:, :, top:top + tile, left:left + tile] += win
    return (acc / norm)[:, :, :h, :w]


def confidence_from_probs(p: torch.Tensor) -> torch.Tensor:
    """1 - normalised entropy of the class distribution: 1 = certain, 0 = uniform."""
    ent = -(p * p.clamp_min(1e-8).log()).sum(1)
    return 1 - ent / math.log(p.shape[1])


def ood_score(sar: np.ndarray, stats: dict | None) -> float | None:
    """Largest z-score of the scene HH/HV mean and std w.r.t. the training scenes."""
    if not stats:
        return None
    feats = np.concatenate([sar.mean(axis=(1, 2)), sar.std(axis=(1, 2))])
    return float(np.abs((feats - np.array(stats["mean"])) / np.array(stats["std"])).max())


def vectorize(class_map: np.ndarray, conf: np.ndarray, scale_xy, tolerance: float = 1.5,
              min_area: int = 16, low_conf: float = 0.5) -> dict:
    """Raster -> simplified polygons (Douglas-Peucker). Coordinates are scene pixels (x, y)."""
    import cv2

    feats = []
    for cls, name in enumerate(SKADI_CLASSES):
        mask = (class_map == cls).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1 or cv2.contourArea(cnt) < min_area:
                continue  # holes are attached to their parent below
            rings = [cnt]
            child = hierarchy[i][2]
            while child != -1:
                if cv2.contourArea(contours[child]) >= min_area:
                    rings.append(contours[child])
                child = hierarchy[child][0]
            region = np.zeros_like(mask)
            cv2.drawContours(region, rings, -1, 1, thickness=cv2.FILLED)
            score = float(conf[(region > 0) & (mask > 0)].mean())
            coords = []
            for ring in rings:
                simple = cv2.approxPolyDP(ring, tolerance, True)[:, 0, :].astype(float)
                if len(simple) < 3:
                    continue
                pts = (simple * np.array(scale_xy)).round(1).tolist()
                coords.append(pts + [pts[0]])
            if coords:
                feats.append({"type": "Feature",
                              "properties": {"class": name, "confidence": round(score, 3),
                                             "low_confidence": score < low_conf},
                              "geometry": {"type": "Polygon", "coordinates": coords}})
    return {"type": "FeatureCollection", "features": feats}


def load_scene_inputs(path: Path, mode: str, factor: int | None):
    """Network input for a ready-to-train scene: resized to 512x512, or downsampled by `factor`."""
    if mode != "sar":
        raise NotImplementedError("infer.py currently feeds HH/HV only; use a '--inputs sar' model")
    with open_scene(path) as ds:
        hh = np.asarray(ds[HH].values, dtype=np.float32)
        hv = np.asarray(ds[HV].values, dtype=np.float32)
    if factor is None:
        sar = np.stack([resize_area(hh)[0], resize_area(hv)[0]])
    else:
        x = torch.from_numpy(np.nan_to_num(np.stack([hh, hv])))[None]
        sar = F.avg_pool2d(x, factor, ceil_mode=True)[0].numpy()
    return sar.astype(np.float32), hh.shape


def quicklook(path: Path, sar, cls_map, conf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    ax[0].imshow(sar[0], cmap="gray", vmin=-2.5, vmax=2.5); ax[0].set_title("HH")
    ax[1].imshow(sar[1], cmap="gray", vmin=-2.5, vmax=2.5); ax[1].set_title("HV")
    cmap = ListedColormap(["#1f4e79", "#9ecae1", "#f7f7f7"])
    im = ax[2].imshow(cls_map, cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
    ax[2].set_title("ice map")
    cb = fig.colorbar(im, ax=ax[2], ticks=[0, 1, 2], fraction=0.046)
    cb.ax.set_yticklabels(SKADI_CLASSES)
    im = ax[3].imshow(conf, cmap="magma", vmin=0, vmax=1); ax[3].set_title("confidence")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", type=Path, required=True, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("products"))
    ap.add_argument("--factor", type=int, default=None,
                    help="downsample by this factor and tile instead of resizing to 512x512 "
                         "(10 turns the 80 m grid into the 800 m the network was trained on)")
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--tolerance", type=float, default=1.5, help="Douglas-Peucker tolerance in map pixels")
    ap.add_argument("--ood-threshold", type=float, default=4.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    group = torch.zeros(len(SKADI_CLASSES), len(SOD_TO_SKADI), device=device)
    for sod, skadi in enumerate(SOD_TO_SKADI):
        group[skadi, sod] = 1
    args.out.mkdir(parents=True, exist_ok=True)

    for scene in args.scene:
        sar, orig_shape = load_scene_inputs(scene, ckpt["inputs"], args.factor)
        x = torch.from_numpy(sar)[None].to(device)
        with torch.no_grad():
            p6 = tiled_predict(model, x, overlap=args.overlap) if args.factor else model(x).float().softmax(1)
        p3 = torch.einsum("ks,bshw->bkhw", group, p6)
        cls3 = p3.argmax(1)[0].cpu().numpy().astype(np.uint8)
        conf = confidence_from_probs(p3)[0].cpu().numpy()
        score = ood_score(sar, ckpt.get("scene_stats"))
        h, w = cls3.shape
        product = vectorize(cls3, conf, (orig_shape[1] / w, orig_shape[0] / h), args.tolerance)
        product["properties"] = {
            "scene": scene.stem, "scene_shape": list(orig_shape), "map_shape": [h, w],
            "mean_confidence": round(float(conf.mean()), 3),
            "ood_score": None if score is None else round(score, 2),
            "out_of_distribution": bool(score is not None and score > args.ood_threshold),
        }
        (args.out / f"{scene.stem}.geojson").write_text(json.dumps(product))
        np.savez_compressed(args.out / f"{scene.stem}.npz", ice=cls3, sod=p6.argmax(1)[0].cpu().numpy().astype(np.uint8),
                            confidence=conf.astype(np.float16))
        quicklook(args.out / f"{scene.stem}.png", sar, cls3, conf)
        size_kb = (args.out / f"{scene.stem}.geojson").stat().st_size / 1024
        print(f"{scene.stem}: {len(product['features'])} polygons, {size_kb:.0f} kB, "
              f"mean confidence {conf.mean():.2f}, OOD score {score if score is None else round(score, 2)}")


if __name__ == "__main__":
    main()
