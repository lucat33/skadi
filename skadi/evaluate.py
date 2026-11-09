"""Evaluate a checkpoint on the AI4Arctic test scenes with the TinyIceNet protocol.

    python -m skadi.evaluate --name skadi_sar

Writes results/<name>/test_metrics.json. The headline number is the weighted F1
over all labelled pixels of the 20 test scenes, computed on the 512x512 resized
scenes exactly like the baseline. With --native-src the predictions are also
upsampled to the original 80 m grid and scored against the full-resolution chart
(the official AutoICE way).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .constants import N_SOD, TINYICENET_PAPER
from .data import INPUT_MODES, load_cache
from .engine import evaluate, load_checkpoint, pick_device, predict_logits
from .metrics import collapse, confusion, scores
from .models import count_macs, count_params


@torch.no_grad()
def cpu_latency(model, in_ch: int, repeats: int = 10) -> float:
    """Median single-scene latency in ms on the CPU (4 threads at most)."""
    model = model.to("cpu").eval()
    x = torch.zeros(1, in_ch, 512, 512)
    times = []
    for _ in range(repeats + 2):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1000)
    return float(sorted(times[2:])[len(times[2:]) // 2])


@torch.no_grad()
def native_resolution_scores(model, scenes, mode, device, src: Path, tta: bool) -> dict:
    from .preprocess import open_scene, read_sod

    cm = torch.zeros(N_SOD, N_SOD, dtype=torch.long)
    for i, name in enumerate(scenes.names):
        path = src / f"{name}.nc"
        with open_scene(path) as ds:
            sod = read_sod(ds, path)
        if sod is None:
            continue
        x = scenes.inputs(torch.tensor([i]), mode, device)
        logits = predict_logits(model, x, tta).cpu()
        pred = F.interpolate(logits, size=sod.shape, mode="bilinear", align_corners=False).argmax(1)[0]
        cm += confusion(pred, torch.from_numpy(sod.astype("int64")))
    out = scores(cm)
    out["skadi3"] = scores(collapse(cm))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", default=None, help="default: checkpoints/<name>/best.pt")
    ap.add_argument("--test-cache", default="data/cache512/test")
    ap.add_argument("--native-src", type=Path, default=None, help="folder with the test .nc files")
    ap.add_argument("--tta", action="store_true", help="average the four flip variants")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    ckpt_path = args.checkpoint or Path("checkpoints") / args.name / "best.pt"
    model, ckpt = load_checkpoint(ckpt_path, device)
    mode = ckpt["inputs"]
    in_ch = INPUT_MODES[mode]
    scenes = load_cache(args.test_cache)

    m = evaluate(model, scenes, mode, device, tta=args.tta, per_scene=True)
    m.update({
        "run": args.name, "arch": ckpt["arch"], "inputs": mode, "tta": args.tta,
        "checkpoint_epoch": ckpt.get("epoch"), "val_f1": ckpt.get("val", {}).get("f1_weighted"),
        "test_scenes": len(scenes),
        "params": count_params(model), "gmacs": count_macs(model, in_ch) / 1e9,
        "tinyicenet_paper": TINYICENET_PAPER,
    })
    if args.native_src is not None:
        m["native_80m"] = native_resolution_scores(model, scenes, mode, device, args.native_src, args.tta)
    m["cpu_latency_ms"] = cpu_latency(model, in_ch)

    out = Path("results") / args.name / ("test_metrics_tta.json" if args.tta else "test_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    delta = m["f1_weighted"] - TINYICENET_PAPER["f1_qat_int8"]
    print(f"{args.name}: test F1 (weighted, 6 SOD classes) = {m['f1_weighted']:.3f}  "
          f"[TinyIceNet paper: {TINYICENET_PAPER['f1_qat_int8']:.3f}, difference {delta:+.3f}]")
    print(f"  macro F1 {m['f1_macro']:.2f} | accuracy {m['accuracy']:.2f} | "
          f"OW/FYI/MYI F1 {m['skadi3']['f1_weighted']:.2f}")
    print(f"  {m['params'] / 1e3:.1f} k parameters | {m['gmacs']:.2f} GMACs | "
          f"CPU latency {m['cpu_latency_ms']:.0f} ms")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
