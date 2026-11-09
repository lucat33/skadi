"""Train a sea-ice segmentation network on the 512x512 AI4Arctic cache.

    python -m skadi.train --name skadi_sar
    python -m skadi.train --name tinyicenet_repro --arch tinyicenet --no-pretrained

Weights go to checkpoints/<name>/ (not versioned), logs and metrics to
results/<name>/ (versioned).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import INPUT_MODES, Augment, class_pixel_counts, load_cache, split_train_val
from .engine import EMA, evaluate, pick_device, save_checkpoint
from .losses import CEDiceLoss, class_weights
from .models import build_model, count_macs, count_params


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name")
    ap.add_argument("--train-cache", default="data/cache512/train")
    ap.add_argument("--arch", default="skadi", choices=["skadi", "tinyicenet"])
    ap.add_argument("--inputs", default="sar", choices=list(INPUT_MODES))
    ap.add_argument("--no-pretrained", action="store_true", help="do not start from ImageNet weights")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--iters", type=int, default=200, help="iterations per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03, help="fraction of the run used for warm-up")
    ap.add_argument("--dice-weight", type=float, default=0.5)
    ap.add_argument("--weight-power", type=float, default=0.5, help="0 = unweighted CE, 1 = inverse frequency")
    ap.add_argument("--myi-boost", type=float, default=1.5)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--val-scenes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    return ap.parse_args(argv)


def lr_lambda(step: int, total: int, warmup: float, floor: float = 0.01) -> float:
    w = max(1, int(total * warmup))
    if step < w:
        return (step + 1) / w
    t = (step - w) / max(1, total - w)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))


def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    scenes = load_cache(args.train_cache)
    train_idx, val_idx = split_train_val(len(scenes), args.val_scenes, seed=0)  # split never changes
    val = scenes.subset(val_idx)
    train_idx = np.array(train_idx)
    counts = class_pixel_counts(scenes.sod[torch.as_tensor(train_idx)])
    weights = class_weights(counts, args.weight_power, args.myi_boost)
    print("train scenes:", len(train_idx), "| val scenes:", len(val))
    print("class pixel share (%):", (100 * counts / counts.sum()).round(decimals=2).tolist())
    print("loss class weights   :", weights.round(decimals=3).tolist())

    in_ch = INPUT_MODES[args.inputs]
    model = build_model(args.arch, in_ch, pretrained=not args.no_pretrained).to(device)
    n_params, macs = count_params(model), count_macs(model, in_ch)
    print(f"{args.arch}: {n_params / 1e3:.1f} k parameters, {macs / 1e9:.2f} GMACs @ {in_ch}x512x512")
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    ema = EMA(model)
    augment = None if args.no_augment else Augment()
    criterion = CEDiceLoss(weights, args.dice_weight, args.label_smoothing).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * args.iters
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: lr_lambda(s, total, args.warmup))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    out_dir = Path("results") / args.name
    ckpt_dir = Path("checkpoints") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"arch": args.arch, "inputs": args.inputs, "n_classes": 6, "args": vars(args),
            "params": n_params, "macs": macs, "class_weights": weights.tolist(),
            "val_scenes": val.names,
            # scene-level radiometric statistics of the training set, used by the
            # out-of-distribution check at inference time (Skadi document, Sec. 3.5)
            "scene_stats": scene_statistics(scenes.sar[torch.as_tensor(train_idx)])}
    (out_dir / "config.json").write_text(json.dumps({k: v for k, v in meta.items()}, indent=2))
    log = (out_dir / "log.jsonl").open("w")

    best, t0 = -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for _ in range(args.iters):
            idx = torch.from_numpy(rng.choice(train_idx, args.batch, replace=len(train_idx) < args.batch))
            x = scenes.inputs(idx, args.inputs, device)
            y = scenes.labels(idx, device)
            if augment is not None:
                x, y = augment(x, y)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            ema.update(model)
            run_loss += loss.item()

        with torch.autocast(device.type, enabled=use_amp):
            m = evaluate(ema.model, val, args.inputs, device)
        rec = {"epoch": epoch, "loss": run_loss / args.iters, "lr": sched.get_last_lr()[0],
               "val_f1": m["f1_weighted"], "val_f1_macro": m["f1_macro"],
               "val_f1_skadi3": m["skadi3"]["f1_weighted"], "minutes": (time.time() - t0) / 60}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        flag = ""
        if m["f1_weighted"] > best:
            best = m["f1_weighted"]
            save_checkpoint(ckpt_dir / "best.pt", ema.model, {**meta, "epoch": epoch, "val": m})
            flag = " *"
        print(f"epoch {epoch:3d}/{args.epochs} loss {rec['loss']:.4f} | val F1 {m['f1_weighted']:.2f} "
              f"(macro {m['f1_macro']:.2f}, 3-class {m['skadi3']['f1_weighted']:.2f}) "
              f"| {rec['minutes']:.1f} min{flag}", flush=True)
    save_checkpoint(ckpt_dir / "last.pt", ema.model, {**meta, "epoch": args.epochs, "val": m})
    print(f"best validation F1: {best:.3f} -> {ckpt_dir / 'best.pt'}")


def scene_statistics(sar: torch.Tensor) -> dict:
    """Mean and spread over scenes of the per-scene HH/HV mean and standard deviation."""
    x = sar.float()
    feats = torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)  # (N, 4)
    return {"mean": feats.mean(0).tolist(), "std": feats.std(0).clamp_min(1e-3).tolist()}


if __name__ == "__main__":
    main()
