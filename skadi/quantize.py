"""INT8 quantisation: post-training (PTQ) and quantisation-aware training (QAT).

    python -m skadi.quantize --name skadi_sar

Starting from checkpoints/<name>/best.pt this script
  1. evaluates the FP32 model on the test scenes,
  2. builds an INT8 model with plain post-training quantisation (calibration only),
  3. fine-tunes the network with fake quantisation in the loop (QAT), keeps the
     epoch with the best validation F1 and converts it to a real INT8 model,
  4. evaluates the INT8 models on the CPU and writes results/<name>/quantization.json.

Weights and activations are both 8 bit (per-channel symmetric weights, affine
activations), which is what the on-board accelerator of the Skadi document runs.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.ao.quantization import get_default_qat_qconfig_mapping, get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx

from .data import INPUT_MODES, Augment, load_cache, split_train_val
from .engine import evaluate, load_checkpoint, pick_device
from .losses import CEDiceLoss
from .train import lr_lambda

warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao")

BACKEND = "x86" if "x86" in torch.backends.quantized.supported_engines else "fbgemm"


def state_size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def summary(m: dict) -> dict:
    return {"f1_weighted": m["f1_weighted"], "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
            "f1_per_class": m["f1_per_class"], "f1_skadi3": m["skadi3"]["f1_weighted"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-cache", default="data/cache512/train")
    ap.add_argument("--test-cache", default="data/cache512/test")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--calib-scenes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = pick_device(args.device)
    torch.backends.quantized.engine = BACKEND
    ckpt_dir = Path("checkpoints") / args.name
    fp32, ckpt = load_checkpoint(ckpt_dir / "best.pt", "cpu")
    mode = ckpt["inputs"]
    in_ch = INPUT_MODES[mode]
    example = (torch.zeros(1, in_ch, 512, 512),)

    scenes = load_cache(args.train_cache)
    train_idx, val_idx = split_train_val(len(scenes), len(ckpt["val_scenes"]), seed=0)
    val = scenes.subset(val_idx)
    assert val.names == ckpt["val_scenes"], "validation split differs from the FP32 training run"
    train_idx = np.array(train_idx)
    test = load_cache(args.test_cache)

    report = {"run": args.name, "backend": BACKEND, "bits": {"weights": 8, "activations": 8}}
    report["fp32"] = {**summary(evaluate(copy.deepcopy(fp32).to(device), test, mode, device)),
                      "size_mb": state_size_mb(fp32)}
    print(f"FP32      test F1 {report['fp32']['f1_weighted']:.3f} | {report['fp32']['size_mb']:.2f} MB")

    # ---- post-training quantisation: observe activations on training scenes, then convert
    ptq = prepare_fx(copy.deepcopy(fp32).eval(), get_default_qconfig_mapping(BACKEND), example)
    calib = rng.choice(train_idx, min(args.calib_scenes, len(train_idx)), replace=False)
    with torch.no_grad():
        for start in range(0, len(calib), 8):
            ptq(scenes.inputs(torch.from_numpy(calib[start:start + 8]), mode, "cpu"))
    ptq = convert_fx(ptq)
    report["ptq_int8"] = {**summary(evaluate(ptq, test, mode, "cpu")), "size_mb": state_size_mb(ptq)}
    print(f"PTQ INT8  test F1 {report['ptq_int8']['f1_weighted']:.3f} | {report['ptq_int8']['size_mb']:.2f} MB")

    # ---- quantisation-aware training
    qat = prepare_qat_fx(copy.deepcopy(fp32).train(), get_default_qat_qconfig_mapping(BACKEND), example)
    qat.to(device)
    augment = Augment()
    criterion = CEDiceLoss(torch.tensor(ckpt["class_weights"]), ckpt["args"]["dice_weight"]).to(device)
    optim = torch.optim.AdamW(qat.parameters(), lr=args.lr, weight_decay=0.0)
    total = args.epochs * args.iters
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: lr_lambda(s, total, 0.05))
    best, best_state, history, t0 = -1.0, None, [], time.time()
    for epoch in range(1, args.epochs + 1):
        qat.train()
        if epoch > max(1, int(0.5 * args.epochs)):  # second half: frozen BatchNorm statistics
            qat.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
        if epoch > max(1, int(0.75 * args.epochs)):  # last quarter: frozen quantisation ranges
            qat.apply(torch.ao.quantization.disable_observer)
        run_loss = 0.0
        for _ in range(args.iters):
            idx = torch.from_numpy(rng.choice(train_idx, args.batch, replace=len(train_idx) < args.batch))
            x, y = augment(scenes.inputs(idx, mode, device), scenes.labels(idx, device))
            loss = criterion(qat(x), y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat.parameters(), 5.0)
            optim.step()
            sched.step()
            run_loss += loss.item()
        m = evaluate(qat, val, mode, device)  # fake-quantised forward pass
        history.append({"epoch": epoch, "loss": run_loss / args.iters, "val_f1": m["f1_weighted"]})
        flag = ""
        if m["f1_weighted"] > best:
            best, best_state, flag = m["f1_weighted"], copy.deepcopy(qat.state_dict()), " *"
        print(f"QAT epoch {epoch:2d}/{args.epochs} loss {run_loss / args.iters:.4f} | "
              f"val F1 {m['f1_weighted']:.2f} | {(time.time() - t0) / 60:.1f} min{flag}", flush=True)

    qat.load_state_dict(best_state)
    int8 = convert_fx(qat.to("cpu").eval())
    full = evaluate(int8, test, mode, "cpu", per_scene=True)
    report["qat_int8"] = {**summary(full), "size_mb": state_size_mb(int8), "val_f1": best,
                          "per_scene_f1": full["per_scene_f1"], "confusion": full["confusion"]}
    report["qat_history"] = history
    print(f"QAT INT8  test F1 {report['qat_int8']['f1_weighted']:.3f} | {report['qat_int8']['size_mb']:.2f} MB")

    torch.save(int8.state_dict(), ckpt_dir / "int8_state.pt")
    try:  # self-contained TorchScript file: runs without this code base
        torch.jit.save(torch.jit.trace(int8, example), str(ckpt_dir / "int8_scripted.pt"))
        report["torchscript"] = str(ckpt_dir / "int8_scripted.pt")
    except Exception as err:
        print(f"TorchScript export skipped: {err!r}")
    out = Path("results") / args.name / "quantization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
