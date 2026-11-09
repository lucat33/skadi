"""Build docs/RESULTS.md and the figures in docs/figures/ from the files in results/.

    python scripts/make_report.py

Every number in the tables is read from the JSON files written by
skadi.evaluate and skadi.quantize: nothing is typed by hand.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
RESULTS, DOCS, FIGS = ROOT / "results", ROOT / "docs", ROOT / "docs" / "figures"
PAPER = {"f1_fp32": 75.168, "f1_qat_int8": 75.216, "params": 146_600, "gmacs": 2.97}
SOD = ["open water", "new ice", "young ice", "thin FYI", "thick FYI", "old ice"]
LABELS = {
    "skadi_sar": "Skadi U-Net (HH, HV)",
    "tinyicenet_repro": "TinyIceNet, our re-training",
    "skadi_geo": "Skadi U-Net + geometry/position/month",
    "abl_scratch": "Skadi U-Net, no ImageNet init",
    "abl_noaug": "Skadi U-Net, no augmentation",
    "abl_ce_only": "Skadi U-Net, plain cross-entropy",
    "abl_ia": "Skadi U-Net + incidence angle",
}
# fixed colour per entity (never per rank)
COLOR = {"skadi_sar": "#2a78d6", "tinyicenet_repro": "#eb6834", "skadi_geo": "#1baf7a",
         "paper": "#52514e"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def load(name, file):
    path = RESULTS / name / file
    return json.loads(path.read_text()) if path.exists() else None


def label(name):
    return LABELS.get(name, name)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def fig_curves(runs):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    shown = 0
    for name in runs:
        log = RESULTS / name / "log.jsonl"
        if name not in COLOR or not log.exists():
            continue
        rec = [json.loads(line) for line in log.read_text().splitlines() if line]
        ax.plot([r["epoch"] for r in rec], [r["val_f1"] for r in rec], color=COLOR[name], lw=2,
                label=label(name))
        shown += 1
    if not shown:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation F1 (%)")
    ax.set_title("Validation F1 during training (30 held-out training scenes)")
    ax.legend(loc="lower right")
    return save(fig, "training_curves.png")


def fig_per_class(metrics):
    names = [n for n in ("skadi_sar", "tinyicenet_repro") if n in metrics]
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        vals = metrics[name]["f1_per_class"]
        x = np.arange(len(SOD)) + (k - (len(names) - 1) / 2) * width
        ax.bar(x, vals, width * 0.92, color=COLOR[name], label=label(name))
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks(np.arange(len(SOD)), SOD)
    ax.set_ylim(0, 105)
    ax.set_ylabel("F1 (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Per-class F1 on the 20 test scenes")
    ax.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.12))
    return save(fig, "per_class_f1.png")


def fig_confusion(m, name):
    cm = np.array(m["confusion"], dtype=float)
    rows = cm.sum(1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0) * 100
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, "#86b6ef", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=100)
    ax.grid(False)
    for i in range(6):
        for j in range(6):
            if rows[i, 0] > 0:
                ax.text(j, i, f"{norm[i, j]:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 55 else INK)
    ax.set_xticks(range(6), SOD, rotation=35, ha="right")
    ax.set_yticks(range(6), SOD)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ice chart")
    ax.set_title(f"{label(name)}: confusion matrix (% of each chart class)")
    return save(fig, f"confusion_{name}.png")


def fig_tradeoff(metrics, quant):
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.scatter([PAPER["gmacs"]], [PAPER["f1_qat_int8"]], s=70, color=COLOR["paper"], marker="D", zorder=3)
    ax.annotate("TinyIceNet (paper, INT8 weights)", (PAPER["gmacs"], PAPER["f1_qat_int8"]),
                textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=9)
    for name in ("skadi_sar", "tinyicenet_repro", "skadi_geo"):
        if name not in metrics:
            continue
        m = metrics[name]
        f1 = quant[name]["qat_int8"]["f1_weighted"] if name in quant else m["f1_weighted"]
        tag = "INT8" if name in quant else "FP32"
        ax.scatter([m["gmacs"]], [f1], s=80, color=COLOR[name], zorder=3)
        ax.annotate(f"{label(name)} ({tag})", (m["gmacs"], f1), textcoords="offset points",
                    xytext=(8, 6), fontsize=9)
    ax.set_xlim(0, 3.6)
    ax.set_xlabel("multiply-accumulates per 512x512 scene (G)")
    ax.set_ylabel("test F1 (%)")
    ax.set_title("Accuracy against computational cost")
    return save(fig, "accuracy_vs_cost.png")


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=160)
    plt.close(fig)
    return f"figures/{name}"


def main():
    runs = [n for n in LABELS if (RESULTS / n).exists()]
    runs += sorted(p.name for p in RESULTS.glob("*") if p.is_dir() and p.name not in LABELS)
    metrics = {n: m for n in runs if (m := load(n, "test_metrics.json"))}
    tta = {n: m for n in runs if (m := load(n, "test_metrics_tta.json"))}
    quant = {n: q for n in runs if (q := load(n, "quantization.json"))}
    if not metrics:
        raise SystemExit("no results/<run>/test_metrics.json found: run scripts/run_experiments.py first")

    md = ["# Results", "",
          "Generated by `scripts/make_report.py` from the JSON files in `results/`. "
          "F1 is the weighted F1 over the six SOD classes on all labelled pixels of the 20 AI4Arctic "
          "test scenes, computed on the 512x512 resized scenes (the protocol of TinyIceNet).", ""]

    rows = [["TinyIceNet, as reported in the paper (FP32)", "146.6 k", "2.97", f"{PAPER['f1_fp32']:.2f}",
             "n/a", "n/a", "n/a", "n/a"]]
    for n, m in metrics.items():
        native = m.get("native_80m", {}).get("f1_weighted")
        rows.append([label(n), f"{m['params'] / 1e3:.1f} k", f"{m['gmacs']:.2f}", f"{m['f1_weighted']:.2f}",
                     f"{tta[n]['f1_weighted']:.2f}" if n in tta else "n/a",
                     f"{native:.2f}" if native is not None else "n/a",
                     f"{m['f1_macro']:.2f}", f"{m['skadi3']['f1_weighted']:.2f}"])
    md += ["## Full-precision models", "",
           table(["Model", "Parameters", "GMACs", "F1", "F1 with flip TTA", "F1 at 80 m", "Macro F1",
                  "OW/FYI/MYI F1"], rows), ""]

    if quant:
        rows = [["TinyIceNet, as reported in the paper", f"{PAPER['f1_fp32']:.2f}", "24.32",
                 f"{PAPER['f1_qat_int8']:.2f}", "n/a"]]
        for n, q in quant.items():
            rows.append([label(n), f"{q['fp32']['f1_weighted']:.2f}", f"{q['ptq_int8']['f1_weighted']:.2f}",
                         f"{q['qat_int8']['f1_weighted']:.2f}", f"{q['qat_int8']['size_mb']:.2f} MB"])
        md += ["## INT8 quantisation", "",
               "Our INT8 models quantise weights and activations to 8 bit; the paper quantises the weights.", "",
               table(["Model", "FP32 F1", "PTQ INT8 F1", "QAT INT8 F1", "INT8 size"], rows), ""]

    main_run = "skadi_sar" if "skadi_sar" in metrics else next(iter(metrics))
    m = metrics[main_run]
    rows = [[SOD[c], f"{m['support'][c]:,}", f"{m['precision_per_class'][c]:.1f}",
             f"{m['recall_per_class'][c]:.1f}", f"{m['f1_per_class'][c]:.1f}"] for c in range(6)]
    md += [f"## Per-class scores of {label(main_run)}", "",
           table(["Class", "Test pixels", "Precision", "Recall", "F1"], rows), ""]
    s3 = m["skadi3"]
    rows = [[c, f"{s3['support'][i]:,}", f"{s3['precision_per_class'][i]:.1f}",
             f"{s3['recall_per_class'][i]:.1f}", f"{s3['f1_per_class'][i]:.1f}"]
            for i, c in enumerate(["OW", "FYI", "MYI"])]
    md += ["Skadi product classes (probabilities of the SOD classes summed per group):", "",
           table(["Class", "Test pixels", "Precision", "Recall", "F1"], rows), ""]
    if "per_scene_f1" in m:
        rows = [[k, f"{v:.1f}"] for k, v in sorted(m["per_scene_f1"].items())]
        md += ["## Per-scene F1", "", table(["Test scene", "F1"], rows), ""]

    figs = [fig_curves(runs), fig_per_class(metrics), fig_confusion(m, main_run), fig_tradeoff(metrics, quant)]
    md += ["## Figures", ""] + [f"![{Path(f).stem}]({f})\n" for f in figs if f]
    DOCS.mkdir(exist_ok=True)
    (DOCS / "RESULTS.md").write_text("\n".join(md))
    print(f"wrote {DOCS / 'RESULTS.md'} and {sum(f is not None for f in figs)} figures")


if __name__ == "__main__":
    main()
