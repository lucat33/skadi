"""Run the whole study with one command (from the repository root):

    python scripts/run_experiments.py                # main runs
    python scripts/run_experiments.py --ablations    # + ablation runs
    python scripts/run_experiments.py --quick        # 2-minute dry run to check the setup

Steps: build the 512x512 cache -> train -> evaluate on the 20 test scenes ->
INT8 quantisation -> tables and figures in docs/. Finished steps are skipped,
so the script can be stopped and restarted at any time.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name -> extra arguments of skadi.train
MAIN_RUNS = {
    "skadi_sar": [],  # the model of the Skadi document: HH + HV only
    "tinyicenet_repro": ["--arch", "tinyicenet"],  # baseline architecture, same recipe
    "skadi_geo": ["--inputs", "sar+geo"],  # + incidence angle, position and month
}
ABLATIONS = {
    "abl_scratch": ["--no-pretrained"],
    "abl_noaug": ["--no-augment"],
    "abl_ce_only": ["--dice-weight", "0", "--weight-power", "0", "--myi-boost", "1"],
    "abl_ia": ["--inputs", "sar+ia"],
}
QUANTIZE = ["skadi_sar", "tinyicenet_repro"]


def run(*cmd):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"), help="folder with the training .nc scenes")
    ap.add_argument("--test-data", type=Path, default=Path("data/test"), help="folder with the test .nc scenes")
    ap.add_argument("--cache", type=Path, default=Path("data/cache512"))
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny budgets, results are meaningless")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16, help="lower it if the GPU runs out of memory")
    args = ap.parse_args()

    train_cache, test_cache = args.cache / "train", args.cache / "test"
    if not (ROOT / args.test_data).exists() or not list((ROOT / args.test_data).glob("*_reference.nc")):
        sys.exit(f"Test scenes not found in {args.test_data}. Download them first:\n"
                 "    python scripts/download_data.py test")
    run("-m", "skadi.preprocess", "--src", args.data, "--dst", train_cache, "--workers", args.workers)
    run("-m", "skadi.preprocess", "--src", args.test_data, "--dst", test_cache, "--workers", args.workers)

    budget = ["--epochs", "2", "--iters", "10", "--batch", "4"] if args.quick else ["--batch", args.batch]
    qbudget = budget
    runs = {**MAIN_RUNS, **(ABLATIONS if args.ablations else {})}
    common = ["--train-cache", train_cache]
    for name, extra in runs.items():
        if not (ROOT / "checkpoints" / name / "last.pt").exists():
            try:
                run("-m", "skadi.train", "--name", name, *common, *extra, *budget)
            except subprocess.CalledProcessError:
                if "geo" in name or "ia" in name:  # scenes without incidence angle / lat-lon
                    print(f"skipping {name}: auxiliary channels not available")
                    continue
                raise
        if not (ROOT / "results" / name / "test_metrics.json").exists():
            run("-m", "skadi.evaluate", "--name", name, "--test-cache", test_cache,
                "--native-src", args.test_data)
        if not (ROOT / "results" / name / "test_metrics_tta.json").exists():
            run("-m", "skadi.evaluate", "--name", name, "--test-cache", test_cache, "--tta")
    for name in QUANTIZE:
        if not (ROOT / "results" / name / "quantization.json").exists():
            run("-m", "skadi.quantize", "--name", name, *common, "--test-cache", test_cache, *qbudget)
    run("scripts/make_report.py")
    print("\nAll done. Tables: docs/RESULTS.md, figures: docs/figures/.\n"
          "Commit the results/ and docs/ folders to keep them with the code.")


if __name__ == "__main__":
    main()
