"""Create small fake scenes with the same layout as AI4Arctic ready-to-train files.

Only for checking that the code runs end to end (tests, CI, a laptop without the
dataset). The backscatter statistics are invented: numbers obtained on these
scenes say nothing about real sea ice.

    python scripts/make_synthetic_data.py --dst data_synth --train 12 --test 3
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

# (HH mean, HV mean) per SOD class in normalised units, loosely ordered like real ice
CLASS_STATS = [(-1.4, -0.8), (-0.9, -0.7), (-0.4, -0.5), (0.0, -0.2), (0.4, 0.2), (0.9, 1.0)]


def make_scene(rng, lines, samples):
    field = gaussian_filter(rng.randn(lines, samples), sigma=min(lines, samples) / 12)
    field = (field - field.min()) / (field.max() - field.min())
    present = np.sort(rng.choice(6, size=rng.randint(2, 5), replace=False))
    sod = present[np.minimum((field * len(present)).astype(int), len(present) - 1)].astype(np.uint8)
    hh = np.zeros((lines, samples), np.float32)
    hv = np.zeros((lines, samples), np.float32)
    for c, (m_hh, m_hv) in enumerate(CLASS_STATS):
        hh[sod == c], hv[sod == c] = m_hh, m_hv
    ramp = np.linspace(0.3, -0.3, samples, dtype=np.float32)[None, :]
    hh += ramp + 0.35 * rng.randn(lines, samples).astype(np.float32)
    hv += 0.5 * ramp + 0.35 * rng.randn(lines, samples).astype(np.float32)
    land = gaussian_filter(rng.randn(lines, samples), sigma=40) > 0.035  # masked region
    sod[land] = 255
    hh[:, :8] = np.nan  # border noise as in real scenes
    hv[:, :8] = np.nan
    sod[:, :8] = 255
    ia = np.broadcast_to(np.linspace(-1.5, 1.5, samples, dtype=np.float32)[None, :], (lines, samples))
    lat0, lon0 = rng.uniform(62, 82), rng.uniform(-80, 20)
    lat = lat0 + np.linspace(0, 3, 6)[:, None] + np.zeros((1, 21))  # tie points, laid out like the image
    lon = lon0 + np.linspace(0, 8, 21)[None, :] + np.zeros((6, 1))
    return hh, hv, sod, ia.copy(), lat, lon


def write(path, hh, hv, sod, ia, lat, lon, with_labels=True):
    d2 = ("sar_lines", "sar_samples")
    g2 = ("sar_sample_2dgrid_points", "sar_line_2dgrid_points")
    var = {"nersc_sar_primary": (d2, hh), "nersc_sar_secondary": (d2, hv),
           "sar_incidenceangle": (d2, ia),
           "sar_grid2d_latitude": (g2, lat), "sar_grid2d_longitude": (g2, lon)}
    if with_labels:
        var["SOD"] = (d2, sod)
    xr.Dataset(var, attrs={"pixel_spacing": 80}).to_netcdf(path, engine="h5netcdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", type=Path, default=Path("data_synth"))
    ap.add_argument("--train", type=int, default=12)
    ap.add_argument("--test", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    (args.dst / "train").mkdir(parents=True, exist_ok=True)
    (args.dst / "test").mkdir(parents=True, exist_ok=True)
    for i in range(args.train):
        scene = make_scene(rng, rng.randint(600, 900), rng.randint(700, 1100))
        write(args.dst / "train" / f"2019{1 + i % 12:02d}15T1000{i:02d}_dmi_prep.nc", *scene)
    for i in range(args.test):
        scene = make_scene(rng, rng.randint(600, 900), rng.randint(700, 1100))
        name = f"2021{1 + (5 * i) % 12:02d}20T1200{i:02d}_cis_prep"
        write(args.dst / "test" / f"{name}.nc", *scene, with_labels=False)  # like the real test set
        xr.Dataset({"SOD": (("sar_lines", "sar_samples"), scene[2])}).to_netcdf(
            args.dst / "test" / f"{name}_reference.nc", engine="h5netcdf")
    print(f"wrote {args.train} train and {args.test} test scenes to {args.dst}")


if __name__ == "__main__":
    main()
