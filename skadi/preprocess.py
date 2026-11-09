"""Convert AI4Arctic ready-to-train netCDF scenes into a compact 512x512 cache.

We follow the protocol of TinyIceNet (arXiv:2603.03075, Sec. 4.2 and 5.2): every
scene (about 5000x5000 pixels at 80 m) is resized to 512x512, i.e. roughly 800 m
per pixel, and both training and testing happen on the resized scenes.

    python -m skadi.preprocess --src data --dst data/cache512/train
    python -m skadi.preprocess --src data/test --dst data/cache512/test

Besides HH/HV and the SOD chart we also store what a satellite knows about its
own acquisition without any other sensor: incidence angle, latitude/longitude
and the acquisition month. They are only used by the optional "+geo" model.
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import IGNORE

SIZE = 512
HH, HV = "nersc_sar_primary", "nersc_sar_secondary"
IA_NAMES = ("sar_incidenceangle", "sar_incidenceangles", "sar_grid_incidenceangle")


def open_scene(path: Path):
    import xarray as xr

    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception:  # fall back to whatever backend is installed
        return xr.open_dataset(path)


def resize_area(img: np.ndarray, size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware area resize. Returns the image and the fraction of valid input per pixel."""
    valid = np.isfinite(img)
    x = torch.from_numpy(np.where(valid, img, 0.0).astype(np.float32))[None, None]
    v = torch.from_numpy(valid.astype(np.float32))[None, None]
    num = F.interpolate(x, size=(size, size), mode="area")[0, 0]
    den = F.interpolate(v, size=(size, size), mode="area")[0, 0]
    out = torch.where(den > 0, num / den.clamp_min(1e-6), torch.zeros_like(num))
    return out.numpy(), den.numpy()


def resize_nearest(lab: np.ndarray, size: int = SIZE) -> np.ndarray:
    x = torch.from_numpy(lab.astype(np.float32))[None, None]
    return F.interpolate(x, size=(size, size), mode="nearest")[0, 0].numpy().astype(lab.dtype)


def resize_bilinear(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    x = torch.from_numpy(img.astype(np.float32))[None, None]
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=True)[0, 0].numpy()


def read_sod(ds, path: Path):
    """SOD chart of a scene. Test scenes keep it in a separate *_reference.nc file."""
    if "SOD" in ds:
        return ds["SOD"].values
    ref = path.with_name(path.stem + "_reference.nc")
    if ref.exists():
        with open_scene(ref) as rds:
            return rds["SOD"].values
    return None


def read_incidence_angle(ds, shape):
    for name in IA_NAMES:
        if name not in ds:
            continue
        ia = np.asarray(ds[name].values, dtype=np.float32)
        if ia.shape == shape:
            return ia
        if ia.ndim == 1 and ia.size == shape[1]:  # one value per range sample
            return np.broadcast_to(ia[None, :], shape).copy()
    return None


def read_latlon(ds):
    if "sar_grid2d_latitude" in ds and "sar_grid2d_longitude" in ds:
        lat = np.asarray(ds["sar_grid2d_latitude"].values, dtype=np.float32)
        lon = np.asarray(ds["sar_grid2d_longitude"].values, dtype=np.float32)
        dims = ds["sar_grid2d_latitude"].dims
        # the tie-point grid is stored (samples, lines) in most files: make it (lines, samples)
        if len(dims) == 2 and "sample" in dims[0] and "line" in dims[1]:
            lat, lon = lat.T, lon.T
        return lat, lon
    return None, None


def month_from_name(name: str) -> int:
    m = re.match(r"(\d{4})(\d{2})(\d{2})T", name)
    return int(m.group(2)) if m else 0


def process_scene(path: Path, dst: Path, size: int = SIZE) -> dict:
    out = dst / (path.stem + ".npz")
    with open_scene(path) as ds:
        hh = np.asarray(ds[HH].values, dtype=np.float32)
        hv = np.asarray(ds[HV].values, dtype=np.float32)
        sod = read_sod(ds, path)
        ia = read_incidence_angle(ds, hh.shape)
        lat, lon = read_latlon(ds)

    hh_r, frac = resize_area(hh, size)
    hv_r, _ = resize_area(hv, size)
    data = {
        "sar": np.stack([hh_r, hv_r]).astype(np.float16),
        "sar_valid": (frac > 0.5),
        "orig_shape": np.array(hh.shape, dtype=np.int32),
        "month": np.int8(month_from_name(path.name)),
    }
    if sod is not None:
        sod = sod.astype(np.uint8)
        data["sod"] = resize_nearest(sod, size)
    if ia is not None:
        data["ia"] = resize_area(ia, size)[0].astype(np.float16)
    if lat is not None:
        data["lat"] = resize_bilinear(lat, size).astype(np.float16)
        # interpolate longitude on the unit circle to survive the +-180 degree wrap
        rad = np.deg2rad(lon)
        data["lon_sin"] = resize_bilinear(np.sin(rad), size).astype(np.float16)
        data["lon_cos"] = resize_bilinear(np.cos(rad), size).astype(np.float16)
    np.savez_compressed(out, **data)
    valid = int((data["sod"] != IGNORE).sum()) if sod is not None else -1
    return {"name": path.stem, "shape": hh.shape, "labelled_px": valid,
            "has_ia": ia is not None, "has_geo": lat is not None}


def list_scenes(src: Path) -> list[Path]:
    return sorted(p for p in src.glob("*.nc") if not p.stem.endswith("_reference"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder with *_prep.nc scenes")
    ap.add_argument("--dst", type=Path, required=True, help="output folder for the .npz cache")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    scenes = list_scenes(args.src)
    todo = [p for p in scenes if args.overwrite or not (args.dst / (p.stem + ".npz")).exists()]
    print(f"{len(scenes)} scenes in {args.src}, {len(todo)} to process")

    done = no_label = no_ia = no_geo = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_scene, p, args.dst, args.size): p for p in todo}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                info = fut.result()
            except Exception as err:  # a corrupted download must not stop the whole run
                print(f"FAILED {path.name}: {err!r}")
                continue
            done += 1
            no_label += info["labelled_px"] < 0
            no_ia += not info["has_ia"]
            no_geo += not info["has_geo"]
            if done % 25 == 0 or done == len(todo):
                print(f"[{done}/{len(todo)}] {info['name']} {info['shape']}")
    print(f"done: {done} scenes | without SOD: {no_label} | without incidence angle: {no_ia} "
          f"| without lat/lon: {no_geo}")


if __name__ == "__main__":
    main()
