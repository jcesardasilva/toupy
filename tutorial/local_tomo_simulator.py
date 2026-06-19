#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local- (interior / ROI-) tomography testbed generator.
======================================================

Goal
----
Develop a local-tomography variant of the two-pass method WITHOUT new beamtime,
on data where the ground truth is already known.

Physics shortcut (why cropping the detector IS a valid local-tomo simulation)
-----------------------------------------------------------------------------
In parallel-beam tomography the detector axis perpendicular to the rotation axis
(here: the COLUMNS, x) is the tomographically critical one; each x-row of the
projection stack is an independent 2-D slice reconstruction in the x-z plane.
A *local* (interior) acquisition simply uses a detector narrower than the object
in x: the central rays still integrate through material that, at other angles,
lies OUTSIDE the field of view (FOV).  So if we take FULL projections and keep
only a central x-window, the kept pixels still carry those external
contributions exactly -- they are physically-correct truncated data.  And the
central region of the existing full reconstruction is EXACT GROUND TRUTH.

Truncation is applied in x (columns) by default.  Optional y (rows) truncation
just limits the axial FOV; it is far less fundamental and off by default.

What this writes
----------------
A truncated projection set ``PXCTalignedprojections_localfov<pct>[_jitter..].npz``
carrying the original metadata PLUS a ``fov`` record (the crop window relative to
the chosen base grid, and the full grid size) so that:
  * twopass_real_data.py can reconstruct it (the filename tag routes its OUT_DIR);
  * a later script can pull the matching ground-truth ROI from the full
    reconstruction with ``extract_groundtruth_roi``.

Base crop
---------
The validated full reconstruction used CROP_X=CROP_Y=55.  To keep the
ground-truth ROI in register, this script first applies the SAME base crop
(``--base-crop-x/-y``, default 55) to the raw projections, then truncates the
FOV centred on that grid.  Reconstruct the output with CROP_X=CROP_Y=0.

Usage
-----
  python tutorial/local_tomo_simulator.py --fov-frac-x 0.5
  python tutorial/local_tomo_simulator.py --fov-frac-x 0.4 --jitter-px 0.5
  python tutorial/local_tomo_simulator.py --selftest
"""

import os
import sys
import argparse
import numpy as np

try:
    from scipy.ndimage import fourier_shift
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

_HERE = os.path.dirname(os.path.abspath(__file__))
IN_FILE = os.path.join(_HERE, "PXCTalignedprojections.npz")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def _centered_window(n, frac):
    """Central window [lo, hi) keeping ~frac of n samples, symmetric about the
    centre.  frac>=1 -> full range."""
    if frac >= 1.0:
        return 0, n
    w = int(round(frac * n))
    w = max(1, min(n, w))
    lo = (n - w) // 2
    return lo, lo + w


def crop_fov(proj, frac_x, frac_y=1.0):
    """
    Truncate the FOV of a projection stack to a central window.

    Parameters
    ----------
    proj : (N_ang, Ny, Nx)  full projections [rad].
    frac_x : float  fraction of columns (x, perpendicular to rotation axis) to
        keep -- the tomographically critical interior truncation.
    frac_y : float  fraction of rows (y, along rotation axis) to keep (1.0=full).

    Returns
    -------
    trunc : (N_ang, ny, nx) truncated projections (a view-derived copy).
    meta  : dict with the crop window and full size (relative to `proj`).
    """
    _, Ny, Nx = proj.shape
    x0, x1 = _centered_window(Nx, frac_x)
    y0, y1 = _centered_window(Ny, frac_y)
    trunc = np.ascontiguousarray(proj[:, y0:y1, x0:x1])
    meta = dict(fov_x0=x0, fov_x1=x1, fov_y0=y0, fov_y1=y1,
                full_Nx=Nx, full_Ny=Ny,
                fov_frac_x=float(frac_x), fov_frac_y=float(frac_y))
    return trunc, meta


def apply_jitter(proj, sigma_px, axes="x", seed=0):
    """Independent Gaussian sub-pixel shift per projection (Fourier shift), to
    test alignment-under-truncation.  Mirrors perturb_alignment.apply_jitter."""
    if sigma_px <= 0:
        return proj, np.zeros(proj.shape[0]), np.zeros(proj.shape[0])
    if not _HAVE_SCIPY:
        sys.exit("scipy required for --jitter-px (pip install scipy).")
    rng = np.random.default_rng(seed)
    N_ang = proj.shape[0]
    dx = rng.normal(0.0, sigma_px, N_ang) if "x" in axes else np.zeros(N_ang)
    dy = rng.normal(0.0, sigma_px, N_ang) if "y" in axes else np.zeros(N_ang)
    out = np.empty_like(proj, dtype=np.float32)
    for a in range(N_ang):
        ft = np.fft.fft2(proj[a])
        out[a] = np.fft.ifft2(fourier_shift(ft, (dy[a], dx[a]))).real
    return out, dx, dy


def extract_groundtruth_roi(vol_full, meta):
    """
    Pull the ground-truth ROI sub-volume from a FULL reconstruction that was
    built on the base grid (i.e. with the same base crop, FOV un-truncated).

    The reconstructed slice spans x and z identically (Nz == Nx), so the
    interior FOV is the central [x0:x1] window in BOTH x and z; y uses the
    (usually full) row window.

    vol_full : (Nz, Ny, Nx) full reconstruction (delta).
    meta     : the `fov` dict written by crop_fov / this script.
    Returns the (nz, ny, nx) ground-truth ROI.
    """
    Nz, Ny, Nx = vol_full.shape
    if (Nx, Ny) != (meta["full_Nx"], meta["full_Ny"]):
        raise ValueError(
            f"full volume (Nx,Ny)=({Nx},{Ny}) does not match the fov record "
            f"({meta['full_Nx']},{meta['full_Ny']}). Reconstruct the full set "
            f"with the same base crop the simulator used.")
    x0, x1 = meta["fov_x0"], meta["fov_x1"]
    y0, y1 = meta["fov_y0"], meta["fov_y1"]
    # z shares the x window (square slices); clip to volume in case Nz != Nx.
    z0, z1 = x0, min(x1, Nz)
    return vol_full[z0:z1, y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", default=IN_FILE)
    ap.add_argument("--fov-frac-x", type=float, default=0.5,
                    help="central fraction of columns (x) to keep (interior "
                         "truncation). 0.5 = keep central 50%%.")
    ap.add_argument("--fov-frac-y", type=float, default=1.0,
                    help="central fraction of rows (y) to keep (axial FOV; "
                         "1.0 = full).")
    ap.add_argument("--base-crop-x", type=int, default=55,
                    help="base crop applied first to match the full recon grid "
                         "(twopass_real_data CROP_X). Reconstruct output w/ CROP_X=0.")
    ap.add_argument("--base-crop-y", type=int, default=55)
    ap.add_argument("--jitter-px", type=float, default=0.0,
                    help="optional per-projection Gaussian jitter to test "
                         "alignment-under-truncation.")
    ap.add_argument("--jitter-axes", default="x")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output npz (default: auto-named "
                    "next to the input).")
    args = ap.parse_args()

    if not os.path.isfile(args.in_file):
        sys.exit(f"input not found: {args.in_file}")
    data = np.load(args.in_file, allow_pickle=True)
    proj = data["projections"].astype(np.float32)
    extra = {k: data[k] for k in data.files if k != "projections"}
    N_ang, Ny0, Nx0 = proj.shape

    print("=" * 66)
    print("Local-tomography testbed — truncated projection generator")
    print("=" * 66)
    print(f"  input : {args.in_file}   shape {proj.shape}")

    # 1) base crop to the full-recon grid
    bcx, bcy = args.base_crop_x, args.base_crop_y
    if bcx or bcy:
        sy = slice(bcy or None, -bcy if bcy else None)
        sx = slice(bcx or None, -bcx if bcx else None)
        proj = np.ascontiguousarray(proj[:, sy, sx])
        print(f"  base crop ({bcy}px rows, {bcx}px cols) -> {proj.shape} "
              f"(this is the full-recon grid; reconstruct output w/ CROP=0)")

    # 2) FOV truncation
    trunc, meta = crop_fov(proj, args.fov_frac_x, args.fov_frac_y)
    print(f"  FOV truncation: keep x[{meta['fov_x0']}:{meta['fov_x1']}] "
          f"({meta['fov_frac_x']:.0%} of {meta['full_Nx']}) , "
          f"y[{meta['fov_y0']}:{meta['fov_y1']}] "
          f"({meta['fov_frac_y']:.0%} of {meta['full_Ny']})")
    print(f"  truncated shape: {trunc.shape}")

    # 3) optional jitter
    jtag = ""
    if args.jitter_px > 0:
        trunc, dx, dy = apply_jitter(trunc, args.jitter_px, args.jitter_axes,
                                     args.seed)
        rms = float(np.sqrt(np.mean(dx**2 + dy**2)))
        jtag = f"_jitter{args.jitter_px:.2f}{args.jitter_axes}"
        meta.update(jitter_px=args.jitter_px, jitter_axes=args.jitter_axes,
                    jitter_dx=dx, jitter_dy=dy)
        print(f"  + jitter sigma={args.jitter_px:.2f}px ({args.jitter_axes}): "
              f"RMS {rms:.3f}px")

    # 4) save
    pct = int(round(meta["fov_frac_x"] * 100))
    out = args.out or args.in_file.replace(
        ".npz", f"_localfov{pct}{jtag}.npz")
    save = dict(extra)
    save["projections"] = trunc.astype(np.float32)
    # flatten the fov record into top-level keys (npz-friendly)
    for k, v in meta.items():
        save[f"fov_{k}" if not k.startswith(("fov_", "jitter_")) else k] = v
    np.savez_compressed(out, **save)
    print(f"\n  saved: {out}")
    print(f"  ground truth: central ROI of the FULL reconstruction "
          f"(use extract_groundtruth_roi with the saved fov_* record).")
    print(f"  reconstruct : set DATA_FILE to the above and CROP_X=CROP_Y=0.")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def selftest():
    print("local_tomo_simulator self-test")
    rng = np.random.default_rng(0)
    N_ang, Ny, Nx = 5, 40, 60
    # phantom: a known central block + external structure outside the FOV in x
    proj = rng.normal(0, 0.01, (N_ang, Ny, Nx)).astype(np.float32)
    proj[:, 10:30, 26:34] += 1.0           # central (will be inside FOV)
    proj[:, 10:30, 2:8] += 0.7             # far-left external (outside FOV)

    # centred window keeps the middle and is symmetric
    trunc, meta = crop_fov(proj, 0.5, 1.0)
    nx = meta["fov_x1"] - meta["fov_x0"]
    centered = (meta["fov_x0"] == (Nx - nx) // 2)
    half_ok = nx == Nx // 2
    central_kept = np.allclose(
        trunc[:, 10:30, :],
        proj[:, 10:30, meta["fov_x0"]:meta["fov_x1"]])
    external_dropped = meta["fov_x0"] > 8   # the x[2:8] block is outside FOV

    # ground-truth ROI extraction from a fake full volume (Nz=Nx, Ny, Nx)
    vol = np.arange(Nx * Ny * Nx, dtype=float).reshape(Nx, Ny, Nx)
    roi = extract_groundtruth_roi(vol, meta)
    roi_shape_ok = roi.shape == (nx, Ny, nx)
    roi_matches = np.array_equal(
        roi, vol[meta["fov_x0"]:meta["fov_x0"]+nx, :,
                 meta["fov_x0"]:meta["fov_x1"]])

    # full frac keeps everything; y truncation works
    full, _ = crop_fov(proj, 1.0, 1.0)
    full_ok = full.shape == proj.shape
    _, my = crop_fov(proj, 1.0, 0.5)
    y_ok = (my["fov_y1"] - my["fov_y0"]) == Ny // 2

    # jitter is sub-pixel, x-only, changes the data
    if _HAVE_SCIPY:
        jt, dx, dy = apply_jitter(trunc, 0.5, "x", 0)
        jit_ok = np.allclose(dy, 0) and not np.allclose(jt, trunc, atol=1e-3)
    else:
        jit_ok = True
        print("  (scipy absent: skipping jitter check)")

    checks = {
        "window centred": centered, "keeps ~50%": half_ok,
        "central content preserved": central_kept,
        "external dropped": external_dropped,
        "ROI shape": roi_shape_ok, "ROI matches full": roi_matches,
        "frac=1 keeps all": full_ok, "y truncation": y_ok,
        "jitter x-only & changes data": jit_ok,
    }
    for k, v in checks.items():
        print(f"  {k:32s}: {v}")
    ok = all(checks.values())
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main()
