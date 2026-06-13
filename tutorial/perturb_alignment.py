#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignment-sensitivity test: is residual mis-alignment limiting the resolution?
==============================================================================

Idea
----
The projections in PXCTalignedprojections.npz are already aligned.  This script
makes copies with a *controlled* sub-pixel random shift added to each
projection (simulating residual mis-alignment), so you can reconstruct each
copy and measure how much the FSC degrades:

  * If a small jitter (sigma < ~1 px) markedly worsens the FSC resolution,
    then alignment IS your ceiling -> better registration (e.g. the matched
    NUFFT re-projection in Toupy) would buy resolution.
  * If the FSC barely moves up to ~1 px, you are NOT alignment-limited -> the
    62.6 nm ceiling is set by dose / phase-retrieval, and improving alignment
    will not help.

The shift is applied per projection via a Fourier (sub-pixel, non-blurring)
shift.  Each projection gets an independent random shift, so the even/odd FSC
half-sets see independent mis-alignment realisations -- exactly the realistic
"residual jitter" scenario the FSC is meant to detect.

Workflow
--------
  1.  python tutorial/perturb_alignment.py
        -> writes PXCTalignedprojections_jitter<sigma>px_<axes>.npz for each
           sigma in JITTER_LIST_PX.
  2.  For each file (and the ORIGINAL as the sigma=0 reference): set DATA_FILE
      in twopass_real_data.py / slurm_twopass.sh, run FSC_HALF=0 and =1, then
      fsc_threeway_comparison.py to get the 1/2-bit resolution.
  3.  Plot 1/2-bit resolution vs sigma.  A steep rise => alignment-limited;
      a flat curve => not alignment-limited.

Self-test:  python tutorial/perturb_alignment.py --selftest
"""

import os
import sys
import argparse
import numpy as np
from scipy.ndimage import fourier_shift

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
IN_FILE = os.path.join(_HERE, "PXCTalignedprojections.npz")

# Jitter amplitudes to generate (std of the per-projection Gaussian shift, px).
# 0.25 / 0.5 / 1.0 px brackets "good residual" to "poor" alignment.
JITTER_LIST_PX = [0.25, 0.5, 1.0]

# Which detector axis to perturb:
#   'x'  -> horizontal (perpendicular to rotation axis): the tomographically
#           critical shift that consistency alignment corrects (recommended).
#   'y'  -> vertical (along rotation axis).
#   'xy' -> both.
JITTER_AXES = "x"

SEED = 0          # reproducible; same seed across sigma => proportional shifts


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def apply_jitter(proj, sigma_px, axes, seed):
    """
    Add an independent Gaussian sub-pixel shift to each projection.

    Parameters
    ----------
    proj : (N_ang, Ny, Nx) float
    sigma_px : float        std of the shift [pixels]
    axes : str              'x', 'y' or 'xy'
    seed : int

    Returns
    -------
    out : (N_ang, Ny, Nx) float32   shifted projections
    dx, dy : (N_ang,) applied shifts [px]   (axis2=x, axis1=y)
    """
    rng = np.random.default_rng(seed)
    N_ang, Ny, Nx = proj.shape
    dx = rng.normal(0.0, sigma_px, N_ang) if "x" in axes else np.zeros(N_ang)
    dy = rng.normal(0.0, sigma_px, N_ang) if "y" in axes else np.zeros(N_ang)

    out = np.empty_like(proj, dtype=np.float32)
    for a in range(N_ang):
        # Fourier shift: tuple is per-axis (axis0=y(rows), axis1=x(cols)).
        ft = np.fft.fft2(proj[a])
        out[a] = np.fft.ifft2(fourier_shift(ft, (dy[a], dx[a]))).real.astype(np.float32)
    return out, dx, dy


def main():
    if not os.path.exists(IN_FILE):
        sys.exit(f"Input not found: {IN_FILE}")

    data = np.load(IN_FILE, allow_pickle=True)
    proj = data["projections"].astype(np.float32)
    meta = {k: data[k] for k in data.files if k != "projections"}
    N_ang, Ny, Nx = proj.shape

    print("=" * 64)
    print("Alignment-sensitivity test — jittered projection sets")
    print("=" * 64)
    print(f"  input : {IN_FILE}")
    print(f"  shape : {proj.shape}   axes perturbed: {JITTER_AXES}")
    print(f"  sigma list (px): {JITTER_LIST_PX}\n")
    print("  Reference (sigma=0) is the ORIGINAL file; reconstruct it too.\n")

    for sigma in JITTER_LIST_PX:
        out, dx, dy = apply_jitter(proj, sigma, JITTER_AXES, SEED)
        rms = float(np.sqrt(np.mean(dx**2 + dy**2)))
        out_file = IN_FILE.replace(
            ".npz", f"_jitter{sigma:.2f}px_{JITTER_AXES}.npz")
        np.savez_compressed(out_file, projections=out, **meta)
        print(f"  sigma={sigma:.2f} px ({JITTER_AXES}): "
              f"RMS shift {rms:.3f} px,  max |shift| {np.abs(np.r_[dx,dy]).max():.2f} px")
        print(f"      -> {out_file}")

    print("\nNext: reconstruct ORIGINAL (sigma=0) and each jittered file "
          "(FSC_HALF 0 & 1),\n      then plot 1/2-bit resolution vs sigma.")


# ---------------------------------------------------------------------------
# Self-test (mechanics, on a tiny synthetic npz)
# ---------------------------------------------------------------------------
def selftest():
    print("perturb_alignment self-test")
    rng = np.random.default_rng(1)
    N_ang, Ny, Nx = 6, 32, 40
    proj = np.zeros((N_ang, Ny, Nx), dtype=np.float32)
    proj[:, 10:22, 12:28] = 1.0                         # a block in each proj

    # sigma=0 -> identity (within FFT round-off)
    out0, dx0, dy0 = apply_jitter(proj, 0.0, "x", SEED)
    id_ok = np.allclose(out0, proj, atol=1e-5) and np.allclose(dx0, 0)

    # sigma>0 -> RMS shift matches sigma (statistically), output differs, x-only
    out1, dx1, dy1 = apply_jitter(proj, 0.5, "x", SEED)
    x_only = np.allclose(dy1, 0.0)
    differs = not np.allclose(out1, proj, atol=1e-3)
    shape_ok = out1.shape == proj.shape and out1.dtype == np.float32

    # a known integer shift reproduces np.roll (sanity of direction)
    one = proj[0:1].copy()
    sh, _, _ = apply_jitter(one, 0.0, "x", SEED)         # identity check again
    shifted = np.fft.ifft2(fourier_shift(np.fft.fft2(proj[0]), (0, 3))).real
    roll_ok = np.allclose(shifted, np.roll(proj[0], 3, axis=1), atol=1e-4)

    print(f"  sigma=0 identity : {id_ok}")
    print(f"  x-only shift     : {x_only}")
    print(f"  output differs   : {differs}")
    print(f"  shape/dtype ok   : {shape_ok}")
    print(f"  integer shift == np.roll : {roll_ok}")
    ok = id_ok and x_only and differs and shape_ok and roll_ok
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    main()
