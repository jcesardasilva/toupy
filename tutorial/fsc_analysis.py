#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier Shell Correlation (FSC) analysis for PXCT two-pass reconstructions.
============================================================================

Computes and plots FSC curves that estimate the spatial resolution of:
  (a) FBP reconstructions from two half-datasets
  (b) Two-pass reconstructions from two half-datasets
  (c) Cross-FSC between FBP and two-pass (diagnostic only)

Usage
-----
1. Run twopass_real_data.py twice with FSC_HALF = 0 and FSC_HALF = 1.
   Each run produces a twopass_reconstruction.npz in its output directory.

2. Run this script, passing the two .npz files as positional arguments:

       # Explicit paths (typical for SLURM runs with OUT_DIR per job):
       python fsc_analysis.py results_<jobid0>/twopass_reconstruction.npz \
                              results_<jobid1>/twopass_reconstruction.npz

       # Default paths (tutorial directory layout):
       python fsc_analysis.py
       # expects twopass_real_figures_half0/twopass_reconstruction.npz
       #     and twopass_real_figures_half1/twopass_reconstruction.npz

3. Outputs (written next to the half-0 file in a 'fsc/' subfolder by default,
   or to the directory set with --out-dir):
       fsc/fsc_curves.png
       fsc/fsc_data.npz

Resolution interpretation
-------------------------
The FSC measures cross-correlation between two independent measurements of
the same object as a function of spatial frequency.  Resolution is reported
at two standard thresholds:
  - 0.5 threshold          (conservative, traditional)
  - 1/2-bit criterion      (frequency-adaptive, recommended for tomography)
    1/2-bit(n) = (0.2071 + 1.9102/sqrt(n)) / (1.2071 + 0.9102/sqrt(n))
    where n = number of voxels in the shell.

The resolution in physical units:
    d_min  [nm] = pixel_size [nm] / spatial_freq_at_threshold

Key insight
-----------
If FSC_tp > FSC_fbp at high spatial frequencies, the two-pass method
provides better agreement between the two independent half-datasets →
it is genuinely resolving more structure, not just fitting noise.
"""

import argparse
import gc
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

_DEFAULT_HALF0 = os.path.join(_HERE, "twopass_real_figures_half0",
                               "twopass_reconstruction.npz")
_DEFAULT_HALF1 = os.path.join(_HERE, "twopass_real_figures_half1",
                               "twopass_reconstruction.npz")
_DEFAULT_OUT   = os.path.join(_HERE, "twopass_real_figures_fsc")

ap = argparse.ArgumentParser(
    description="Fourier Shell Correlation analysis for two-pass PXCT reconstructions.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples
--------
# Explicit .npz paths (typical SLURM run):
  python fsc_analysis.py results_33091306/twopass_reconstruction.npz \\
                         results_33091307/twopass_reconstruction.npz

# Default paths (tutorial directory layout):
  python fsc_analysis.py

# Custom output directory:
  python fsc_analysis.py half0.npz half1.npz --out-dir /scratch/fsc_output
""",
)
ap.add_argument(
    "half0",
    nargs="?",
    default=_DEFAULT_HALF0,
    help=f"Path to the FSC_HALF=0 reconstruction .npz  (default: {_DEFAULT_HALF0})",
)
ap.add_argument(
    "half1",
    nargs="?",
    default=_DEFAULT_HALF1,
    help=f"Path to the FSC_HALF=1 reconstruction .npz  (default: {_DEFAULT_HALF1})",
)
ap.add_argument(
    "--out-dir",
    default=None,
    help=(
        "Directory for output figure and data file.  "
        "Defaults to a 'fsc' subfolder next to the half-0 file, "
        f"or '{_DEFAULT_OUT}' when using the default paths."
    ),
)
ap.add_argument(
    "--apod-width",
    type=int,
    default=20,
    metavar="PX",
    help=(
        "Width in pixels of the Tukey (cosine-tapered flat-top) apodization "
        "applied to each volume before FFT.  Matches Toupy's "
        "FourierShellCorr default (apod_width=20).  "
        "Set to 0 to disable apodization — faster but can over-estimate "
        "resolution because edge artefacts correlate between half-datasets.  "
        "Default: 20"
    ),
)
args = ap.parse_args()

FILE_HALF0  = os.path.abspath(args.half0)
FILE_HALF1  = os.path.abspath(args.half1)
APOD_WIDTH  = args.apod_width   # 0 = no apodization

if args.out_dir is not None:
    OUT_DIR = os.path.abspath(args.out_dir)
elif args.half0 == _DEFAULT_HALF0:
    OUT_DIR = _DEFAULT_OUT
else:
    # Place output next to the half-0 file in a 'fsc' subfolder
    OUT_DIR = os.path.join(os.path.dirname(FILE_HALF0), "fsc")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print(f"Loading half-0 reconstruction from:\n  {FILE_HALF0}")
if not os.path.isfile(FILE_HALF0):
    sys.exit(
        f"ERROR: {FILE_HALF0} not found.\n"
        "Pass the correct path as the first positional argument:\n"
        "  python fsc_analysis.py <path/to/half0.npz> <path/to/half1.npz>\n"
        "Or run twopass_real_data.py with FSC_HALF = 0 first."
    )

print(f"Loading half-1 reconstruction from:\n  {FILE_HALF1}")
if not os.path.isfile(FILE_HALF1):
    sys.exit(
        f"ERROR: {FILE_HALF1} not found.\n"
        "Pass the correct path as the second positional argument:\n"
        "  python fsc_analysis.py <path/to/half0.npz> <path/to/half1.npz>\n"
        "Or run twopass_real_data.py with FSC_HALF = 1 first."
    )

d0 = np.load(FILE_HALF0, allow_pickle=True)
d1 = np.load(FILE_HALF1, allow_pickle=True)

fbp0 = d0["delta_fbp"].astype(np.float32)
fbp1 = d1["delta_fbp"].astype(np.float32)
tp0  = d0["delta_tp"].astype(np.float32)
tp1  = d1["delta_tp"].astype(np.float32)

PIXEL_SIZE = float(d0["psize"])   # [m]
PIXEL_NM   = PIXEL_SIZE * 1e9     # [nm]

assert fbp0.shape == fbp1.shape, "Volume shapes must match between half-datasets"
Nz, Ny, Nx = fbp0.shape

print(f"Volume shape: ({Nz}, {Ny}, {Nx}),  pixel = {PIXEL_NM:.2f} nm")
print()

# ---------------------------------------------------------------------------
# FSC computation
# ---------------------------------------------------------------------------

def _tukey_1d(n: int, apod: int) -> np.ndarray:
    """
    1-D Tukey (cosine-tapered flat-top) apodization window.

    The window equals 1 in the central ``n - 2*apod`` samples and tapers
    smoothly from 1 → 0 via a raised-cosine ramp over ``apod`` samples at
    each end.  This matches the window produced by Toupy's
    ``FourierShellCorr._make_1d_tukey``.

    Parameters
    ----------
    n : int   — window length (number of samples)
    apod : int — taper width in pixels at each end

    Returns
    -------
    win : ndarray, shape (n,), dtype float32
    """
    win = np.ones(n, dtype=np.float32)
    if apod <= 0 or 2 * apod >= n:
        return win
    t = np.arange(apod, dtype=np.float32)
    taper = 0.5 * (1.0 - np.cos(np.pi * t / apod))   # 0 → 1 over apod pts
    win[:apod]  = taper
    win[-apod:] = taper[::-1]
    return win


def compute_fsc(vol1: np.ndarray, vol2: np.ndarray,
                n_shells: int = 200, apod_width: int = 20):
    """
    Compute the Fourier Shell Correlation between two 3-D volumes.

    Parameters
    ----------
    vol1, vol2 : ndarray, shape (Nz, Ny, Nx)
    n_shells : int
        Number of radial shells.
    apod_width : int
        Pixels of Tukey taper applied to each axis before FFT, matching
        Toupy's FourierShellCorr default.  Set to 0 to disable.

    Returns
    -------
    freq_nyq : ndarray, shape (n_shells,)
        Spatial frequency in units of Nyquist (0 = DC, 0.5 = Nyquist).
    fsc : ndarray, shape (n_shells,)
        FSC values in [−1, 1].
    n_voxels : ndarray, shape (n_shells,)
        Number of Fourier voxels in each shell (for threshold criteria).

    Memory notes
    ------------
    Uses rfftn (real FFT) which exploits the real-valued input to produce
    a half-spectrum of shape (Nz, Ny, Nx//2+1) in complex64 — roughly 4×
    smaller than a full complex128 fftn.  The radial coordinate R and the
    apodization window are both built via broadcasting (three tiny 1-D
    arrays) to avoid large temporary allocations.
    """
    Nz, Ny, Nx = vol1.shape

    # Apply separable Tukey apodization (matching Toupy's convention)
    # via broadcasting so we never materialise the full 3-D window array.
    v1 = vol1.astype(np.float32)
    v2 = vol2.astype(np.float32)
    if apod_width > 0:
        wz = _tukey_1d(Nz, apod_width)[:, None, None]
        wy = _tukey_1d(Ny, apod_width)[None, :, None]
        wx = _tukey_1d(Nx, apod_width)[None, None, :]
        v1 *= wz;  v1 *= wy;  v1 *= wx
        v2 *= wz;  v2 *= wy;  v2 *= wx

    # rfftn: input float32 → output complex64, shape (Nz, Ny, Nx//2+1)
    # FSC is a ratio so the missing conjugate half cancels in numerator/denom.
    F1 = np.fft.rfftn(v1)
    F2 = np.fft.rfftn(v2)
    del v1, v2   # free before building R

    # Radial coordinate via broadcasting — no meshgrid, no giant temporaries
    kz = np.fft.fftfreq(Nz).astype(np.float32)[:, None, None]   # (Nz,1,1)
    ky = np.fft.fftfreq(Ny).astype(np.float32)[None, :, None]   # (1,Ny,1)
    kx = np.fft.rfftfreq(Nx).astype(np.float32)[None, None, :]  # (1,1,Nx//2+1)
    R  = np.sqrt(kz**2 + ky**2 + kx**2)   # broadcast → (Nz,Ny,Nx//2+1)

    r_max   = 0.5    # Nyquist
    r_edges = np.linspace(0.0, r_max, n_shells + 1)
    r_mid   = 0.5 * (r_edges[:-1] + r_edges[1:])

    fsc      = np.zeros(n_shells, dtype=np.float64)
    n_voxels = np.zeros(n_shells, dtype=np.int64)

    for i, (r_lo, r_hi) in enumerate(zip(r_edges[:-1], r_edges[1:])):
        mask = (R >= r_lo) & (R < r_hi)
        n    = int(mask.sum())
        if n == 0:
            continue
        f1 = F1[mask]
        f2 = F2[mask]
        num   = float(np.real(np.sum(f1 * f2.conj())))
        denom = float(np.sqrt(np.sum(np.abs(f1)**2) * np.sum(np.abs(f2)**2)))
        fsc[i]      = num / denom if denom > 0 else 0.0
        n_voxels[i] = n

    return r_mid, fsc, n_voxels


def half_bit_criterion(n_voxels: np.ndarray) -> np.ndarray:
    """
    1/2-bit information criterion for resolution (van Heel & Schatz 2005).

    T_{1/2}(N) = (0.2071 + 1.9102/sqrt(N)) / (1.2071 + 0.9102/sqrt(N))

    Asymptote at large N: 0.2071/1.2071 ≈ 0.171
    This is the primary resolution threshold recommended by van Heel &
    Schatz (2005) for half-dataset FSC.
    """
    sqn = np.sqrt(n_voxels.astype(float) + 1e-30)
    return (0.2071 + 1.9102 / sqn) / (1.2071 + 0.9102 / sqn)


def one_bit_criterion(n_voxels: np.ndarray) -> np.ndarray:
    """
    1-bit information criterion for resolution (van Heel & Schatz 2005).

    T_1(N) = (0.5000 + 2.4142/sqrt(N)) / (1.5000 + 1.4142/sqrt(N))

    Asymptote at large N: 0.5/1.5 ≈ 0.333
    More conservative (higher threshold) than the ½-bit criterion; the
    0.5 fixed-threshold rule of thumb approximates this curve at large
    shell populations.
    """
    sqn = np.sqrt(n_voxels.astype(float) + 1e-30)
    return (0.5000 + 2.4142 / sqn) / (1.5000 + 1.4142 / sqn)


def resolution_at_threshold(freq_nyq, fsc, threshold):
    """
    Find the spatial frequency where FSC first crosses threshold (from above).
    Returns frequency in Nyquist units (0.5 = Nyquist) and resolution in nm.
    """
    for i in range(len(fsc) - 1):
        if fsc[i] >= threshold > fsc[i + 1]:
            # Linear interpolation
            f0, f1 = freq_nyq[i], freq_nyq[i + 1]
            v0, v1 = fsc[i], fsc[i + 1]
            f_cross = f0 + (threshold - v0) / (v1 - v0) * (f1 - f0)
            if f_cross > 0:
                return f_cross, PIXEL_NM / f_cross
    return np.nan, np.nan   # never crossed


# ---------------------------------------------------------------------------
# Compute FSC for FBP and two-pass
# ---------------------------------------------------------------------------

_apod_str = f"{APOD_WIDTH} px" if APOD_WIDTH > 0 else "none"
print(f"Apodization : {_apod_str}  "
      f"({'Tukey taper, matching Toupy convention' if APOD_WIDTH > 0 else 'disabled — may over-estimate resolution'})")
print()

print("Computing FSC — FBP half-0 vs half-1 …")
r_mid, fsc_fbp, n_vox = compute_fsc(fbp0, fbp1, apod_width=APOD_WIDTH)
gc.collect()

print("Computing FSC — two-pass half-0 vs half-1 …")
_, fsc_tp, _ = compute_fsc(tp0, tp1, apod_width=APOD_WIDTH)
gc.collect()

# Also compute cross-FSC (diagnostic: where do FBP and TP agree/disagree?)
print("Computing cross-FSC — FBP half-0 vs two-pass half-0 …")
_, fsc_cross, _ = compute_fsc(fbp0, tp0, apod_width=APOD_WIDTH)
gc.collect()

half_bit = half_bit_criterion(n_vox)
one_bit  = one_bit_criterion(n_vox)

# Resolution at fixed FSC = 0.5 threshold
f05_fbp, d05_fbp = resolution_at_threshold(r_mid, fsc_fbp, 0.5)
f05_tp,  d05_tp  = resolution_at_threshold(r_mid, fsc_tp,  0.5)

# Resolution at ½-bit criterion (primary van Heel threshold)
hb_fbp = next((r_mid[i] for i in range(len(fsc_fbp))
               if fsc_fbp[i] < half_bit[i]), np.nan)
hb_tp  = next((r_mid[i] for i in range(len(fsc_tp))
               if fsc_tp[i]  < half_bit[i]), np.nan)
d_hb_fbp = PIXEL_NM / hb_fbp if not np.isnan(hb_fbp) else np.nan
d_hb_tp  = PIXEL_NM / hb_tp  if not np.isnan(hb_tp)  else np.nan

# Resolution at 1-bit criterion (more conservative)
ob_fbp = next((r_mid[i] for i in range(len(fsc_fbp))
               if fsc_fbp[i] < one_bit[i]), np.nan)
ob_tp  = next((r_mid[i] for i in range(len(fsc_tp))
               if fsc_tp[i]  < one_bit[i]), np.nan)
d_ob_fbp = PIXEL_NM / ob_fbp if not np.isnan(ob_fbp) else np.nan
d_ob_tp  = PIXEL_NM / ob_tp  if not np.isnan(ob_tp)  else np.nan

print()
print("=" * 66)
print("  Resolution summary")
print("=" * 66)
print(f"  {'Method':<14}  {'FSC=0.5':>10}  {'1-bit':>10}  {'1/2-bit':>10}")
print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")
print(f"  {'FBP':<14}  {d05_fbp:>8.1f} nm  {d_ob_fbp:>8.1f} nm  {d_hb_fbp:>8.1f} nm")
print(f"  {'Two-pass':<14}  {d05_tp:>8.1f} nm  {d_ob_tp:>8.1f} nm  {d_hb_tp:>8.1f} nm")
print("=" * 66)
print()

# Physical frequency axis (cycles per nm)
freq_nm = r_mid / PIXEL_NM   # cycles/nm

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                          gridspec_kw={"wspace": 0.35})

# ── Left: FSC vs spatial frequency ────────────────────────────────────────
ax = axes[0]
ax.plot(r_mid, fsc_fbp,   color="C0",        lw=1.8, label="FBP")
ax.plot(r_mid, fsc_tp,    color="C1",        lw=1.8, label="Two-pass")
ax.plot(r_mid, fsc_cross, color="C2",        lw=1.2, ls="--",
        label="Cross (FBP₀ vs TP₀)", alpha=0.7)
ax.plot(r_mid, one_bit,   color="steelblue", lw=1.3, ls="-.",
        label="1-bit criterion")
ax.plot(r_mid, half_bit,  color="dimgray",   lw=1.3, ls=":",
        label="½-bit criterion")
ax.axhline(0.5,           color="black",     lw=0.8, ls="--", alpha=0.5,
           label="FSC = 0.5 (ref.)")

# Mark ½-bit crossings (primary, solid tick) and 1-bit crossings (secondary)
for f_c, col, tag, ls in [
        (hb_fbp, "C0",        f"FBP ½-bit {d_hb_fbp:.1f} nm",  ":"),
        (hb_tp,  "C1",        f"TP  ½-bit {d_hb_tp:.1f} nm",   ":"),
        (ob_fbp, "C0",        f"FBP 1-bit {d_ob_fbp:.1f} nm",  "-."),
        (ob_tp,  "C1",        f"TP  1-bit {d_ob_tp:.1f} nm",   "-.")]:
    if not np.isnan(f_c):
        ax.axvline(f_c, color=col, lw=0.8, ls=ls, alpha=0.6)
        y_txt = 0.08 if "1-bit" in tag else 0.30
        ax.text(f_c + 0.004, y_txt, tag, color=col, fontsize=6.5,
                rotation=90, va="bottom")

ax.set_xlim(0, 0.5)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Spatial frequency  [Nyquist units]")
ax.set_ylabel("Fourier Shell Correlation")
ax.set_title("Half-dataset FSC")
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, alpha=0.3)

# Secondary x-axis in nm⁻¹
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ticks_nyq = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
ax2.set_xticks(ticks_nyq)
ax2.set_xticklabels([f"{1/(t*PIXEL_NM*2):.1f}" for t in ticks_nyq], fontsize=7)
ax2.set_xlabel("Spatial frequency  [nm⁻¹]", fontsize=8)

# ── Right: Resolution summary bar chart ───────────────────────────────────
# Bar order (left → right per method): ½-bit (prominent) | 1-bit | FSC=0.5
ax = axes[1]
methods = ["FBP", "Two-pass"]
d_hb    = [d_hb_fbp,  d_hb_tp]   # ½-bit  — primary (van Heel recommended)
d_ob    = [d_ob_fbp,  d_ob_tp]   # 1-bit  — more conservative
d_05    = [d05_fbp,   d05_tp]    # FSC=0.5 — legacy reference

x  = np.arange(len(methods))
w  = 0.24   # narrower to fit three bars

# ½-bit: solid, full saturation — the recommended threshold
bars_hb = ax.bar(x - w, d_hb, w,
                 label="½-bit crit. (recommended)",
                 color=["C0", "C1"], alpha=1.0)
# 1-bit: slightly lighter
bars_ob = ax.bar(x,     d_ob, w,
                 label="1-bit crit.",
                 color=["C0", "C1"], alpha=0.60)
# FSC=0.5: faded + hatched to mark it as legacy
bars_05 = ax.bar(x + w, d_05, w,
                 label="FSC = 0.5  (legacy)",
                 color=["C0", "C1"], alpha=0.30, hatch="//")

for bar in list(bars_hb) + list(bars_ob) + list(bars_05):
    h = bar.get_height()
    if not np.isnan(h):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=7)

ax.axhline(2 * PIXEL_NM, color="gray", ls="--", lw=1.0, alpha=0.8,
           label=f"2-pixel limit ({2*PIXEL_NM:.1f} nm)")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("Resolution  [nm]")
ax.set_title("Resolution comparison\n(smaller = better)")
ax.legend(fontsize=7.5)
ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    f"Fourier Shell Correlation — PXCT two-pass vs FBP\n"
    f"pixel = {PIXEL_NM:.2f} nm  |  volume {Nz}×{Ny}×{Nx}  "
    f"|  apod = {_apod_str}",
    fontsize=10)

fpath = os.path.join(OUT_DIR, "fsc_curves.png")
fig.savefig(fpath, dpi=150, bbox_inches="tight")
print(f"Saved FSC figure: {fpath}")

# ---------------------------------------------------------------------------
# Save numerical FSC data
# ---------------------------------------------------------------------------

np.savez_compressed(
    os.path.join(OUT_DIR, "fsc_data.npz"),
    freq_nyq   = r_mid,
    fsc_fbp    = fsc_fbp,
    fsc_tp     = fsc_tp,
    fsc_cross  = fsc_cross,
    half_bit   = half_bit,
    one_bit    = one_bit,
    n_voxels   = n_vox,
    pixel_nm   = PIXEL_NM,
    apod_width = APOD_WIDTH,
    res_hb_fbp = d_hb_fbp,
    res_hb_tp  = d_hb_tp,
    res_ob_fbp = d_ob_fbp,
    res_ob_tp  = d_ob_tp,
    res_05_fbp = d05_fbp,
    res_05_tp  = d05_tp,
)
print(f"Saved FSC data:   {OUT_DIR}/fsc_data.npz")
