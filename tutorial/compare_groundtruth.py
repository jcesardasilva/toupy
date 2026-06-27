#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground-truth comparison for local-tomo two-pass reconstruction
==============================================================

Compares the local-tomo result (from twopass_local_data.py) against the
ground-truth ROI extracted from the full-FOV reconstruction (twopass_real_data.py).

Usage
-----
  python tutorial/compare_groundtruth.py \\
      --local  tutorial/twopass_local_figures_localfov20/twopass_local_reconstruction.npz \\
      --full   tutorial/twopass_real_figures/twopass_reconstruction.npz

What counts as "ground truth"
------------------------------
The full reconstruction was done on the complete (non-truncated) projections.
The central sub-volume that corresponds to the local-tomo FOV is therefore an
exact reference: it was not affected by truncation.  We extract that ROI using
the fov_* metadata saved in the local-tomo npz and compare it voxel-by-voxel
against the local-tomo result.

Figures produced
----------------
  fig_gt_cuts.png          — central slices: FBP-init vs two-pass vs ground truth
  fig_gt_profiles.png      — 1-D line profiles through the ROI centre
  fig_gt_histogram.png     — δ histogram comparison (captures DC bias / scale)
  fig_gt_scatter.png       — pixel scatter plot two-pass vs ground truth
  fig_gt_difference.png    — (two-pass − GT) and (FBP − GT) difference maps
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ap = argparse.ArgumentParser(
    description="Compare local-tomo two-pass result against full-FOV ground truth.")
ap.add_argument("--local", required=True,
                help=".npz from twopass_local_data.py "
                     "(contains delta_tp_roi, delta_fbp_roi, fov_* metadata)")
ap.add_argument("--full",  required=True,
                help=".npz from twopass_real_data.py  "
                     "(contains delta_tp as the full-FOV ground truth)")
ap.add_argument("--out",   default=None,
                help="Output directory for figures (default: next to --local file)")
args = ap.parse_args()

for f in (args.local, args.full):
    if not os.path.isfile(f):
        sys.exit(f"File not found: {f}")

OUT_DIR = args.out or os.path.dirname(os.path.abspath(args.local))
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print(f"Loading local-tomo result : {args.local}")
loc  = np.load(args.local, allow_pickle=True)

print(f"Loading full-FOV reference: {args.full}")
full = np.load(args.full,  allow_pickle=True)

# Local-tomo ROI volumes (pre-extracted by twopass_local_data.py)
delta_tp_roi  = loc["delta_tp_roi"].astype(np.float32)
delta_fbp_roi = loc["delta_fbp_roi"].astype(np.float32)

# FOV metadata needed to cut the same ROI from the full reconstruction
fov_x0    = int(loc["fov_x0"])
fov_x1    = int(loc["fov_x1"])
fov_y0    = int(loc["fov_y0"])
fov_y1    = int(loc["fov_y1"])
fov_z0    = int(loc["fov_z0"])
fov_z1    = int(loc["fov_z1"])
full_Nx   = int(loc["full_Nx"])
full_Ny   = int(loc["full_Ny"])
PIXEL_SIZE = float(loc["psize"])

print(f"  ROI  : ({delta_tp_roi.shape})  x[{fov_x0}:{fov_x1}] "
      f"y[{fov_y0}:{fov_y1}] z[{fov_z0}:{fov_z1}]")

# Ground-truth = central ROI of the FULL two-pass reconstruction
delta_gt_full = full["delta_tp"].astype(np.float32)   # (Nz, Ny, Nx)
Nz_gt, Ny_gt, Nx_gt = delta_gt_full.shape

if (Nx_gt, Ny_gt) != (full_Nx, full_Ny):
    sys.exit(
        f"Full reconstruction shape ({Nx_gt},{Ny_gt}) does not match the "
        f"fov metadata ({full_Nx},{full_Ny}).  Make sure both were produced "
        f"with the same base crop (CROP_X=CROP_Y=55 in twopass_real_data.py "
        f"and --base-crop-x/y 55 in local_tomo_simulator.py).")

delta_gt = delta_gt_full[fov_z0:fov_z1, fov_y0:fov_y1, fov_x0:fov_x1]
print(f"  GT   : {delta_gt.shape}  (from full reconstruction ROI)")

if delta_gt.shape != delta_tp_roi.shape:
    sys.exit(
        f"Shape mismatch: local ROI {delta_tp_roi.shape} vs GT {delta_gt.shape}. "
        f"The FOV metadata and the full reconstruction must be from the same "
        f"base-crop run.")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

diff_tp  = delta_tp_roi  - delta_gt
diff_fbp = delta_fbp_roi - delta_gt

def _stats(name, vol, ref):
    d = vol - ref
    mask = ref > 0
    rms  = float(np.sqrt(np.mean(d**2)))
    bias = float(np.mean(d))
    # Normalised RMS error w.r.t. 99th percentile of GT
    p99  = float(np.percentile(ref[ref > 0], 99)) if mask.any() else 1.0
    nrms = rms / p99
    # Pearson correlation (all voxels)
    v1 = vol.ravel().astype(np.float64)
    v2 = ref.ravel().astype(np.float64)
    corr = float(np.corrcoef(v1, v2)[0, 1])
    print(f"  {name:20s}:  RMS-err={rms:.3e}  bias={bias:+.3e}  "
          f"NRMS={nrms:.3f}  Pearson r={corr:.5f}")
    return rms, bias, nrms, corr

print("\nQuantitative metrics (vs ground truth):")
rms_fbp, bias_fbp, nrms_fbp, corr_fbp = _stats("FBP-init ROI",   delta_fbp_roi, delta_gt)
rms_tp,  bias_tp,  nrms_tp,  corr_tp  = _stats("Two-pass ROI",   delta_tp_roi,  delta_gt)
improvement_rms  = (rms_fbp  - rms_tp)  / rms_fbp  * 100
improvement_corr = (corr_tp  - corr_fbp)
print(f"\n  Two-pass vs FBP-init:  RMS reduced by {improvement_rms:.1f}%  "
      f"Δcorr = {improvement_corr:+.5f}")
print()

# ---------------------------------------------------------------------------
# Shared colour scales
# ---------------------------------------------------------------------------

CMAP = "gray"
_pos = delta_gt[delta_gt > 0]
vmax = float(np.percentile(_pos, 99.5)) if _pos.size > 0 else 1e-5
vmin = 0.0

iz = delta_gt.shape[0] // 2
iy = delta_gt.shape[1] // 2
ix = delta_gt.shape[2] // 2

px_nm = PIXEL_SIZE * 1e9

# ---------------------------------------------------------------------------
# Figure 1: Three-way orthogonal cuts
# ---------------------------------------------------------------------------

fig1, axes = plt.subplots(3, 3, figsize=(13, 11),
                           gridspec_kw={"hspace": 0.40, "wspace": 0.30})
rows = [
    (delta_fbp_roi[iz],       delta_tp_roi[iz],      delta_gt[iz],
     f"xy  (z={iz+fov_z0})"),
    (delta_fbp_roi[:,iy,:],   delta_tp_roi[:,iy,:],  delta_gt[:,iy,:],
     f"xz  (y={iy+fov_y0})"),
    (delta_fbp_roi[:,:,ix].T, delta_tp_roi[:,:,ix].T, delta_gt[:,:,ix].T,
     f"yz  (x={ix+fov_x0})"),
]
titles = ["FBP-init", "Two-pass", "Ground truth (full FOV)"]
for row, (fbp_cut, tp_cut, gt_cut, lbl) in enumerate(rows):
    for col, (img, ttl) in enumerate(zip([fbp_cut, tp_cut, gt_cut], titles)):
        im = axes[row, col].imshow(img, cmap=CMAP, vmin=vmin, vmax=vmax,
                                   origin="lower")
        axes[row, col].set_title(f"{ttl}  [{lbl}]", fontsize=7)
        axes[row, col].axis("off")
        plt.colorbar(im, ax=axes[row, col], fraction=0.04, label="δ")
fig1.suptitle(
    "Local-tomo ROI: FBP-init vs two-pass vs ground truth (full-FOV two-pass)\n"
    f"Pearson r  FBP={corr_fbp:.4f}  two-pass={corr_tp:.4f}  "
    f"NRMS  FBP={nrms_fbp:.3f}  two-pass={nrms_tp:.3f}",
    fontsize=9)
p1 = os.path.join(OUT_DIR, "fig_gt_cuts.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
print(f"  Saved: {p1}")

# ---------------------------------------------------------------------------
# Figure 2: Line profiles
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4),
                            gridspec_kw={"wspace": 0.40})
profile_x = np.arange(delta_gt.shape[2]) * px_nm
profile_z = np.arange(delta_gt.shape[0]) * px_nm
profile_y = np.arange(delta_gt.shape[1]) * px_nm

axes2[0].plot(profile_x, delta_gt[iz, iy, :]*1e6, 'k-', lw=1.8, label="GT")
axes2[0].plot(profile_x, delta_tp_roi[iz, iy, :]*1e6, 'b-', lw=1.2, label="two-pass")
axes2[0].plot(profile_x, delta_fbp_roi[iz, iy, :]*1e6, 'r--', lw=1.0, label="FBP")
axes2[0].set_xlabel("x [nm]"); axes2[0].set_ylabel("δ ×10⁻⁶")
axes2[0].set_title(f"x-profile  (z=mid, y=mid)", fontsize=8)
axes2[0].legend(fontsize=7); axes2[0].grid(True, alpha=0.3)

axes2[1].plot(profile_z, delta_gt[:, iy, ix]*1e6, 'k-', lw=1.8, label="GT")
axes2[1].plot(profile_z, delta_tp_roi[:, iy, ix]*1e6, 'b-', lw=1.2, label="two-pass")
axes2[1].plot(profile_z, delta_fbp_roi[:, iy, ix]*1e6, 'r--', lw=1.0, label="FBP")
axes2[1].set_xlabel("z [nm]"); axes2[1].set_ylabel("δ ×10⁻⁶")
axes2[1].set_title(f"z-profile  (y=mid, x=mid)", fontsize=8)
axes2[1].legend(fontsize=7); axes2[1].grid(True, alpha=0.3)

axes2[2].plot(profile_y, delta_gt[iz, :, ix]*1e6, 'k-', lw=1.8, label="GT")
axes2[2].plot(profile_y, delta_tp_roi[iz, :, ix]*1e6, 'b-', lw=1.2, label="two-pass")
axes2[2].plot(profile_y, delta_fbp_roi[iz, :, ix]*1e6, 'r--', lw=1.0, label="FBP")
axes2[2].set_xlabel("y [nm]"); axes2[2].set_ylabel("δ ×10⁻⁶")
axes2[2].set_title(f"y-profile  (z=mid, x=mid)", fontsize=8)
axes2[2].legend(fontsize=7); axes2[2].grid(True, alpha=0.3)

fig2.suptitle("Central line profiles vs ground truth", fontsize=9)
fig2.tight_layout()
p2 = os.path.join(OUT_DIR, "fig_gt_profiles.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
print(f"  Saved: {p2}")

# ---------------------------------------------------------------------------
# Figure 3: δ histograms
# ---------------------------------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, vmax * 1.05, 120)
ax3.hist(delta_gt.ravel() * 1e6,      bins=bins*1e6, histtype='step',
         color='k', lw=1.8, label=f"Ground truth  (full FOV)")
ax3.hist(delta_tp_roi.ravel() * 1e6,  bins=bins*1e6, histtype='step',
         color='b', lw=1.2, label=f"Two-pass ROI  (r={corr_tp:.4f})")
ax3.hist(delta_fbp_roi.ravel() * 1e6, bins=bins*1e6, histtype='step',
         color='r', lw=1.0, ls='--', label=f"FBP-init ROI  (r={corr_fbp:.4f})")
ax3.set_xlabel("δ ×10⁻⁶"); ax3.set_ylabel("Voxel count")
ax3.set_title("δ histogram — DC bias and scale check\n"
              "(distributions should overlap if quantitatively accurate)", fontsize=9)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
p3 = os.path.join(OUT_DIR, "fig_gt_histogram.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
print(f"  Saved: {p3}")

# ---------------------------------------------------------------------------
# Figure 4: Scatter plot two-pass vs GT  (subsample for speed)
# ---------------------------------------------------------------------------

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5),
                            gridspec_kw={"wspace": 0.40})
rng = np.random.default_rng(0)
n_sample = min(50_000, delta_gt.size)
idx      = rng.choice(delta_gt.size, n_sample, replace=False)
gt_s     = delta_gt.ravel()[idx] * 1e6
tp_s     = delta_tp_roi.ravel()[idx] * 1e6
fbp_s    = delta_fbp_roi.ravel()[idx] * 1e6
diag     = np.array([min(gt_s.min(), tp_s.min(), fbp_s.min()),
                     max(gt_s.max(), tp_s.max(), fbp_s.max())])

for ax, (ys, lbl, col) in zip(axes4, [
        (tp_s,  f"Two-pass  r={corr_tp:.4f}", 'steelblue'),
        (fbp_s, f"FBP-init  r={corr_fbp:.4f}", 'tomato')]):
    ax.scatter(gt_s, ys, s=1, alpha=0.15, color=col, rasterized=True)
    ax.plot(diag, diag, 'k--', lw=1.0, label="ideal (y=x)")
    ax.set_xlabel("δ_GT ×10⁻⁶"); ax.set_ylabel("δ_recon ×10⁻⁶")
    ax.set_title(lbl, fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)
    ax.set_aspect('equal')
fig4.suptitle("Voxel scatter: reconstruction vs ground truth\n"
              "(points on the diagonal = perfect quantitative agreement)", fontsize=9)
p4 = os.path.join(OUT_DIR, "fig_gt_scatter.png")
fig4.savefig(p4, dpi=150, bbox_inches="tight")
print(f"  Saved: {p4}")

# ---------------------------------------------------------------------------
# Figure 5: Difference maps  (two-pass − GT)  and  (FBP − GT)
# ---------------------------------------------------------------------------

dlim = np.percentile(np.abs(diff_tp), 99.5)

fig5, axes5 = plt.subplots(2, 3, figsize=(14, 8),
                            gridspec_kw={"hspace": 0.45, "wspace": 0.30})
diff_rows = [
    (diff_tp,  "Two-pass − GT"),
    (diff_fbp, "FBP-init − GT"),
]
for row, (dd, rlbl) in enumerate(diff_rows):
    cuts = [
        (dd[iz],       f"xy (z={iz+fov_z0})"),
        (dd[:,iy,:],   f"xz (y={iy+fov_y0})"),
        (dd[:,:,ix].T, f"yz (x={ix+fov_x0})"),
    ]
    for col, (img, lbl) in enumerate(cuts):
        im = axes5[row, col].imshow(img, cmap="bwr",
                                    vmin=-dlim, vmax=dlim, origin="lower")
        axes5[row, col].set_title(f"{rlbl}  [{lbl}]", fontsize=7)
        axes5[row, col].axis("off")
        plt.colorbar(im, ax=axes5[row, col], fraction=0.04, label="Δδ")
fig5.suptitle("Difference maps vs ground truth  "
              f"(colour scale ±{dlim*1e6:.2f} ×10⁻⁶)", fontsize=9)
p5 = os.path.join(OUT_DIR, "fig_gt_difference.png")
fig5.savefig(p5, dpi=150, bbox_inches="tight")
print(f"  Saved: {p5}")

plt.close("all")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("GROUND-TRUTH COMPARISON SUMMARY")
print("=" * 65)
print(f"  ROI shape            : {delta_tp_roi.shape}")
print(f"  FOV fraction         : {(fov_x1-fov_x0)/full_Nx:.0%} x "
      f"{(fov_y1-fov_y0)/full_Ny:.0%} y")
print()
print(f"  {'Metric':<22} {'FBP-init':>12} {'Two-pass':>12}  {'Improvement':>12}")
print(f"  {'-'*60}")
print(f"  {'RMS error':<22} {rms_fbp*1e6:>11.4f}  {rms_tp*1e6:>11.4f}  "
      f"{improvement_rms:>+10.1f}%")
print(f"  {'Bias (mean Δδ)':<22} {bias_fbp*1e6:>+11.4f}  {bias_tp*1e6:>+11.4f}  "
      f"{'(×10⁻⁶)':>12}")
print(f"  {'Norm. RMS (vs 99th %)':<22} {nrms_fbp:>12.4f}  {nrms_tp:>12.4f}")
print(f"  {'Pearson r':<22} {corr_fbp:>12.5f}  {corr_tp:>12.5f}  "
      f"{improvement_corr:>+10.5f}")
print()
print(f"  Figures saved to : {OUT_DIR}/")
print("=" * 65)
