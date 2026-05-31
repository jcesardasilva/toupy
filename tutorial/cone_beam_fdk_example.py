#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cone-beam FDK reconstruction example
=====================================

End-to-end demonstration of the toupy cone-beam pipeline:

  1.  Build a 3-D Shepp-Logan phantom
  2.  Define the acquisition geometry with ConeBeamGeometry
  3.  Forward-project with cone_project  (simulate measurements)
  4.  Inspect ConeBeamRegistration utilities (CoR estimation)
  5.  Reconstruct step-by-step:
        fdk_weight -> fdk_filter -> fdk_backproject
  6.  Reconstruct with the one-call wrapper: fdk_reconstruct
  7.  Use the high-level display wrapper:  fdk_tomo_recons
  8.  Compare phantom central slices with reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt

# ── toupy imports ─────────────────────────────────────────────────────────────
from toupy.simulation import phantom3d
from toupy.tomo.geometry import ConeBeamGeometry
from toupy.tomo.cone_projector import cone_project
from toupy.tomo.fdk import fdk_weight, fdk_filter, fdk_backproject, fdk_reconstruct
from toupy.tomo.tomorecons import fdk_tomo_recons
from toupy.registration.cone_registration import ConeBeamRegistration, rebin_fan_to_parallel


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Acquisition geometry
# ═══════════════════════════════════════════════════════════════════════════════

# Small grid for fast execution in this demo (~2 min on a laptop CPU)
N_PHANTOM = 64    # transaxial voxels
N_V       = 48    # axial (height) voxels / detector rows
N_U       = 64    # detector columns
N_ANGLES  = 180   # projection angles over 360°

# Physical units: millimetres
SOD = 500.0          # source-to-object distance
SDD = 1000.0         # source-to-detector distance
DET_PIXEL = 0.5      # detector pixel pitch [mm]

angles = np.linspace(0, 360, N_ANGLES, endpoint=False)

geom = ConeBeamGeometry(
    SOD=SOD,
    SDD=SDD,
    det_pixel_size=DET_PIXEL,
    n_u=N_U,
    n_v=N_V,
    angles=angles,
)
geom.validate()

print("Geometry summary")
print("  SOD              : {:.1f} mm".format(geom.SOD))
print("  SDD              : {:.1f} mm".format(geom.SDD))
print("  Magnification    : {:.2f}".format(geom.magnification))
print("  Effective voxel  : {:.4f} mm".format(geom.effective_pixel_size))
print("  Detector (u,v)   : ({}, {})".format(geom.n_u, geom.n_v))
print("  Angles           : {} views".format(len(geom.angles)))
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Build phantom and forward-project
# ═══════════════════════════════════════════════════════════════════════════════

print("Building 3-D Shepp-Logan phantom …")
phantom = phantom3d(N=N_PHANTOM, n_v=N_V)       # shape (n_v, N, N)
print("  phantom shape:", phantom.shape)

# cone_project expects volume shape (n_v, N, N)
print("Forward projecting (this may take ~1-2 min on CPU) …")
projections = cone_project(phantom, geom)        # shape (n_angles, n_v, n_u)
print("  projections shape:", projections.shape)
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Registration utilities
# ═══════════════════════════════════════════════════════════════════════════════

reg = ConeBeamRegistration(geom)

# Horizontal CoR estimation from the central detector row
# (sinogram shape must be (n_u, n_angles))
central_sinogram = projections[:, N_V // 2, :].T    # (n_u, n_angles)
cor_shift = reg.estimate_cor(central_sinogram)
print("Centre-of-rotation estimate")
print("  CoR shift (object voxels): {:.4f}".format(cor_shift))
print("  (0 expected for a perfectly centred phantom)")
print()

# If the CoR is non-zero you would call align_horizontal / align_vertical.
# Here the phantom is already perfectly centred, so we skip the correction loop.
# Example of how you would call them:
#
#   shiftstack = np.zeros((2, N_ANGLES))
#   shiftstack = reg.align_horizontal(projections, shiftstack,
#                                     maxit=10, pixtol=0.01,
#                                     freqcutoff=0.8, shiftmeth='fourier',
#                                     filtertype='ram-lak', circle=True)
#   shiftstack = reg.align_vertical(projections, shiftstack,
#                                   maxit=10, pixtol=0.01,
#                                   polyorder=1, deltax=5)


# ═══════════════════════════════════════════════════════════════════════════════
# 4b. Fan-to-parallel rebinning  (enables parallel-beam registration tools)
# ═══════════════════════════════════════════════════════════════════════════════
# The FDK algorithm first applies a cosine (fan) weighting, then runs an
# FBP-like ramp filter — so the raw cone-beam projections look nothing like
# parallel-beam sinograms.  The parallel-beam registration routines (CoR
# estimation, horizontal / vertical alignment) rely on the characteristic
# sinusoidal trace of every object feature, which is only present in a
# parallel-beam sinogram.
#
# rebin_fan_to_parallel() converts the central detector row (a 1-D fan-beam
# sinogram) to an equivalent parallel-beam sinogram via the exact geometric
# mapping:
#
#   φ = θ + γ  (mod π),   s = SOD · sin(γ),   γ = arctan(u / SDD)
#
# where θ is the source angle and u the fan detector coordinate.
# After rebinning every feature traces out a proper sinusoid s(φ) = x₀ cos φ
# + y₀ sin φ, making all parallel-beam alignment tools valid.

print("Fan-to-parallel rebinning of the central detector row …")

# Central row: shape (n_angles, n_u)
central_row = projections[:, N_V // 2, :]   # (n_angles, n_u)

# Rebin — returns parallel sinogram and coordinate arrays
sino_par, s_coords, phi_deg = rebin_fan_to_parallel(
    central_row,
    geom,
    average_redundant=True,   # average the two views of each line in 360° scan
)
# sino_par : (n_u, n_phi),  each column is one parallel projection angle
# s_coords : (n_u,)  offset axis [mm]
# phi_deg  : (n_phi,) angle axis [degrees, 0 … 180)

print("  central_row shape  :", central_row.shape)
print("  sino_par shape     :", sino_par.shape)
print("  s range  [mm]      : [{:.3f}, {:.3f}]".format(s_coords.min(), s_coords.max()))
print("  phi range [deg]    : [{:.1f}, {:.1f}]".format(phi_deg.min(), phi_deg.max()))
print()

# You can now pass sino_par directly to the parallel-beam CoR estimator:
#
#   from toupy.registration.registration import estimate_rot_axis
#   cor_parallel = estimate_rot_axis(sino_par, ...)
#
# or reconstruct the central slice with mod_iradon:
#
#   from toupy.tomo.iradon import mod_iradon
#   central_slice = mod_iradon(sino_par, phi_deg)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Step-by-step FDK reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

print("Step-by-step FDK reconstruction …")

# Step 1: cosine weighting
proj_weighted = fdk_weight(projections, geom)
print("  After fdk_weight  :", proj_weighted.shape)

# Step 2: ramp filtering (row-by-row along u)
proj_filtered = fdk_filter(proj_weighted, filter_type="ram-lak", freqcutoff=1.0)
print("  After fdk_filter  :", proj_filtered.shape)

# Step 3: 3-D divergent-ray back-projection
volume_steps = fdk_backproject(proj_filtered, geom, output_size=N_PHANTOM)
print("  After fdk_backproject:", volume_steps.shape)
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  One-call wrapper: fdk_reconstruct
# ═══════════════════════════════════════════════════════════════════════════════

print("One-call fdk_reconstruct …")
# Use a windowed ramp filter to suppress Gibbs ringing at sharp phantom edges.
# 'shepp-logan' is a mild sinc window; 'hamming' / 'hann' cut off more aggressively.
# Pure 'ram-lak' gives the sharpest result but with the most ringing.
volume = fdk_reconstruct(
    projections,
    geom,
    filter_type="shepp-logan",   # try 'ram-lak', 'hann', 'hamming', 'cosine'
    freqcutoff=1.0,
    output_size=N_PHANTOM,
    cuda=False,
)
print("  volume shape:", volume.shape)
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  High-level display wrapper: fdk_tomo_recons
# ═══════════════════════════════════════════════════════════════════════════════

# fdk_tomo_recons is intended for interactive sessions where you want an
# automatic display of the central slice and optional autosave.
print("High-level fdk_tomo_recons …")
volume_hl = fdk_tomo_recons(
    projections,
    geom,
    filtertype="ram-lak",
    freqcutoff=1.0,
    output_size=N_PHANTOM,
    circle=False,        # set True to apply a cylindrical mask
    showrecons=False,    # set True to pop up the central-slice figure
    colormap="bone",
    vmin_plot=None,
    vmax_plot=None,
    autosave=False,
)
print("  volume_hl shape:", volume_hl.shape)
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Visual comparison
# ═══════════════════════════════════════════════════════════════════════════════

mid_z  = N_V // 2          # central axial slice
mid_xy = N_PHANTOM // 2    # central coronal slice

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 0 — phantom
axes[0, 0].imshow(phantom[mid_z], cmap="bone", origin="lower")
axes[0, 0].set_title("Phantom — axial slice")
axes[0, 0].set_xlabel("x [voxels]"); axes[0, 0].set_ylabel("y [voxels]")

axes[0, 1].imshow(phantom[:, mid_xy, :], cmap="bone", origin="lower")
axes[0, 1].set_title("Phantom — coronal slice")
axes[0, 1].set_xlabel("x [voxels]"); axes[0, 1].set_ylabel("z [voxels]")

axes[0, 2].imshow(projections[0], cmap="bone", origin="lower")
axes[0, 2].set_title("Projection — angle 0°")
axes[0, 2].set_xlabel("u [pixels]"); axes[0, 2].set_ylabel("v [pixels]")

axes[0, 3].imshow(projections[:, mid_z, :], cmap="bone", origin="lower",
                  aspect="auto")
axes[0, 3].set_title("Fan sinogram — central row")
axes[0, 3].set_xlabel("u [pixels]"); axes[0, 3].set_ylabel("angle index")

# Row 1 — reconstruction
recon_clim = dict(
    vmin=np.percentile(volume, 2),
    vmax=np.percentile(volume, 98),
)

axes[1, 0].imshow(volume[mid_z], cmap="bone", origin="lower", **recon_clim)
axes[1, 0].set_title("FDK recon — axial slice")
axes[1, 0].set_xlabel("x [voxels]"); axes[1, 0].set_ylabel("y [voxels]")

axes[1, 1].imshow(volume[:, mid_xy, :], cmap="bone", origin="lower", **recon_clim)
axes[1, 1].set_title("FDK recon — coronal slice")
axes[1, 1].set_xlabel("x [voxels]"); axes[1, 1].set_ylabel("z [voxels]")

# Difference (phantom and recon may differ slightly in scale — normalise each)
def norm01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)

diff_axial   = norm01(phantom[mid_z]) - norm01(volume[mid_z])
diff_coronal = norm01(phantom[:, mid_xy, :]) - norm01(volume[:, mid_xy, :])
clim_diff = dict(vmin=-0.1, vmax=0.1, cmap="seismic")

im3 = axes[1, 2].imshow(diff_axial,   origin="lower", **clim_diff)
axes[1, 2].set_title("Difference — axial")
axes[1, 2].set_xlabel("x [voxels]"); axes[1, 2].set_ylabel("y [voxels]")
plt.colorbar(im3, ax=axes[1, 2])

im4 = axes[1, 3].imshow(diff_coronal, origin="lower", **clim_diff)
axes[1, 3].set_title("Difference — coronal")
axes[1, 3].set_xlabel("x [voxels]"); axes[1, 3].set_ylabel("z [voxels]")
plt.colorbar(im4, ax=axes[1, 3])

plt.tight_layout()
plt.savefig("cone_beam_fdk_result.png", dpi=120)
plt.show()
print("Figure saved to cone_beam_fdk_result.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Fan-to-parallel rebinned sinogram
# ═══════════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

axes2[0].imshow(projections[:, mid_z, :], cmap="bone", origin="lower", aspect="auto")
axes2[0].set_title("Fan-beam sinogram — central row")
axes2[0].set_xlabel("u [detector pixels]")
axes2[0].set_ylabel("source angle index  (0 … 360°)")

extent_par = [phi_deg.min(), phi_deg.max(), s_coords.min(), s_coords.max()]
axes2[1].imshow(sino_par, cmap="bone", origin="lower", aspect="auto",
                extent=extent_par)
axes2[1].set_title("Parallel-beam sinogram — after rebinning")
axes2[1].set_xlabel("φ [degrees]  (0 … 180°)")
axes2[1].set_ylabel("s [mm]  (parallel offset)")

plt.tight_layout()
plt.savefig("cone_beam_rebinned_sinogram.png", dpi=120)
plt.show()
print("Figure saved to cone_beam_rebinned_sinogram.png")
