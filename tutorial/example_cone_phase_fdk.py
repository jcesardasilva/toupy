#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: cone-beam propagation phase-contrast tomography (phase retrieval + FDK).

Demonstrates that the single-distance non-linear phase retrieval works in
cone-beam geometry through the Fresnel scaling theorem, feeding the FDK
reconstruction of ``toupy.tomo``:

    1.  A homogeneous 3-D phantom is forward-projected with the cone-beam
        projector to give the projected phase per angle.
    2.  Each projection is propagated (parallel-beam Fresnel at the *effective*
        distance) to synthesise the measured cone-beam intensity.
    3.  Each projection is phase-retrieved two ways --- TIE-Hom and the
        non-linear iterative solver (with TV) --- in the effective frame.
    4.  Each retrieved phase stack is reconstructed with FDK and compared to
        the ground-truth volume (and to FDK of the clean projected phase).

Run with:
    python tutorial/example_cone_phase_fdk.py
"""

import os
import sys
import time
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import cupy as cp          # noqa: F401
    GPU = True
    print("CuPy detected — pass cuda=True for the GPU path.")
except ImportError:
    GPU = False
    print("CuPy not found — running on CPU.")

from toupy.simulation import phantom3d
from toupy.tomo import (ConeBeamGeometry, cone_project, fdk_reconstruct,
                        cone_phase_retrieval_fdk, effective_fresnel_distance)
from toupy.restoration import tie_hom

# ---------------------------------------------------------------------------
# 0. Parameters (a representative lab/micro-CT cone-beam geometry)
# ---------------------------------------------------------------------------
ENERGY_keV = 17.0
WAVELENGTH = 1.23984193e-6 / (ENERGY_keV * 1e3)
DELTA_BETA = 50.0
N        = 96            # transaxial detector columns / recon grid
N_V      = 32            # detector rows (axial)
N_ANG    = 120
MARGIN   = 14           # transaxial vacuum margin (captures fringes)
Z_MARGIN = 5            # axial vacuum margin
MAX_PHASE = 1.0         # peak projected phase [rad] (avoid wrapping)

geom = ConeBeamGeometry(
    SOD=0.15, SDD=0.30, det_pixel_size=2e-6,    # M = 2, eff. pixel = 1 um
    n_u=N, n_v=N_V,
    angles=np.linspace(0, 360, N_ANG, endpoint=False),
)
Z_EFF = effective_fresnel_distance(geom)
PX_EFF = geom.effective_pixel_size
NF = PX_EFF**2 / (WAVELENGTH * Z_EFF)
print("Magnification      : {:.2f}".format(geom.magnification))
print("Effective pixel    : {:.0f} nm".format(PX_EFF * 1e9))
print("Effective distance : {:.1f} mm  (Fresnel NF = {:.3f})".format(Z_EFF * 1e3, NF))

# ---------------------------------------------------------------------------
# 1. Homogeneous phantom inside the FOV (vacuum margins)
# ---------------------------------------------------------------------------
core = N - 2 * MARGIN
core_z = N_V - 2 * Z_MARGIN
vol_core = phantom3d(N=core, n_v=core_z, phantom_type="Modified Shepp-Logan")
volume = np.zeros((N_V, N, N))
zo, o = (N_V - core_z) // 2, (N - core) // 2
volume[zo:zo + core_z, o:o + core, o:o + core] = vol_core

# ---------------------------------------------------------------------------
# 2. Forward model: cone projection -> projected phase -> Fresnel intensity
# ---------------------------------------------------------------------------
print("\nForward projecting (cone beam)...")
proj_phase = cone_project(volume, geom)              # (N_ANG, N_V, N)
proj_phase *= MAX_PHASE / proj_phase.max()           # scale peak phase
print("  projected phase: shape {}, peak {:.2f} rad".format(
    proj_phase.shape, proj_phase.max()))

c = (-1.0 / DELTA_BETA + 1j)


def fresnel_intensity(phi, z, px, pad=2):
    """Parallel-beam Fresnel intensity of a homogeneous projection (vacuum=1)."""
    nv, nu = phi.shape
    my, mx = nv * pad, nu * pad
    oy, ox = (my - nv) // 2, (mx - nu) // 2
    fy, fx = fftfreq(my, d=px), fftfreq(mx, d=px)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    H = np.exp(1j * np.pi * WAVELENGTH * z * (FY**2 + FX**2))
    field = np.ones((my, mx), dtype=complex)
    field[oy:oy + nv, ox:ox + nu] = np.exp(c * phi)
    return (np.abs(ifft2(fft2(field) * H))**2)[oy:oy + nv, ox:ox + nu]


rng = np.random.default_rng(0)
intensity = np.array([fresnel_intensity(proj_phase[k], Z_EFF, PX_EFF)
                      for k in range(N_ANG)])
intensity = np.maximum(intensity + rng.normal(0, 0.005 * intensity.mean(),
                                              intensity.shape), 0)

# ---------------------------------------------------------------------------
# 3. Reconstructions
# ---------------------------------------------------------------------------
def fdk(stack):
    return fdk_reconstruct(stack, geom)


print("\nFDK of the clean projected phase (reference)...")
rec_clean = fdk(proj_phase)

print("TIE-Hom retrieval + FDK...")
t = time.perf_counter()
tie_stack = np.array([
    tie_hom(intensity[k], Z_EFF, WAVELENGTH, PX_EFF, delta_beta=DELTA_BETA)
    for k in range(N_ANG)])
rec_tie = fdk(tie_stack)
print("  done in {:.1f} s".format(time.perf_counter() - t))

print("Iterative retrieval + FDK (cone_phase_retrieval_fdk)...")
t = time.perf_counter()
rec_iter = cone_phase_retrieval_fdk(
    intensity, geom, WAVELENGTH, DELTA_BETA,
    n_iter=80, reg_tv=2e-3, verbose=True)
print("  done in {:.1f} s".format(time.perf_counter() - t))

# ---------------------------------------------------------------------------
# 4. Metrics + figure (central axial slice)
# ---------------------------------------------------------------------------
def nrm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-30)


def rmse(a):
    return np.sqrt(np.mean((nrm(a) - nrm(volume))**2))


print("\n--- Volume RMSE (normalised, vs ground-truth phantom) ---")
print("  FDK(clean projected phase) : {:.4f}".format(rmse(rec_clean)))
print("  FDK(TIE-Hom retrieved)     : {:.4f}".format(rmse(rec_tie)))
print("  FDK(iterative retrieved)   : {:.4f}".format(rmse(rec_iter)))

zc = N_V // 2
# FDK returns the refractive-index contrast on its own absolute scale, so each
# panel is normalised to [0, 1] for a fair visual comparison (this matches how
# the RMSE above is computed).
panels = [
    (volume[zc],    "Ground-truth slice"),
    (rec_tie[zc],   "FDK  •  TIE-Hom"),
    (rec_iter[zc],  "FDK  •  iterative (+TV)"),
    (rec_clean[zc], "FDK  •  clean phase\n(reference)"),
]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
for ax, (img, title) in zip(axes, panels):
    im = ax.imshow(nrm(img), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(
    "Cone-beam phase-contrast tomography (retrieval + FDK)  "
    "({:.0f} keV, M={:.1f}, eff.\\ pixel {:.0f} nm, z_eff={:.0f} mm, "
    "NF={:.3f})".format(ENERGY_keV, geom.magnification, PX_EFF * 1e9,
                        Z_EFF * 1e3, NF),
    fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("cone_phase_fdk.png", dpi=150, bbox_inches="tight")
print("\nSaved cone_phase_fdk.png")
print("Done.")
