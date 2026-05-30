#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: nonlinear iterative phase retrieval vs TIE-Hom (Paganin).

TIE-Hom is a *first-order* (single-fringe) phase-retrieval filter.  When the
propagation distance is large enough that the projection shows **several
Fresnel fringes** (small Fresnel number), or when δ/β is large, the Paganin
filter blurs the image and the recovered grey levels are no longer
quantitative.

This example shows how to refine the TIE-Hom result with
``iterative_phase_retrieval``, which inverts the **exact** Fresnel forward
model (angular-spectrum propagation) by nonlinear conjugate-gradient
minimisation, using TIE-Hom as the initial guess.  It demonstrates:

    1.  Single-distance refinement in the multi-fringe regime
        (TIE-Hom blurs → iterative sharpens and becomes quantitative).
    2.  Multi-distance NONLINEAR holotomography, which stays valid for
        strong phase where the linear CTF breaks down.

Run with:
    python tutorial/example_iterative_phase_retrieval.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import cupy as cp          # noqa: F401
    GPU_AVAILABLE = True
    print("CuPy detected — GPU path available (pass cuda=True).")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found — running on CPU.")

from toupy.simulation import phantom
from toupy.restoration import tie_hom, iterative_phase_retrieval

# ---------------------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------------------
N          = 256
PIXEL_SIZE = 50e-9
ENERGY_keV = 17.0
WAVELENGTH = 1.23984193e-6 / (ENERGY_keV * 1e3)
DELTA_BETA = 20.0                       # homogeneous object

# Multi-fringe regime: a practical (effective) propagation distance that
# produces several Fresnel fringes per edge — exactly where TIE-Hom struggles.
Z_SINGLE   = 0.8e-3                      # 0.8 mm  -> NF ≈ 0.043
Z_MULTI    = [3e-5, 1.5e-4, 6e-4, 2.4e-3]   # holotomographic distance series

NF = PIXEL_SIZE**2 / (WAVELENGTH * Z_SINGLE)
print("Wavelength       : {:.4e} m".format(WAVELENGTH))
print("Fresnel NF (z={:.1f}mm) : {:.3f}  (multi-fringe)".format(Z_SINGLE * 1e3, NF))

# ---------------------------------------------------------------------------
# 1. Homogeneous phantom INSIDE the field of view (vacuum margin) so that the
#    propagation fringes are captured by the detector.
# ---------------------------------------------------------------------------
MARGIN = 64
core   = N - 2 * MARGIN
obj    = phantom(core, phantom_type="Modified Shepp-Logan") * 1.0   # ~1 rad peak
phase_true = np.zeros((N, N))
phase_true[MARGIN:MARGIN + core, MARGIN:MARGIN + core] = obj
mask = phase_true > 0           # object support (for quantitative metrics)


# ---------------------------------------------------------------------------
# 2. Forward model: homogeneous object propagated with the angular spectrum.
# ---------------------------------------------------------------------------
def simulate(distance, pad=2, seed=7):
    """Return the noisy detector intensity for the homogeneous phantom."""
    M = N * pad
    o = (M - N) // 2
    fy = np.fft.fftfreq(M, d=PIXEL_SIZE)
    fx = np.fft.fftfreq(M, d=PIXEL_SIZE)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    H = np.exp(1j * np.pi * WAVELENGTH * distance * (FY**2 + FX**2))
    c = (-1.0 / DELTA_BETA + 1j)
    field = np.zeros((M, M), dtype=complex)
    field[o:o + N, o:o + N] = np.exp(c * phase_true)
    field[:o, :] = 1.0; field[o + N:, :] = 1.0       # vacuum surround
    field[:, :o] = 1.0; field[:, o + N:] = 1.0
    psz = np.fft.ifft2(np.fft.fft2(field) * H)
    I = (np.abs(psz)**2)[o:o + N, o:o + N]
    rng = np.random.default_rng(seed)
    return np.maximum(I + rng.normal(0, 0.005 * I.mean(), I.shape), 0)


def rmse_abs(a, b):
    return np.sqrt(np.mean((a[mask] - b[mask])**2))


def corr(a, b):
    return np.corrcoef(a[mask], b[mask])[0, 1]


# ---------------------------------------------------------------------------
# 3. Single-distance: TIE-Hom vs iterative refinement
# ---------------------------------------------------------------------------
print("\n--- Single distance (multi-fringe) ---")
I_single = simulate(Z_SINGLE)

phase_tie = tie_hom(I_single, Z_SINGLE, WAVELENGTH, PIXEL_SIZE,
                    delta_beta=DELTA_BETA, regularisation=1e-3)

# Total-variation regularisation (reg_tv) flattens the residual Fresnel
# ripples in the homogeneous interior while preserving the sharp edges —
# giving a single-distance result close to the multi-distance one.
phase_iter = iterative_phase_retrieval(
    I_single, Z_SINGLE, WAVELENGTH, PIXEL_SIZE,
    delta_beta=DELTA_BETA, init=phase_tie, n_iter=300,
    reg_smooth=0.0, reg_tv=3e-3, verbose=True,
)

print("  TIE-Hom    : abs-RMSE = {:.4f} rad   corr = {:.4f}".format(
    rmse_abs(phase_tie, phase_true), corr(phase_tie, phase_true)))
print("  Iterative  : abs-RMSE = {:.4f} rad   corr = {:.4f}".format(
    rmse_abs(phase_iter, phase_true), corr(phase_iter, phase_true)))

# ---------------------------------------------------------------------------
# 4. Multi-distance NONLINEAR holotomography
# ---------------------------------------------------------------------------
print("\n--- Multi-distance nonlinear holotomography ---")
I_multi = np.array([simulate(z) for z in Z_MULTI])
phase_holo = iterative_phase_retrieval(
    I_multi, Z_MULTI, WAVELENGTH, PIXEL_SIZE,
    delta_beta=DELTA_BETA, n_iter=300, reg_smooth=1e-4,
)
print("  Holo (NL)  : abs-RMSE = {:.4f} rad   corr = {:.4f}".format(
    rmse_abs(phase_holo, phase_true), corr(phase_holo, phase_true)))

# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------
vmin, vmax = phase_true.min(), phase_true.max() * 1.05
panels = [
    (phase_true,  "Ground truth"),
    (phase_tie,   "TIE-Hom\n(blurred, multi-fringe)"),
    (phase_iter,  "Iterative (single z)\nfull Fresnel model"),
    (phase_holo,  "Iterative (4 z)\nnonlinear holotomography"),
]
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, (img, title) in zip(axes, panels):
    im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(
    "Nonlinear iterative phase retrieval  "
    "({:.0f} keV, pixel {:.0f} nm, z = {:.1f} mm, NF = {:.3f}, δ/β = {:.0f})".format(
        ENERGY_keV, PIXEL_SIZE * 1e9, Z_SINGLE * 1e3, NF, DELTA_BETA),
    fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("iterative_phase_retrieval.png", dpi=150, bbox_inches="tight")
print("\nSaved iterative_phase_retrieval.png")

# Line profile through the object centre (quantitative comparison)
row = N // 2
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(phase_true[row], "k-", lw=2, label="ground truth")
ax.plot(phase_tie[row],  "C3--", label="TIE-Hom")
ax.plot(phase_iter[row], "C0-", label="iterative (single z)")
ax.plot(phase_holo[row], "C2-", label="iterative (multi z)")
ax.set_xlabel("pixel"); ax.set_ylabel("phase [rad]")
ax.set_title("Central line profile — quantitative grey levels")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("iterative_phase_profile.png", dpi=150)
print("Saved iterative_phase_profile.png")
print("\nDone.")
