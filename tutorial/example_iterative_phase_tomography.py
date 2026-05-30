#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: full phase-contrast TOMOGRAPHY pipeline with iterative phase retrieval.

This extends ``example_iterative_phase_retrieval.py`` (single 2-D projection)
to a complete tomographic reconstruction:

    1.  A homogeneous slice (projected refractive-index map) is rotated;
        its line integrals form the clean phase sinogram.
    2.  For every projection angle the wavefield is propagated (Fresnel /
        angular spectrum) to give the measured intensity — for a single
        distance AND for a 4-distance holotomographic series.
    3.  Each projection is phase-retrieved three ways:
            - TIE-Hom (Paganin)
            - iterative, single distance      (full Fresnel model + TV)
            - iterative, 4 distances          (nonlinear holotomography)
    4.  The retrieved sinograms are reconstructed by FBP and compared to the
        ground-truth slice.

Physics note
------------
For a single slice the object is invariant along the rotation axis, so the
2-D Fresnel propagation reduces EXACTLY to a 1-D problem along the detector.
Each projection is therefore retrieved as a 1-D line (fast), and the loop over
~150 angles runs in well under a minute on a CPU.  Pass ``cuda=True`` to the
retrieval calls to use the GPU.

The object is scaled so the maximum projected phase is ~1.2 rad: tomographic
line integrals accumulate phase over the whole ray, and keeping it below ~pi
avoids phase-wrapping ambiguity in single-distance retrieval.

Run with:
    python tutorial/example_iterative_phase_tomography.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from tqdm import tqdm
except ImportError:                       # pragma: no cover
    def tqdm(x, **k):
        return x

try:
    import cupy as cp          # noqa: F401
    GPU_AVAILABLE = True
    print("CuPy detected — pass cuda=True to the retrieval calls for GPU.")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found — running on CPU.")

from toupy.simulation import phantom
from toupy.restoration import (tie_hom, iterative_phase_retrieval,
                               suggest_holo_distances)

# ---------------------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------------------
N          = 192
PIXEL_SIZE = 50e-9
ENERGY_keV = 17.0
WAVELENGTH = 1.23984193e-6 / (ENERGY_keV * 1e3)
DELTA_BETA = 20.0
N_ANGLES   = 150
THETA      = np.linspace(0, 180, N_ANGLES, endpoint=False)

Z_SINGLE   = 0.8e-3                          # NF ≈ 0.043 (multi-fringe)
MARGIN     = 48                             # vacuum margin (captures fringes)
N_ITER     = 120
MAX_PHASE  = 1.2                            # peak projected phase [rad]

# Holotomographic distance series chosen automatically from the CTF-zero
# interleaving rules (geometric spacing, zero-free shortest distance).
Z_MULTI, holo_info = suggest_holo_distances(
    PIXEL_SIZE, WAVELENGTH, n=4, delta_beta=DELTA_BETA, return_info=True)

NF = PIXEL_SIZE**2 / (WAVELENGTH * Z_SINGLE)
print("Fresnel NF (single z = {:.1f} mm) : {:.3f}".format(Z_SINGLE * 1e3, NF))
print("Suggested 4-z series [µm]   : {}  (ratio R = {:.2f})".format(
    ", ".join("{:.1f}".format(z * 1e6) for z in Z_MULTI), holo_info["ratio"]))
print("  per-distance NF           : {}".format(
    ", ".join("{:.3f}".format(v) for v in holo_info["NF"])))
print("  worst combined transfer   : {:.3f}  (gap-free if > ~0.02)".format(
    holo_info["min_coverage"]))

# ---------------------------------------------------------------------------
# 1. Ground-truth slice and clean phase sinogram
# ---------------------------------------------------------------------------
slice_raw  = phantom(N, phantom_type="Modified Shepp-Logan")
sino_clean = radon(slice_raw, theta=THETA, circle=True)
scale      = MAX_PHASE / sino_clean.max()
slice_true = slice_raw * scale
sino_clean = sino_clean * scale
n_det      = sino_clean.shape[0]
L          = n_det + 2 * MARGIN
print("Sinogram (n_det x n_ang):", sino_clean.shape,
      " max projected phase = {:.2f} rad".format(sino_clean.max()))


# ---------------------------------------------------------------------------
# 2. 1-D Fresnel forward model for one projection line
# ---------------------------------------------------------------------------
def propagate_line(proj, distance):
    """Return the detector intensity (length L, vacuum=1 margins) for one
    homogeneous projection line propagated over *distance*."""
    pe = np.zeros(L)
    pe[MARGIN:MARGIN + n_det] = proj
    M = L * 2
    o = (M - L) // 2
    fx = np.fft.fftfreq(M, d=PIXEL_SIZE)
    H = np.exp(1j * np.pi * WAVELENGTH * distance * fx**2)
    c = (-1.0 / DELTA_BETA + 1j)
    field = np.ones(M, dtype=complex)              # vacuum = 1 outside object
    field[o:o + L] = np.exp(c * pe)
    inten = np.abs(np.fft.ifft2(
        np.fft.fft2(field.reshape(1, -1)) * H.reshape(1, -1)))**2
    return inten[0][o:o + L]


rng = np.random.default_rng(0)


def add_noise(I, level=0.005):
    return np.maximum(I + rng.normal(0, level * I.mean(), I.shape), 0)


# ---------------------------------------------------------------------------
# 3. Per-angle phase retrieval (TIE-Hom, iterative 1z, iterative 4z)
# ---------------------------------------------------------------------------
sino_tie  = np.zeros_like(sino_clean)
sino_it1  = np.zeros_like(sino_clean)
sino_it4  = np.zeros_like(sino_clean)

print("\nRetrieving {} projections (3 methods)...".format(N_ANGLES))
t0 = time.perf_counter()
for k in tqdm(range(N_ANGLES)):
    col = sino_clean[:, k]

    # --- single distance ---
    I1 = add_noise(propagate_line(col, Z_SINGLE))[None, :]
    sino_tie[:, k] = tie_hom(
        I1, Z_SINGLE, WAVELENGTH, PIXEL_SIZE, delta_beta=DELTA_BETA
    )[0][MARGIN:MARGIN + n_det]
    sino_it1[:, k] = iterative_phase_retrieval(
        I1, Z_SINGLE, WAVELENGTH, PIXEL_SIZE, delta_beta=DELTA_BETA,
        n_iter=N_ITER, reg_tv=3e-3,
    )[0][MARGIN:MARGIN + n_det]

    # --- multi distance (holotomography) ---
    I4 = add_noise(np.array([propagate_line(col, z) for z in Z_MULTI]))
    sino_it4[:, k] = iterative_phase_retrieval(
        I4[:, None, :], Z_MULTI, WAVELENGTH, PIXEL_SIZE, delta_beta=DELTA_BETA,
        n_iter=N_ITER, reg_tv=1e-3,
    )[0][MARGIN:MARGIN + n_det]
print("  retrieval done in {:.1f} s".format(time.perf_counter() - t0))

# ---------------------------------------------------------------------------
# 4. Tomographic reconstruction (FBP)
# ---------------------------------------------------------------------------
def fbp(sino):
    return iradon(sino, theta=THETA, filter_name="ramp", circle=True)


rec_true = fbp(sino_clean)
rec_tie  = fbp(sino_tie)
rec_it1  = fbp(sino_it1)
rec_it4  = fbp(sino_it4)


def normalise(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-30)


def rmse(a):
    return np.sqrt(np.mean((normalise(a) - normalise(slice_true))**2))


print("\n--- Tomogram RMSE (normalised, vs ground-truth slice) ---")
print("  FBP(clean sinogram)        : {:.4f}".format(rmse(rec_true)))
print("  FBP(TIE-Hom retrieved)     : {:.4f}".format(rmse(rec_tie)))
print("  FBP(iterative single-z)    : {:.4f}".format(rmse(rec_it1)))
print("  FBP(iterative 4-z holo)    : {:.4f}".format(rmse(rec_it4)))

# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
panels = [
    (slice_true, "Ground-truth slice"),
    (rec_tie,    "FBP  •  TIE-Hom\nphase retrieval"),
    (rec_it1,    "FBP  •  iterative single-z\n(full Fresnel + TV)"),
    (rec_it4,    "FBP  •  iterative 4-z\n(nonlinear holotomography)"),
]
vmin, vmax = slice_true.min(), slice_true.max() * 1.05
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
for ax, (img, title) in zip(axes, panels):
    im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(
    "Phase-contrast tomography: TIE-Hom vs iterative phase retrieval  "
    "({:.0f} keV, {:.0f} nm, z = {:.1f} mm, NF = {:.3f}, δ/β = {:.0f})".format(
        ENERGY_keV, PIXEL_SIZE * 1e9, Z_SINGLE * 1e3, NF, DELTA_BETA),
    fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("iterative_phase_tomography.png", dpi=150, bbox_inches="tight")
print("\nSaved iterative_phase_tomography.png")

# Line profile through the reconstructed slice centre
row = N // 2
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(slice_true[row], "k-", lw=2, label="ground truth")
ax.plot(rec_tie[row], "C3--", label="TIE-Hom")
ax.plot(rec_it1[row], "C0-", label="iterative single-z")
ax.plot(rec_it4[row], "C2-", label="iterative 4-z")
ax.set_xlabel("pixel"); ax.set_ylabel("reconstructed value")
ax.set_title("Central line profile through the tomogram")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("iterative_phase_tomography_profile.png", dpi=150)
print("Saved iterative_phase_tomography_profile.png")

# ---------------------------------------------------------------------------
# 6. Frequency-space view — WHY single-distance has gaps and multi-z fills them
# ---------------------------------------------------------------------------
# Homogeneous phase transfer function at distance z:
#     w_z(u) = sin(chi) + eps*cos(chi),   chi = pi*lambda*z*u^2,  eps = 1/(delta/beta)
# Phase is lost where w_z(u) = 0.  The pure-phase part sin(chi) vanishes at
#     chi = n*pi   ->   u_n = sqrt( n / (lambda*z) ),   n = 1, 2, 3, ...
# i.e. in units of the Nyquist frequency u_Nyq = 1/(2*pixel):
#     f_n = u_n / u_Nyq = 2 * sqrt( n * NF ),   NF = pixel^2 / (lambda*z).
# Spacing the distances geometrically makes these zeros INTERLEAVE, so the
# combined coverage never drops to zero.
u_nyq = 1.0 / (2 * PIXEL_SIZE)
u = np.linspace(0, u_nyq, 2000)
fnorm = u / u_nyq
eps = 1.0 / DELTA_BETA


def transfer(z):
    chi = np.pi * WAVELENGTH * z * u**2
    return np.sin(chi) + eps * np.cos(chi)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.2))

# Panel A — single distance (gaps) vs combined multi-z coverage
axA.plot(fnorm, np.abs(transfer(Z_SINGLE)), "C3",
         label="single  z = {:.1f} mm".format(Z_SINGLE * 1e3))
combined = np.sqrt(np.mean([transfer(z)**2 for z in Z_MULTI], axis=0))
axA.plot(fnorm, combined, "C2", lw=2.2,
         label=r"4-z combined  $\sqrt{\langle w_z^2\rangle}$")
axA.axhline(0, color="k", lw=0.5)
axA.set_xlabel(r"spatial frequency  $u/u_{\mathrm{Nyq}}$")
axA.set_ylabel("|phase transfer|")
axA.set_title("Single distance dips to zero at its CTF zeros (gaps);\n"
              "the 4-z combination stays bounded away from zero")
axA.legend(fontsize=8)

# Panel B — individual distances: interleaved zeros
for z in Z_MULTI:
    nf = PIXEL_SIZE**2 / (WAVELENGTH * z)
    axB.plot(fnorm, np.abs(transfer(z)),
             label="z = {:6.0f} µm  (NF = {:.2f})".format(z * 1e6, nf))
axB.set_xlabel(r"spatial frequency  $u/u_{\mathrm{Nyq}}$")
axB.set_ylabel(r"$|w_z(u)|$")
axB.set_title(r"Individual distances: zeros at $u_n=\sqrt{n/(\lambda z)}$"
              "\nshort z → low-freq, long z → high-freq")
axB.legend(fontsize=8)

plt.tight_layout()
plt.savefig("ctf_frequency_coverage.png", dpi=150)
print("Saved ctf_frequency_coverage.png")
print("\nDone.")
