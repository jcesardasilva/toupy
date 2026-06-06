#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: magnified inline-holography multi-distance CTF (ESRF ID16A style).

Simulates a focused-beam (cone-beam) holotomography projection acquired at four
defocus positions and reconstructs the phase with the full
``toupy.restoration.holo_ctf_reconstruct`` pipeline:

    flat-field  ->  rescale to a common pixel  ->  sub-pixel align  ->  CTF

Demonstrations:
  * the four positions' magnifications / effective pixels / Fresnel fringes and
    a **frequency-coverage diagnostic** of the multi-distance CTF;
  * the **regularisation / cupping** trade-off (``alpha``): too large an alpha
    attenuates the low frequencies and bows the interior grey levels;
  * dynamic **eigen-flat** correction recovering a **drifting beam** that plain
    flat-fielding cannot remove.

Run with:
    python tutorial/example_holo_ctf_id16a.py
"""

import os
import sys
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import affine_transform, fourier_shift, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toupy.simulation import phantom
from toupy.restoration import holo_ctf_reconstruct, holo_geometry

# ---------------------------------------------------------------------------
# 0. Geometry (focused beam: 4 defocus positions, different magnifications)
# ---------------------------------------------------------------------------
ENERGY_keV = 17.0
WAVELENGTH = 1.23984193e-6 / (ENERGY_keV * 1e3)
DELTA_BETA = 50.0
ZD = 0.40                                        # focus-detector distance [m]
DET_PIXEL = 1.0e-6                               # detector pixel [m]
ZS = np.array([50.0, 85.0, 130.0, 200.0]) * 1e-3  # focus-sample distances [m]
# (a wide z_eff spread, ~44-100 mm, gives the low-frequency coverage needed for
#  quantitative grey levels; a too-small max distance leaves interior cupping)

geom = holo_geometry(ZS, ZD, DET_PIXEL)
NF = geom.effective_pixel_size**2 / (WAVELENGTH * geom.effective_distance)
print("Magnifications     :", np.round(geom.magnification, 2))
print("Effective pixel nm :", np.round(geom.effective_pixel_size * 1e9, 1))
print("Effective dist mm  :", np.round(geom.effective_distance * 1e3, 2))
print("Effective NF       :", np.round(NF, 4))

N_DET = 128
N_FINE = 512
PX1 = geom.effective_pixel_size.min()

# ---------------------------------------------------------------------------
# 1. Object + forward model
# ---------------------------------------------------------------------------
CORE = 80
obj = phantom(CORE, "Modified Shepp-Logan") * 0.3
phase_fine = np.zeros((N_FINE, N_FINE))
o = (N_FINE - CORE) // 2
phase_fine[o:o + CORE, o:o + CORE] = obj
c = (-1.0 / DELTA_BETA + 1j)


def propagate(psi, z, px, pad=2):
    n = psi.shape[0]; m = n * pad; off = (m - n) // 2
    f = fftfreq(m, d=px); FY, FX = np.meshgrid(f, f, indexing="ij")
    H = np.exp(1j * np.pi * WAVELENGTH * z * (FY**2 + FX**2))
    fld = np.ones((m, m), dtype=complex); fld[off:off + n, off:off + n] = psi
    return (np.abs(ifft2(fft2(fld) * H))**2)[off:off + n, off:off + n]


def to_detector(img_fine, scale):
    img = gaussian_filter(img_fine, 0.5 * scale) if scale > 1 else img_fine
    ni = img.shape[0]
    off = (ni - 1) / 2.0 - scale * (N_DET - 1) / 2.0
    return affine_transform(img, [scale, scale], offset=[off, off],
                            output_shape=(N_DET, N_DET), order=3, mode="nearest")


yy, xx = np.mgrid[0:N_FINE, 0:N_FINE]
beam0 = (1.0 + 0.3 * np.sin(2 * np.pi * xx / N_FINE * 4)
         * np.cos(2 * np.pi * yy / N_FINE * 4) + 0.1 * xx / N_FINE)
mode1 = 0.15 * np.sin(2 * np.pi * yy / N_FINE * 3)
mode2 = 0.15 * np.cos(2 * np.pi * (xx + yy) / N_FINE * 5)
DARK = 50.0
shifts = np.array([[0, 0], [1.3, -0.7], [-0.8, 1.1], [0.6, 2.2]])

gt = phase_fine[(N_FINE - N_DET) // 2:(N_FINE + N_DET) // 2,
                (N_FINE - N_DET) // 2:(N_FINE + N_DET) // 2]
mask = gt > 0.01 * gt.max()


def holograms(beam_modes=None, n_flats=1, seed=0):
    """Return (samples, references) for a 4-distance acquisition.

    beam_modes : None for a stable beam, else amplitudes of the drift modes."""
    rng = np.random.default_rng(seed)
    S, R = [], []
    for i in range(4):
        field = propagate(np.exp(c * phase_fine), geom.effective_distance[i], PX1)
        field = np.real(ifft2(fourier_shift(fft2(field), shifts[i])))
        s = geom.effective_pixel_size[i] / PX1
        if beam_modes is None:
            beam_s = to_detector(beam0, s)
            flats = (to_detector(beam0, s) + DARK)[None]
        else:
            w = rng.normal(0, 1, 2)
            beam_s = to_detector(beam0 + w[0] * mode1 + w[1] * mode2, s)
            flats = np.array([
                to_detector(beam0 + rng.normal(0, 1) * mode1
                            + rng.normal(0, 1) * mode2, s) + DARK
                for _ in range(n_flats)])
        S.append(beam_s * to_detector(field, s) + DARK)
        R.append(flats)
    S = np.array(S) + rng.normal(0, 0.003, (4, N_DET, N_DET))
    return S, np.array(R)


def reco(samples, references, flat_method="simple", alpha=1e-4,
         method="ctf", refine_align=False):
    p = holo_ctf_reconstruct(samples, references, DARK, ZS, ZD, DET_PIXEL,
                             WAVELENGTH, alpha=alpha, delta_beta=DELTA_BETA,
                             flat_method=flat_method, n_eigen=4, method=method,
                             align=True, align_blur=2.0, refine_align=refine_align)
    return p - p[~mask].mean()


def corr(p):
    return np.corrcoef(p[mask], gt[mask])[0, 1]


# ---------------------------------------------------------------------------
# 2. Stable-beam acquisition: linear CTF vs non-linear refinement (cupping)
# ---------------------------------------------------------------------------
S_stable, R_stable = holograms(beam_modes=None)
p_ctf = reco(S_stable, R_stable, method="ctf")                       # cupping
p_good = reco(S_stable, R_stable, method="nonlinear", refine_align=True)
print("\n--- quantitative grey levels (stable beam) ---")
print("  linear CTF                : corr={:.3f}  (interior cupping)".format(corr(p_ctf)))
print("  non-linear + refine-align : corr={:.3f}  (quantitative interior)".format(corr(p_good)))

# ---------------------------------------------------------------------------
# 3. Drifting beam: simple flat-field vs eigen-flat
# ---------------------------------------------------------------------------
S_drift, R_drift = holograms(beam_modes=True, n_flats=10, seed=1)
p_simple = reco(S_drift, R_drift, flat_method="simple", alpha=1e-2)
p_eigen = reco(S_drift, R_drift, flat_method="eigen", alpha=1e-2)
print("\n--- beam drift: flat-field method ---")
print("  simple flat (mean) : corr={:.3f}  (drift residual fringes)".format(corr(p_simple)))
print("  eigen-flat         : corr={:.3f}  (drift modelled & removed)".format(corr(p_eigen)))

# ---------------------------------------------------------------------------
# 4. Frequency-coverage diagnostic of the multi-distance CTF
# ---------------------------------------------------------------------------
u = np.linspace(0, 1.0 / (2 * PX1), 1500)
fnorm = u / (1.0 / (2 * PX1))
eps = 1.0 / DELTA_BETA
w = [np.sin(np.pi * WAVELENGTH * z * u**2) + eps * np.cos(np.pi * WAVELENGTH * z * u**2)
     for z in geom.effective_distance]
combined = np.sqrt(np.mean([wi**2 for wi in w], axis=0))

# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------
_, info = holo_ctf_reconstruct(S_stable, R_stable, DARK, ZS, ZD, DET_PIXEL,
                               WAVELENGTH, alpha=1e-4, delta_beta=DELTA_BETA,
                               return_intermediates=True)
fig = plt.figure(figsize=(16, 8))
for i in range(4):
    ax = fig.add_subplot(2, 4, i + 1)
    ax.imshow(info["aligned"][i], cmap="gray")
    ax.set_title("hologram {}  (M={:.1f}, NF={:.3f})".format(
        i + 1, geom.magnification[i], NF[i]), fontsize=8)
    ax.axis("off")
axA = fig.add_subplot(2, 4, 5)
for i, wi in enumerate(w):
    axA.plot(fnorm, np.abs(wi), lw=0.8, label="z{} ({:.0f}mm)".format(
        i + 1, geom.effective_distance[i] * 1e3))
axA.plot(fnorm, combined, "k", lw=2, label=r"$\sqrt{\langle w^2\rangle}$")
axA.set_xlabel(r"$u/u_{\rm Nyq}$", fontsize=8)
axA.set_ylabel("|CTF transfer|", fontsize=8)
axA.set_title("frequency coverage (4 distances)", fontsize=9)
axA.legend(fontsize=6, ncol=2)
for ax_i, (img, title) in zip(
    (6, 7), [(gt, "Ground-truth phase"),
             (p_good, "Non-linear + refine-align")]):
    ax = fig.add_subplot(2, 4, ax_i)
    im = ax.imshow(img, cmap="gray"); ax.axis("off")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
axP = fig.add_subplot(2, 4, 8)
axP.plot(gt[N_DET // 2], "k-", lw=2, label="truth")
axP.plot(p_good[N_DET // 2], "C0-", label="non-linear (quantitative)")
axP.plot(p_ctf[N_DET // 2], "C3--", lw=1, label="linear CTF (cupping)")
axP.set_title("central line profile", fontsize=9)
axP.legend(fontsize=7)
plt.suptitle(
    "Magnified inline-holography multi-distance CTF (ID16A)  "
    "(corr = {:.3f})".format(corr(p_good)), fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("holo_ctf_id16a.png", dpi=150, bbox_inches="tight")
print("\nSaved holo_ctf_id16a.png")
print("Done.")
