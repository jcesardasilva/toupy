#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: magnified inline-holography multi-distance CTF (ESRF ID16A style).

Simulates a focused-beam (cone-beam) holotomography projection acquired at four
defocus positions and reconstructs the phase with the full
``toupy.restoration.holo_ctf_reconstruct`` pipeline:

    flat-field  ->  rescale to a common pixel  ->  sub-pixel align  ->  CTF

The four positions have **different magnifications** (hence different effective
pixel sizes and Fresnel fringes), each hologram carries a **structured beam**
and **dark** offset (removed by flat-fielding), and each is **misaligned** by a
sub-pixel amount (removed by registration).

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
ZD = 0.10                                       # focus-detector distance [m]
DET_PIXEL = 1.0e-6                              # detector pixel [m]
ZS = np.array([12.5, 16.7, 25.0, 40.0]) * 1e-3  # focus-sample distances [m]

geom = holo_geometry(ZS, ZD, DET_PIXEL)
print("Magnifications     :", np.round(geom.magnification, 2))
print("Effective pixel nm :", np.round(geom.effective_pixel_size * 1e9, 1))
print("Effective dist mm  :", np.round(geom.effective_distance * 1e3, 2))
print("Effective NF       :", np.round(
    geom.effective_pixel_size**2 / (WAVELENGTH * geom.effective_distance), 4))

N_DET = 128                                     # detector size
N_FINE = 512                                    # simulation grid (object plane)
PX1 = geom.effective_pixel_size.min()           # finest (closest to focus)

# ---------------------------------------------------------------------------
# 1. Object (weak-phase, compact enough to fit the smallest field of view)
# ---------------------------------------------------------------------------
CORE = 80                                       # 80 * PX1 ~ 10 um < FOV_1 (16 um)
obj = phantom(CORE, "Modified Shepp-Logan") * 0.3
phase_fine = np.zeros((N_FINE, N_FINE))
o = (N_FINE - CORE) // 2
phase_fine[o:o + CORE, o:o + CORE] = obj
c = (-1.0 / DELTA_BETA + 1j)                     # homogeneous object


# ---------------------------------------------------------------------------
# 2. Forward model: propagate (fine grid) -> detector (magnified, band-limited)
# ---------------------------------------------------------------------------
def propagate(psi, z, px, pad=2):
    n = psi.shape[0]; m = n * pad; off = (m - n) // 2
    f = fftfreq(m, d=px); FY, FX = np.meshgrid(f, f, indexing="ij")
    H = np.exp(1j * np.pi * WAVELENGTH * z * (FY**2 + FX**2))
    fld = np.ones((m, m), dtype=complex); fld[off:off + n, off:off + n] = psi
    return (np.abs(ifft2(fft2(fld) * H))**2)[off:off + n, off:off + n]


def to_detector(img_fine, scale):
    """Anti-aliased magnification downsample (px1 -> px_eff_i), centre-preserving."""
    img = gaussian_filter(img_fine, 0.5 * scale) if scale > 1 else img_fine
    ni = img.shape[0]
    off = (ni - 1) / 2.0 - scale * (N_DET - 1) / 2.0
    return affine_transform(img, [scale, scale], offset=[off, off],
                            output_shape=(N_DET, N_DET), order=3, mode="nearest")


# structured beam on the fine grid + a per-position sub-pixel misalignment
yy, xx = np.mgrid[0:N_FINE, 0:N_FINE]
beam_fine = (1.0 + 0.3 * np.sin(2 * np.pi * xx / N_FINE * 4)
             * np.cos(2 * np.pi * yy / N_FINE * 4) + 0.1 * xx / N_FINE)
DARK = 50.0
shifts = np.array([[0, 0], [1.3, -0.7], [-0.8, 1.1], [0.6, 2.2]])   # px (common grid)

print("\nSimulating four holograms (flat-field + magnification + misalignment)...")
samples, references = [], []
rng = np.random.default_rng(0)
for i in range(4):
    field = propagate(np.exp(c * phase_fine), geom.effective_distance[i], PX1)
    field = np.real(ifft2(fourier_shift(fft2(field), shifts[i])))      # misalign
    s = geom.effective_pixel_size[i] / PX1
    beam_i = to_detector(beam_fine, s)
    hologram = to_detector(field, s)
    samples.append(beam_i * hologram + DARK)            # S = beam * sample + dark
    references.append(beam_i + DARK)                    # R = beam + dark (empty)
samples = np.array(samples) + rng.normal(0, 0.003, (4, N_DET, N_DET))
references = np.array(references)

# ---------------------------------------------------------------------------
# 3. Reconstruction (the whole pipeline in one call)
# ---------------------------------------------------------------------------
print("Reconstructing (flat-field -> rescale -> align -> multi-distance CTF)...")
phase, info = holo_ctf_reconstruct(
    samples, references, DARK, ZS, ZD, DET_PIXEL, WAVELENGTH,
    alpha=1e-2, delta_beta=DELTA_BETA, align=True, align_blur=2.0,
    return_intermediates=True)

# ground truth on the common (finest) grid
gt = phase_fine[(N_FINE - N_DET) // 2:(N_FINE + N_DET) // 2,
                (N_FINE - N_DET) // 2:(N_FINE + N_DET) // 2]
mask = gt > 0.01 * gt.max()
phase = phase - phase[~mask].mean()


def nrm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-30)


corr = np.corrcoef(phase[mask], gt[mask])[0, 1]
print("\nCommon pixel size  : {:.0f} nm".format(info["common_pixel_size"] * 1e9))
print("Retrieved phase corr: {:.4f}".format(corr))
print("Retrieved range     : [{:.3f}, {:.3f}] rad  (truth peak {:.2f})".format(
    phase.min(), phase.max(), gt.max()))

# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 7))
# top row: the four flat-fielded holograms (different fringes / magnification)
for i in range(4):
    ax = fig.add_subplot(2, 4, i + 1)
    ax.imshow(info["rescaled"][i], cmap="gray")
    ax.set_title("hologram {}  (M={:.1f}, NF={:.3f})".format(
        i + 1, geom.magnification[i],
        geom.effective_pixel_size[i]**2 /
        (WAVELENGTH * geom.effective_distance[i])), fontsize=8)
    ax.axis("off")
# bottom row: truth, retrieved, difference
for ax_i, (img, title) in zip(
    (5, 6, 7),
    [(gt, "Ground-truth phase"),
     (phase, "Retrieved (multi-distance CTF)"),
     (phase - gt, "Difference")]):
    ax = fig.add_subplot(2, 4, ax_i)
    im = ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
ax = fig.add_subplot(2, 4, 8)
ax.plot(gt[N_DET // 2], "k-", lw=2, label="truth")
ax.plot(phase[N_DET // 2], "C0-", label="retrieved")
ax.set_title("central line profile", fontsize=9)
ax.legend(fontsize=8)
plt.suptitle(
    "Magnified inline-holography multi-distance CTF (ID16A style)  "
    "(corr = {:.3f})".format(corr), fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("holo_ctf_id16a.png", dpi=150, bbox_inches="tight")
print("\nSaved holo_ctf_id16a.png")
print("Done.")
