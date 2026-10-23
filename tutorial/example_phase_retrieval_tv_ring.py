#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example: Phase retrieval, ring correction, and TV reconstruction.

Demonstrates the full synchrotron / ptychographic nanotomography
pre-processing pipeline using simulated data:

    1.  Create a Shepp-Logan phantom as the projected phase object.
    2.  Simulate free-space propagation (TIE forward model) to get
        intensity projections, then recover the phase with:
            - TIE-Hom  (Paganin 2002)
            - CTF inversion
    3.  Inject synthetic ring artifacts into the sinogram and remove
        them with:
            - Wavelet-FFT filter (Münch 2009)
            - Titarenko analytical correction
    4.  Reconstruct with:
            - Standard FBP (Ram-Lak filter)
            - TV-regularised FBP (Chambolle-Pock)
    5.  Display and compare all results.

Run with:
    python tutorial/example_phase_retrieval_tv_ring.py
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")      # headless; change to 'TkAgg' for interactive
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.transform import radon, iradon

# toupy imports
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# GPU availability check (informational only — all functions fall back to
# CPU automatically when CuPy is not installed)
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected — GPU path enabled.")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found — running on CPU only.")

from toupy.simulation import phantom
from toupy.restoration import tie_hom, ctf_retrieve
from toupy.restoration import remove_rings_wavelet_fft, remove_rings_titarenko
from toupy.tomo import tv_reconstruction, mod_iradon

# ---------------------------------------------------------------------------
# 0. Experimental parameters
# ---------------------------------------------------------------------------
# Detector pixel / energy are representative of a zone-plate nanotomography
# setup at 17 keV (e.g. ID16A at ESRF).
#
# PROPAGATION DISTANCES for phase retrieval:
#   Phase-contrast methods require the Fresnel number NF = pixel²/(λ·z) to
#   be in the range 0.1–2 for meaningful contrast transfer.  For 50 nm pixels
#   at 17 keV this corresponds to an *effective* propagation distance of
#   roughly 30–200 µm.  In zone-plate optics this "effective z" arises from
#   the focal-length geometry of the objective, not the physical
#   sample-to-detector separation.  We choose:
#
#     z_TIE = 40 µm  →  NF ≈ 0.86  (near-field: best TIE-Hom validity)
#     z_CTF = 30 µm  →  NF ≈ 1.14  (near-field: widest first CTF passband)
#
# Choosing a SHORT distance (large NF) keeps the higher-order Fresnel
# fringes weak, so the first-order TIE / CTF inversions remain accurate.
# Longer distances increase contrast but also the model mismatch (and the
# negative low-frequency halo of single-distance retrieval).
# ---------------------------------------------------------------------------

N          = 256                # phantom / detector size [pixels]
PIXEL_SIZE = 50e-9              # effective pixel size at sample [m]  (50 nm)
ENERGY_keV = 17.0               # X-ray energy [keV]
DELTA_BETA = 50.0               # delta/beta ratio — moderate-Z material at 17 keV
                                # (lower δ/β ⇒ stronger relative absorption ⇒
                                #  more faithful single-distance TIE-Hom)
N_ANGLES   = 180                # number of projection angles
THETA      = np.linspace(0, 180, N_ANGLES, endpoint=False)  # [degrees]

Z_TIE = 40e-6                   # effective propagation distance for TIE-Hom [m]
Z_CTF = 30e-6                   # effective propagation distance for CTF [m]

# Multi-distance / holotomographic CTF: several propagation distances whose
# CTF zeros interleave.  All stay below the single-FFT Fresnel-sampling limit
# z_crit = N·pixel²/λ ≈ 8.8 mm so the simulated propagation is alias-free.
Z_HOLO = [30e-6, 150e-6, 600e-6, 2.4e-3]   # [m]

# Derived
hc_eV_m    = 1.23984193e-6      # hc [eV·m]
WAVELENGTH = hc_eV_m / (ENERGY_keV * 1e3)   # [m]

print("Wavelength        : {:.4e} m".format(WAVELENGTH))
print("Fresnel NF (TIE)  : {:.3f}".format(PIXEL_SIZE**2 / (WAVELENGTH * Z_TIE)))
print("Fresnel NF (CTF)  : {:.3f}".format(PIXEL_SIZE**2 / (WAVELENGTH * Z_CTF)))

# ---------------------------------------------------------------------------
# 1. Phantom: projected phase and absorption (Paganin homogeneous assumption)
#
#    For TIE-Hom (Paganin) retrieval to work the object must be "homogeneous":
#    absorption and phase are linked by  mu*T = 2 * phi / (delta/beta).
#    We use the Shepp-Logan phantom values as the PHASE map and derive the
#    absorption from it.  Peak phase ~5 rad is typical for bone at 17 keV
#    over ~60 µm sample thickness.
#
#    For CTF we need a WEAK-phase object (|phi| << 1 rad) because the CTF
#    inversion is a linearised approximation valid only in that regime.
#    We therefore demonstrate CTF with a separate, scaled-down phantom.
# ---------------------------------------------------------------------------

phase_true    = phantom(N, phantom_type="Modified Shepp-Logan") * 5.0  # [rad]
phase_true_wk = phantom(N, phantom_type="Modified Shepp-Logan") * 0.3  # weak-phase for CTF

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
im = ax.imshow(phase_true, cmap="gray")
ax.set_title("Ground-truth projected phase [rad]")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("phase_true.png", dpi=150)
print("Saved phase_true.png")

# ---------------------------------------------------------------------------
# 2. Simulate free-space propagation using the angular spectrum method.
#
#    This is the physically correct forward model for any object thickness.
#    The linearised TIE forward model (I ≈ 1 - z*λ/(4π)·∇²φ) is only valid
#    for very small phase values and would produce unphysical (negative)
#    intensities for |φ| >> 1 rad.
#
#    For TIE-Hom: mixed object following the Paganin homogeneity assumption
#        ψ_in = exp(-mu_T/2) · exp(i·φ),   mu_T = 2·φ / (δ/β)
#    For CTF:     pure phase object (weakly scattering)
#        ψ_in = exp(i·φ_weak)
# ---------------------------------------------------------------------------

print("\n--- Simulating propagation (angular spectrum) ---")


def propagate_angular_spectrum(psi_in, distance, wavelength, pixel_size, pad=1):
    """Free-space propagation via the paraxial angular spectrum method.

    Returns the intensity |ψ_prop|² at the detector plane.
    Valid for any phase magnitude; no linear approximations.

    ``pad`` (>1) zero/vacuum-pads the field before propagation to avoid
    circular wrap-around of the Fresnel kernel at longer distances, then
    crops back to the original field of view.
    """
    ny, nx = psi_in.shape
    if pad > 1:
        my, mx = ny * pad, nx * pad
        oy, ox = (my - ny) // 2, (mx - nx) // 2
        field = np.ones((my, mx), dtype=complex)   # vacuum = 1
        field[oy:oy + ny, ox:ox + nx] = psi_in
    else:
        my, mx, oy, ox, field = ny, nx, 0, 0, psi_in
    fy = np.fft.fftfreq(my, d=pixel_size)
    fx = np.fft.fftfreq(mx, d=pixel_size)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    # Fresnel (paraxial) propagator
    H = np.exp(1j * np.pi * wavelength * distance * (FY**2 + FX**2))
    psi_prop = np.fft.ifft2(np.fft.fft2(field) * H)
    intensity = np.abs(psi_prop)**2
    return intensity[oy:oy + ny, ox:ox + nx] if pad > 1 else intensity


# --- TIE-Hom simulation: mixed object obeying Paganin's assumption ---
# Absorption is linked to phase by the homogeneity condition:
#   mu*T = 2 * phi / (delta/beta)
# Both contribute to the transmitted wave:  psi = exp(-mu_T/2 + i*phi)
mu_T    = 2.0 * phase_true / DELTA_BETA
psi_mix = np.exp(-mu_T / 2.0 + 1j * phase_true)
I_tiehom = propagate_angular_spectrum(psi_mix, Z_TIE, WAVELENGTH, PIXEL_SIZE)
rng_ph = np.random.default_rng(7)
I_tiehom_noisy = I_tiehom + rng_ph.normal(0, 0.005 * I_tiehom.mean(), I_tiehom.shape)
I_tiehom_noisy = np.maximum(I_tiehom_noisy, 0)

# --- CTF simulation: pure weak-phase object (|phi| << 1 rad) ---
# CTF is the linearised approximation  I ≈ 1 + 2*CTF*phi, valid only for
# |phi| << 1 rad.  We use a separate scaled phantom (max 0.3 rad).
psi_wk  = np.exp(1j * phase_true_wk)
I_ctf   = propagate_angular_spectrum(psi_wk, Z_CTF, WAVELENGTH, PIXEL_SIZE)
I_ctf_noisy = I_ctf + rng_ph.normal(0, 0.002 * I_ctf.mean(), I_ctf.shape)
I_ctf_noisy = np.maximum(I_ctf_noisy, 0)

# --- Holotomographic (multi-distance) CTF simulation ---
# A HOMOGENEOUS weak object (same δ/β coupling as TIE-Hom) imaged at several
# distances.  The absorption term lets the multi-distance CTF recover the low
# frequencies / DC that single-distance CTF cannot.
mu_T_wk    = 2.0 * phase_true_wk / DELTA_BETA
psi_wk_mix = np.exp(-mu_T_wk / 2.0 + 1j * phase_true_wk)
I_holo = np.array([
    propagate_angular_spectrum(psi_wk_mix, z, WAVELENGTH, PIXEL_SIZE, pad=2)
    for z in Z_HOLO
])
I_holo_noisy = np.maximum(
    I_holo + rng_ph.normal(0, 0.002 * I_holo.mean(), I_holo.shape), 0)

print("  TIE-Hom intensity range : [{:.4f}, {:.4f}]".format(
    I_tiehom_noisy.min(), I_tiehom_noisy.max()))
print("  CTF     intensity range : [{:.4f}, {:.4f}]".format(
    I_ctf_noisy.min(), I_ctf_noisy.max()))
print("  Holo    distances [µm]  : {}".format(
    [round(z * 1e6, 1) for z in Z_HOLO]))

# ---------------------------------------------------------------------------
# 3. Phase retrieval — single projection
# ---------------------------------------------------------------------------

print("\n--- Phase retrieval (single projection) ---")

from scipy.ndimage import gaussian_filter

# TIE-Hom: pre-smooth the intensity by ~1.5 pixels before retrieval.
#
# WHY: the angular spectrum propagation at NF≈0.49 produces Fresnel
# diffraction fringes at the sharp edges of the Shepp-Logan phantom.
# The Paganin filter (derived from the near-field TIE approximation)
# misinterprets these high-frequency edge fringes as phase, creating a
# halo / ringing artefact around the object boundary.  A mild Gaussian
# pre-filter mimics the finite PSF of a real detector and brings the
# measured intensity closer to the smooth TIE model prediction.
I_tiehom_smooth = gaussian_filter(I_tiehom_noisy, sigma=0.8)

phase_tiehom = tie_hom(
    I_tiehom_smooth,
    distance=Z_TIE,
    wavelength=WAVELENGTH,
    pixel_size=PIXEL_SIZE,
    delta_beta=DELTA_BETA,
    regularisation=1e-3,
)

# CTF: a moderate Tikhonov α ≈ 1e-2 gives the most robust inversion here.
# A single propagation distance transfers phase only in a BAND around the
# first CTF maximum (χ ≈ π/2); too small an α amplifies noise near the CTF
# zeros without recovering the (unrecoverable) low frequencies.
phase_ctf = ctf_retrieve(
    I_ctf_noisy,
    distance=Z_CTF,
    wavelength=WAVELENGTH,
    pixel_size=PIXEL_SIZE,
    alpha=1e-2,
)

# CTF background correction:
#
# Single-distance CTF is fundamentally a BAND-PASS phase filter: it cannot
# recover DC or the lowest spatial frequencies because sin(χ→0) = 0.  For
# the large smooth Shepp-Logan ellipses (χ ≈ 10⁻³ rad) the absolute phase
# level is NOT retrieved — only the mid/high-frequency structure and the
# correct contrast polarity.  (Full low-frequency recovery requires either
# multiple propagation distances / holotomography, or far-field distances.)
#
# We set the vacuum background (phase_true_wk ≈ 0) to zero in the CTF output
# so the phase RELATIVE to vacuum is displayed correctly.
bg_mask        = (phase_true_wk < 0.01 * phase_true_wk.max())  # vacuum pixels
phase_ctf     -= phase_ctf[bg_mask].mean()

# --- Holotomographic (multi-distance, homogeneous) CTF retrieval ---
# Passing a list of distances + delta_beta switches ctf_retrieve into the
# multi-distance homogeneous mode, which fills the single-distance CTF zeros
# AND recovers the low frequencies / DC through the absorption term.
phase_holo = ctf_retrieve(
    I_holo_noisy,
    distance=Z_HOLO,
    wavelength=WAVELENGTH,
    pixel_size=PIXEL_SIZE,
    alpha=1e-3,
    delta_beta=DELTA_BETA,
)

print("  TIE-Hom phase range  : [{:.3f}, {:.3f}] rad".format(
    phase_tiehom.min(), phase_tiehom.max()))
print("  CTF     phase range  : [{:.4f}, {:.4f}] rad".format(
    phase_ctf.min(), phase_ctf.max()))
print("  Holo-CTF phase range : [{:.4f}, {:.4f}] rad  (truth 0–0.3)".format(
    phase_holo.min(), phase_holo.max()))

fig, axes = plt.subplots(1, 4, figsize=(17, 4))
display_data = [
    (phase_true,    "Ground truth\n(phase, 0–5 rad)"),
    (phase_tiehom,  "TIE-Hom retrieved\n(mixed object)"),
    (phase_ctf,     "Single-distance CTF\n(weak phase, band-pass)"),
    (phase_holo,    "Holotomographic CTF\n(4 distances + δ/β)"),
]
for ax, (img, title) in zip(axes, display_data):
    im = ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("phase_retrieval_comparison.png", dpi=150)
print("Saved phase_retrieval_comparison.png")

# ---------------------------------------------------------------------------
# 4. Build a full sinogram from phase projections
#    (simulate ptychographic tomography: we already have the phase per angle)
# ---------------------------------------------------------------------------

print("\n--- Building sinogram from phase projections ---")

sino_clean = radon(phase_true, theta=THETA, circle=False)
print("  Sinogram shape (n_pixels x n_angles):", sino_clean.shape)

# ---------------------------------------------------------------------------
# 5. Inject ring artifacts
#    Rings appear as additive stripes in the sinogram (column-constant offset)
# ---------------------------------------------------------------------------

print("\n--- Injecting ring artifacts ---")

rng = np.random.default_rng(42)
n_rings = 15
ring_positions = rng.integers(10, sino_clean.shape[0] - 10, size=n_rings)
ring_amplitudes = rng.uniform(-0.3, 0.3, size=n_rings) * sino_clean.std()

sino_rings = sino_clean.copy()
for pos, amp in zip(ring_positions, ring_amplitudes):
    width = rng.integers(1, 4)
    sino_rings[max(0, pos - width): pos + width, :] += amp

# Add Poisson-like noise (simulating photon statistics)
sino_noisy = sino_rings + rng.normal(0, 0.02 * sino_clean.std(),
                                      sino_clean.shape)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, sino, title in zip(
    axes,
    [sino_clean, sino_rings, sino_noisy],
    ["Clean sinogram", "With ring artifacts", "With rings + noise"],
):
    ax.imshow(sino, cmap="gray", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Angle index")
    ax.set_ylabel("Detector pixel")
plt.tight_layout()
plt.savefig("sinogram_comparison.png", dpi=150)
print("Saved sinogram_comparison.png")

# ---------------------------------------------------------------------------
# 6. Ring artifact correction
# ---------------------------------------------------------------------------

print("\n--- Ring artifact correction ---")

sino_wavelet   = remove_rings_wavelet_fft(sino_noisy, level=3, sigma=3)
sino_titarenko = remove_rings_titarenko(sino_noisy, size=51)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, sino, title in zip(
    axes,
    [sino_noisy, sino_wavelet, sino_titarenko],
    ["Noisy + rings", "Wavelet-FFT corrected", "Titarenko corrected"],
):
    ax.imshow(sino, cmap="gray", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Angle index")
    ax.set_ylabel("Detector pixel")
plt.tight_layout()
plt.savefig("ring_correction_comparison.png", dpi=150)
print("Saved ring_correction_comparison.png")

# ---------------------------------------------------------------------------
# 7. Tomographic reconstruction: FBP vs TV
# ---------------------------------------------------------------------------

print("\n--- Tomographic reconstruction ---")

# FBP from clean sinogram (reference)
recons_fbp_clean = iradon(sino_clean, theta=THETA, filter_name="ramp",
                           circle=False)

# FBP from ring-corrupted sinogram
recons_fbp_rings = iradon(sino_noisy, theta=THETA, filter_name="ramp",
                           circle=False)

# FBP after wavelet-FFT ring correction
recons_fbp_corrected = iradon(sino_wavelet, theta=THETA, filter_name="ramp",
                               circle=False)

# TV reconstruction from ring-corrected sinogram
print("  Running TV reconstruction (100 iterations)...")
recons_tv = tv_reconstruction(
    sino_wavelet,
    theta=THETA,
    lam=0.01,
    n_iter=100,
    output_size=N,
    verbose=True,
)

print("  TV reconstruction done.")

# ---------------------------------------------------------------------------
# 8. Final comparison figure
# ---------------------------------------------------------------------------

vmin = phase_true.min() * 0.9
vmax = phase_true.max() * 1.1

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

panels = [
    (phase_true,           "Ground truth"),
    (recons_fbp_clean,     "FBP — clean sino"),
    (recons_fbp_rings,     "FBP — with rings"),
    (recons_fbp_corrected, "FBP — ring corrected"),
    (phase_tiehom,         "TIE-Hom phase"),
    (phase_ctf,            "Single-distance CTF"),
    (phase_holo,           "Holotomographic CTF"),
    (recons_tv,            "TV — ring corrected\n(λ=0.01, 100 iter)"),
]

for idx, (img, title) in enumerate(panels):
    r, c = divmod(idx, 4)
    ax = fig.add_subplot(gs[r, c])
    im = ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(
    "Synchrotron nanotomography pipeline\n"
    "({:.0f} keV, pixel = {:.0f} nm, z_TIE = {:.0f} µm, δ/β = {:.0f})".format(
        ENERGY_keV, PIXEL_SIZE * 1e9, Z_TIE * 1e6, DELTA_BETA
    ),
    fontsize=11,
)
plt.savefig("full_pipeline_comparison.png", dpi=150, bbox_inches="tight")
print("Saved full_pipeline_comparison.png")

# ---------------------------------------------------------------------------
# 9. Quantitative metrics
# ---------------------------------------------------------------------------

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def normalise(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-30)


ref = normalise(phase_true)
print("\n--- RMSE (normalised) ---")
for label, img in [
    ("FBP clean",      recons_fbp_clean),
    ("FBP rings",      recons_fbp_rings),
    ("FBP corrected",  recons_fbp_corrected),
    ("TV  corrected",  recons_tv),
]:
    print("  {:20s}  RMSE = {:.5f}".format(label, rmse(ref, normalise(img))))

# Phase-retrieval fidelity for the weak-phase object (CTF vs holo-CTF):
print("\n--- Phase retrieval fidelity (weak-phase object, vs truth) ---")
for label, img in [
    ("Single-distance CTF", phase_ctf),
    ("Holotomographic CTF", phase_holo),
]:
    r = np.corrcoef(img.ravel(), phase_true_wk.ravel())[0, 1]
    print("  {:22s}  RMSE = {:.5f}   corr = {:+.3f}".format(
        label, rmse(normalise(phase_true_wk), normalise(img)), r))

print("\nAll figures saved. Done.")

# ---------------------------------------------------------------------------
# 10. GPU demonstration (runs only when CuPy is available)
#     Shows the cuda=True interface and timing comparison.
# ---------------------------------------------------------------------------

if GPU_AVAILABLE:
    print("\n" + "=" * 60)
    print("GPU demonstration")
    print("=" * 60)

    # Build a larger stack to make the GPU advantage visible
    N_GPU     = 512
    N_ANG_GPU = 360
    theta_gpu = np.linspace(0, 180, N_ANG_GPU, endpoint=False)

    rng2     = np.random.default_rng(7)
    ph_stack = phantom(N_GPU)[np.newaxis] * np.ones((10, N_GPU, N_GPU)) * 8.0
    ph_stack += rng2.normal(0, 0.01, ph_stack.shape)

    # Simulate propagation for the stack (mixed object, Paganin assumption)
    mu_T_gpu = 2.0 * ph_stack / DELTA_BETA
    psi_gpu  = np.exp(-mu_T_gpu / 2.0 + 1j * ph_stack)
    I_stack  = np.array([
        propagate_angular_spectrum(psi_gpu[k], Z_TIE, WAVELENGTH, PIXEL_SIZE)
        for k in range(ph_stack.shape[0])
    ])

    print("\n  Phase retrieval — stack of 10 × {}×{} images".format(N_GPU, N_GPU))

    t0 = time.perf_counter()
    _ = tie_hom(I_stack, Z_TIE, WAVELENGTH, PIXEL_SIZE,
                delta_beta=DELTA_BETA, cuda=False)
    t_cpu = time.perf_counter() - t0
    print("    CPU  TIE-Hom  : {:.3f} s".format(t_cpu))

    t0 = time.perf_counter()
    _ = tie_hom(I_stack, Z_TIE, WAVELENGTH, PIXEL_SIZE,
                delta_beta=DELTA_BETA, cuda=True)
    t_gpu = time.perf_counter() - t0
    print("    GPU  TIE-Hom  : {:.3f} s  (speedup {:.1f}×)".format(
        t_gpu, t_cpu / max(t_gpu, 1e-6)))

    # Ring correction timing on the larger sinogram
    sino_gpu_test = radon(phantom(N_GPU), theta=theta_gpu, circle=False)
    rings_test    = sino_gpu_test + rng2.normal(0, 0.05, sino_gpu_test.shape)
    rings_test[N_GPU // 3, :] += 0.5  # inject one strong ring

    print("\n  Ring correction — sinogram {}×{}".format(*rings_test.shape))

    t0 = time.perf_counter()
    _ = remove_rings_wavelet_fft(rings_test, level=3, sigma=2.5, cuda=False)
    t_cpu = time.perf_counter() - t0
    print("    CPU  wavelet-FFT : {:.3f} s".format(t_cpu))

    t0 = time.perf_counter()
    _ = remove_rings_wavelet_fft(rings_test, level=3, sigma=2.5, cuda=True)
    t_gpu = time.perf_counter() - t0
    print("    GPU  wavelet-FFT : {:.3f} s  (speedup {:.1f}×)".format(
        t_gpu, t_cpu / max(t_gpu, 1e-6)))

    # TV reconstruction timing
    sino_tv_gpu = radon(phantom(N_GPU), theta=theta_gpu, circle=False)
    print("\n  TV reconstruction — {}×{} image, {} angles, 30 iter".format(
        N_GPU, N_GPU, N_ANG_GPU))

    t0 = time.perf_counter()
    _ = tv_reconstruction(sino_tv_gpu, theta_gpu, lam=0.01, n_iter=30,
                          output_size=N_GPU, cuda=False)
    t_cpu = time.perf_counter() - t0
    print("    CPU  TV : {:.3f} s".format(t_cpu))

    t0 = time.perf_counter()
    _ = tv_reconstruction(sino_tv_gpu, theta_gpu, lam=0.01, n_iter=30,
                          output_size=N_GPU, cuda=True)
    t_gpu = time.perf_counter() - t0
    print("    GPU  TV : {:.3f} s  (speedup {:.1f}×)".format(
        t_gpu, t_cpu / max(t_gpu, 1e-6)))

else:
    print("\nSkipping GPU demo (CuPy not available).")
    print("Install CuPy matching your CUDA version, e.g.:")
    print("  pip install cupy-cuda12x")
