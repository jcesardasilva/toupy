#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass multislice tomographic reconstruction — step-by-step tutorial
=======================================================================

This self-contained tutorial demonstrates the full workflow of the two-pass
method introduced in ``toupy/tomo/multislice.py`` and ``toupy/tomo/twopass.py``
using a **simulated 3-D phantom**.

Physical scenario
-----------------
We simulate a soft-X-ray ptychographic tomography (PXCT) experiment at
λ = 1 nm (~1.24 keV) with 5 nm voxels.  At this wavelength the Fresnel
number for 10–50 nm sample features is F ~ 0.1–7, so diffraction *within*
the sample is non-negligible and the projection approximation (used by FBP)
begins to fail.

Workflow
--------
1.  Create a 3-D phantom (ground truth δ, β).
2.  Simulate exit waves at N angles using the multislice forward model
    (the "measured" data).
3.  Extract projected-phase maps and reconstruct with standard FBP
    (Pass 1 / projection approximation).
4.  Refine with the multislice gradient-descent loop (Pass 2).
5.  Compare: ground truth vs. FBP vs. two-pass.

Dependencies
------------
    numpy, scipy, matplotlib, scikit-image

The two new Toupy modules are loaded directly from their file paths so that
the tutorial can run even in environments where the complete Toupy package
(with optional dependencies such as fabio) is not installed.

Run
---
    python tutorial/twopass_tutorial.py

Estimated wall-clock time: 3–8 minutes on a modern laptop (CPU only).
Reduce ``N_ANGLES``, ``N_SLICES``, or ``N_ITER`` to speed it up.
"""

# ---------------------------------------------------------------------------
# 0.  Imports and module loading
# ---------------------------------------------------------------------------

import os, sys, time, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering; change to "TkAgg" etc. if needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import rotate as ndimage_rotate
from skimage.transform import iradon      # FBP reference implementation

# ── Load toupy.tomo.multislice directly (avoids triggering toupy.__init__) ─
_HERE = os.path.dirname(os.path.abspath(__file__))
_TOMO = os.path.join(_HERE, "..", "toupy", "tomo")

def _load_module(dotted_name, fpath):
    spec = importlib.util.spec_from_file_location(dotted_name, fpath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

ms_mod = _load_module("toupy.tomo.multislice",
                      os.path.join(_TOMO, "multislice.py"))

MultisliceEngine           = ms_mod.MultisliceEngine
extract_slices_from_volume = ms_mod.extract_slices_from_volume
scatter_gradient_to_volume = ms_mod.scatter_gradient_to_volume

# ── Optional PyTorch backend (GPU/CPU acceleration for Pass 2) ─────────────
# Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU (multi-threaded torch)
# Falls back silently to NumPy when torch is not installed.
TORCH_AVAILABLE = False
try:
    import torch   # noqa: F401 — only needed to verify installation
    ms_torch_mod = _load_module("toupy.tomo.multislice_torch",
                                os.path.join(_TOMO, "multislice_torch.py"))
    TorchMultisliceEngine  = ms_torch_mod.TorchMultisliceEngine
    extract_slices_torch   = ms_torch_mod.extract_slices_torch
    scatter_gradient_torch = ms_torch_mod.scatter_gradient_torch
    tv_grad_torch          = ms_torch_mod.tv_grad_torch
    TorchAdamState         = ms_torch_mod.TorchAdamState
    select_device          = ms_torch_mod.select_device
    device_info            = ms_torch_mod.device_info
    warmup_device          = ms_torch_mod.warmup_device
    TORCH_AVAILABLE        = True
    DEVICE                 = select_device()
    print(f"PyTorch backend available — device: {device_info(DEVICE)}")
    print("  Pass 2 will run on the GPU/CPU torch backend.\n")
except (ImportError, Exception) as _e:
    print(f"PyTorch backend not available ({_e}); using NumPy backend.\n")

print("Modules loaded successfully.\n")

# ---------------------------------------------------------------------------
# 1.  Physical parameters
# ---------------------------------------------------------------------------

WAVELENGTH  = 1.0e-9     # [m]  λ = 1 nm  (~1.24 keV soft X-ray)
PIXEL_SIZE  = 5.0e-9     # [m]  5 nm isotropic voxel
N           = 64          # cubic volume side (voxels)  → 320 nm total
K0          = 2.0 * np.pi / WAVELENGTH   # wave-number [m⁻¹]

# Angular sampling
N_ANGLES    = 18                                 # projections (every 10°)
THETA       = np.linspace(0, 170, N_ANGLES)      # [degrees]

# Multislice discretisation
N_SLICES    = 16          # slabs per projection  (↑ from 10 → 4 px/slab, reduces z-staircase)
SLICE_DZ    = N * PIXEL_SIZE / N_SLICES          # [m] thickness per slab

# Optimisation (Pass 2)
N_ITER      = 50          # gradient-descent iterations  (↑ from 20 for better convergence)
LR          = 5e-4        # Adam peak learning rate
LAMBDA_TV   = 1e-5        # TV regularisation weight (0 to disable; tune for your noise level)
WARMUP_ITERS = 5          # linear LR warm-up length [iterations]

# Output directory for figures
OUT_DIR     = os.path.join(_HERE, "twopass_figures")
os.makedirs(OUT_DIR, exist_ok=True)

print("Physical parameters")
print(f"  λ          = {WAVELENGTH*1e9:.1f} nm")
print(f"  pixel size = {PIXEL_SIZE*1e9:.0f} nm")
print(f"  volume     = {N}³  ({N*PIXEL_SIZE*1e9:.0f} nm)³")
print(f"  k₀         = {K0:.3e} m⁻¹")
print(f"  Δz/slab    = {SLICE_DZ*1e9:.0f} nm  ({N_SLICES} slabs)")
print()

# Fresnel-number estimate for sphere edge features (radius r = 8 pixels)
r_nm   = 8 * PIXEL_SIZE * 1e9
L_nm   = N * PIXEL_SIZE * 1e9
F_edge = (r_nm ** 2) / (WAVELENGTH * 1e9 * L_nm)
print(f"  Fresnel number for r = {r_nm:.0f} nm edge over L = {L_nm:.0f} nm:  F = {F_edge:.2f}")
print(f"  (F ≪ 1 → strong diffraction; F ≫ 1 → projection limit)\n")

# ---------------------------------------------------------------------------
# 2.  Create 3-D phantom  (ground truth)
# ---------------------------------------------------------------------------

print("Step 1 — Creating 3-D phantom …", flush=True)

cz, cy, cx = N // 2, N // 2, N // 2

def sphere_mask(radius, centre=None, fill=1.0):
    """Return a 3-D float mask for a sphere inside an N³ grid."""
    if centre is None:
        centre = (cz, cy, cx)
    z, y, x = np.mgrid[0:N, 0:N, 0:N]
    r2 = (z - centre[0])**2 + (y - centre[1])**2 + (x - centre[2])**2
    mask = (r2 <= radius**2).astype(np.float64)
    return mask * fill

# Main cell body: sphere, radius 22 voxels
delta_true = sphere_mask(22, fill=4.0e-4)
beta_true  = sphere_mask(22, fill=4.0e-7)

# Inclusion A (higher δ): sphere of radius 8, offset to (+16, 0, 0) from centre
delta_true += sphere_mask(8, centre=(cz, cy, cx + 16), fill=4.0e-4)  # +δ
beta_true  += sphere_mask(8, centre=(cz, cy, cx + 16), fill=4.0e-7)

# Inclusion B (lower δ): sphere of radius 8, offset to (-14, 0, +8) from centre
delta_true += sphere_mask(7, centre=(cz - 6, cy, cx - 14), fill=-2.0e-4)  # −δ punch
beta_true  += sphere_mask(7, centre=(cz - 6, cy, cx - 14), fill=0.0)

# Physical constraints
np.clip(delta_true, 0.0, None, out=delta_true)
np.clip(beta_true,  0.0, None, out=beta_true)

print(f"  δ range: [{delta_true.min():.2e}, {delta_true.max():.2e}]")
print(f"  β range: [{beta_true.min():.2e}, {beta_true.max():.2e}]")

# Maximum accumulated phase (projection approximation)
max_phase_rad = K0 * PIXEL_SIZE * delta_true.sum(axis=0).max()
print(f"  Max projected phase (centre column) ≈ {max_phase_rad:.2f} rad")
print()

# ---------------------------------------------------------------------------
# 3.  Simulate exit waves (multislice forward model)
# ---------------------------------------------------------------------------

print("Step 2 — Simulating exit waves at all angles …", flush=True)

engine = MultisliceEngine(
    shape           = (N, N),
    pixel_size      = PIXEL_SIZE,
    wavelength      = WAVELENGTH,
    slice_thickness = SLICE_DZ,
)

probe = np.ones((N, N), dtype=complex)    # plane-wave illumination

u_measured    = np.zeros((N_ANGLES, N, N), dtype=complex)   # multislice exit waves
u_proj_approx = np.zeros((N_ANGLES, N, N), dtype=complex)   # projection approx.
phase_stack   = np.zeros((N_ANGLES, N, N), dtype=np.float64)# extracted phases

t0 = time.time()
for ai, theta in enumerate(THETA):
    # ── Multislice (physically correct) ──────────────────────────────────
    delta_sl, beta_sl, delta_means = extract_slices_from_volume(
        delta_true, beta_true, theta, n_slices=N_SLICES)
    u_ms = engine.forward(delta_sl, beta_sl, probe, delta_means=delta_means)
    u_measured[ai] = u_ms

    # ── Projection approximation ─────────────────────────────────────────
    # Accumulate transmission through all slabs without Fresnel propagation
    T_proj = np.ones((N, N), dtype=complex)
    for j in range(N_SLICES):
        T_proj *= engine.transmission(delta_sl[j], beta_sl[j])
    u_proj_approx[ai] = T_proj * probe

    # ── Phase for FBP ────────────────────────────────────────────────────
    # Use the *multislice* exit wave to extract the projected phase
    # (as would be done with ptychographic reconstruction in practice)
    phase_stack[ai] = np.angle(u_measured[ai])

    if (ai + 1) % 6 == 0:
        print(f"  angle {ai+1}/{N_ANGLES}  (θ={theta:.0f}°)", flush=True)

print(f"  Forward simulation done in {time.time()-t0:.1f} s\n")

# Residual between projection-approximation and multislice exit waves
diff_rms = np.sqrt(np.mean(np.abs(u_measured - u_proj_approx)**2))
print(f"  RMS |u_MS − u_proj_approx| = {diff_rms:.4e}")
print(f"  (This residual is what the two-pass corrects for)\n")

# ---------------------------------------------------------------------------
# 4.  FBP reconstruction  (Pass 1)
# ---------------------------------------------------------------------------

print("Step 3 — FBP reconstruction (Pass 1) …", flush=True)

# FBP convention (skimage.transform.iradon):
#   iradon(sinogram, theta, filter_name='ramp', circle=True)
#   sinogram shape: (n_detector_bins, n_angles)
#   output shape:   (n_bins, n_bins)    [for circle=True]
#
# Phase convention:  φ(x,y,θ) = −k₀ · pixel_size · Σ_z δ(x,y,z)
#   ⟹  sinogram(x, θ) = phase_stack[θ, y_fixed, x]
#   FBP of sinogram gives φ_3D(z, x) [see Note in docstring of pass1]
#   Conversion: δ = −φ_3D / (k₀ · pixel_size)

t0 = time.time()
delta_fbp = np.zeros((N, N, N), dtype=np.float64)   # shape (Nz, Ny, Nx)

for iy in range(N):
    # sinogram[:, iy, :] → shape (N_ANGLES, N) → transpose → (N, N_ANGLES)
    sino = phase_stack[:, iy, :].T          # (N, N_ANGLES)
    recons_slice = iradon(sino, theta=THETA,
                          filter_name="ramp",
                          circle=True)       # (N, N)
    # recons_slice[iz, ix]: FBP output in the (z-depth, x-transverse) plane
    # delta_fbp[iz, iy, ix]
    delta_fbp[:, iy, :] = recons_slice

# Convert projected phase → δ
delta_fbp = (-delta_fbp / (K0 * PIXEL_SIZE)).clip(0, None)

# ── Orientation correction ────────────────────────────────────────────────
# scipy.ndimage.rotate with axes=(2, 0) and a positive angle θ maps
# output coordinates as:
#   x_in = x_out·cos θ + z_out·sin θ
#   z_in = −x_out·sin θ + z_out·cos θ    (centered about N/2)
# At θ = 90° this gives z_in = −(x_out − N/2) + N/2 = N − x_out, i.e. the
# output x-axis samples the *reversed* z-axis of the original volume.
# skimage.iradon interprets the sinogram position as the detector coordinate
# and places it along the column (x) axis of the reconstruction; the row
# axis (axis 0 of recons_slice) therefore ends up reversed relative to our
# z-axis convention.  Flipping axis 0 corrects this.
delta_fbp = np.ascontiguousarray(delta_fbp[::-1, :, :])

# β not retrieved (no amplitude sinogram in this demo; set β = δ * 1e-3)
beta_fbp = delta_fbp * 1e-3

print(f"  FBP done in {time.time()-t0:.1f} s")
print(f"  δ_FBP range: [{delta_fbp.min():.2e}, {delta_fbp.max():.2e}]")
print()

# ---------------------------------------------------------------------------
# 5.  Total-variation (TV) regulariser  (NumPy version — used as fallback)
# ---------------------------------------------------------------------------

def tv_grad(vol, eps=1e-8):
    """
    Gradient of the anisotropic (L1) total-variation regulariser.

    TV(vol) = Σ_axis Σ_i |vol[i+1,...] − vol[i,...]|      (forward differences)

    A smooth surrogate |x| ≈ x / sqrt(x² + eps²) (Charbonnier) is used so
    the gradient is defined everywhere and never blows up.

    Physical interpretation
    -----------------------
    At a flat-material interior: neighbouring differences are small but
    non-zero (oscillation artefacts), and the gradient pushes those voxels
    toward the local average → suppresses low-frequency oscillations.
    At a sharp material boundary: the gradient is bounded (≈ ±1 per axis),
    so the material edge is not eroded — the regulariser is edge-preserving.

    This makes anisotropic TV ideal for piecewise-constant objects such as
    the biological/materials specimens common in tomographic imaging.

    Parameters
    ----------
    vol : ndarray, real, shape (Nz, Ny, Nx)
    eps : float
        Smoothing constant for |·|.  Should be << typical voxel difference.

    Returns
    -------
    g : ndarray, same shape and dtype as vol
        ∂TV/∂vol
    """
    g = np.zeros_like(vol)
    for ax in range(vol.ndim):
        sl_lo = [slice(None)] * vol.ndim
        sl_hi = [slice(None)] * vol.ndim
        sl_lo[ax] = slice(None, -1)   # indices 0 … N−2
        sl_hi[ax] = slice(1, None)    # indices 1 … N−1
        d  = vol[tuple(sl_hi)] - vol[tuple(sl_lo)]      # forward difference
        sd = d / np.sqrt(d ** 2 + eps ** 2)             # smooth sign  ≈ sign(d)
        # Chain rule: ∂|d|/∂vol[i] = −sign(d),  ∂|d|/∂vol[i+1] = +sign(d)
        g[tuple(sl_lo)] -= sd
        g[tuple(sl_hi)] += sd
    return g


# ---------------------------------------------------------------------------
# 6.  Minimal Adam optimiser  (NumPy version — used as fallback)
# ---------------------------------------------------------------------------

class _Adam:
    """Lightweight Adam optimiser state for one ndarray variable."""
    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0

    def step(self, param, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat  = self.m / (1 - self.b1 ** self.t)
        v_hat  = self.v / (1 - self.b2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param

# ---------------------------------------------------------------------------
# 7.  Two-pass refinement  (Pass 2)
# ---------------------------------------------------------------------------

print("Step 4 — Multislice refinement (Pass 2) …", flush=True)
_backend = "torch/" + str(DEVICE) if TORCH_AVAILABLE else "numpy"
print(f"  backend={_backend}, n_iter={N_ITER}, lr={LR}, n_slices={N_SLICES}, "
      f"lambda_tv={LAMBDA_TV:.0e}, warmup={WARMUP_ITERS} iters, "
      f"matter_correction=True\n")

# Start from FBP initial guess
delta_tp = delta_fbp.copy()
beta_tp  = beta_fbp.copy()

loss_history = []
t_total = time.time()

if TORCH_AVAILABLE:
    # ── PyTorch path (GPU/CPU) ─────────────────────────────────────────────
    import torch

    # Warm up the device (avoids JIT-compilation latency inside first iter)
    warmup_device(DEVICE)

    # Create the torch engine (same geometry as the NumPy engine above)
    torch_engine = TorchMultisliceEngine(
        shape           = (N, N),
        pixel_size      = PIXEL_SIZE,
        wavelength      = WAVELENGTH,
        slice_thickness = SLICE_DZ,
        device          = DEVICE,
    )

    # Move volumes and measured data to the device as float32 / complex64
    delta_tp_t = torch.from_numpy(delta_tp.astype(np.float32)).to(DEVICE)
    beta_tp_t  = torch.from_numpy(beta_tp.astype(np.float32)).to(DEVICE)
    probe_t    = torch.from_numpy(probe.astype(np.complex64)).to(DEVICE)
    u_meas_t   = torch.from_numpy(u_measured.astype(np.complex64)).to(DEVICE)
    # (N_ANGLES, N, N) complex64

    # Adam optimiser states (all on device)
    adam_d = TorchAdamState(delta_tp_t.shape, lr=LR, device=DEVICE)
    adam_b = TorchAdamState(beta_tp_t.shape,  lr=LR, device=DEVICE)

    for it in range(N_ITER):
        t_iter = time.time()
        total_loss = 0.0
        grad_delta = torch.zeros_like(delta_tp_t)
        grad_beta  = torch.zeros_like(beta_tp_t)

        # ── LR schedule: linear warm-up → cosine decay ────────────────
        if it < WARMUP_ITERS:
            lr_t = LR * (it + 1) / WARMUP_ITERS
        else:
            progress = (it - WARMUP_ITERS) / max(N_ITER - WARMUP_ITERS, 1)
            lr_t = LR * 0.5 * (1.0 + np.cos(np.pi * progress))
        adam_d.lr = adam_b.lr = lr_t

        for ai, theta in enumerate(THETA):
            # ── Extract rotated slices + matter-corrected δ̄ ──────────
            delta_sl, beta_sl, delta_means = extract_slices_torch(
                delta_tp_t, beta_tp_t, theta, n_slices=N_SLICES)

            # ── Forward + adjoint multislice ──────────────────────────
            loss_i, gd_i, gb_i, _ = torch_engine.loss_and_gradient(
                delta_sl, beta_sl, probe_t,
                u_meas_t[ai],
                delta_means=delta_means,
            )
            total_loss += loss_i

            # ── Back-rotate gradient slices → 3-D volume ──────────────
            gd_vol, gb_vol = scatter_gradient_torch(
                gd_i, gb_i, theta, delta_tp_t.shape, n_slices=N_SLICES)
            grad_delta += gd_vol
            grad_beta  += gb_vol

        # Normalise by number of angles
        grad_delta /= N_ANGLES
        grad_beta  /= N_ANGLES

        # ── TV regularisation ─────────────────────────────────────────
        if LAMBDA_TV > 0:
            grad_delta += LAMBDA_TV * tv_grad_torch(delta_tp_t)
            grad_beta  += LAMBDA_TV * tv_grad_torch(beta_tp_t)

        # ── Adam update ────────────────────────────────────────────────
        adam_d.step(delta_tp_t, grad_delta)
        adam_b.step(beta_tp_t,  grad_beta)

        # ── Positivity constraint ──────────────────────────────────────
        delta_tp_t.clamp_(min=0.0)
        beta_tp_t.clamp_(min=0.0)

        loss_history.append(total_loss)
        elapsed = time.time() - t_iter

        # Bring scalars to CPU for printing
        d_min = float(delta_tp_t.min())
        d_max = float(delta_tp_t.max())
        print(f"  Iter {it+1:3d}/{N_ITER}  loss={total_loss:.4e}  "
              f"lr={lr_t:.2e}  t={elapsed:.1f}s  "
              f"δ∈[{d_min:.2e},{d_max:.2e}]", flush=True)

    # Transfer final volumes back to NumPy (float64) for figures/RMSE
    delta_tp = delta_tp_t.cpu().numpy().astype(np.float64)
    beta_tp  = beta_tp_t.cpu().numpy().astype(np.float64)

else:
    # ── NumPy fallback path ────────────────────────────────────────────────
    adam_d = _Adam(delta_tp.shape, lr=LR)
    adam_b = _Adam(beta_tp.shape,  lr=LR)

    for it in range(N_ITER):
        t_iter = time.time()
        total_loss = 0.0
        grad_delta = np.zeros_like(delta_tp)
        grad_beta  = np.zeros_like(beta_tp)

        # ── LR schedule: linear warm-up → cosine decay ────────────────
        if it < WARMUP_ITERS:
            lr_t = LR * (it + 1) / WARMUP_ITERS
        else:
            progress = (it - WARMUP_ITERS) / max(N_ITER - WARMUP_ITERS, 1)
            lr_t = LR * 0.5 * (1.0 + np.cos(np.pi * progress))
        adam_d.lr = adam_b.lr = lr_t

        for ai, theta in enumerate(THETA):
            # ── Extract rotated slices + matter-corrected δ̄ ──────────
            delta_sl, beta_sl, delta_means = extract_slices_from_volume(
                delta_tp, beta_tp, theta, n_slices=N_SLICES)

            # ── Forward + adjoint multislice ──────────────────────────
            loss_i, gd_i, gb_i, _ = engine.loss_and_gradient(
                delta_sl, beta_sl, probe,
                u_measured[ai],
                delta_means=delta_means,
            )
            total_loss += loss_i

            # ── Back-rotate gradient slices → 3-D volume ──────────────
            gd_vol, gb_vol = scatter_gradient_to_volume(
                gd_i, gb_i, theta, delta_tp.shape, n_slices=N_SLICES)
            grad_delta += gd_vol
            grad_beta  += gb_vol

        # Normalise by number of angles
        grad_delta /= N_ANGLES
        grad_beta  /= N_ANGLES

        # ── TV regularisation ─────────────────────────────────────────
        if LAMBDA_TV > 0:
            grad_delta += LAMBDA_TV * tv_grad(delta_tp)
            grad_beta  += LAMBDA_TV * tv_grad(beta_tp)

        # ── Adam update ────────────────────────────────────────────────
        adam_d.step(delta_tp, grad_delta)
        adam_b.step(beta_tp,  grad_beta)

        # ── Positivity constraint ──────────────────────────────────────
        np.clip(delta_tp, 0.0, None, out=delta_tp)
        np.clip(beta_tp,  0.0, None, out=beta_tp)

        loss_history.append(total_loss)
        elapsed = time.time() - t_iter
        print(f"  Iter {it+1:3d}/{N_ITER}  loss={total_loss:.4e}  "
              f"lr={lr_t:.2e}  t={elapsed:.1f}s  "
              f"δ∈[{delta_tp.min():.2e},{delta_tp.max():.2e}]", flush=True)

print(f"\n  Pass 2 done in {time.time()-t_total:.1f} s total.\n")

# ---------------------------------------------------------------------------
# 8.  Quantitative evaluation
# ---------------------------------------------------------------------------

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def rel_err(a, b):
    return rmse(a, b) / (np.abs(b).mean() + 1e-30)

rmse_fbp = rmse(delta_fbp, delta_true)
rmse_tp  = rmse(delta_tp,  delta_true)
print("Quantitative comparison (δ vs ground truth)")
print(f"  RMSE  FBP    : {rmse_fbp:.3e}   rel = {rel_err(delta_fbp, delta_true)*100:.1f}%")
print(f"  RMSE  2-pass : {rmse_tp:.3e}   rel = {rel_err(delta_tp,  delta_true)*100:.1f}%")
print(f"  Improvement  : {(1 - rmse_tp/rmse_fbp)*100:.1f}%\n")

# ---------------------------------------------------------------------------
# 9.  Figures
# ---------------------------------------------------------------------------

print("Generating figures …", flush=True)

CMAP_DELTA = "inferno"
CMAP_PHASE = "bwr"

# ── Helper: show three orthogonal slices of a 3-D volume ─────────────────
def ortho_slices(vol, title="", cmap="inferno", vmin=None, vmax=None, ax=None):
    """Show xy, xz, yz mid-planes as a horizontal strip on the given axes."""
    cz_, cy_, cx_ = [s // 2 for s in vol.shape]
    strip = np.concatenate([vol[cz_, :, :],      # xy plane (top view)
                            vol[:, cy_, :],       # xz plane (side view)
                            vol[:, :, cx_].T],    # yz plane (front view)
                           axis=1)
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(strip, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return im

# ── Figure 1: Ground-truth phantom, FBP, two-pass (orthogonal strips) ────
vmax_d = delta_true.max() * 1.05

fig1, axes = plt.subplots(3, 1, figsize=(9, 7),
                           gridspec_kw={"hspace": 0.35})
im0 = ortho_slices(delta_true, title="Ground truth δ(r)  —  "
                   "left: xy mid-plane | centre: xz | right: yz",
                   cmap=CMAP_DELTA, vmin=0, vmax=vmax_d, ax=axes[0])
im1 = ortho_slices(delta_fbp,
                   title="FBP reconstruction  (Pass 1, projection approximation)",
                   cmap=CMAP_DELTA, vmin=0, vmax=vmax_d, ax=axes[1])
im2 = ortho_slices(delta_tp,
                   title=f"Two-pass reconstruction (after {N_ITER} iterations)",
                   cmap=CMAP_DELTA, vmin=0, vmax=vmax_d, ax=axes[2])
plt.colorbar(im0, ax=axes[0], fraction=0.012, label="δ")
plt.colorbar(im1, ax=axes[1], fraction=0.012, label="δ")
plt.colorbar(im2, ax=axes[2], fraction=0.012, label="δ")
fig1.suptitle("Tomographic reconstruction comparison\n"
              f"λ={WAVELENGTH*1e9:.0f} nm  |  pixel={PIXEL_SIZE*1e9:.0f} nm  "
              f"|  {N_ANGLES} angles  |  {N_SLICES} MS slabs",
              fontsize=9)
fpath1 = os.path.join(OUT_DIR, "fig1_comparison.png")
fig1.savefig(fpath1, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath1}")

# ── Figure 2: Exit waves at three angles ─────────────────────────────────
angles_to_show = [0, N_ANGLES // 3, 2 * N_ANGLES // 3]
fig2, axes2 = plt.subplots(3, 4, figsize=(12, 8),
                            gridspec_kw={"hspace": 0.45, "wspace": 0.35})

for row, ai in enumerate(angles_to_show):
    theta_i = THETA[ai]
    u_ms  = u_measured[ai]
    u_pa  = u_proj_approx[ai]

    axes2[row, 0].imshow(np.abs(u_ms), cmap="gray", origin="lower")
    axes2[row, 0].set_title(f"θ={theta_i:.0f}°  |u_MS|", fontsize=7)

    ph_ms = np.angle(u_ms)
    ph_pa = np.angle(u_pa)
    diff  = ph_ms - ph_pa

    lim = max(np.abs(ph_ms).max(), np.abs(ph_pa).max())
    axes2[row, 1].imshow(ph_ms, cmap=CMAP_PHASE, vmin=-lim, vmax=lim, origin="lower")
    axes2[row, 1].set_title("∠(u_MS)", fontsize=7)

    axes2[row, 2].imshow(ph_pa, cmap=CMAP_PHASE, vmin=-lim, vmax=lim, origin="lower")
    axes2[row, 2].set_title("∠(u_proj)", fontsize=7)

    dlim = np.abs(diff).max()
    im_d = axes2[row, 3].imshow(diff, cmap=CMAP_PHASE, vmin=-dlim, vmax=dlim,
                                origin="lower")
    axes2[row, 3].set_title("Δφ = ∠u_MS − ∠u_proj", fontsize=7)
    plt.colorbar(im_d, ax=axes2[row, 3], fraction=0.04)

    for ax in axes2[row]:
        ax.axis("off")

fig2.suptitle("Exit waves: multislice vs projection approximation\n"
              "(Column 4 shows the diffraction artefact that FBP ignores)",
              fontsize=9)
fpath2 = os.path.join(OUT_DIR, "fig2_exit_waves.png")
fig2.savefig(fpath2, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath2}")

# ── Figure 3: Convergence curve ───────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 3.5))
ax3.semilogy(range(1, len(loss_history) + 1), loss_history, "o-",
             lw=1.8, ms=4, color="steelblue")
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Total squared-residual loss")
ax3.set_title("Pass 2 convergence", fontsize=9)
ax3.grid(True, which="both", alpha=0.35)
fig3.tight_layout()
fpath3 = os.path.join(OUT_DIR, "fig3_convergence.png")
fig3.savefig(fpath3, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath3}")

# ── Figure 4: Central-slice line profiles ────────────────────────────────
# Profile through the horizontal centre of the xz plane
iy_mid  = N // 2
iz_mid  = N // 2
profile_x = np.arange(N) * PIXEL_SIZE * 1e9   # [nm]

true_row = delta_true[iz_mid, iy_mid, :]
fbp_row  = delta_fbp[iz_mid,  iy_mid, :]
tp_row   = delta_tp[iz_mid,   iy_mid, :]

fig4, axes4 = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"wspace": 0.45})

# Left: xy mid-plane image with profile line
cmap_opts = dict(cmap=CMAP_DELTA, vmin=0, vmax=vmax_d, origin="lower",
                 extent=[0, N * PIXEL_SIZE * 1e9, 0, N * PIXEL_SIZE * 1e9])
axes4[0].imshow(delta_true[iz_mid], **cmap_opts)
axes4[0].axhline(iy_mid * PIXEL_SIZE * 1e9, color="lime", lw=1, ls="--",
                 label="profile line")
axes4[0].set_xlabel("x [nm]"); axes4[0].set_ylabel("y [nm]")
axes4[0].set_title("Ground truth δ  (xy mid-plane)", fontsize=8)
axes4[0].legend(fontsize=7)

# Right: line profiles
axes4[1].plot(profile_x, true_row * 1e4, "k-",  lw=2.0, label="Ground truth")
axes4[1].plot(profile_x, fbp_row  * 1e4, "r--", lw=1.5, label=f"FBP  (RMSE={rmse_fbp:.2e})")
axes4[1].plot(profile_x, tp_row   * 1e4, "b-",  lw=1.5, label=f"Two-pass  (RMSE={rmse_tp:.2e})")
axes4[1].set_xlabel("x [nm]")
axes4[1].set_ylabel("δ  ×10⁻⁴")
axes4[1].set_title("Central line profile (xz mid-plane, y=centre)", fontsize=8)
axes4[1].legend(fontsize=7)
axes4[1].grid(True, alpha=0.35)

fig4.suptitle(f"Two-pass improvement: {(1 - rmse_tp/rmse_fbp)*100:.1f}% lower RMSE",
              fontsize=9)
fpath4 = os.path.join(OUT_DIR, "fig4_profiles.png")
fig4.savefig(fpath4, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath4}")

# ── Figure 5: Error maps ──────────────────────────────────────────────────
err_fbp = np.abs(delta_fbp - delta_true)
err_tp  = np.abs(delta_tp  - delta_true)
emax    = max(err_fbp.max(), err_tp.max())

fig5, axes5 = plt.subplots(2, 3, figsize=(11, 6.5),
                            gridspec_kw={"hspace": 0.45, "wspace": 0.30})

for row, (vol, err, label) in enumerate([
        (delta_fbp, err_fbp, "FBP"),
        (delta_tp,  err_tp,  "Two-pass")]):

    im_r = axes5[row, 0].imshow(vol[iz_mid],  cmap=CMAP_DELTA,
                                vmin=0, vmax=vmax_d, origin="lower")
    axes5[row, 0].set_title(f"{label}  δ  (xy mid)", fontsize=8)
    plt.colorbar(im_r, ax=axes5[row, 0], fraction=0.04)

    im_e1 = axes5[row, 1].imshow(err[iz_mid], cmap="hot",
                                 vmin=0, vmax=emax, origin="lower")
    axes5[row, 1].set_title(f"{label}  |error|  (xy mid)", fontsize=8)
    plt.colorbar(im_e1, ax=axes5[row, 1], fraction=0.04)

    im_e2 = axes5[row, 2].imshow(err[:, iy_mid, :], cmap="hot",
                                 vmin=0, vmax=emax, origin="lower")
    axes5[row, 2].set_title(f"{label}  |error|  (xz mid)", fontsize=8)
    plt.colorbar(im_e2, ax=axes5[row, 2], fraction=0.04)

    for ax in axes5[row]:
        ax.axis("off")

fig5.suptitle("Absolute error maps  |δ_recons − δ_true|", fontsize=9)
fpath5 = os.path.join(OUT_DIR, "fig5_error_maps.png")
fig5.savefig(fpath5, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath5}")

# ── Summary figure (compact, suitable for publication) ───────────────────
fig6 = plt.figure(figsize=(14, 3.8))
gs   = gridspec.GridSpec(1, 5, figure=fig6, wspace=0.30)

panels = [
    (delta_true[iz_mid], "Ground truth", CMAP_DELTA, 0, vmax_d),
    (delta_fbp[iz_mid],  "FBP (Pass 1)", CMAP_DELTA, 0, vmax_d),
    (delta_tp[iz_mid],   f"Two-pass\n(Pass 2, {N_ITER} iter)",
                          CMAP_DELTA, 0, vmax_d),
    (err_fbp[iz_mid],    f"FBP error\nRMSE={rmse_fbp:.2e}", "hot", 0, emax),
    (err_tp[iz_mid],     f"2-pass error\nRMSE={rmse_tp:.2e}", "hot", 0, emax),
]
for k, (img, ttl, cm, lo, hi) in enumerate(panels):
    ax = fig6.add_subplot(gs[k])
    im = ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, origin="lower")
    ax.set_title(ttl, fontsize=7.5)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.055, pad=0.03)

fig6.suptitle(
    f"Two-pass multislice reconstruction  —  "
    f"λ={WAVELENGTH*1e9:.0f} nm, {PIXEL_SIZE*1e9:.0f} nm voxel, "
    f"{N_ANGLES} angles, {N_SLICES} slabs",
    fontsize=8.5)
fpath6 = os.path.join(OUT_DIR, "fig6_summary.png")
fig6.savefig(fpath6, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath6}")

plt.close("all")

# ---------------------------------------------------------------------------
# 10.  Summary printout
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("TUTORIAL SUMMARY")
print("=" * 60)
print(f"  Compute backend     : {_backend}")
print(f"  Wavelength          : {WAVELENGTH*1e9:.1f} nm")
print(f"  Voxel size          : {PIXEL_SIZE*1e9:.0f} nm")
print(f"  Volume              : {N}³ = {N**3:,} voxels")
print(f"  Projection angles   : {N_ANGLES}")
print(f"  MS slabs/angle      : {N_SLICES}  (Δz = {SLICE_DZ*1e9:.0f} nm)")
print(f"  Pass 2 iterations   : {N_ITER}")
print()
print(f"  FBP   RMSE          : {rmse_fbp:.3e}")
print(f"  Two-pass RMSE       : {rmse_tp:.3e}")
print(f"  Improvement         : {(1 - rmse_tp/rmse_fbp)*100:.1f}%")
print()
print(f"  Figures saved to    : {OUT_DIR}/")
print("=" * 60)
