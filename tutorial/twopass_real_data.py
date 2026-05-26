#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass multislice reconstruction on real PXCT data
=====================================================

Template for applying the two-pass multislice method to experimental
ptychographic X-ray computed tomography (PXCT) data stored in
``PXCTalignedprojections.npz``.

Data file layout
----------------
  'projections' : float32, shape (N_ANGLES, Ny, Nx)
      Projected phase maps φ(x, y, θ) in radians, already phase-retrieved
      and aligned.  Sign convention: φ = −k₀ · pixel_size · Σ_z δ(z),
      i.e. positive δ gives negative phase.
  'wavelen'     : scalar float, wavelength [m]
  'psize'       : scalar float, pixel size [m]
  'theta'       : float64, shape (N_ANGLES,), tilt angles [degrees]

Physical note
-------------
Fresnel number for this dataset:
  F = psize² / (wavelen × Nz × psize) ≈ 0.29   (Nz = Nx from FBP)
F < 1 → within-sample diffraction is non-negligible → two-pass method
is physically well motivated.

Assumptions
-----------
* Pure-phase object: amplitude of exit wave ≈ 1, so
      u_measured(x, y, θ) = exp(i · φ(x, y, θ))
  Adjust if you have the amplitude channel from ptychography.
* Isotropic pixel size (same in x, y, z).
* Phase convention matches the sign above; flip the sign of delta_fbp
  conversion if your ptychography code uses the opposite convention.

Computational note
------------------
FBP (Pass 1):
  iradon is a CPU-only operation (~0.7 s/slice × 394 slices ≈ 5 min
  sequential).  The script parallelises across y-slices automatically
  using joblib threads, cutting this to ~2–3 min on an 8-core M2.
  Install joblib if not already present:  pip install joblib

Pass 2 (multislice refinement):
  With 450 angles and 394 × 493 frames, one iteration takes
  ~5–10 min on CPU and ~25–40 s on MPS/CUDA.  Start with N_ITER = 5
  and a coarser N_SLICES = 8 to validate, then increase.

Run
---
  python tutorial/twopass_real_data.py
"""

# ---------------------------------------------------------------------------
# 0.  Imports and module loading
# ---------------------------------------------------------------------------

import os, sys, time, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.transform import iradon
try:
    from joblib import Parallel, delayed as joblib_delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False
    print("joblib not found — FBP will run sequentially (pip install joblib to speed up)")

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

# ── Optional PyTorch backend ───────────────────────────────────────────────
TORCH_AVAILABLE = False
try:
    import torch
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
    print(f"PyTorch backend: {device_info(DEVICE)}\n")
except (ImportError, Exception) as _e:
    print(f"PyTorch not available ({_e}); using NumPy backend.\n")

# ---------------------------------------------------------------------------
# 1.  Load data
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(_HERE, "PXCTalignedprojections.npz")
print(f"Loading data from: {DATA_FILE}")

data        = np.load(DATA_FILE)
phase_stack = data["projections"].astype(np.float64)   # (N_ANGLES, Ny, Nx) [rad]
WAVELENGTH  = float(data["wavelen"])                   # [m]
PIXEL_SIZE  = float(data["psize"])                     # [m]
THETA       = data["theta"].astype(np.float64)         # [degrees]

N_ANGLES, Ny, Nx = phase_stack.shape
K0 = 2.0 * np.pi / WAVELENGTH

# FBP always produces a square (Nx × Nx) reconstruction per y-slice
Nz = Nx

print(f"  N_ANGLES = {N_ANGLES}")
print(f"  (Ny, Nx) = ({Ny}, {Nx})  — non-square detector")
print(f"  Nz       = {Nz}  (set equal to Nx by FBP)")
print(f"  θ range  : [{THETA.min():.1f}, {THETA.max():.1f}]°  "
      f"(step ≈ {np.diff(THETA).mean():.2f}°)")
print(f"  λ        = {WAVELENGTH*1e9:.4f} nm")
print(f"  pixel    = {PIXEL_SIZE*1e9:.2f} nm")
print(f"  k₀       = {K0:.4e} m⁻¹")
print(f"  φ range  : [{phase_stack.min():.3f}, {phase_stack.max():.3f}] rad")

# Fresnel number: feature = one pixel, thickness = Nz pixels
F = PIXEL_SIZE**2 / (WAVELENGTH * Nz * PIXEL_SIZE)
print(f"\n  Fresnel number (pixel-scale features over full thickness):")
print(f"    F = pixel² / (λ · Nz · pixel) = {F:.3f}")
if F < 1:
    print(f"    F < 1 → strong diffraction regime → two-pass correction is important")
else:
    print(f"    F > 1 → projection approximation is reasonable")
print()

# ---------------------------------------------------------------------------
# 2.  User-tunable reconstruction parameters
# ---------------------------------------------------------------------------

# ── Multislice discretisation ──────────────────────────────────────────────
# Rule of thumb: each slab should be thinner than the depth of focus
#   DoF ≈ pixel² / λ = F * Nz * pixel  →  N_SLICES ≈ Nz / DoF_in_pixels
# For this dataset DoF ≈ F*Nz ≈ 143 pixels → start with Nz/16 ≈ 31 px/slab.
# Increase N_SLICES for higher accuracy (but proportionally slower).
N_SLICES    = 16
SLICE_DZ    = Nz * PIXEL_SIZE / N_SLICES          # [m] slab thickness

# ── Pass 2 optimisation ────────────────────────────────────────────────────
# With 450 angles + 394×493 frames, one iteration ≈ 30–60 s on MPS/CUDA.
# Start small (N_ITER = 5) to check convergence, then increase.
N_ITER       = 20
LR           = 5e-4       # Adam peak learning rate
LAMBDA_TV    = 1e-5       # TV regularisation weight (0 to disable)
WARMUP_ITERS = 3          # linear LR warm-up iterations

# ── Angle subsampling (for fast prototyping) ───────────────────────────────
# Set ANGLE_STEP = 1 to use all angles; 2 = every other angle, etc.
# Using all 450 angles gives the best quality reconstruction.
ANGLE_STEP   = 1
theta_use    = THETA[::ANGLE_STEP]
phase_use    = phase_stack[::ANGLE_STEP]          # (N_use, Ny, Nx)
N_use        = len(theta_use)

if ANGLE_STEP > 1:
    print(f"  [Subsampling] Using {N_use}/{N_ANGLES} angles "
          f"(every {ANGLE_STEP}th)\n")

# Output directory
OUT_DIR = os.path.join(_HERE, "twopass_real_figures")
os.makedirs(OUT_DIR, exist_ok=True)

print("Reconstruction parameters")
print(f"  N_SLICES   = {N_SLICES}  (slab Δz = {SLICE_DZ*1e9:.1f} nm = "
      f"{Nz//N_SLICES} pixels)")
print(f"  N_ITER     = {N_ITER}")
print(f"  LR         = {LR}")
print(f"  LAMBDA_TV  = {LAMBDA_TV}")
print(f"  backend    = {'torch/' + str(DEVICE) if TORCH_AVAILABLE else 'numpy'}")
print()

# ---------------------------------------------------------------------------
# 3.  FBP reconstruction  (Pass 1)
# ---------------------------------------------------------------------------

print("Step 1 — FBP reconstruction (Pass 1) …", flush=True)
t0 = time.time()

# ── Parallelise across y-slices (each is independent) ─────────────────────
# Using threads rather than processes avoids data-copy overhead for the
# shared phase_use array.  Speedup ≈ 2× on an 8-core M2 (iradon internally
# uses numpy which already uses some threading, limiting further gains).
def _fbp_slice(iy):
    return iradon(phase_use[:, iy, :].T, theta=theta_use,
                  filter_name="ramp", circle=True)   # (Nx, Nx)

if _JOBLIB:
    n_jobs = -1   # use all available CPU cores
    print(f"  Parallel FBP (joblib threads, n_jobs={n_jobs}) …", flush=True)
    slices = Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib_delayed(_fbp_slice)(iy) for iy in range(Ny))
    delta_fbp = np.stack(slices, axis=1)          # (Nz, Ny, Nx)
else:
    print("  Sequential FBP …", flush=True)
    delta_fbp = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    for iy in range(Ny):
        delta_fbp[:, iy, :] = _fbp_slice(iy)

# Convert projected phase → δ:   φ = −k₀ · pixel · Σδ   →   δ = −φ / (k₀ · pixel)
# If your ptychography code uses the opposite sign convention, remove the minus.
delta_fbp = -delta_fbp / (K0 * PIXEL_SIZE)

# ── Orientation correction ────────────────────────────────────────────────
# scipy.ndimage.rotate with axes=(2, 0) reverses the z-axis relative to
# the FBP convention of skimage.iradon.  Flip axis 0 to correct.
delta_fbp = np.ascontiguousarray(delta_fbp[::-1, :, :])

# Positivity constraint: δ ≥ 0 for standard materials
# Comment out if your sample has negative contrast regions.
delta_fbp = delta_fbp.clip(0, None)

# β not available from phase-only data; initialise as a small fraction of δ
beta_fbp = delta_fbp * 1e-3

print(f"  FBP done in {time.time()-t0:.1f} s")
print(f"  δ_FBP range: [{delta_fbp.min():.3e}, {delta_fbp.max():.3e}]")
print()

# ---------------------------------------------------------------------------
# 4.  Form complex exit waves for Pass 2
# ---------------------------------------------------------------------------
# Pure-phase object approximation: amplitude = 1.
# u_meas(x, y, θ) = exp(i · φ(x, y, θ))
# If you have an amplitude channel from ptychography, use:
#   u_meas = amplitude * exp(1j * phase)

u_measured = np.exp(1j * phase_use.astype(np.complex128)).astype(np.complex64)
# shape: (N_use, Ny, Nx),  complex64

probe = np.ones((Ny, Nx), dtype=np.complex64)   # plane-wave illumination

# ---------------------------------------------------------------------------
# 5.  Total-variation regulariser  (NumPy fallback)
# ---------------------------------------------------------------------------

def tv_grad(vol, eps=1e-8):
    """Gradient of anisotropic Charbonnier TV regulariser."""
    g = np.zeros_like(vol)
    for ax in range(vol.ndim):
        sl_lo = [slice(None)] * vol.ndim
        sl_hi = [slice(None)] * vol.ndim
        sl_lo[ax] = slice(None, -1)
        sl_hi[ax] = slice(1, None)
        d  = vol[tuple(sl_hi)] - vol[tuple(sl_lo)]
        sd = d / np.sqrt(d ** 2 + eps ** 2)
        g[tuple(sl_lo)] -= sd
        g[tuple(sl_hi)] += sd
    return g


# ---------------------------------------------------------------------------
# 6.  Minimal Adam optimiser  (NumPy fallback)
# ---------------------------------------------------------------------------

class _Adam:
    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

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

print("Step 2 — Multislice refinement (Pass 2) …", flush=True)
_backend = "torch/" + str(DEVICE) if TORCH_AVAILABLE else "numpy"
print(f"  backend={_backend}, n_iter={N_ITER}, lr={LR}, "
      f"n_slices={N_SLICES}, lambda_tv={LAMBDA_TV:.0e}, "
      f"warmup={WARMUP_ITERS} iters\n")

delta_tp = delta_fbp.copy()
beta_tp  = beta_fbp.copy()

loss_history = []
t_total = time.time()

if TORCH_AVAILABLE:
    # ── PyTorch path ───────────────────────────────────────────────────────
    warmup_device(DEVICE)

    torch_engine = TorchMultisliceEngine(
        shape           = (Ny, Nx),
        pixel_size      = PIXEL_SIZE,
        wavelength      = WAVELENGTH,
        slice_thickness = SLICE_DZ,
        device          = DEVICE,
    )

    delta_tp_t = torch.from_numpy(delta_tp.astype(np.float32)).to(DEVICE)
    beta_tp_t  = torch.from_numpy(beta_tp.astype(np.float32)).to(DEVICE)
    probe_t    = torch.from_numpy(probe).to(DEVICE)
    u_meas_t   = torch.from_numpy(u_measured).to(DEVICE)   # (N_use, Ny, Nx)

    adam_d = TorchAdamState(delta_tp_t.shape, lr=LR, device=DEVICE)
    adam_b = TorchAdamState(beta_tp_t.shape,  lr=LR, device=DEVICE)

    for it in range(N_ITER):
        t_iter = time.time()
        total_loss = 0.0
        grad_delta = torch.zeros_like(delta_tp_t)
        grad_beta  = torch.zeros_like(beta_tp_t)

        # LR schedule
        if it < WARMUP_ITERS:
            lr_t = LR * (it + 1) / WARMUP_ITERS
        else:
            progress = (it - WARMUP_ITERS) / max(N_ITER - WARMUP_ITERS, 1)
            lr_t = LR * 0.5 * (1.0 + np.cos(np.pi * progress))
        adam_d.lr = adam_b.lr = lr_t

        for ai, theta in enumerate(theta_use):
            delta_sl, beta_sl, delta_means = extract_slices_torch(
                delta_tp_t, beta_tp_t, theta, n_slices=N_SLICES)

            loss_i, gd_i, gb_i, _ = torch_engine.loss_and_gradient(
                delta_sl, beta_sl, probe_t,
                u_meas_t[ai],
                delta_means=delta_means,
            )
            total_loss += loss_i

            gd_vol, gb_vol = scatter_gradient_torch(
                gd_i, gb_i, theta, delta_tp_t.shape, n_slices=N_SLICES)
            grad_delta += gd_vol
            grad_beta  += gb_vol

        grad_delta /= N_use
        grad_beta  /= N_use

        if LAMBDA_TV > 0:
            grad_delta += LAMBDA_TV * tv_grad_torch(delta_tp_t)
            grad_beta  += LAMBDA_TV * tv_grad_torch(beta_tp_t)

        adam_d.step(delta_tp_t, grad_delta)
        adam_b.step(beta_tp_t,  grad_beta)

        delta_tp_t.clamp_(min=0.0)
        beta_tp_t.clamp_(min=0.0)

        loss_history.append(total_loss)
        elapsed = time.time() - t_iter
        print(f"  Iter {it+1:3d}/{N_ITER}  loss={total_loss:.4e}  "
              f"lr={lr_t:.2e}  t={elapsed:.1f}s  "
              f"δ∈[{float(delta_tp_t.min()):.2e},{float(delta_tp_t.max()):.2e}]",
              flush=True)

    delta_tp = delta_tp_t.cpu().numpy().astype(np.float64)
    beta_tp  = beta_tp_t.cpu().numpy().astype(np.float64)

else:
    # ── NumPy fallback ─────────────────────────────────────────────────────
    engine = MultisliceEngine(
        shape           = (Ny, Nx),
        pixel_size      = PIXEL_SIZE,
        wavelength      = WAVELENGTH,
        slice_thickness = SLICE_DZ,
    )

    adam_d = _Adam(delta_tp.shape, lr=LR)
    adam_b = _Adam(beta_tp.shape,  lr=LR)

    for it in range(N_ITER):
        t_iter = time.time()
        total_loss = 0.0
        grad_delta = np.zeros_like(delta_tp)
        grad_beta  = np.zeros_like(beta_tp)

        if it < WARMUP_ITERS:
            lr_t = LR * (it + 1) / WARMUP_ITERS
        else:
            progress = (it - WARMUP_ITERS) / max(N_ITER - WARMUP_ITERS, 1)
            lr_t = LR * 0.5 * (1.0 + np.cos(np.pi * progress))
        adam_d.lr = adam_b.lr = lr_t

        for ai, theta in enumerate(theta_use):
            delta_sl, beta_sl, delta_means = extract_slices_from_volume(
                delta_tp, beta_tp, theta, n_slices=N_SLICES)

            loss_i, gd_i, gb_i, _ = engine.loss_and_gradient(
                delta_sl, beta_sl, probe,
                u_measured[ai],
                delta_means=delta_means,
            )
            total_loss += loss_i

            gd_vol, gb_vol = scatter_gradient_to_volume(
                gd_i, gb_i, theta, delta_tp.shape, n_slices=N_SLICES)
            grad_delta += gd_vol
            grad_beta  += gb_vol

        grad_delta /= N_use
        grad_beta  /= N_use

        if LAMBDA_TV > 0:
            grad_delta += LAMBDA_TV * tv_grad(delta_tp)
            grad_beta  += LAMBDA_TV * tv_grad(beta_tp)

        adam_d.step(delta_tp, grad_delta)
        adam_b.step(beta_tp,  grad_beta)

        np.clip(delta_tp, 0.0, None, out=delta_tp)
        np.clip(beta_tp,  0.0, None, out=beta_tp)

        loss_history.append(total_loss)
        elapsed = time.time() - t_iter
        print(f"  Iter {it+1:3d}/{N_ITER}  loss={total_loss:.4e}  "
              f"lr={lr_t:.2e}  t={elapsed:.1f}s  "
              f"δ∈[{delta_tp.min():.2e},{delta_tp.max():.2e}]", flush=True)

print(f"\n  Pass 2 done in {time.time()-t_total:.1f} s total.\n")

# ---------------------------------------------------------------------------
# 8.  Save reconstructed volumes
# ---------------------------------------------------------------------------

np.savez_compressed(
    os.path.join(OUT_DIR, "twopass_reconstruction.npz"),
    delta_fbp = delta_fbp.astype(np.float32),
    delta_tp  = delta_tp.astype(np.float32),
    beta_tp   = beta_tp.astype(np.float32),
    wavelength = WAVELENGTH,
    psize      = PIXEL_SIZE,
    theta      = theta_use,
)
print(f"  Volumes saved to: {OUT_DIR}/twopass_reconstruction.npz")

# ---------------------------------------------------------------------------
# 9.  Figures
# ---------------------------------------------------------------------------

print("Generating figures …", flush=True)

CMAP = "gray"

# Central slice indices
iz_mid = Nz // 2
iy_mid = Ny // 2
ix_mid = Nx // 2

# Common colour scale: use two-pass range for fairness
vmin = 0.0
vmax = np.percentile(delta_tp[delta_tp > 0], 99) if delta_tp.max() > 0 else delta_fbp.max()

# ── Figure 1: Three orthogonal cuts — FBP vs two-pass ────────────────────
fig1, axes = plt.subplots(3, 2, figsize=(10, 11),
                           gridspec_kw={"hspace": 0.40, "wspace": 0.30})

cuts = [
    (delta_fbp[iz_mid],     delta_tp[iz_mid],     f"xy  (z={iz_mid})"),
    (delta_fbp[:, iy_mid, :], delta_tp[:, iy_mid, :], f"xz  (y={iy_mid})"),
    (delta_fbp[:, :, ix_mid].T, delta_tp[:, :, ix_mid].T, f"yz  (x={ix_mid})"),
]

for row, (fbp_cut, tp_cut, label) in enumerate(cuts):
    for col, (img, ttl) in enumerate([
            (fbp_cut, f"FBP  [{label}]"),
            (tp_cut,  f"Two-pass  [{label}]")]):
        im = axes[row, col].imshow(img, cmap=CMAP, vmin=vmin, vmax=vmax,
                                   origin="lower")
        axes[row, col].set_title(ttl, fontsize=8)
        axes[row, col].axis("off")
        plt.colorbar(im, ax=axes[row, col], fraction=0.04, label="δ")

fig1.suptitle(
    f"PXCT two-pass reconstruction\n"
    f"λ={WAVELENGTH*1e9:.3f} nm  |  pixel={PIXEL_SIZE*1e9:.1f} nm  "
    f"|  {N_use} angles  |  {N_SLICES} MS slabs  |  {N_ITER} iters",
    fontsize=9)
fpath1 = os.path.join(OUT_DIR, "fig1_orthogonal_cuts.png")
fig1.savefig(fpath1, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath1}")

# ── Figure 2: Difference map (two-pass − FBP) ─────────────────────────────
diff = delta_tp - delta_fbp
dlim = np.percentile(np.abs(diff), 99)

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4),
                            gridspec_kw={"wspace": 0.35})
diff_cuts = [
    (diff[iz_mid],        f"xy  (z={iz_mid})"),
    (diff[:, iy_mid, :],  f"xz  (y={iy_mid})"),
    (diff[:, :, ix_mid].T, f"yz  (x={ix_mid})"),
]
for ax, (img, lbl) in zip(axes2, diff_cuts):
    im = ax.imshow(img, cmap="bwr", vmin=-dlim, vmax=dlim, origin="lower")
    ax.set_title(f"Δδ = two-pass − FBP  [{lbl}]", fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Δδ")
fig2.suptitle("Difference map (two-pass − FBP)", fontsize=9)
fpath2 = os.path.join(OUT_DIR, "fig2_difference_maps.png")
fig2.savefig(fpath2, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath2}")

# ── Figure 3: Convergence curve ───────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 3.5))
ax3.semilogy(range(1, len(loss_history) + 1), loss_history,
             "o-", lw=1.8, ms=4, color="steelblue")
ax3.set_xlabel("Iteration")
ax3.set_ylabel("Total squared-residual loss")
ax3.set_title("Pass 2 convergence", fontsize=9)
ax3.grid(True, which="both", alpha=0.35)
fig3.tight_layout()
fpath3 = os.path.join(OUT_DIR, "fig3_convergence.png")
fig3.savefig(fpath3, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath3}")

# ── Figure 4: Central line profiles ──────────────────────────────────────
profile_x = np.arange(Nx) * PIXEL_SIZE * 1e9   # [nm]
profile_z = np.arange(Nz) * PIXEL_SIZE * 1e9   # [nm]

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4),
                            gridspec_kw={"wspace": 0.40})

# Horizontal profile through (iz_mid, iy_mid, :)
axes4[0].plot(profile_x, delta_fbp[iz_mid, iy_mid, :] * 1e6,
              "r--", lw=1.5, label="FBP")
axes4[0].plot(profile_x, delta_tp[iz_mid,  iy_mid, :] * 1e6,
              "b-",  lw=1.5, label="Two-pass")
axes4[0].set_xlabel("x [nm]")
axes4[0].set_ylabel("δ  ×10⁻⁶")
axes4[0].set_title(f"Horizontal profile  (z=mid, y=mid)", fontsize=8)
axes4[0].legend(fontsize=8)
axes4[0].grid(True, alpha=0.35)

# Vertical profile through (:, iy_mid, ix_mid)
axes4[1].plot(profile_z, delta_fbp[:, iy_mid, ix_mid] * 1e6,
              "r--", lw=1.5, label="FBP")
axes4[1].plot(profile_z, delta_tp[:,  iy_mid, ix_mid] * 1e6,
              "b-",  lw=1.5, label="Two-pass")
axes4[1].set_xlabel("z [nm]")
axes4[1].set_ylabel("δ  ×10⁻⁶")
axes4[1].set_title(f"Vertical profile  (y=mid, x=mid)", fontsize=8)
axes4[1].legend(fontsize=8)
axes4[1].grid(True, alpha=0.35)

fig4.suptitle("Central line profiles: FBP vs two-pass", fontsize=9)
fpath4 = os.path.join(OUT_DIR, "fig4_profiles.png")
fig4.savefig(fpath4, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath4}")

# ── Figure 5: Projections — measured vs re-projected ─────────────────────
# Re-project the two-pass volume at a few angles to check consistency.
# Uses simple projection approximation (fast, for visual check only).
n_show = min(3, N_use)
angle_idx = np.linspace(0, N_use - 1, n_show, dtype=int)

fig5, axes5 = plt.subplots(n_show, 3, figsize=(11, 3.5 * n_show),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.30})
if n_show == 1:
    axes5 = axes5[np.newaxis, :]

for row, ai in enumerate(angle_idx):
    phi_meas = phase_use[ai]                    # measured phase
    # Re-projected phase from FBP volume (projection approx.):
    from scipy.ndimage import rotate as ndrotate
    d_rot = ndrotate(delta_tp[:, iy_mid, :], theta_use[ai],
                     axes=(1, 0), reshape=False, order=1)
    phi_reproj = -K0 * PIXEL_SIZE * d_rot.sum(axis=0)  # 1-D, at y=iy_mid

    lim = np.abs(phi_meas[iy_mid]).max() * 1.1

    im0 = axes5[row, 0].imshow(phi_meas, cmap="bwr",
                                vmin=-np.abs(phi_meas).max(),
                                vmax= np.abs(phi_meas).max(),
                                origin="lower")
    axes5[row, 0].set_title(f"θ={theta_use[ai]:.1f}°  measured φ", fontsize=7)
    axes5[row, 0].axis("off")
    plt.colorbar(im0, ax=axes5[row, 0], fraction=0.04)

    axes5[row, 1].plot(phi_meas[iy_mid], "r-",  lw=1.2, label="measured")
    axes5[row, 1].plot(phi_reproj,        "b--", lw=1.2, label="re-projected")
    axes5[row, 1].set_xlabel("x [pixel]")
    axes5[row, 1].set_ylabel("φ [rad]")
    axes5[row, 1].set_title(f"Profile y=mid", fontsize=7)
    axes5[row, 1].legend(fontsize=6)
    axes5[row, 1].grid(True, alpha=0.3)

    resid = phi_meas[iy_mid] - phi_reproj
    axes5[row, 2].plot(resid, "g-", lw=1.2)
    axes5[row, 2].axhline(0, color="k", lw=0.8, ls="--")
    axes5[row, 2].set_xlabel("x [pixel]")
    axes5[row, 2].set_ylabel("Δφ [rad]")
    axes5[row, 2].set_title("Residual meas − re-proj", fontsize=7)
    axes5[row, 2].grid(True, alpha=0.3)

fig5.suptitle("Self-consistency check: measured vs re-projected phases",
              fontsize=9)
fpath5 = os.path.join(OUT_DIR, "fig5_reprojection_check.png")
fig5.savefig(fpath5, dpi=150, bbox_inches="tight")
print(f"  Saved: {fpath5}")

plt.close("all")

# ---------------------------------------------------------------------------
# 10.  Summary
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("RECONSTRUCTION SUMMARY")
print("=" * 65)
print(f"  Compute backend     : {_backend}")
print(f"  Wavelength          : {WAVELENGTH*1e9:.4f} nm")
print(f"  Pixel size          : {PIXEL_SIZE*1e9:.2f} nm")
print(f"  Volume shape        : ({Nz}, {Ny}, {Nx})  [Nz, Ny, Nx]")
print(f"  Angles used         : {N_use}  (of {N_ANGLES} total)")
print(f"  MS slabs / angle    : {N_SLICES}  (Δz = {SLICE_DZ*1e9:.1f} nm)")
print(f"  Pass 2 iterations   : {N_ITER}")
print(f"  Fresnel number      : {F:.3f}")
print(f"  δ_FBP  range        : [{delta_fbp.min():.3e}, {delta_fbp.max():.3e}]")
print(f"  δ_TP   range        : [{delta_tp.min():.3e},  {delta_tp.max():.3e}]")
print(f"  Loss (final iter)   : {loss_history[-1]:.4e}")
print(f"  Figures saved to    : {OUT_DIR}/")
print("=" * 65)
