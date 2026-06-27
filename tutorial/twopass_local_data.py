#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass multislice reconstruction for local (interior / ROI) tomography
=========================================================================

Use this script on datasets produced by ``local_tomo_simulator.py``.

Local-tomography physics
------------------------
The detector captures only a central FOV window (columns x0:x1) narrower than
the object.  Those truncated projections still carry the correct phase integral
through the interior of the FOV, so the FULL extended-grid volume is a valid
forward model: simulate on the full grid, compare only the measured FOV strip.

Algorithm
---------
Pass 1 (FBP):
  Run FBP on the truncated projections → small ROI volume (nz, ny, nx).
  Embed the ROI in a zero-padded full-grid volume (Nz_full, Ny_full, Nx_full).

Pass 2 (multislice gradient descent):
  Forward model operates on the FULL grid.  After each synthetic projection,
  only pixels inside the measured FOV window enter the data loss.  Pixels
  outside the FOV see zero residual → no gradient contribution → the halo is
  regularised purely by TV.  This lets the halo self-consistently account for
  material that passes through the beam at other angles but lies outside the
  measured FOV.

Input
-----
A .npz produced by ``local_tomo_simulator.py`` that carries:
  'projections'  : float32 (N_ang, ny, nx)  — truncated phase projections [rad]
  'fov_fov_x0/x1/y0/y1'  : int   — FOV window in the base-cropped grid
  'fov_full_Nx/Ny'        : int   — full base-cropped grid dimensions
  'wavelen', 'psize', 'theta'      — standard metadata

Usage
-----
  # Generate input (50 % FOV, base crop already applied by simulator):
  python tutorial/local_tomo_simulator.py --fov-frac-x 0.5

  # Reconstruct (set DATA_FILE below, or pass via env / edit this script):
  python tutorial/twopass_local_data.py
"""

# ---------------------------------------------------------------------------
# 0.  Imports
# ---------------------------------------------------------------------------

import os, sys, time, importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.transform import iradon
try:
    from joblib import Parallel, delayed as joblib_delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False

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

# Point DATA_FILE at any .npz produced by local_tomo_simulator.py.
# The default looks for the 50 % FOV file next to this script.
DATA_FILE = os.path.join(
    _HERE, "PXCTalignedprojections_localfov50.npz")

if not os.path.isfile(DATA_FILE):
    sys.exit(
        f"Data file not found:\n  {DATA_FILE}\n"
        "Generate it first with:\n"
        "  python tutorial/local_tomo_simulator.py --fov-frac-x 0.5")

print(f"Loading data from: {DATA_FILE}")
data = np.load(DATA_FILE, allow_pickle=True)

phase_trunc = data["projections"].astype(np.float32)  # (N_ang, ny, nx)
WAVELENGTH  = float(data["wavelen"])
PIXEL_SIZE  = float(data["psize"])
THETA       = data["theta"].astype(np.float64)

N_ANGLES, ny_trunc, nx_trunc = phase_trunc.shape
K0 = 2.0 * np.pi / WAVELENGTH

# ── FOV metadata written by local_tomo_simulator.py ───────────────────────
# The simulator saves each crop-window key with the prefix "fov_fov_" because
# crop_fov already prepends "fov_" to keys that don't start with it.
# Both naming conventions are accepted here for robustness.
def _fov_key(d, name):
    """Return d[name] trying both the double-prefix and single-prefix forms."""
    for k in (name, "fov_" + name):
        if k in d:
            return int(d[k])
    raise KeyError(f"FOV key '{name}' not found in npz (keys: {list(d.keys())})")

FOV_X0    = _fov_key(data, "fov_x0")
FOV_X1    = _fov_key(data, "fov_x1")
FOV_Y0    = _fov_key(data, "fov_y0")
FOV_Y1    = _fov_key(data, "fov_y1")
Nx_full   = _fov_key(data, "full_Nx")
Ny_full   = _fov_key(data, "full_Ny")
Nz_full   = Nx_full   # FBP always produces square (Nx×Nx) slices

print(f"  N_ANGLES         = {N_ANGLES}")
print(f"  Truncated (ny,nx)= ({ny_trunc}, {nx_trunc})")
print(f"  Full grid (Ny,Nx)= ({Ny_full}, {Nx_full})  Nz={Nz_full}")
print(f"  FOV window       : x[{FOV_X0}:{FOV_X1}]  y[{FOV_Y0}:{FOV_Y1}]")
print(f"  FOV fraction     : {(FOV_X1-FOV_X0)/Nx_full:.0%} x  "
      f"{(FOV_Y1-FOV_Y0)/Ny_full:.0%} y")
print(f"  θ range          : [{THETA.min():.1f}, {THETA.max():.1f}]°")
print(f"  λ = {WAVELENGTH*1e9:.4f} nm   pixel = {PIXEL_SIZE*1e9:.2f} nm")

F = PIXEL_SIZE**2 / (WAVELENGTH * Nz_full * PIXEL_SIZE)
print(f"  Fresnel number F = {F:.3f}  "
      f"({'diffraction regime → two-pass important' if F<1 else 'projection approx. reasonable'})")
print()

# ---------------------------------------------------------------------------
# 2.  Reconstruction parameters
# ---------------------------------------------------------------------------

# ── Multislice ─────────────────────────────────────────────────────────────
N_SLICES    = 64
SLICE_DZ    = Nz_full * PIXEL_SIZE / N_SLICES

# ── Pass 2 optimisation ────────────────────────────────────────────────────
N_ITER       = 100
LR           = 5e-6
LAMBDA_TV    = 1e-5    # TV weight — applies to the full extended grid
WARMUP_ITERS = 3

# TV weight multiplier for halo voxels (outside the ROI in x).
# >1 means stronger regularisation on the coarse outer annulus.
# 1.0 = uniform TV everywhere (safe default for first runs).
LAMBDA_TV_HALO = 5.0

OPTIMIZE_BETA = False  # beta ~ 1e-3*delta for hard X-rays; saving VRAM

# ── FBP backend ────────────────────────────────────────────────────────────
FBP_METHOD  = 'auto'   # 'auto' | 'iradon' | 'gpu'

# ── Zero-padding before FBP ────────────────────────────────────────────────
# The standard local-tomo trick: before FBP, zero-pad each truncated
# projection (sinogram row) to the FULL detector width (Nx_full columns).
# This pushes the truncation cupping artifact outward to the boundary of
# the inscribed circle, leaving the interior less biased.  The resulting
# FBP volume covers the full grid (Nz_full, Ny_full, Nx_full) — the same
# size as the Pass-2 extended grid — giving a much better initial condition.
# Set False to use the old approach (ROI-only FBP embedded in zeros).
# NOTE: padded FBP turned out to be counterproductive — the ramp filter
# creates a bright ring at the hard FOV boundary that lands inside the ROI
# and contaminates Pass 2.  Keep False until a smoother taper is implemented.
PADDED_FBP  = False

# ── Per-angle phase offset model ───────────────────────────────────────────
# Each truncated projection is missing the phase contribution from material
# OUTSIDE the FOV: φ_measured(θ) = φ_true(θ) − φ_exterior(θ).
#
# NEGATIVE RESULT (do not re-enable without changing the basis): fitting one
# real SCALAR c_θ per angle does NOT fix the interior bias.  A uniform δ-bias
# Δδ over the circular FOV projects to a CHORD-LENGTH profile 2√(R²−t²) in the
# detector — peaked at centre, zero at the edges, the same for every angle —
# NOT a flat offset.  A scalar c_θ is flat across the detector and lives in an
# orthogonal subspace, so it cannot absorb the bias; moreover a constant added
# to every projection is killed by the ramp filter (DC→0).  Empirically r went
# 0.49→0.48 with this on.  Kept for reference; leave OFF.
PHASE_OFFSET      = False
OFFSET_LR         = 0.05   # Adam LR for c_θ [rad/step]

# ── DC anchoring via a known-value (air) sub-region ────────────────────────
# The interior problem is unique only up to a smooth additive function
# (Courdurier/Kudyakov), UNLESS the value is known on some sub-region.  Air /
# pores give us exactly that: δ_air = 0.  Each iteration we estimate the air
# floor as a low percentile of the in-FOV δ and subtract it from the whole
# volume (then re-clamp ≥0).  Because the data term is flat along the DC
# null-space, the optimiser does not push the offset back, so this is stable
# and pins the absolute scale.  This is the lever that actually removes the
# ~+5e-6 uniform bias seen in the difference maps.
DC_ANCHOR      = True
AIR_PERCENTILE = 2.0   # percentile of in-FOV positive δ taken as the air floor

# ── Angle subsampling (prototyping) ────────────────────────────────────────
ANGLE_STEP   = 1

theta_use  = THETA[::ANGLE_STEP]
phase_use  = phase_trunc[::ANGLE_STEP]
N_use      = len(theta_use)
if ANGLE_STEP > 1:
    print(f"  [Subsampling] Using {N_use}/{N_ANGLES} angles (every {ANGLE_STEP}th)\n")

# ── Output directory ────────────────────────────────────────────────────────
_data_stem = os.path.splitext(os.path.basename(DATA_FILE))[0]
_data_tag  = _data_stem.replace("PXCTalignedprojections", "").lstrip("_")
OUT_DIR = os.path.join(_HERE, f"twopass_local_figures_{_data_tag}")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"  Output dir : {OUT_DIR}")

print("Reconstruction parameters")
print(f"  N_SLICES      = {N_SLICES}  (slab Δz = {SLICE_DZ*1e9:.1f} nm)")
print(f"  N_ITER        = {N_ITER}   LR = {LR}")
print(f"  LAMBDA_TV     = {LAMBDA_TV}  LAMBDA_TV_HALO = {LAMBDA_TV_HALO}")
print(f"  OPTIMIZE_BETA = {OPTIMIZE_BETA}")
print(f"  PHASE_OFFSET  = {PHASE_OFFSET}"
      + (f"  OFFSET_LR = {OFFSET_LR}" if PHASE_OFFSET else ""))
print(f"  DC_ANCHOR     = {DC_ANCHOR}"
      + (f"  AIR_PERCENTILE = {AIR_PERCENTILE}" if DC_ANCHOR else ""))
print(f"  FBP_METHOD    = {FBP_METHOD}  PADDED_FBP = {PADDED_FBP}")
print(f"  backend       = {'torch/'+str(DEVICE) if TORCH_AVAILABLE else 'numpy'}")
print()

# ---------------------------------------------------------------------------
# 3.  FBP  (Pass 1)
# ---------------------------------------------------------------------------

print("Step 1 — FBP (Pass 1) …", flush=True)
t0 = time.time()

# z always equals x for FBP (square slices)
FOV_Z0, FOV_Z1 = FOV_X0, FOV_X1

# Build the sinogram that FBP will actually see:
# PADDED_FBP=True  → zero-pad each projection to the full detector width.
#   Classic local-tomo trick: the ramp filter then sees zeros outside the FOV
#   rather than a hard edge, so the cupping bias is pushed to the boundary of
#   the inscribed circle instead of contaminating the interior.
#   FBP output: (Nz_full, Ny_full, Nx_full) — same size as the Pass-2 grid.
# PADDED_FBP=False → run FBP only on the small truncated FOV, then embed.
if PADDED_FBP:
    phase_fbp = np.zeros((N_use, Ny_full, Nx_full), dtype=np.float32)
    phase_fbp[:, FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] = phase_use
    Ny_fbp, Nx_fbp = Ny_full, Nx_full
    print(f"  Zero-padded sinogram: {phase_use.shape} → {phase_fbp.shape}  "
          f"(FOV at x[{FOV_X0}:{FOV_X1}])")
else:
    phase_fbp = phase_use
    Ny_fbp, Nx_fbp = ny_trunc, nx_trunc
    print(f"  Unpadded sinogram: {phase_fbp.shape}  (ROI-only FBP)")

_method = FBP_METHOD
if _method == 'auto':
    _method = 'gpu' if (TORCH_AVAILABLE and DEVICE.type == 'cuda') else 'iradon'

_fbp_used_gpu = False

if _method == 'gpu' and TORCH_AVAILABLE and DEVICE.type == 'cuda':
    import torch
    import torch.nn.functional as F_nn

    N_ang_f, Ny_f, Nx_f = phase_fbp.shape
    sino = torch.as_tensor(phase_fbp, dtype=torch.float32, device=DEVICE)
    sino = sino.permute(1, 0, 2).contiguous()
    n_pad  = max(64, int(2**np.ceil(np.log2(2*Nx_f))))
    freqs  = torch.fft.rfftfreq(n_pad, device=DEVICE).float()
    ramp   = 2.0 * freqs
    sino_f = torch.fft.rfft(sino, n=n_pad, dim=-1) * ramp[None, None, :]
    filt   = torch.fft.irfft(sino_f, n=n_pad, dim=-1)[..., :Nx_f]
    R = (Nx_f - 1) / 2.0
    x_lin = torch.linspace(-R, R, Nx_f, device=DEVICE)
    z_lin = torch.linspace(-R, R, Nx_f, device=DEVICE)
    Z, X  = torch.meshgrid(z_lin, x_lin, indexing='ij')
    cos_t = torch.as_tensor(np.cos(np.deg2rad(theta_use)), dtype=torch.float32, device=DEVICE)
    sin_t = torch.as_tensor(np.sin(np.deg2rad(theta_use)), dtype=torch.float32, device=DEVICE)
    t_all    = (X[None]*cos_t[:,None,None] + Z[None]*sin_t[:,None,None]) / R
    grid_all = torch.stack([t_all, torch.zeros_like(t_all)], dim=-1)
    recon_g = torch.zeros(Ny_f, Nx_f, Nx_f, device=DEVICE)
    for iy in range(Ny_f):
        fi      = filt[iy].unsqueeze(1).unsqueeze(1)
        sampled = F_nn.grid_sample(fi, grid_all, mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
        recon_g[iy] = sampled.squeeze(1).sum(0)
    recon_g *= np.pi / (2*N_ang_f)
    recon_g *= (X**2 + Z**2 <= R**2).unsqueeze(0)
    delta_fbp_out = recon_g.permute(1, 0, 2).cpu().numpy()
    delta_fbp_out = np.ascontiguousarray(delta_fbp_out[::-1])
    _fbp_used_gpu = True
    del sino, filt, recon_g, grid_all

else:
    _slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if _slurm_cpus is not None:
        n_jobs = int(_slurm_cpus)
    elif _JOBLIB:
        n_jobs = min(Ny_fbp, os.cpu_count() or 1)
    else:
        n_jobs = 1
    print(f"  CPU FBP (n_jobs={n_jobs}) …", flush=True)

    def _fbp_slice(iy):
        return iradon(phase_fbp[:, iy, :].T, theta=theta_use,
                      filter_name="ramp", circle=True)

    if _JOBLIB and n_jobs != 1:
        slices       = Parallel(n_jobs=n_jobs, prefer="threads")(
                           joblib_delayed(_fbp_slice)(iy) for iy in range(Ny_fbp))
        delta_fbp_out = np.stack(slices, axis=1)
    else:
        delta_fbp_out = np.zeros((Nx_fbp, Ny_fbp, Nx_fbp), dtype=np.float64)
        for iy in range(Ny_fbp):
            delta_fbp_out[:, iy, :] = _fbp_slice(iy)

# Phase → δ, positivity
delta_fbp_out = (-delta_fbp_out / (K0 * PIXEL_SIZE)).astype(np.float32)
delta_fbp_out = delta_fbp_out.clip(0, None)

# Soft circular mask (suppress FBP inscribed-circle boundary ringing)
_Nz_m, _Nx_m = delta_fbp_out.shape[0], delta_fbp_out.shape[2]
_Zm = np.arange(_Nz_m) - _Nz_m / 2.0
_Xm = np.arange(_Nx_m) - _Nx_m / 2.0
_ZZ, _XX = np.meshgrid(_Zm, _Xm, indexing='ij')
_r_max  = (min(_Nz_m, _Nx_m) - 1) / 2.0
_r_fade = max(3, int(0.05 * _r_max))
_mask2d = np.clip((_r_max - np.sqrt(_ZZ**2 + _XX**2)) / _r_fade, 0.0, 1.0)
delta_fbp_out *= _mask2d[:, np.newaxis, :]
del _Zm, _Xm, _ZZ, _XX, _mask2d

print(f"  FBP done in {time.time()-t0:.1f} s  shape={delta_fbp_out.shape}")
print(f"  δ_FBP range: [{delta_fbp_out.min():.3e}, {delta_fbp_out.max():.3e}]")

# ---------------------------------------------------------------------------
# 4.  Build the full-grid initial volume
# ---------------------------------------------------------------------------

print("Step 2 — Building full-grid initial volume …", flush=True)

if PADDED_FBP:
    # FBP already covers the full grid
    delta_full = delta_fbp_out
    print(f"  Padded FBP covers the full grid: {delta_full.shape}")
else:
    # Embed ROI-only FBP in a zero-padded full grid
    delta_full = np.zeros((Nz_full, Ny_full, Nx_full), dtype=np.float32)
    delta_full[FOV_Z0:FOV_Z1, FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] = delta_fbp_out
    print(f"  ROI embedded at z[{FOV_Z0}:{FOV_Z1}] y[{FOV_Y0}:{FOV_Y1}] "
          f"x[{FOV_X0}:{FOV_X1}] in {delta_full.shape}")

beta_full = delta_full * 1e-3

# ---------------------------------------------------------------------------
# 5.  Complex measured exit waves  (truncated FOV only)
# ---------------------------------------------------------------------------

u_meas_trunc = np.exp(1j * phase_use.astype(np.complex128)).astype(np.complex64)
# shape: (N_use, ny_trunc, nx_trunc)

# Full-size probe (plane wave on the extended detector)
probe_full = np.ones((Ny_full, Nx_full), dtype=np.complex64)

# ---------------------------------------------------------------------------
# 6.  TV regulariser (NumPy fallback — same as twopass_real_data.py)
# ---------------------------------------------------------------------------

def tv_grad(vol, eps=1e-8):
    gz = np.zeros_like(vol); gz[:-1]       = vol[1:]       - vol[:-1]
    gy = np.zeros_like(vol); gy[:, :-1, :] = vol[:, 1:, :] - vol[:, :-1, :]
    gx = np.zeros_like(vol); gx[:, :, :-1] = vol[:, :, 1:] - vol[:, :, :-1]
    norm = np.sqrt(gz**2 + gy**2 + gx**2 + eps)
    tv_val = float(np.sum(norm))
    gz /= norm; gy /= norm; gx /= norm
    grad = np.empty_like(vol)
    grad[0,  :, :]  = -gz[0,  :, :]
    grad[1:, :, :]  =  gz[:-1, :, :] - gz[1:, :, :]
    grad[:,  0, :]  -= gy[:,  0, :]
    grad[:, 1:, :]  += gy[:, :-1, :] - gy[:, 1:, :]
    grad[:, :,  0]  -= gx[:, :,  0]
    grad[:, :, 1:]  += gx[:, :, :-1] - gx[:, :, 1:]
    return grad, tv_val


class _Adam:
    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0

    def step(self, param, grad):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*grad**2
        param -= self.lr * (self.m/(1-self.b1**self.t)) / (
                  np.sqrt(self.v/(1-self.b2**self.t)) + self.eps)
        return param

# ---------------------------------------------------------------------------
# 7.  Pass 2 — multislice refinement on the extended grid
# ---------------------------------------------------------------------------

print("Step 3 — Multislice refinement on extended grid (Pass 2) …", flush=True)
_backend = "torch/" + str(DEVICE) if TORCH_AVAILABLE else "numpy"
print(f"  backend={_backend}  n_iter={N_ITER}  lr={LR}  "
      f"n_slices={N_SLICES}  lambda_tv={LAMBDA_TV:.0e}\n")

delta_tp = delta_full.copy()
beta_tp  = beta_full.copy()
loss_history = []
t_total = time.time()

if TORCH_AVAILABLE:
    # ── PyTorch path ───────────────────────────────────────────────────────
    warmup_device(DEVICE)

    torch_engine = TorchMultisliceEngine(
        shape           = (Ny_full, Nx_full),
        pixel_size      = PIXEL_SIZE,
        wavelength      = WAVELENGTH,
        slice_thickness = SLICE_DZ,
        device          = DEVICE,
    )

    delta_tp_t = torch.from_numpy(delta_tp.astype(np.float32)).to(DEVICE)
    beta_tp_t  = torch.from_numpy(beta_tp.astype(np.float32)).to(DEVICE)
    probe_t    = torch.from_numpy(probe_full).to(DEVICE)
    # u_meas stays on CPU and is sliced per angle; move each slice on demand.

    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    adam_d = TorchAdamState(delta_tp_t.shape, lr=LR, device=DEVICE)
    adam_b = (TorchAdamState(beta_tp_t.shape, lr=LR, device=DEVICE)
              if OPTIMIZE_BETA else None)

    # Per-angle phase offsets (CPU numpy — tiny, N_use scalars)
    c_offsets   = np.zeros(N_use, dtype=np.float64)
    adam_c      = _Adam((N_use,), lr=OFFSET_LR) if PHASE_OFFSET else None

    # Halo TV mask: 1.0 inside ROI x-window, LAMBDA_TV_HALO outside
    if LAMBDA_TV_HALO != 1.0 and LAMBDA_TV > 0:
        _tv_halo_mask = torch.ones(Nz_full, Ny_full, Nx_full,
                                   dtype=torch.float32, device=DEVICE)
        _tv_halo_mask[:, :, :FOV_X0] = LAMBDA_TV_HALO
        _tv_halo_mask[:, :, FOV_X1:] = LAMBDA_TV_HALO
        _tv_halo_mask[:FOV_Z0, :, :] = LAMBDA_TV_HALO
        _tv_halo_mask[FOV_Z1:, :, :] = LAMBDA_TV_HALO
    else:
        _tv_halo_mask = None

    for it in range(N_ITER):
        t_iter = time.time()
        total_loss = 0.0
        grad_delta  = torch.zeros_like(delta_tp_t)
        grad_beta   = torch.zeros_like(beta_tp_t) if OPTIMIZE_BETA else None
        grad_c      = np.zeros(N_use, dtype=np.float64)

        # LR schedule (linear warmup + cosine decay)
        if it < WARMUP_ITERS:
            lr_t = LR * (it + 1) / WARMUP_ITERS
        else:
            progress = (it - WARMUP_ITERS) / max(N_ITER - WARMUP_ITERS, 1)
            lr_t = LR * 0.5 * (1.0 + np.cos(np.pi * progress))
        adam_d.lr = lr_t
        if OPTIMIZE_BETA:
            adam_b.lr = lr_t

        for ai, theta in enumerate(theta_use):
            delta_sl, beta_sl, delta_means = extract_slices_torch(
                delta_tp_t, beta_tp_t, theta, n_slices=N_SLICES)

            U_exit, wf = torch_engine.forward(
                delta_sl, beta_sl, probe_t,
                delta_means=delta_means,
                store_wavefields=True)

            u_meas_ai = torch.from_numpy(u_meas_trunc[ai]).to(DEVICE)

            if PHASE_OFFSET:
                # Apply per-angle phase phasor to the synthetic exit wave.
                # exp(i*c) * U_exit[FOV] ≈ U_exit[FOV] + φ_exterior correction.
                phasor    = complex(np.cos(c_offsets[ai]), np.sin(c_offsets[ai]))
                phasor_t  = torch.tensor(phasor, dtype=torch.complex64, device=DEVICE)
                U_fov     = U_exit[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1]
                u_synth_c = phasor_t * U_fov           # phase-shifted synthetic
                res_fov   = u_synth_c - u_meas_ai
                # Gradient of loss w.r.t. c_i:
                # ∂L/∂c = -Im[sum(conj(res) * u_synth_c)]
                grad_c[ai] = -float(
                    (res_fov.conj() * u_synth_c).sum().imag.cpu())
                # For volume backward: residual that flows into engine is
                # conj(phasor)*res_fov (= U_exit[FOV] - conj(phasor)*u_meas_ai)
                # → masked measured wavefield trick with phase-rotated u_meas:
                u_meas_full_t = U_exit.clone().detach()
                u_meas_full_t[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] = (
                    phasor_t.conj() * u_meas_ai)
                loss_i = 0.5 * float(res_fov.abs().pow(2).sum().cpu())
            else:
                res_fov       = U_exit[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] - u_meas_ai
                u_meas_full_t = U_exit.clone().detach()
                u_meas_full_t[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] = u_meas_ai
                loss_i        = 0.5 * float(res_fov.abs().pow(2).sum().cpu())

            total_loss += loss_i

            gd_i, gb_i, _ = torch_engine.gradient(
                wf, u_meas_full_t, delta_means=delta_means)

            gd_vol, gb_vol = scatter_gradient_torch(
                gd_i, gb_i, theta, delta_tp_t.shape, n_slices=N_SLICES,
                compute_beta=OPTIMIZE_BETA)
            grad_delta.add_(gd_vol)
            if OPTIMIZE_BETA:
                grad_beta.add_(gb_vol)
            del gd_vol, gb_vol, U_exit, wf, u_meas_full_t

        grad_delta /= N_use
        if OPTIMIZE_BETA:
            grad_beta /= N_use

        # Update per-angle offsets
        if PHASE_OFFSET:
            adam_c.step(c_offsets, grad_c / N_use)

        tv_str = ""
        if LAMBDA_TV > 0:
            tv_gd_raw, tv_val_d = tv_grad_torch(delta_tp_t)
            if _tv_halo_mask is not None:
                tv_gd_raw = tv_gd_raw * _tv_halo_mask
            grad_delta += LAMBDA_TV * tv_gd_raw
            tv_val_b = 0.0
            if OPTIMIZE_BETA:
                tv_gb_raw, tv_val_b = tv_grad_torch(beta_tp_t)
                if _tv_halo_mask is not None:
                    tv_gb_raw = tv_gb_raw * _tv_halo_mask
                grad_beta += LAMBDA_TV * tv_gb_raw
            tv_str = f"  tv={LAMBDA_TV*(tv_val_d+tv_val_b):.3e}"

        adam_d.step(delta_tp_t, grad_delta)
        delta_tp_t.clamp_(min=0.0)
        if OPTIMIZE_BETA:
            adam_b.step(beta_tp_t, grad_beta)
            beta_tp_t.clamp_(min=0.0)

        # DC anchor: pin the air floor (low percentile of in-FOV δ) to 0
        air_dc = 0.0
        if DC_ANCHOR:
            _roi_flat = delta_tp_t[FOV_Z0:FOV_Z1, FOV_Y0:FOV_Y1,
                                   FOV_X0:FOV_X1].reshape(-1)
            _roi_pos = _roi_flat[_roi_flat > 0]
            if _roi_pos.numel() > 0:
                if _roi_pos.numel() > 500_000:        # torch.quantile element cap
                    _idx = torch.randint(0, _roi_pos.numel(), (500_000,),
                                         device=DEVICE)
                    _roi_pos = _roi_pos[_idx]
                air_dc = float(torch.quantile(_roi_pos, AIR_PERCENTILE / 100.0))
                delta_tp_t.sub_(air_dc).clamp_(min=0.0)

        loss_history.append(total_loss)
        elapsed = time.time() - t_iter
        off_str = (f"  c∈[{c_offsets.min():.3f},{c_offsets.max():.3f}]rad"
                   if PHASE_OFFSET else "")
        if DC_ANCHOR:
            off_str += f"  air_dc={air_dc:.2e}"
        print(f"  Iter {it+1:3d}/{N_ITER}  loss={total_loss:.4e}{tv_str}  "
              f"lr={lr_t:.2e}  t={elapsed:.1f}s{off_str}  "
              f"δ_ROI∈[{float(delta_tp_t[FOV_Z0:FOV_Z1,:,FOV_X0:FOV_X1].min()):.2e},"
              f"{float(delta_tp_t[FOV_Z0:FOV_Z1,:,FOV_X0:FOV_X1].max()):.2e}]",
              flush=True)

    delta_tp = delta_tp_t.cpu().numpy().astype(np.float32)
    beta_tp  = beta_tp_t.cpu().numpy().astype(np.float32)
    delta_tp_t = beta_tp_t = probe_t = None
    grad_delta = grad_beta = adam_d = adam_b = None
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

else:
    # ── NumPy fallback ─────────────────────────────────────────────────────
    engine = MultisliceEngine(
        shape           = (Ny_full, Nx_full),
        pixel_size      = PIXEL_SIZE,
        wavelength      = WAVELENGTH,
        slice_thickness = SLICE_DZ,
    )

    adam_d  = _Adam(delta_tp.shape, lr=LR)
    adam_b  = _Adam(beta_tp.shape, lr=LR) if OPTIMIZE_BETA else None
    c_offsets = np.zeros(N_use, dtype=np.float64)
    adam_c    = _Adam((N_use,), lr=OFFSET_LR) if PHASE_OFFSET else None

    for it in range(N_ITER):
        t_iter = time.time()
        total_loss = 0.0
        grad_delta = np.zeros_like(delta_tp)
        grad_beta  = np.zeros_like(beta_tp) if OPTIMIZE_BETA else None
        grad_c     = np.zeros(N_use, dtype=np.float64)

        if it < WARMUP_ITERS:
            lr_t = LR * (it + 1) / WARMUP_ITERS
        else:
            progress = (it - WARMUP_ITERS) / max(N_ITER - WARMUP_ITERS, 1)
            lr_t = LR * 0.5 * (1.0 + np.cos(np.pi * progress))
        adam_d.lr = lr_t
        if OPTIMIZE_BETA:
            adam_b.lr = lr_t

        for ai, theta in enumerate(theta_use):
            delta_sl, beta_sl, delta_means = extract_slices_from_volume(
                delta_tp, beta_tp, theta, n_slices=N_SLICES)

            U_exit, wf = engine.forward(
                delta_sl, beta_sl, probe_full,
                delta_means=delta_means,
                store_wavefields=True)

            u_meas_ai = u_meas_trunc[ai]

            if PHASE_OFFSET:
                phasor    = np.exp(1j * c_offsets[ai])
                U_fov     = U_exit[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1]
                u_synth_c = phasor * U_fov
                res_fov   = u_synth_c - u_meas_ai
                grad_c[ai] = -float(np.sum(np.conj(res_fov) * u_synth_c).imag)
                u_meas_full = U_exit.copy()
                u_meas_full[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] = (
                    np.conj(phasor) * u_meas_ai)
                loss_i = 0.5 * float(np.sum(np.abs(res_fov)**2))
            else:
                res_fov     = U_exit[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] - u_meas_ai
                u_meas_full = U_exit.copy()
                u_meas_full[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] = u_meas_ai
                loss_i      = 0.5 * float(np.sum(np.abs(res_fov)**2))

            total_loss += loss_i

            gd_i, gb_i, _ = engine.gradient(wf, u_meas_full, delta_means=delta_means)

            gd_vol, gb_vol = scatter_gradient_to_volume(
                gd_i, gb_i, theta, delta_tp.shape, n_slices=N_SLICES)
            grad_delta += gd_vol
            if OPTIMIZE_BETA:
                grad_beta += gb_vol

        grad_delta /= N_use
        if OPTIMIZE_BETA:
            grad_beta /= N_use
        if PHASE_OFFSET:
            adam_c.step(c_offsets, grad_c / N_use)

        tv_str = ""
        if LAMBDA_TV > 0:
            tv_gd, tv_val_d = tv_grad(delta_tp)
            grad_delta += LAMBDA_TV * tv_gd
            tv_val_b = 0.0
            if OPTIMIZE_BETA:
                tv_gb, tv_val_b = tv_grad(beta_tp)
                grad_beta += LAMBDA_TV * tv_gb
            tv_str = f"  tv={LAMBDA_TV*(tv_val_d+tv_val_b):.3e}"

        adam_d.step(delta_tp, grad_delta)
        np.clip(delta_tp, 0.0, None, out=delta_tp)
        if OPTIMIZE_BETA:
            adam_b.step(beta_tp, grad_beta)
            np.clip(beta_tp, 0.0, None, out=beta_tp)

        # DC anchor: pin the air floor (low percentile of in-FOV δ) to 0
        air_dc = 0.0
        if DC_ANCHOR:
            _roi_pos = delta_tp[FOV_Z0:FOV_Z1, FOV_Y0:FOV_Y1, FOV_X0:FOV_X1]
            _roi_pos = _roi_pos[_roi_pos > 0]
            if _roi_pos.size > 0:
                air_dc = float(np.percentile(_roi_pos, AIR_PERCENTILE))
                delta_tp -= air_dc
                np.clip(delta_tp, 0.0, None, out=delta_tp)

        loss_history.append(total_loss)
        elapsed = time.time() - t_iter
        _roi = delta_tp[FOV_Z0:FOV_Z1, :, FOV_X0:FOV_X1]
        off_str = (f"  c∈[{c_offsets.min():.3f},{c_offsets.max():.3f}]rad"
                   if PHASE_OFFSET else "")
        if DC_ANCHOR:
            off_str += f"  air_dc={air_dc:.2e}"
        print(f"  Iter {it+1:3d}/{N_ITER}  loss={total_loss:.4e}{tv_str}  "
              f"lr={lr_t:.2e}  t={elapsed:.1f}s{off_str}  "
              f"δ_ROI∈[{_roi.min():.2e},{_roi.max():.2e}]", flush=True)

print(f"\n  Pass 2 done in {time.time()-t_total:.1f} s total.\n")

# ---------------------------------------------------------------------------
# 8.  Save results
# ---------------------------------------------------------------------------

# Extract the converged ROI sub-volume
delta_tp_roi = delta_tp[FOV_Z0:FOV_Z1, FOV_Y0:FOV_Y1, FOV_X0:FOV_X1]
delta_fbp_roi = delta_full[FOV_Z0:FOV_Z1, FOV_Y0:FOV_Y1, FOV_X0:FOV_X1]

np.savez_compressed(
    os.path.join(OUT_DIR, "twopass_local_reconstruction.npz"),
    # Full extended-grid volumes
    delta_fbp_full = delta_full.astype(np.float32),
    delta_tp_full  = delta_tp.astype(np.float32),
    # ROI sub-volumes (for ground-truth comparison)
    delta_fbp_roi  = delta_fbp_roi.astype(np.float32),
    delta_tp_roi   = delta_tp_roi.astype(np.float32),
    # Per-angle phase offsets (rad); zeros if PHASE_OFFSET=False
    c_offsets  = c_offsets,
    theta      = theta_use,
    # Metadata
    wavelength = WAVELENGTH,
    psize      = PIXEL_SIZE,
    fov_x0=FOV_X0, fov_x1=FOV_X1,
    fov_y0=FOV_Y0, fov_y1=FOV_Y1,
    fov_z0=FOV_Z0, fov_z1=FOV_Z1,
    full_Nx=Nx_full, full_Ny=Ny_full,
)
print(f"  Volumes saved to: {OUT_DIR}/twopass_local_reconstruction.npz")

# ---------------------------------------------------------------------------
# 9.  Figures
# ---------------------------------------------------------------------------

print("Generating figures …", flush=True)
CMAP = "gray"

# Colour scale from the converged ROI
_pos = delta_tp_roi[delta_tp_roi > 0]
vmin = 0.0
vmax = float(np.percentile(_pos, 99.5)) if _pos.size > 0 else 1e-5
del _pos

iz_mid = delta_tp_roi.shape[0] // 2
iy_mid = delta_tp_roi.shape[1] // 2
ix_mid = delta_tp_roi.shape[2] // 2

# ── Figure 1: Three ROI orthogonal cuts — FBP-init vs two-pass ───────────
fig1, axes = plt.subplots(3, 2, figsize=(10, 11),
                           gridspec_kw={"hspace": 0.40, "wspace": 0.30})
cuts = [
    (delta_fbp_roi[iz_mid],           delta_tp_roi[iz_mid],
     f"ROI xy (z={iz_mid+FOV_Z0})"),
    (delta_fbp_roi[:, iy_mid, :],     delta_tp_roi[:, iy_mid, :],
     f"ROI xz (y={iy_mid+FOV_Y0})"),
    (delta_fbp_roi[:, :, ix_mid].T,   delta_tp_roi[:, :, ix_mid].T,
     f"ROI yz (x={ix_mid+FOV_X0})"),
]
for row, (fbp_cut, tp_cut, label) in enumerate(cuts):
    for col, (img, ttl) in enumerate([
            (fbp_cut, f"FBP-init  [{label}]"),
            (tp_cut,  f"Two-pass  [{label}]")]):
        im = axes[row, col].imshow(img, cmap=CMAP, vmin=vmin, vmax=vmax,
                                   origin="lower")
        axes[row, col].set_title(ttl, fontsize=8)
        axes[row, col].axis("off")
        plt.colorbar(im, ax=axes[row, col], fraction=0.04, label="δ")
fig1.suptitle(
    f"Local-tomo two-pass  FOV {(FOV_X1-FOV_X0)/Nx_full:.0%} × {(FOV_Y1-FOV_Y0)/Ny_full:.0%}\n"
    f"λ={WAVELENGTH*1e9:.3f} nm  pixel={PIXEL_SIZE*1e9:.1f} nm  "
    f"{N_use} angles  {N_SLICES} MS slabs  {N_ITER} iters",
    fontsize=9)
fig1.savefig(os.path.join(OUT_DIR, "fig1_roi_orthogonal_cuts.png"),
             dpi=150, bbox_inches="tight")
print(f"  Saved: fig1_roi_orthogonal_cuts.png")

# ── Figure 2: Full extended-grid mid-slice (shows halo) ───────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5),
                            gridspec_kw={"wspace": 0.35})
iz_full = Nz_full // 2
_pos_full = delta_tp[delta_tp > 0]
vmax_full = float(np.percentile(_pos_full, 99.5)) if _pos_full.size > 0 else vmax
for ax, (img, ttl) in zip(axes2, [
        (delta_full[iz_full], "FBP-init (full grid)"),
        (delta_tp[iz_full],   "Two-pass (full grid)")]):
    im = ax.imshow(img, cmap=CMAP, vmin=0, vmax=vmax_full, origin="lower")
    # FOV rectangle overlay
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((FOV_X0-0.5, FOV_Y0-0.5),
                            FOV_X1-FOV_X0, FOV_Y1-FOV_Y0,
                            edgecolor='r', facecolor='none', lw=1.2,
                            label='measured FOV'))
    ax.set_title(f"{ttl}  xy (z={iz_full})", fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.04, label="δ")
    ax.legend(fontsize=7, loc='upper right')
fig2.suptitle("Extended-grid reconstruction (red = measured FOV window)", fontsize=9)
fig2.savefig(os.path.join(OUT_DIR, "fig2_full_grid_midslice.png"),
             dpi=150, bbox_inches="tight")
print(f"  Saved: fig2_full_grid_midslice.png")

# ── Figure 3: Convergence ─────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 3.5))
ax3.semilogy(range(1, len(loss_history)+1), loss_history, "o-",
             lw=1.8, ms=4, color="steelblue")
ax3.set_xlabel("Iteration"); ax3.set_ylabel("FOV squared-residual loss")
ax3.set_title("Pass 2 convergence (loss over measured FOV only)", fontsize=9)
ax3.grid(True, which="both", alpha=0.35)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "fig3_convergence.png"), dpi=150, bbox_inches="tight")
print(f"  Saved: fig3_convergence.png")

# ── Figure 4: Difference map (two-pass − FBP-init) in the ROI ────────────
diff = delta_tp_roi - delta_fbp_roi
dlim = np.percentile(np.abs(diff), 99)
fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4),
                            gridspec_kw={"wspace": 0.35})
for ax, (img, lbl) in zip(axes4, [
        (diff[iz_mid],          f"ROI xy (z={iz_mid+FOV_Z0})"),
        (diff[:, iy_mid, :],    f"ROI xz (y={iy_mid+FOV_Y0})"),
        (diff[:, :, ix_mid].T,  f"ROI yz (x={ix_mid+FOV_X0})")]):
    im = ax.imshow(img, cmap="bwr", vmin=-dlim, vmax=dlim, origin="lower")
    ax.set_title(f"Δδ two-pass − FBP  [{lbl}]", fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Δδ")
fig4.suptitle("ROI difference map (two-pass − FBP init)", fontsize=9)
fig4.savefig(os.path.join(OUT_DIR, "fig4_roi_difference.png"),
             dpi=150, bbox_inches="tight")
print(f"  Saved: fig4_roi_difference.png")

# ── Figure 5: Per-angle phase offsets ─────────────────────────────────────
if PHASE_OFFSET and np.any(c_offsets != 0):
    fig5, ax5 = plt.subplots(figsize=(8, 3.5))
    ax5.plot(theta_use, np.rad2deg(c_offsets), "o-", ms=2, lw=1.0,
             color="steelblue")
    ax5.axhline(0, color="k", lw=0.8, ls="--")
    ax5.set_xlabel("Projection angle θ [°]")
    ax5.set_ylabel("Phase offset c_θ [°]")
    ax5.set_title("Per-angle phase offsets\n"
                  "(absorbs missing exterior line integral per projection)",
                  fontsize=9)
    ax5.grid(True, alpha=0.35)
    fig5.tight_layout()
    fig5.savefig(os.path.join(OUT_DIR, "fig5_phase_offsets.png"),
                 dpi=150, bbox_inches="tight")
    print(f"  Saved: fig5_phase_offsets.png  "
          f"(offset range: [{np.rad2deg(c_offsets.min()):.2f}°, "
          f"{np.rad2deg(c_offsets.max()):.2f}°])")

plt.close("all")

# ---------------------------------------------------------------------------
# 10.  Summary
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("LOCAL-TOMO RECONSTRUCTION SUMMARY")
print("=" * 65)
print(f"  Backend           : {_backend}")
print(f"  Full grid shape   : ({Nz_full}, {Ny_full}, {Nx_full})")
print(f"  ROI shape         : {delta_tp_roi.shape}")
print(f"  FOV fraction      : {(FOV_X1-FOV_X0)/Nx_full:.0%} x "
      f"{(FOV_Y1-FOV_Y0)/Ny_full:.0%} y")
print(f"  Angles            : {N_use}  (of {N_ANGLES})")
print(f"  MS slabs          : {N_SLICES}  (Δz = {SLICE_DZ*1e9:.1f} nm)")
print(f"  Pass 2 iterations : {N_ITER}")
print(f"  Fresnel number    : {F:.3f}")
print(f"  δ_ROI range       : [{delta_tp_roi.min():.3e}, {delta_tp_roi.max():.3e}]")
print(f"  Loss (final)      : {loss_history[-1]:.4e}")
print(f"  Results           : {OUT_DIR}/")
print("=" * 65)
print()
print("Ground-truth comparison:")
print("  Run the companion script (works standalone, no variable import needed):")
print(f"    python tutorial/compare_groundtruth.py \\")
print(f"      --local  {os.path.join(OUT_DIR, 'twopass_local_reconstruction.npz')} \\")
print(f"      --full   tutorial/twopass_real_figures/twopass_reconstruction.npz")
