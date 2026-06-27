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
print(f"  FBP_METHOD    = {FBP_METHOD}")
print(f"  backend       = {'torch/'+str(DEVICE) if TORCH_AVAILABLE else 'numpy'}")
print()

# ---------------------------------------------------------------------------
# 3.  FBP on truncated projections  (Pass 1)
# ---------------------------------------------------------------------------

print("Step 1 — FBP on truncated projections (Pass 1) …", flush=True)
t0 = time.time()

# FBP gives a ROI volume (nz=nx_trunc, ny_trunc, nx_trunc)
nz_roi = nx_trunc

_method = FBP_METHOD
if _method == 'auto':
    _method = 'gpu' if (TORCH_AVAILABLE and DEVICE.type == 'cuda') else 'iradon'

_fbp_used_gpu = False

if _method == 'gpu' and TORCH_AVAILABLE and DEVICE.type == 'cuda':
    # Inline GPU FBP (same algorithm as twopass_real_data._fbp_gpu)
    import torch
    import torch.nn.functional as F_nn

    N_ang_f, Ny_f, Nx_f = phase_use.shape
    sino = torch.as_tensor(phase_use, dtype=torch.float32, device=DEVICE)
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
    recon_roi = torch.zeros(Ny_f, Nx_f, Nx_f, device=DEVICE)
    for iy in range(Ny_f):
        fi      = filt[iy].unsqueeze(1).unsqueeze(1)
        sampled = F_nn.grid_sample(fi, grid_all, mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
        recon_roi[iy] = sampled.squeeze(1).sum(0)
    recon_roi *= np.pi / (2*N_ang_f)
    recon_roi *= (X**2 + Z**2 <= R**2).unsqueeze(0)
    delta_roi = recon_roi.permute(1, 0, 2).cpu().numpy()
    delta_roi = np.ascontiguousarray(delta_roi[::-1])  # z-flip GPU FBP convention
    _fbp_used_gpu = True
    del sino, filt, recon_roi, grid_all

else:
    _slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if _slurm_cpus is not None:
        n_jobs = int(_slurm_cpus)
    elif _JOBLIB:
        n_jobs = min(ny_trunc, os.cpu_count() or 1)
    else:
        n_jobs = 1
    print(f"  CPU FBP (n_jobs={n_jobs}) …", flush=True)

    def _fbp_slice(iy):
        return iradon(phase_use[:, iy, :].T, theta=theta_use,
                      filter_name="ramp", circle=True)

    if _JOBLIB and n_jobs != 1:
        slices    = Parallel(n_jobs=n_jobs, prefer="threads")(
                        joblib_delayed(_fbp_slice)(iy) for iy in range(ny_trunc))
        delta_roi = np.stack(slices, axis=1)
    else:
        delta_roi = np.zeros((nz_roi, ny_trunc, nx_trunc), dtype=np.float64)
        for iy in range(ny_trunc):
            delta_roi[:, iy, :] = _fbp_slice(iy)

# Convert phase → δ
delta_roi = (-delta_roi / (K0 * PIXEL_SIZE)).astype(np.float32)
delta_roi = delta_roi.clip(0, None)

# Soft circular mask on the ROI FBP (boundary ringing)
_Zm = np.arange(nz_roi) - nz_roi / 2.0
_Xm = np.arange(nx_trunc) - nx_trunc / 2.0
_ZZ, _XX = np.meshgrid(_Zm, _Xm, indexing='ij')
_r_max  = (min(nz_roi, nx_trunc) - 1) / 2.0
_r_fade = max(3, int(0.05 * _r_max))
_mask2d = np.clip((np.sqrt(_ZZ**2 + _XX**2) - _r_max + _r_fade) / _r_fade, 0, 1)
_mask2d = 1.0 - _mask2d   # taper from 1 at centre to 0 at boundary
delta_roi *= _mask2d[:, np.newaxis, :]
del _Zm, _Xm, _ZZ, _XX, _mask2d

print(f"  ROI FBP done in {time.time()-t0:.1f} s  shape={delta_roi.shape}")
print(f"  δ_ROI range: [{delta_roi.min():.3e}, {delta_roi.max():.3e}]")

# ---------------------------------------------------------------------------
# 4.  Embed ROI in extended full grid
# ---------------------------------------------------------------------------

print("Step 2 — Embedding ROI in extended full-grid volume …", flush=True)

delta_full = np.zeros((Nz_full, Ny_full, Nx_full), dtype=np.float32)
beta_full  = np.zeros_like(delta_full)

# The ROI FBP volume spans (x: FOV_X0..FOV_X1) in the full grid.
# z shares the same window (square slices from FBP).
FOV_Z0, FOV_Z1 = FOV_X0, FOV_X1
delta_full[FOV_Z0:FOV_Z1, FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] = delta_roi
beta_full[FOV_Z0:FOV_Z1,  FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] = delta_roi * 1e-3

print(f"  Full grid: ({Nz_full}, {Ny_full}, {Nx_full})  "
      f"ROI embedded at z[{FOV_Z0}:{FOV_Z1}] y[{FOV_Y0}:{FOV_Y1}] x[{FOV_X0}:{FOV_X1}]")

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
        grad_delta = torch.zeros_like(delta_tp_t)
        grad_beta  = torch.zeros_like(beta_tp_t) if OPTIMIZE_BETA else None

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
            # Extract slices from full volume, forward propagate
            delta_sl, beta_sl, delta_means = extract_slices_torch(
                delta_tp_t, beta_tp_t, theta, n_slices=N_SLICES)

            U_exit, wf = torch_engine.forward(
                delta_sl, beta_sl, probe_t,
                delta_means=delta_means,
                store_wavefields=True)

            # Build masked measured wavefield:
            # set u_meas_full = U_exit everywhere EXCEPT the FOV window,
            # where it takes the actual measured exit wave.
            # → residual is zero outside FOV → no gradient there.
            u_meas_ai = torch.from_numpy(u_meas_trunc[ai]).to(DEVICE)
            u_meas_full_t = U_exit.clone().detach()
            u_meas_full_t[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] = u_meas_ai

            # Loss: only inside the FOV window
            res_fov  = U_exit[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] - u_meas_ai
            loss_i   = 0.5 * float(res_fov.abs().pow(2).sum().cpu())
            total_loss += loss_i

            # Backward: pass the masked measured wavefield so gradient
            # is zero outside the FOV
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

        loss_history.append(total_loss)
        elapsed = time.time() - t_iter
        print(f"  Iter {it+1:3d}/{N_ITER}  loss={total_loss:.4e}{tv_str}  "
              f"lr={lr_t:.2e}  t={elapsed:.1f}s  "
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

    adam_d = _Adam(delta_tp.shape, lr=LR)
    adam_b = _Adam(beta_tp.shape, lr=LR) if OPTIMIZE_BETA else None

    for it in range(N_ITER):
        t_iter = time.time()
        total_loss = 0.0
        grad_delta = np.zeros_like(delta_tp)
        grad_beta  = np.zeros_like(beta_tp) if OPTIMIZE_BETA else None

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

            # Forward pass
            U_exit, wf = engine.forward(
                delta_sl, beta_sl, probe_full,
                delta_means=delta_means,
                store_wavefields=True)

            # Masked measured wavefield (zero residual outside FOV)
            u_meas_ai    = u_meas_trunc[ai]
            u_meas_full  = U_exit.copy()
            u_meas_full[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] = u_meas_ai

            # Loss (FOV only)
            res_fov    = U_exit[FOV_Y0:FOV_Y1, FOV_X0:FOV_X1] - u_meas_ai
            loss_i     = 0.5 * float(np.sum(np.abs(res_fov)**2))
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

        loss_history.append(total_loss)
        elapsed = time.time() - t_iter
        _roi = delta_tp[FOV_Z0:FOV_Z1, :, FOV_X0:FOV_X1]
        print(f"  Iter {it+1:3d}/{N_ITER}  loss={total_loss:.4e}{tv_str}  "
              f"lr={lr_t:.2e}  t={elapsed:.1f}s  "
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
    # Metadata
    wavelength = WAVELENGTH,
    psize      = PIXEL_SIZE,
    theta      = theta_use,
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
print("  Load the FULL reconstruction (twopass_real_data.py output), then:")
print("  from tutorial.local_tomo_simulator import extract_groundtruth_roi")
print("  fov_meta = dict(fov_x0=FOV_X0, fov_x1=FOV_X1, fov_y0=FOV_Y0,")
print("                  fov_y1=FOV_Y1, full_Nx=Nx_full, full_Ny=Ny_full)")
print("  gt_roi = extract_groundtruth_roi(delta_full_ref, fov_meta)")
