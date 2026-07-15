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

# NOTE on CUDA memory fragmentation: if a *GPU* OOM recurs purely from
# fragmentation (lots of "reserved but unallocated" memory), export
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# BEFORE launching python.  It is NOT enabled here by default because it can
# slow large runs that churn big alloc/free (our per-angle rotation buffers),
# and the GPU OOM was actually resolved by OPTIMIZE_BETA=False.  We only honour
# whatever the user exported.

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

data        = np.load(DATA_FILE, allow_pickle=True)
# float32 phases (O(1) rad) halve host RAM vs float64 with no practical loss.
phase_stack = data["projections"].astype(np.float32)   # (N_ANGLES, Ny, Nx) [rad]
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

# ── Projection boundary crop ───────────────────────────────────────────────
# Ptychographic reconstruction produces unreliable phase values at the edges
# of each projection because the probe footprint overlaps the scan boundary
# (insufficient illumination).  These noisy pixels appear as large residuals
# in the self-consistency plot (residual meas − re-proj at x < CROP_X and
# x > Nx − CROP_X).
#
# Why crop?  Those noisy boundary pixels:
#   1. Produce large loss contributions that dominate the gradient.
#   2. Corrupt neighbouring voxels in the gradient back-projection.
#   3. Prevent the loss from converging below the noise floor.
#
# Tune CROP_X by looking at the self-consistency residual plot produced by a
# prior run: choose the value where the interior residuals are small but the
# boundary spike starts.  Typical values: 50–150 pixels.
# Tune CROP_Y similarly for top/bottom boundary rows.
CROP_X = 55   # pixels to remove from EACH side in the column (x) direction
              # 0 = no crop; the FBP and volume x-extent shrink by 2×CROP_X
CROP_Y = 55   # pixels to remove from EACH side in the row (y) direction
              # 0 = no crop; only the y-extent of projections and volume shrinks
              # Calibrated for PXCTalignedprojections.npz: 55 px each side.

if CROP_X > 0 or CROP_Y > 0:
    _sy = slice(CROP_Y if CROP_Y > 0 else None,
                -CROP_Y if CROP_Y > 0 else None)
    _sx = slice(CROP_X if CROP_X > 0 else None,
                -CROP_X if CROP_X > 0 else None)
    phase_stack = np.ascontiguousarray(phase_stack[:, _sy, _sx])
    _, Ny, Nx   = phase_stack.shape
    Nz          = Nx   # FBP reconstruction is always square in (x, z)
    print(f"  After crop ({CROP_Y}px top/bottom, {CROP_X}px left/right): "
          f"(Ny, Nx, Nz) = ({Ny}, {Nx}, {Nz})\n")

# ── Half-dataset mode — for Fourier Shell Correlation (FSC) ───────────────
# The standard way to estimate spatial resolution from a single dataset is to
# split the projections into two independent halves, reconstruct each
# separately, and compute the FSC between the two volumes.
#
# Workflow:
#   1. Run with FSC_HALF = 0  (even angles: θ[0], θ[2], θ[4], …)
#      → saves  twopass_real_figures_half0/twopass_reconstruction.npz
#   2. Run with FSC_HALF = 1  (odd angles: θ[1], θ[3], θ[5], …)
#      → saves  twopass_real_figures_half1/twopass_reconstruction.npz
#   3. Run  tutorial/fsc_analysis.py  to compute and plot the FSC curves.
#
# On a cluster (A100, ~15 s/iter with all angles → ~7 s/iter per half),
# 100-iter runs finish in < 12 min each — very practical in parallel.
#
# Set to None to use the full dataset (default).
FSC_HALF = None   # None | 0 | 1

# ── Multislice discretisation ──────────────────────────────────────────────
# Rule of thumb: each slab should be thinner than the depth of focus
#   DoF ≈ pixel² / λ = F * Nz * pixel  →  N_SLICES ≈ Nz / DoF_in_pixels
# For this dataset DoF ≈ F*Nz ≈ 143 pixels → start with Nz/16 ≈ 31 px/slab.
# Increase N_SLICES for higher accuracy (but proportionally slower).
# ── Benchmarks on NVIDIA A40 (48 GB), uncropped 493×394×493, 450 angles:
#   N_SLICES=16  →  ~17 s/iter   (time scales linearly with N_SLICES)
#   N_SLICES=64  →  ~68 s/iter
#   N_SLICES=64, with CROP_X=80/CROP_Y=20 (333×354×333)  →  ~40 s/iter
# Memory (persistent tensors): ~3.5 GB regardless of N_SLICES.
# N_SLICES=64 fits comfortably in a 48 GB GPU; use 16 only for fast tests.
N_SLICES    = 64
SLICE_DZ    = Nz * PIXEL_SIZE / N_SLICES          # [m] slab thickness

# ── Pass-1 (FBP) back-projector ─────────────────────────────────────────────
# 'auto'     -> _fbp_gpu on CUDA, else skimage.iradon  (default, unchanged)
# 'iradon'   -> force skimage.iradon (CPU)
# 'gpu'      -> force the custom GPU back-projector (requires CUDA)
# 'gridding' -> experimental NUFFT direct-Fourier back-projector
#               (tutorial/nufft_gridding.py, needs `pip install finufft`).
#               It self-tests against iradon at start-up and AUTOMATICALLY
#               FALLS BACK to iradon if the test fails or finufft is missing,
#               so it can never silently corrupt the reconstruction.
FBP_METHOD  = 'auto'

# ── Pass 2 optimisation ────────────────────────────────────────────────────
# Convergence guide (from experimental loss curves):
#   Loss drops ~65% in first 20 iters, then approaches the noise floor set
#   by measurement noise + ptychography reconstruction error.
# Timing on A40 with N_SLICES=64, cropped data, 450 angles:
#   ~40 s/iter  →  100 iters ≈ 67 min  →  request 2 h on SLURM
# On MacBook Pro MPS with N_SLICES=16: ~300 s/iter — use only for tests.
N_ITER       = 100
LR           = 5e-6       # Adam peak learning rate
                          # Hard X-ray data: δ ~ 1e-5–1e-6; was 5e-4 (suited for δ ~ 1e-3)
LAMBDA_TV    = 1e-5       # TV regularisation weight (0 to disable)
WARMUP_ITERS = 3          # linear LR warm-up iterations

# ── Optimise the absorption (beta) volume? ─────────────────────────────────
# For hard-X-ray phase objects beta ~ 1e-3 * delta and barely affects the
# forward model, yet optimising it costs THREE extra full-volume GPU buffers
# (its gradient + Adam m,v).  Freezing beta (=False) saves ~3x the volume
# size in VRAM with negligible effect on the recovered delta -- essential for
# large volumes on a 40-48 GB GPU.  True reproduces the original behaviour.
OPTIMIZE_BETA = True      # set False for large volumes (memory) ; delta-only

# ── Per-angle noise weighting (statistically-weighted data term) ───────────
# The Pass-2 data term is Sum_theta w_theta |model - measured|^2.  Weighting
# each projection by its inverse noise variance (a maximum-likelihood data
# term) uses the photon budget more efficiently than uniform least-squares,
# down-weighting noisy angles that would otherwise inject noise into the
# volume at full strength.
#   'uniform' -> w_theta = 1   (default; reproduces the unweighted result EXACTLY)
#   'snr'     -> w_theta = 1/sigma_theta^2, sigma from a robust high-frequency
#                noise estimate (MAD of horizontal first differences) per
#                projection.  Weights are normalised to mean 1 so the gradient
#                scale (and hence the tuned LR) is unchanged.
ANGLE_WEIGHT = 'uniform'   # 'uniform' | 'snr'

# ── Angle subsampling (for fast prototyping) ───────────────────────────────
# Set ANGLE_STEP = 1 to use all angles; 2 = every other angle, etc.
# FSC_HALF (above) takes precedence and selects even or odd angles.
ANGLE_STEP   = 1

# ── FSC half-dataset: override optimiser settings automatically ───────────
# With only half the angles the FBP initial condition is noisier.
# Using full-dataset settings (100 iters, LR=5e-6) lets the optimiser drift
# too far from FBP and reduces the FBP–two-pass correlation to ~0.69.
# The gentler settings below keep corr(FBP, two-pass) > 0.90 so that the
# FSC comparison between the two halves remains meaningful.
#
# WARNING: this block SILENTLY overrides N_ITER / LR / LAMBDA_TV set above (or
# injected by slurm_twopass.sh, whose sed only matches the top-level `^LAMBDA_TV`
# assignment, not this indented one).  An FSC run therefore never used the
# LAMBDA_TV printed at the top -- it used 5e-5.  Set FSC_AUTO_GENTLE = False to
# keep the explicit values, which is REQUIRED for the TV sweep: the whole point
# there is to vary LAMBDA_TV (including 0) while holding N_ITER/LR fixed at the
# gentle values, so that TV is the only variable.
FSC_AUTO_GENTLE = True

if FSC_HALF is not None and FSC_AUTO_GENTLE:
    N_ITER    = 30     # was 100 — stops before overfitting the half-angle set
    LR        = 2e-6   # was 5e-6 — smaller steps
    LAMBDA_TV = 5e-5   # was 1e-5 — stronger TV keeps the solution close to FBP
    print(f"  [FSC half-{FSC_HALF}] Overriding optimiser: "
          f"N_ITER={N_ITER}, LR={LR:.0e}, LAMBDA_TV={LAMBDA_TV:.0e}")
elif FSC_HALF is not None:
    print(f"  [FSC half-{FSC_HALF}] FSC_AUTO_GENTLE=False — keeping explicit "
          f"N_ITER={N_ITER}, LR={LR:.0e}, LAMBDA_TV={LAMBDA_TV:.0e}")

if FSC_HALF is not None:
    _all_idx   = np.arange(N_ANGLES)
    _half_idx  = _all_idx[FSC_HALF::2]    # even-half: 0,2,4,…  odd-half: 1,3,5,…
    theta_use  = THETA[_half_idx]
    phase_use  = phase_stack[_half_idx]
    N_use      = len(theta_use)
    print(f"  [FSC half-{FSC_HALF}] Using {N_use}/{N_ANGLES} angles "
          f"(every other, starting at index {FSC_HALF})\n")
else:
    theta_use  = THETA[::ANGLE_STEP]
    phase_use  = phase_stack[::ANGLE_STEP]
    N_use      = len(theta_use)
    if ANGLE_STEP > 1:
        print(f"  [Subsampling] Using {N_use}/{N_ANGLES} angles "
              f"(every {ANGLE_STEP}th)\n")

# Output directory.  Tags keep DIFFERENT runs from overwriting each other:
#   - dataset tag: derived from the input filename, so each DATA_FILE (the big
#     volume, jittered copies, etc.) writes to its OWN folder automatically.
#     The canonical 'PXCTalignedprojections.npz' yields no tag, preserving the
#     original 'twopass_real_figures...' name (backward compatible).
#   - FBP back-projector tag: '_grid' when FBP_METHOD='gridding' (else none);
#   - weighting tag:          '_snr' for non-uniform ANGLE_WEIGHT (else none);
#   - half tag:               '_half0' / '_half1' when FSC_HALF is set.
_data_stem = os.path.splitext(os.path.basename(DATA_FILE))[0]
_data_tag  = _data_stem.replace("PXCTalignedprojections", "").lstrip("_")
_dtag = f"_{_data_tag}" if _data_tag else ""        # '', '_big', '_jitter0.50px_x', ...
_ftag = "_grid" if FBP_METHOD == "gridding" else ""
_wtag = "" if ANGLE_WEIGHT == "uniform" else f"_{ANGLE_WEIGHT}"
_out_suffix = f"_half{FSC_HALF}" if FSC_HALF is not None else ""
# NOTE: keep this a SINGLE line — slurm_twopass.sh patches it with
# `sed s|^OUT_DIR=.*|...|`, which only replaces one line.
OUT_DIR = os.path.join(_HERE, f"twopass_real_figures{_dtag}{_ftag}{_wtag}{_out_suffix}")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"  Output dir : {OUT_DIR}")

print("Reconstruction parameters")
print(f"  N_SLICES   = {N_SLICES}  (slab Δz = {SLICE_DZ*1e9:.1f} nm = "
      f"{Nz//N_SLICES} pixels)")
print(f"  N_ITER     = {N_ITER}")
print(f"  LR         = {LR}")
print(f"  LAMBDA_TV  = {LAMBDA_TV}")
print(f"  OPTIMIZE_BETA = {OPTIMIZE_BETA}")
print(f"  ANGLE_WEIGHT = {ANGLE_WEIGHT}")
print(f"  FBP_METHOD = {FBP_METHOD}")
print(f"  backend    = {'torch/' + str(DEVICE) if TORCH_AVAILABLE else 'numpy'}")
print()

# ---------------------------------------------------------------------------
# 3.  FBP reconstruction  (Pass 1)
# ---------------------------------------------------------------------------

print("Step 1 — FBP reconstruction (Pass 1) …", flush=True)
t0 = time.time()

# ── GPU FBP (preferred when a CUDA/MPS device is available) ───────────────
# Implements filtered back-projection entirely in PyTorch so that the full
# pipeline stays on the GPU without any CPU bottleneck.
#
# Algorithm (matches skimage.transform.iradon with filter='ramp', circle=True):
#   1. Ramp-filter each sinogram row in the Fourier domain (rfft → × |ω| → irfft)
#   2. Backproject: for each angle θ_i, interpolate the filtered sinogram
#      at detector coordinate t = x·cos(θ_i) + z·sin(θ_i) and accumulate.
#      grid_sample provides bilinear interpolation — the same kernel used in
#      the multislice rotation, so no extra dependency.
#   3. Apply the FBP normalisation factor π/(2·N_angles).
#   4. Zero pixels outside the inscribed circle (circle=True).
#
# The geometry is identical for every y-slice, so the sampling grid is
# precomputed once for all N_angles and then reused Ny times.
# At N_angles=450, Ny=~350, Nz=Nx=~330 on an A40 this takes ~2–4 s.

def _fbp_gpu(phase_arr, theta_deg, dev):
    """
    GPU filtered back-projection.

    Parameters
    ----------
    phase_arr : ndarray (N_angles, Ny, Nx)
    theta_deg : ndarray (N_angles,) in degrees
    dev       : torch.device

    Returns
    -------
    recon : ndarray (Nz=Nx, Ny, Nx)
    """
    import torch
    import torch.nn.functional as F_nn

    N_ang, Ny_, Nx_ = phase_arr.shape
    Nz_ = Nx_

    # ── 1.  Sinogram → GPU  (Ny, N_angles, Nx) ─────────────────────────────
    sino = torch.as_tensor(phase_arr, dtype=torch.float32, device=dev)
    sino = sino.permute(1, 0, 2).contiguous()          # (Ny, N_ang, Nx)

    # ── 2.  Ramp filter in Fourier domain ───────────────────────────────────
    n_pad  = max(64, int(2 ** np.ceil(np.log2(2 * Nx_))))
    freqs  = torch.fft.rfftfreq(n_pad, device=dev).float()
    ramp   = 2.0 * freqs                               # Ram-Lak: |ω| on one-sided spectrum
    sino_f = torch.fft.rfft(sino, n=n_pad, dim=-1) * ramp[None, None, :]
    filt   = torch.fft.irfft(sino_f, n=n_pad, dim=-1)[..., :Nx_]  # (Ny, N_ang, Nx)

    # ── 3.  Precompute sampling grid for all angles ─────────────────────────
    #  Pixel (z, x) maps to detector coordinate t = x·cos(θ) + z·sin(θ),
    #  normalised to [-1, 1] for grid_sample (align_corners=True convention).
    R      = (Nx_ - 1) / 2.0
    x_lin  = torch.linspace(-R, R, Nx_, device=dev)
    z_lin  = torch.linspace(-R, R, Nz_, device=dev)
    Z, X   = torch.meshgrid(z_lin, x_lin, indexing='ij')       # (Nz, Nx)

    cos_t  = torch.as_tensor(np.cos(np.deg2rad(theta_deg)),
                              dtype=torch.float32, device=dev)  # (N_ang,)
    sin_t  = torch.as_tensor(np.sin(np.deg2rad(theta_deg)),
                              dtype=torch.float32, device=dev)

    # t_all: (N_ang, Nz, Nx),  normalised detector coordinate for every angle
    t_all     = (X[None] * cos_t[:, None, None] +
                 Z[None] * sin_t[:, None, None]) / R
    grid_all  = torch.stack([t_all,
                              torch.zeros_like(t_all)], dim=-1) # (N_ang, Nz, Nx, 2)

    # ── 4.  Backproject — loop over Ny, vectorise over N_angles ─────────────
    recon = torch.zeros(Ny_, Nz_, Nx_, device=dev)
    for iy in range(Ny_):
        # Treat each angle's filtered row as a (1 × Nx) 2-D image,
        # then sample at all (Nz, Nx) output positions simultaneously.
        fi      = filt[iy].unsqueeze(1).unsqueeze(1)            # (N_ang, 1, 1, Nx)
        sampled = F_nn.grid_sample(fi, grid_all, mode='bilinear',
                                   padding_mode='zeros',
                                   align_corners=True)           # (N_ang, 1, Nz, Nx)
        recon[iy] = sampled.squeeze(1).sum(0)                   # (Nz, Nx)

    # ── 5.  Normalise + inscribed-circle mask ────────────────────────────────
    recon *= np.pi / (2 * N_ang)
    recon *= (X**2 + Z**2 <= R**2).unsqueeze(0)                 # circle=True

    return recon.permute(1, 0, 2).cpu().numpy()                 # (Nz, Ny, Nx)


# ── Choose backend ──────────────────────────────────────────────────────────
# Track which FBP implementation was used: the custom _fbp_gpu (t = x·cosθ +
# z·sinθ) and skimage.iradon produce z-flipped volumes relative to each other.
# Only _fbp_gpu needs the z-axis flip below to reach the natural (true-sample)
# orientation that the Pass-2 multislice forward model expects.  Feeding the
# wrong z-orientation makes Pass 2 fight the FBP initial guess and produces
# spurious +/- sector artefacts in the xz/yz difference maps.
# Resolve FBP_METHOD ('auto' -> gpu on CUDA else iradon).
_method = FBP_METHOD
if _method == 'auto':
    _method = 'gpu' if (TORCH_AVAILABLE and DEVICE.type == 'cuda') else 'iradon'
if _method == 'gpu' and not (TORCH_AVAILABLE and DEVICE.type == 'cuda'):
    print("  [FBP] 'gpu' requested but CUDA unavailable; using iradon.",
          flush=True)
    _method = 'iradon'

# Experimental gridding backend: load + self-test, fall back to iradon on any
# failure (missing finufft, failed orientation self-test, import error).
_grid_mod, _grid_calib = None, None
if _method == 'gridding':
    try:
        _grid_mod = _load_module(
            'nufft_gridding', os.path.join(_HERE, 'nufft_gridding.py'))
        # Calibrate at the REAL reconstruction dimensions so the amplitude
        # scale (gridding -> iradon) is exact; orientation is N-independent.
        _ok, _cc, _grid_calib = _grid_mod.self_test(N=Nx, n_ang=N_use)
        if not _ok:
            print(f"  [FBP] gridding self-test did not pass (corr={_cc:.4f}); "
                  f"falling back to iradon.", flush=True)
            _method = 'iradon'
    except Exception as _e:
        print(f"  [FBP] gridding backend unavailable ({_e}); "
              f"falling back to iradon.", flush=True)
        _method = 'iradon'

_fbp_used_gpu = False
if _method == 'gpu':
    print(f"  GPU FBP on {DEVICE} …", flush=True)
    delta_fbp = _fbp_gpu(phase_use, theta_use, DEVICE)
    _fbp_used_gpu = True

elif _method == 'gridding':
    print("  NUFFT gridding FBP (self-test passed) …", flush=True)
    # Output is aligned to the iradon convention -> natural orientation,
    # so it takes NO z-flip below (same as the iradon path).
    delta_fbp = _grid_mod.gridding_reconstruct_volume(
        phase_use, theta_use, calib=_grid_calib)

if _method == 'iradon':
    # ── CPU FBP (joblib threads) ─────────────────────────────────────────────
    # Respect SLURM CPU allocation; avoid saturating all cores on a shared node.
    _slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if _slurm_cpus is not None:
        n_jobs = int(_slurm_cpus)
        print(f"  Parallel FBP (joblib, SLURM n_jobs={n_jobs}) …", flush=True)
    elif _JOBLIB:
        n_jobs = min(Ny, os.cpu_count() or 1)
        print(f"  Parallel FBP (joblib threads, n_jobs={n_jobs}) …", flush=True)
    else:
        n_jobs = 1
        print("  Sequential FBP (install joblib for parallel speedup) …", flush=True)

    def _fbp_slice(iy):
        return iradon(phase_use[:, iy, :].T, theta=theta_use,
                      filter_name="ramp", circle=True)          # (Nx, Nx)

    if _JOBLIB and n_jobs != 1:
        slices    = Parallel(n_jobs=n_jobs, prefer="threads")(
                        joblib_delayed(_fbp_slice)(iy) for iy in range(Ny))
        delta_fbp = np.stack(slices, axis=1)                    # (Nz, Ny, Nx)
    else:
        delta_fbp = np.zeros((Nz, Ny, Nx), dtype=np.float64)
        for iy in range(Ny):
            delta_fbp[:, iy, :] = _fbp_slice(iy)

# Convert projected phase → δ:   φ = −k₀ · pixel · Σδ   →   δ = −φ / (k₀ · pixel)
# If your ptychography code uses the opposite sign convention, remove the minus.
# Keep float32 to halve host RAM (a 988^3 float64 volume is 6 GB).
delta_fbp = (-delta_fbp / (K0 * PIXEL_SIZE)).astype(np.float32)

# ── Orientation correction ────────────────────────────────────────────────
# skimage.iradon already reconstructs in the natural (true-sample) z
# orientation.  The custom _fbp_gpu backprojector (t = x·cosθ + z·sinθ) comes
# out z-flipped relative to iradon, so flip ONLY the GPU-FBP result so that
# both backends feed the SAME orientation into Pass 2.
#
# (Verified: with an asymmetric phantom, corr(iradon, _fbp_gpu[::-1]) ≫
#  corr(iradon, _fbp_gpu), and iradon matches the phantom directly.)
#
# An earlier version flipped unconditionally, which corrupted the iradon
# (MPS/CPU) path and caused +/- sector artefacts in the two-pass − FBP
# difference maps on every non-CUDA run.
if _fbp_used_gpu:
    delta_fbp = np.ascontiguousarray(delta_fbp[::-1, :, :])

# Positivity constraint: δ ≥ 0 for standard materials
# Comment out if your sample has negative contrast regions.
delta_fbp = delta_fbp.clip(0, None)

# ── Circular soft mask — suppress FBP inscribed-circle boundary ringing ──────
# iradon(circle=True) computes only inside the inscribed circle, but leaves
# sharp-edge ringing at the boundary whose amplitude (~8×10⁻⁴) is typically
# 10–100× larger than the true interior δ.  A cosine-tapered mask confined to
# the outermost 5 % of the radius removes that artifact without touching the
# reconstructed interior.
_Zm   = np.arange(Nz) - Nz / 2.0
_Xm   = np.arange(Nx) - Nx / 2.0
_ZZ, _XX = np.meshgrid(_Zm, _Xm, indexing='ij')      # (Nz, Nx)
_r_max   = (min(Nz, Nx) - 1) / 2.0
_r_fade  = max(3, int(0.05 * _r_max))                 # 5 % cosine fade zone
_r       = np.sqrt(_ZZ**2 + _XX**2)
_mask_2d = np.clip((_r_max - _r) / _r_fade, 0.0, 1.0)
# Apply the 2-D mask to every y-slice (broadcast over axis 1)
delta_fbp *= _mask_2d[:, np.newaxis, :]               # (Nz, Ny, Nx)
del _Zm, _Xm, _ZZ, _XX, _r, _mask_2d                 # free temporaries

# β not available from phase-only data; initialise as a small fraction of δ
beta_fbp = delta_fbp * 1e-3

print(f"  FBP done in {time.time()-t0:.1f} s")
print(f"  δ_FBP range after mask: [{delta_fbp.min():.3e}, {delta_fbp.max():.3e}]")
_pos_vals = delta_fbp[delta_fbp > 0]
if _pos_vals.size > 0:
    _p50, _p99 = np.percentile(_pos_vals, [50, 99])
    print(f"  δ_FBP positive pixels: median={_p50:.3e}  99th-pct={_p99:.3e}")
    print(f"  → Suggested LR ≈ {_p99 * 0.1:.1e}  (= 0.1 × 99th-pct of δ_FBP)")
del _pos_vals
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
    """
    Gradient of the smooth isotropic 3-D total-variation functional.

    Identical formula to ``_tv_gradient_3d`` in ``toupy/tomo/twopass.py``
    and to ``tv_grad_torch`` in ``toupy/tomo/multislice_torch.py``.

    Uses forward differences with zero-Neumann boundary conditions and
    their adjoint (backward differences) for the divergence:

        ∇R_TV(v) = −div(∇v / √(|∇v|² + ε))

    The same ``eps`` convention is used in all three implementations so
    that NumPy and PyTorch paths produce numerically identical results.
    """
    # Forward differences with zero-Neumann BC (last diff = 0)
    gz = np.zeros_like(vol); gz[:-1]       = vol[1:]       - vol[:-1]
    gy = np.zeros_like(vol); gy[:, :-1, :] = vol[:, 1:, :] - vol[:, :-1, :]
    gx = np.zeros_like(vol); gx[:, :, :-1] = vol[:, :, 1:] - vol[:, :, :-1]

    # Smoothed local norm + TV value (free: just sum the norm array)
    norm = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2 + eps)
    tv_val = float(np.sum(norm))
    gz /= norm; gy /= norm; gx /= norm

    # Adjoint backward differences: (D_d^* p)_i = p_{i-1} − p_i, p_{-1}=0
    grad = np.empty_like(vol)
    grad[0,  :, :]  = -gz[0,  :, :]
    grad[1:, :, :]  =  gz[:-1, :, :] - gz[1:, :, :]
    grad[:,  0, :]  -= gy[:,  0, :]
    grad[:, 1:, :]  += gy[:, :-1, :] - gy[:, 1:, :]
    grad[:, :,  0]  -= gx[:, :,  0]
    grad[:, :, 1:]  += gx[:, :, :-1] - gx[:, :, 1:]
    return grad, tv_val


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

# ── Per-angle weights ──────────────────────────────────────────────────────
def _compute_angle_weights(phase_arr, mode):
    """
    Per-projection weights for the Pass-2 data term.

    phase_arr : (N_use, Ny, Nx) phase projections.
    mode      : 'uniform' or 'snr'.

    Returns w (N_use,) normalised to mean 1, so that with the existing
    'grad /= N_use' the gradient scale is unchanged for 'uniform'.
    """
    n = phase_arr.shape[0]
    if mode == 'uniform':
        return np.ones(n, dtype=np.float64)
    if mode == 'snr':
        # Robust per-projection noise estimate: MAD of horizontal first
        # differences (high-frequency content) divided by sqrt(2).  MAD is
        # insensitive to sparse real edges, so it tracks the noise floor.
        sig = np.empty(n, dtype=np.float64)
        for i in range(n):
            d = np.diff(phase_arr[i], axis=-1).ravel()
            mad = np.median(np.abs(d - np.median(d)))
            sig[i] = 1.4826 * mad / np.sqrt(2.0)
        sig = np.maximum(sig, np.median(sig) * 1e-3)   # guard zeros
        w = 1.0 / sig ** 2
        w *= n / w.sum()                               # normalise to mean 1
        return w
    raise ValueError(f"Unknown ANGLE_WEIGHT={mode!r}. Use 'uniform' or 'snr'.")


angle_w = _compute_angle_weights(phase_use, ANGLE_WEIGHT)
if ANGLE_WEIGHT != 'uniform':
    print(f"  Per-angle weighting '{ANGLE_WEIGHT}': "
          f"w in [{angle_w.min():.3f}, {angle_w.max():.3f}], "
          f"mean={angle_w.mean():.3f}  "
          f"(down-weighting {int((angle_w < 0.5).sum())} noisy angles)")
else:
    print(f"  Per-angle weighting: uniform (w=1)")

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

    # Free the Pass-1 (FBP) GPU scratch before allocating Pass-2 volumes.
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    adam_d = TorchAdamState(delta_tp_t.shape, lr=LR, device=DEVICE)
    adam_b = (TorchAdamState(beta_tp_t.shape, lr=LR, device=DEVICE)
              if OPTIMIZE_BETA else None)

    for it in range(N_ITER):
        t_iter = time.time()
        total_loss = 0.0
        grad_delta = torch.zeros_like(delta_tp_t)
        grad_beta  = torch.zeros_like(beta_tp_t) if OPTIMIZE_BETA else None

        # LR schedule
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

            loss_i, gd_i, gb_i, _ = torch_engine.loss_and_gradient(
                delta_sl, beta_sl, probe_t,
                u_meas_t[ai],
                delta_means=delta_means,
            )
            w_i = float(angle_w[ai])          # per-angle noise weight (mean 1)
            total_loss += w_i * loss_i

            gd_vol, gb_vol = scatter_gradient_torch(
                gd_i, gb_i, theta, delta_tp_t.shape, n_slices=N_SLICES,
                compute_beta=OPTIMIZE_BETA)   # skip beta rotation when frozen
            # Fused in-place scaled add: same cost as '+= gd_vol' regardless of
            # w_i (no temporary tensor), so 'uniform' has zero overhead.
            grad_delta.add_(gd_vol, alpha=w_i)
            if OPTIMIZE_BETA:
                grad_beta.add_(gb_vol, alpha=w_i)
            del gd_vol, gb_vol            # free rotation transients promptly

        grad_delta /= N_use
        if OPTIMIZE_BETA:
            grad_beta /= N_use

        tv_str = ""
        if LAMBDA_TV > 0:
            tv_gd, tv_val_d = tv_grad_torch(delta_tp_t)
            grad_delta += LAMBDA_TV * tv_gd
            tv_val_b = 0.0
            if OPTIMIZE_BETA:
                tv_gb, tv_val_b = tv_grad_torch(beta_tp_t)
                grad_beta += LAMBDA_TV * tv_gb
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
              f"δ∈[{float(delta_tp_t.min()):.2e},{float(delta_tp_t.max()):.2e}]",
              flush=True)

    # Keep host arrays float32 (half the RAM of float64) — important for large
    # volumes (a 988^3 float64 volume is 6 GB; figures hold several at once).
    delta_tp = delta_tp_t.cpu().numpy().astype(np.float32)
    beta_tp  = beta_tp_t.cpu().numpy().astype(np.float32)

    # Free GPU + large optimiser buffers before figure generation (rebind to
    # None so the references drop and empty_cache can reclaim the VRAM).
    delta_tp_t = beta_tp_t = u_meas_t = probe_t = None
    grad_delta = grad_beta = adam_d = adam_b = None
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

else:
    # ── NumPy fallback ─────────────────────────────────────────────────────
    engine = MultisliceEngine(
        shape           = (Ny, Nx),
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

            loss_i, gd_i, gb_i, _ = engine.loss_and_gradient(
                delta_sl, beta_sl, probe,
                u_measured[ai],
                delta_means=delta_means,
            )
            w_i = float(angle_w[ai])          # per-angle noise weight (mean 1)
            total_loss += w_i * loss_i

            gd_vol, gb_vol = scatter_gradient_to_volume(
                gd_i, gb_i, theta, delta_tp.shape, n_slices=N_SLICES)
            if w_i == 1.0:                     # uniform: plain in-place add
                grad_delta += gd_vol
                if OPTIMIZE_BETA:
                    grad_beta += gb_vol
            else:                             # weighted: fused scaled add
                np.add(grad_delta, w_i * gd_vol, out=grad_delta)
                if OPTIMIZE_BETA:
                    np.add(grad_beta, w_i * gb_vol, out=grad_beta)

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
        print(f"  Iter {it+1:3d}/{N_ITER}  loss={total_loss:.4e}{tv_str}  "
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

# Common colour scale: percentile-based to avoid boundary-ringing artefacts.
# Prefer the two-pass result if it contains positive values; fall back to FBP.
# Never use .max() — it may be dominated by isolated artefact voxels.
vmin = 0.0
_ref_for_scale = delta_tp if delta_tp.max() > 0 else delta_fbp
_pos_ref = _ref_for_scale[_ref_for_scale > 0]
vmax = float(np.percentile(_pos_ref, 99.5)) if _pos_ref.size > 0 else 1e-5
del _ref_for_scale, _pos_ref

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
#
# IMPORTANT: the re-projection here uses the PROJECTION APPROXIMATION
# (simple Beer-Lambert sum of δ along the beam direction), NOT the full
# multislice forward model.  For this dataset with F≈{F:.2f} < 1 (strong
# diffraction regime), the projection approximation is inaccurate and will
# always produce systematic residuals even for a perfect two-pass result.
# Use this figure to check overall orientation and scale — not convergence.
# Convergence quality is shown by Figure 3 (loss curve).
#
# What to look for:
#   - Profile shape roughly matching (same peak positions) → orientation OK
#   - Residual is a smooth ramp across x → low-frequency ptychography phase
#     error; enable PHASE_RAMP_CORR = True to suppress it
#   - Residual is random and small → good agreement
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
