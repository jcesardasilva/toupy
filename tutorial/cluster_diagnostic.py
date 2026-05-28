#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA environment diagnostic and smoke test for the two-pass method.
====================================================================

Run this FIRST on the HPC cluster before submitting the full job.
It checks every dependency and code path used by twopass_real_data.py,
estimates memory requirements, benchmarks timing, and recommends
parameters suited to the available GPU.

Usage
-----
  # Basic: tests environment only (no data file needed)
  python cluster_diagnostic.py

  # Full: also estimates timing for your actual data dimensions
  python cluster_diagnostic.py --data /path/to/PXCTalignedprojections.npz

Output
------
  Prints a structured report.  If anything fails, copy the FULL output
  and share it to get the problem diagnosed.

Expected runtime: < 2 minutes.
"""

import sys, os, time, argparse, math, platform
import numpy as np

SEP  = "=" * 65
SEP2 = "-" * 65

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}"); sys.exit(1)

def fmt_bytes(b):
    for unit in ("B","KB","MB","GB","TB"):
        if b < 1024: return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"

# ---------------------------------------------------------------------------
# 1. System info
# ---------------------------------------------------------------------------
print(SEP)
print("CLUSTER DIAGNOSTIC — two-pass multislice reconstruction")
print(SEP)
print(f"\n[1] System")
print(f"  Python      : {sys.version.split()[0]}  ({sys.executable})")
print(f"  Platform    : {platform.platform()}")
print(f"  CPU cores   : {os.cpu_count()}")

# ---------------------------------------------------------------------------
# 2. Python packages
# ---------------------------------------------------------------------------
print(f"\n[2] Python packages")

required = {
    "numpy":   "np",
    "scipy":   "sp",
    "skimage": "skimage",
    "matplotlib": "matplotlib",
}
for pkg, alias in required.items():
    try:
        mod = __import__(pkg)
        ok(f"{pkg} {mod.__version__}")
    except ImportError:
        fail(f"{pkg} NOT FOUND — install it before running the reconstruction")

try:
    import joblib
    ok(f"joblib {joblib.__version__}  (parallel FBP available)")
except ImportError:
    warn("joblib not found — FBP will run sequentially (pip install joblib)")

# ---------------------------------------------------------------------------
# 3. PyTorch and CUDA
# ---------------------------------------------------------------------------
print(f"\n[3] PyTorch / CUDA")

try:
    import torch
    ok(f"torch {torch.__version__}")
except ImportError:
    fail("torch NOT FOUND — install PyTorch before running on GPU")

cuda_ok = torch.cuda.is_available()
if cuda_ok:
    n_gpu = torch.cuda.device_count()
    ok(f"CUDA available  ({n_gpu} GPU{'s' if n_gpu>1 else ''})")
    for i in range(n_gpu):
        p = torch.cuda.get_device_properties(i)
        total_mb = p.total_memory / 1024**2
        ok(f"  GPU {i}: {p.name}  —  {total_mb:.0f} MB VRAM  "
           f"(compute {p.major}.{p.minor})")
    DEVICE = torch.device("cuda:0")
else:
    warn("CUDA not available on this node — will fall back to CPU")
    warn("  If you expected a GPU, check: module load cuda / srun --gres=gpu:1")
    DEVICE = torch.device("cpu")

mps_ok = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
if mps_ok:
    ok("MPS (Apple Silicon) available")

print(f"  → Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# 4. Key PyTorch operations used by the engine
# ---------------------------------------------------------------------------
print(f"\n[4] Key operation tests on {DEVICE}")

N = 64    # test size

# FFT (used in Fresnel propagator)
try:
    x = torch.randn(N, N, dtype=torch.complex64, device=DEVICE)
    y = torch.fft.ifft2(torch.fft.fft2(x))
    err = (x - y).abs().max().item()
    if err < 1e-4:
        ok(f"fft2 / ifft2  round-trip error = {err:.1e}")
    else:
        warn(f"fft2 / ifft2  round-trip error = {err:.1e}  (unexpectedly large)")
except Exception as e:
    fail(f"fft2 failed: {e}")

# affine_grid + grid_sample (used in rotate_volume_torch)
try:
    import torch.nn.functional as F
    vol = torch.randn(N, N, N, dtype=torch.float32, device=DEVICE)
    vol_b = vol.permute(1, 0, 2).unsqueeze(1)        # (N,1,N,N)
    theta = math.radians(30)
    c, s  = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s, 0.], [s, c, 0.]],
                     dtype=torch.float32, device=DEVICE)
    R_b   = R.unsqueeze(0).expand(N, -1, -1)
    grid  = F.affine_grid(R_b, vol_b.shape, align_corners=True)
    out   = F.grid_sample(vol_b, grid, mode="bilinear",
                          padding_mode="zeros", align_corners=True)
    ok(f"affine_grid + grid_sample  output shape = {tuple(out.shape)}")
except Exception as e:
    fail(f"affine_grid / grid_sample failed: {e}")

# Transmission function  (complex exp)
try:
    delta = torch.rand(N, N, dtype=torch.float32, device=DEVICE) * 1e-4
    beta  = delta * 1e-3
    dz    = 1e-8
    k0    = 3.14e10
    T = torch.exp(-k0 * dz * (beta + 1j * delta))
    ok(f"complex transmission  |T| range = [{T.abs().min():.4f}, {T.abs().max():.4f}]")
except Exception as e:
    fail(f"complex transmission failed: {e}")

# In-place clamp (positivity constraint)
try:
    v = torch.randn(N, N, N, dtype=torch.float32, device=DEVICE)
    v.clamp_(min=0.0)
    ok(f"clamp_ (positivity constraint)  min={v.min():.1f}")
except Exception as e:
    fail(f"clamp_ failed: {e}")

if cuda_ok:
    torch.cuda.synchronize(DEVICE)
ok("All operation tests passed")

# ---------------------------------------------------------------------------
# 5. Load toupy modules
# ---------------------------------------------------------------------------
print(f"\n[5] Toupy module loading")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOMO = os.path.join(_HERE, "..", "toupy", "tomo")

def _load_module(dotted_name, fpath):
    import importlib.util
    spec = importlib.util.spec_from_file_location(dotted_name, fpath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

try:
    ms_mod = _load_module("toupy.tomo.multislice",
                          os.path.join(_TOMO, "multislice.py"))
    ok("toupy.tomo.multislice loaded")
except Exception as e:
    fail(f"multislice.py failed to load: {e}")

try:
    ms_torch_mod = _load_module("toupy.tomo.multislice_torch",
                                os.path.join(_TOMO, "multislice_torch.py"))
    ok("toupy.tomo.multislice_torch loaded")
except Exception as e:
    fail(f"multislice_torch.py failed to load: {e}")

# ---------------------------------------------------------------------------
# 6. End-to-end smoke test (tiny N=32 synthetic data)
# ---------------------------------------------------------------------------
print(f"\n[6] End-to-end smoke test (synthetic, N=32, 4 angles, 3 iters)")

try:
    TorchMultisliceEngine  = ms_torch_mod.TorchMultisliceEngine
    extract_slices_torch   = ms_torch_mod.extract_slices_torch
    scatter_gradient_torch = ms_torch_mod.scatter_gradient_torch
    tv_grad_torch          = ms_torch_mod.tv_grad_torch
    TorchAdamState         = ms_torch_mod.TorchAdamState
    rotate_volume_torch    = ms_torch_mod.rotate_volume_torch

    SN   = 32
    WAVE = 0.2e-9
    PIX  = 28e-9
    DZ   = SN * PIX / 8
    K0   = 2 * math.pi / WAVE
    ANGS = np.array([0., 45., 90., 135.])
    rng  = np.random.default_rng(0)

    # Tiny phantom
    delta_vol = np.zeros((SN, SN, SN), dtype=np.float32)
    delta_vol[SN//4:3*SN//4, SN//4:3*SN//4, SN//4:3*SN//4] = 4e-5
    probe_np  = np.ones((SN, SN), dtype=np.complex64)

    engine = TorchMultisliceEngine(
        shape=(SN, SN), pixel_size=PIX, wavelength=WAVE,
        slice_thickness=DZ, device=DEVICE)

    # Simulate exit waves
    u_meas = []
    for ang in ANGS:
        from toupy.tomo.multislice import extract_slices_from_volume
        ms_mod2 = ms_mod
        d_sl, b_sl, dm = ms_mod2.extract_slices_from_volume(
            delta_vol.astype(np.float64),
            (delta_vol * 1e-3).astype(np.float64), ang, n_slices=8)
        u = engine.forward(d_sl.astype(np.float32),
                           b_sl.astype(np.float32), probe_np, delta_means=dm.astype(np.float32))
        u_meas.append(u.cpu().numpy())
    u_meas = np.array(u_meas)   # (4, SN, SN) complex64

    ok(f"Forward simulation: u_meas shape={u_meas.shape}, "
       f"|u| ∈ [{np.abs(u_meas).min():.3f}, {np.abs(u_meas).max():.3f}]")

    # Pass 2 (3 iters)
    delta_t = torch.zeros((SN, SN, SN), dtype=torch.float32, device=DEVICE)
    beta_t  = torch.zeros((SN, SN, SN), dtype=torch.float32, device=DEVICE)
    probe_t = torch.from_numpy(probe_np).to(DEVICE)
    u_t     = torch.from_numpy(u_meas).to(DEVICE)
    adam_d  = TorchAdamState(delta_t.shape, lr=1e-3, device=DEVICE)
    adam_b  = TorchAdamState(beta_t.shape,  lr=1e-3, device=DEVICE)

    t0 = time.time()
    for it in range(3):
        gd = torch.zeros_like(delta_t)
        gb = torch.zeros_like(beta_t)
        tl = 0.0
        for ai, ang in enumerate(ANGS):
            d_sl, b_sl, dm = extract_slices_torch(delta_t, beta_t, ang, n_slices=8)
            loss, gdi, gbi, _ = engine.loss_and_gradient(
                d_sl, b_sl, probe_t, u_t[ai], delta_means=dm)
            tl += loss
            gd_v, gb_v = scatter_gradient_torch(gdi, gbi, ang, delta_t.shape, n_slices=8)
            gd += gd_v; gb += gb_v
        gd /= len(ANGS); gb /= len(ANGS)
        adam_d.step(delta_t, gd); adam_b.step(beta_t, gb)
        delta_t.clamp_(min=0.); beta_t.clamp_(min=0.)
    if cuda_ok:
        torch.cuda.synchronize(DEVICE)
    t_smoke = time.time() - t0
    ok(f"Pass 2 smoke (3 iters, 4 angles, N=32): {t_smoke:.2f} s  "
       f"loss={tl:.3e}  δ_max={float(delta_t.max()):.2e}")
except Exception as e:
    import traceback
    print(f"  [FAIL] Smoke test failed:\n")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 7. Memory and timing estimates for real data
# ---------------------------------------------------------------------------
print(f"\n[7] Resource estimates for real data")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--data", default=None)
args, _ = parser.parse_known_args()

if args.data and os.path.exists(args.data):
    data   = np.load(args.data, allow_pickle=True)
    WAVE_r = float(data["wavelen"])
    PIX_r  = float(data["psize"])
    THETA_r = data["theta"]
    N_ANG  = len(THETA_r)
    Ny_r, Nx_r = data["projections"].shape[1:]
    Nz_r   = Nx_r
    print(f"  Data: {os.path.basename(args.data)}")
    print(f"  Shape: ({Nz_r}, {Ny_r}, {Nx_r})  — {N_ANG} angles")
else:
    # Use the known dimensions of PXCTalignedprojections.npz
    WAVE_r = 0.2e-9; PIX_r = 28.6e-9
    N_ANG  = 450; Ny_r = 394; Nx_r = 493; Nz_r = Nx_r
    print(f"  (Using known PXCTalignedprojections.npz dimensions — "
          f"pass --data /path/to/file for exact values)")
    print(f"  Shape: ({Nz_r}, {Ny_r}, {Nx_r})  — {N_ANG} angles")

F_r = PIX_r**2 / (WAVE_r * Nz_r * PIX_r)
print(f"  Fresnel number: F = {F_r:.3f}  "
      f"({'diffraction important' if F_r < 1 else 'projection approx. OK'})")

# Memory breakdown (float32 = 4 bytes, complex64 = 8 bytes)
bytes_vol   = Nz_r * Ny_r * Nx_r * 4          # one f32 volume
bytes_cmplx = Ny_r * Nx_r * 8                  # one c64 frame

mem = {
    "delta_tp (f32)":          bytes_vol,
    "beta_tp  (f32)":          bytes_vol,
    "Adam m,v for delta (f32)":2 * bytes_vol,
    "Adam m,v for beta  (f32)":2 * bytes_vol,
    "grad_delta (f32)":        bytes_vol,
    "grad_beta  (f32)":        bytes_vol,
    "u_meas all angles (c64)": N_ANG * bytes_cmplx,
    "probe (c64)":             bytes_cmplx,
}
total_bytes = sum(mem.values())

print(f"\n  Memory breakdown (persistent tensors):")
for k, v in mem.items():
    print(f"    {k:30s}: {fmt_bytes(v)}")
print(f"  {'TOTAL (persistent)':30s}: {fmt_bytes(total_bytes)}")

# Per-iteration working memory (slices + wavefields)
for N_SL in [8, 16, 32]:
    slices_bytes  = N_SL * Ny_r * Nx_r * 4      # f32 slices (delta & beta)
    wf_bytes      = (N_SL + 1) * Ny_r * Nx_r * 8  # c64 wavefields T,W,U
    work_bytes    = 2 * slices_bytes + 3 * wf_bytes
    peak_bytes    = total_bytes + work_bytes
    print(f"  Peak memory N_SLICES={N_SL:2d}: {fmt_bytes(peak_bytes)}")

if cuda_ok:
    gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    print(f"\n  GPU 0 total VRAM: {fmt_bytes(gpu_bytes)}")

    print(f"\n  Recommendation:")
    safe = gpu_bytes * 0.85   # leave 15% headroom
    chosen_slices = None
    for N_SL in [8, 16, 32, 64]:
        slices_bytes = N_SL * Ny_r * Nx_r * 4
        wf_bytes     = (N_SL + 1) * Ny_r * Nx_r * 8
        peak         = total_bytes + 2*slices_bytes + 3*wf_bytes
        if peak < safe:
            chosen_slices = N_SL
    if chosen_slices:
        ok(f"N_SLICES = {chosen_slices} fits comfortably in GPU memory")
    else:
        warn("Even N_SLICES=8 may exceed GPU memory!")
        warn("Options: (a) use gradient checkpointing, "
             "(b) load u_meas one angle at a time, "
             "(c) crop/bin the volume")

    # Check if u_meas fits on GPU
    if N_ANG * bytes_cmplx < safe * 0.3:
        ok(f"u_meas ({N_ANG} × (Ny,Nx) complex64) fits on GPU: "
           f"{fmt_bytes(N_ANG * bytes_cmplx)}")
    else:
        warn(f"u_meas is large ({fmt_bytes(N_ANG * bytes_cmplx)}). "
             f"Consider loading one angle at a time (see --stream-angles flag in job script).")

# ---------------------------------------------------------------------------
# 8. Timing benchmark at real data dimensions
# ---------------------------------------------------------------------------
print(f"\n[8] Timing benchmark (one iteration at real dimensions, 5 angles)")

try:
    N_BENCH = min(5, N_ANG)
    N_SL_B  = 16
    DZ_B    = Nz_r * PIX_r / N_SL_B
    K0_r    = 2 * math.pi / WAVE_r

    eng_b = TorchMultisliceEngine(
        shape=(Ny_r, Nx_r), pixel_size=PIX_r,
        wavelength=WAVE_r, slice_thickness=DZ_B, device=DEVICE)

    delta_b = torch.zeros((Nz_r, Ny_r, Nx_r), dtype=torch.float32, device=DEVICE)
    beta_b  = torch.zeros_like(delta_b)
    probe_b = torch.ones((Ny_r, Nx_r),  dtype=torch.complex64, device=DEVICE)
    # Synthetic measured waves
    u_b     = torch.exp(1j * torch.rand((N_BENCH, Ny_r, Nx_r),
                        dtype=torch.float32, device=DEVICE) - 0.5)

    angles_b = np.linspace(0, 180, N_BENCH, endpoint=False)

    # Warm-up
    for ang in angles_b[:2]:
        d_sl, b_sl, dm = extract_slices_torch(delta_b, beta_b, ang, n_slices=N_SL_B)
        _, _, _, _ = eng_b.loss_and_gradient(d_sl, b_sl, probe_b, u_b[0], delta_means=dm)
    if cuda_ok:
        torch.cuda.synchronize(DEVICE)

    t0 = time.time()
    for ai, ang in enumerate(angles_b):
        d_sl, b_sl, dm = extract_slices_torch(delta_b, beta_b, ang, n_slices=N_SL_B)
        loss, gd, gb, _ = eng_b.loss_and_gradient(
            d_sl, b_sl, probe_b, u_b[ai % N_BENCH], delta_means=dm)
        gd_v, gb_v = scatter_gradient_torch(gd, gb, ang, delta_b.shape, n_slices=N_SL_B)
    if cuda_ok:
        torch.cuda.synchronize(DEVICE)
    t_bench = time.time() - t0

    t_per_angle = t_bench / N_BENCH
    t_per_iter  = t_per_angle * N_ANG
    t_total_20  = t_per_iter * 20

    ok(f"N_SLICES={N_SL_B}, real (Nz,Ny,Nx)=({Nz_r},{Ny_r},{Nx_r})")
    ok(f"  {N_BENCH} angles → {t_bench:.2f} s  ({t_per_angle:.2f} s/angle)")
    ok(f"  Extrapolated: {t_per_iter:.0f} s/iter  ({t_per_iter/60:.1f} min/iter)")
    ok(f"  20 iterations ≈ {t_total_20/60:.0f} min  ({t_total_20/3600:.1f} h)")

    print(f"\n  Recommended SLURM --time: {max(1, int(t_total_20/3600) + 1)}h "
          f"(20 iters + overhead)")

except Exception as e:
    import traceback
    warn(f"Timing benchmark failed (non-fatal):")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 9. Summary and recommendations
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("DIAGNOSTIC SUMMARY")
print(SEP)
print(f"  Device      : {DEVICE}")
if cuda_ok:
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU         : {p.name}  ({p.total_memory/1024**2:.0f} MB)")
print(f"  torch       : {torch.__version__}")
print(f"  All tests   : PASSED")
print()
print("  Suggested parameters for twopass_real_data.py:")
if cuda_ok:
    gpu_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    n_sl   = chosen_slices if chosen_slices else 8
    print(f"    N_SLICES   = {n_sl}")
    print(f"    N_ITER     = 20  (adjust based on convergence)")
    print(f"    ANGLE_STEP = 1   (all {N_ANG} angles)")
    if gpu_mb < 20000:
        print(f"    NOTE: GPU has {gpu_mb:.0f} MB; monitor memory with nvidia-smi")
else:
    print(f"    Running on CPU — expect ~{t_per_iter/60:.0f} min/iter")
    print(f"    Consider requesting a GPU node")
print(SEP)
