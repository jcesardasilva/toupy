#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-scalable two-pass multislice reconstruction via y-chunking.
==================================================================

Why
---
The monolithic two-pass (twopass_real_data.py) keeps the whole volume + Adam
state on one GPU.  For experimental-scale data (e.g. 988x778x988 ~ 3 GB/volume,
or ~1800^3 ~ 21 GB/volume) that exceeds a 40-48 GB GPU.

This driver reconstructs the volume in horizontal BANDS along y (the rotation
axis) and stitches them.  It is correct because, in the multislice forward
model, the tomographic rotation is purely in the x-z plane (y is never mixed)
and the only y-coupling is the 2-D Fresnel propagation, whose transverse
spread over the full thickness is only

    coupling ~ t * lambda / (2 * pixel) = Nz * lambda / 2   [pixels]

(a few pixels at hard-X-ray wavelengths).  Reconstructing each y-band with an
overlap HALO >> coupling and discarding the halo therefore reproduces the
monolithic result, while memory scales with the band height instead of the
full volume.  Bands are independent -> trivially parallel across GPUs / SLURM
array jobs.

Validation
----------
Run the built-in self-test (CPU, NumPy, tiny volume) to confirm the chunked
reconstruction matches the monolithic one in the band interiors:

    python tutorial/twopass_chunked.py --selftest

Usage (production)
------------------
    python tutorial/twopass_chunked.py        # edit the parameters below
"""

import os, sys, time, importlib.util, argparse
import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOMO = os.path.join(_HERE, "..", "toupy", "tomo")


def _load_module(dotted, fpath):
    spec = importlib.util.spec_from_file_location(dotted, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Parameters (production run)
# ---------------------------------------------------------------------------
DATA_FILE   = os.path.join(_HERE, "PXCTalignedprojections_big.npz")
CROP_X      = 110
CROP_Y      = 110
N_SLICES    = 32
N_ITER      = 100
LR          = 5e-6
LAMBDA_TV   = 1e-5
WARMUP_ITERS = 5
OPTIMIZE_BETA = False          # frozen beta strongly recommended at scale
ANGLE_STEP  = 1

# y-chunking
Y_CHUNK     = 150              # interior rows reconstructed per band
Y_HALO      = None            # None -> auto (~3x the Fresnel coupling), in px

DEVICE_PREF = "auto"          # 'auto'|'cuda'|'mps'|'cpu'


# ---------------------------------------------------------------------------
# Isotropic 3-D TV gradient (NumPy) — same formula as tv_grad_torch
# ---------------------------------------------------------------------------
def _tv_grad_numpy(vol, eps=1e-8):
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


# ---------------------------------------------------------------------------
# Core: FBP + Pass-2 on a single band sub-volume
# ---------------------------------------------------------------------------
def _fbp_band_numpy(phase_band, theta, k0, pixel):
    """Per-y-slice iradon FBP of a band -> delta (Nz, Nyb, Nx), natural orient."""
    from skimage.transform import iradon
    N_ang, Nyb, Nx = phase_band.shape
    Nz = Nx
    delta = np.zeros((Nz, Nyb, Nx), dtype=np.float64)
    for iy in range(Nyb):
        delta[:, iy, :] = iradon(phase_band[:, iy, :].T, theta=theta,
                                 filter_name="ramp", circle=True)
    delta = -delta / (k0 * pixel)
    np.clip(delta, 0.0, None, out=delta)
    return delta


def _pass2_numpy(delta0, phase_band, theta, k0, pixel, wavelength, slice_dz,
                 ms_mod, n_slices, n_iter, lr, lambda_tv, warmup, optimize_beta):
    """NumPy Pass-2 on one band.  Returns refined delta (Nz, Nyb, Nx)."""
    MultisliceEngine = ms_mod.MultisliceEngine
    extract = ms_mod.extract_slices_from_volume
    scatter = ms_mod.scatter_gradient_to_volume
    tv_grad = _tv_grad_numpy

    N_ang, Nyb, Nx = phase_band.shape
    engine = MultisliceEngine(shape=(Nyb, Nx), pixel_size=pixel,
                              wavelength=wavelength, slice_thickness=slice_dz)
    probe = np.ones((Nyb, Nx), dtype=np.complex128)
    u_meas = np.exp(1j * phase_band.astype(np.complex128))

    delta = delta0.copy()
    beta = delta * 1e-3

    # minimal Adam
    class _Adam:
        def __init__(s, shp, lr):
            s.lr = lr; s.b1 = 0.9; s.b2 = 0.999; s.eps = 1e-8; s.t = 0
            s.m = np.zeros(shp); s.v = np.zeros(shp)
        def step(s, p, g):
            s.t += 1
            s.m = s.b1*s.m + (1-s.b1)*g
            s.v = s.b2*s.v + (1-s.b2)*g*g
            mh = s.m/(1-s.b1**s.t); vh = s.v/(1-s.b2**s.t)
            p -= s.lr*mh/(np.sqrt(vh)+s.eps); return p

    adam_d = _Adam(delta.shape, lr)
    adam_b = _Adam(beta.shape, lr) if optimize_beta else None

    for it in range(n_iter):
        grad_d = np.zeros_like(delta)
        grad_b = np.zeros_like(beta) if optimize_beta else None
        lr_t = lr*(it+1)/warmup if it < warmup else \
            lr*0.5*(1+np.cos(np.pi*(it-warmup)/max(n_iter-warmup, 1)))
        adam_d.lr = lr_t
        if optimize_beta:
            adam_b.lr = lr_t
        for ai, th in enumerate(theta):
            d_sl, b_sl, dm = extract(delta, beta, th, n_slices=n_slices)
            _, gd_i, gb_i, _ = engine.loss_and_gradient(d_sl, b_sl, probe,
                                                        u_meas[ai], delta_means=dm)
            gd_v, gb_v = scatter(gd_i, gb_i, th, delta.shape, n_slices=n_slices)
            grad_d += gd_v
            if optimize_beta:
                grad_b += gb_v
        grad_d /= N_ang
        if optimize_beta:
            grad_b /= N_ang
        if lambda_tv > 0:
            grad_d += lambda_tv * tv_grad(delta)[0]
            if optimize_beta:
                grad_b += lambda_tv * tv_grad(beta)[0]
        adam_d.step(delta, grad_d); np.clip(delta, 0.0, None, out=delta)
        if optimize_beta:
            adam_b.step(beta, grad_b); np.clip(beta, 0.0, None, out=beta)
    return delta


def _pass2_torch(delta0, phase_band, theta, k0, pixel, wavelength, slice_dz,
                 mst, device, n_slices, n_iter, lr, lambda_tv, warmup,
                 optimize_beta):
    """Torch Pass-2 on one band (GPU/MPS)."""
    import torch
    Engine = mst.TorchMultisliceEngine
    extract = mst.extract_slices_torch
    scatter = mst.scatter_gradient_torch
    tv_grad = mst.tv_grad_torch
    Adam = mst.TorchAdamState

    N_ang, Nyb, Nx = phase_band.shape
    engine = Engine(shape=(Nyb, Nx), pixel_size=pixel, wavelength=wavelength,
                    slice_thickness=slice_dz, device=device)
    probe = torch.ones((Nyb, Nx), dtype=torch.complex64, device=device)
    u_meas = torch.from_numpy(
        np.exp(1j*phase_band.astype(np.complex64))).to(device)
    delta = torch.from_numpy(delta0.astype(np.float32)).to(device)
    beta = (delta * 1e-3).clone()

    adam_d = Adam(delta.shape, lr=lr, device=device)
    adam_b = Adam(beta.shape, lr=lr, device=device) if optimize_beta else None

    for it in range(n_iter):
        grad_d = torch.zeros_like(delta)
        grad_b = torch.zeros_like(beta) if optimize_beta else None
        lr_t = lr*(it+1)/warmup if it < warmup else \
            lr*0.5*(1+np.cos(np.pi*(it-warmup)/max(n_iter-warmup, 1)))
        adam_d.lr = lr_t
        if optimize_beta:
            adam_b.lr = lr_t
        for ai, th in enumerate(theta):
            d_sl, b_sl, dm = extract(delta, beta, th, n_slices=n_slices)
            _, gd_i, gb_i, _ = engine.loss_and_gradient(d_sl, b_sl, probe,
                                                        u_meas[ai], delta_means=dm)
            gd_v, gb_v = scatter(gd_i, gb_i, th, delta.shape,
                                 n_slices=n_slices, compute_beta=optimize_beta)
            grad_d.add_(gd_v)
            if optimize_beta:
                grad_b.add_(gb_v)
            del gd_v, gb_v
        grad_d /= N_ang
        if optimize_beta:
            grad_b /= N_ang
        if lambda_tv > 0:
            grad_d += lambda_tv * tv_grad(delta)[0]
            if optimize_beta:
                grad_b += lambda_tv * tv_grad(beta)[0]
        adam_d.step(delta, grad_d); delta.clamp_(min=0.0)
        if optimize_beta:
            adam_b.step(beta, grad_b); beta.clamp_(min=0.0)
    out = delta.cpu().numpy().astype(np.float64)
    del delta, beta, grad_d, u_meas
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Band scheduler + chunked driver
# ---------------------------------------------------------------------------
def _bands(Ny, chunk, halo):
    """Yield (y0, y1, lo, hi): interior [y0,y1) reconstructed from rows [lo,hi)."""
    for y0 in range(0, Ny, chunk):
        y1 = min(y0 + chunk, Ny)
        lo = max(0, y0 - halo)
        hi = min(Ny, y1 + halo)
        yield y0, y1, lo, hi


def reconstruct_chunked(phase, theta, k0, pixel, wavelength, slice_dz,
                        backend, n_slices, n_iter, lr, lambda_tv, warmup,
                        optimize_beta, chunk, halo, verbose=True):
    """
    Full chunked two-pass.  `backend` is ('numpy', ms_mod) or ('torch', mst, dev).
    Returns delta (Nz, Ny, Nx).
    """
    N_ang, Ny, Nx = phase.shape
    Nz = Nx
    delta_full = np.zeros((Nz, Ny, Nx), dtype=np.float64)

    for bi, (y0, y1, lo, hi) in enumerate(_bands(Ny, chunk, halo)):
        if verbose:
            print(f"  band {bi+1}: interior y[{y0}:{y1}]  with halo y[{lo}:{hi}] "
                  f"({hi-lo} rows)", flush=True)
        phase_band = phase[:, lo:hi, :]
        delta0 = _fbp_band_numpy(phase_band, theta, k0, pixel)
        if backend[0] == "numpy":
            delta_band = _pass2_numpy(delta0, phase_band, theta, k0, pixel,
                                      wavelength, slice_dz, backend[1],
                                      n_slices, n_iter, lr, lambda_tv, warmup,
                                      optimize_beta)
        else:
            delta_band = _pass2_torch(delta0, phase_band, theta, k0, pixel,
                                      wavelength, slice_dz, backend[1],
                                      backend[2], n_slices, n_iter, lr,
                                      lambda_tv, warmup, optimize_beta)
        # write only the interior rows (discard halo)
        delta_full[:, y0:y1, :] = delta_band[:, y0 - lo:y1 - lo, :]
    return delta_full


def auto_halo(Nz, wavelength, pixel):
    """~3x the transverse Fresnel coupling, in pixels (min 16)."""
    coupling_px = Nz * wavelength / (2.0 * pixel)        # = t*lambda/(2 pixel^2)
    return int(max(16, np.ceil(3.0 * coupling_px)))


# ---------------------------------------------------------------------------
# Self-test: chunked vs monolithic (NumPy, tiny volume)
# ---------------------------------------------------------------------------
def selftest():
    print("=" * 64)
    print("twopass_chunked self-test: chunked vs monolithic (NumPy)")
    print("=" * 64)
    ms_mod = _load_module("toupy.tomo.multislice",
                          os.path.join(_TOMO, "multislice.py"))

    rng = np.random.default_rng(0)
    N_ang, Ny, Nx = 16, 40, 48
    wavelength, pixel = 0.2e-9, 28.64e-9
    k0 = 2*np.pi/wavelength
    Nz = Nx
    slice_dz = Nz * pixel / 8

    # synthetic phase projections from a smooth-ish object
    base = np.zeros((Ny, Nx)); base[8:32, 10:38] = 1.0
    phase = np.zeros((N_ang, Ny, Nx))
    theta = np.linspace(0, 180, N_ang, endpoint=False)
    for a in range(N_ang):
        phase[a] = 0.5*base + 0.1*rng.standard_normal((Ny, Nx))

    kw = dict(k0=k0, pixel=pixel, wavelength=wavelength, slice_dz=slice_dz,
              backend=("numpy", ms_mod), n_slices=8, n_iter=8, lr=5e-6,
              lambda_tv=0.0, warmup=2, optimize_beta=False)

    mono = reconstruct_chunked(phase, theta, chunk=Ny, halo=0, verbose=False, **kw)
    chunked = reconstruct_chunked(phase, theta, chunk=12, halo=8, verbose=False, **kw)

    # compare interiors away from the very top/bottom (boundary of the whole vol)
    sl = slice(4, Ny-4)
    a = mono[:, sl, :].ravel(); b = chunked[:, sl, :].ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    rel = float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-30))
    print(f"  chunk=12 halo=8 vs monolithic: corr={corr:.5f}  rel.diff={rel:.3e}")
    ok = corr > 0.999 and rel < 1e-2
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Production main
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("Chunked two-pass reconstruction")
    print("=" * 64)
    data = np.load(DATA_FILE, allow_pickle=True)
    phase = data["projections"].astype(np.float64)
    wavelength = float(data["wavelen"]); pixel = float(data["psize"])
    theta = data["theta"].astype(np.float64)
    k0 = 2*np.pi/wavelength

    if ANGLE_STEP > 1:
        phase = phase[::ANGLE_STEP]; theta = theta[::ANGLE_STEP]

    sy = slice(CROP_Y or None, -CROP_Y or None)
    sx = slice(CROP_X or None, -CROP_X or None)
    phase = np.ascontiguousarray(phase[:, sy, sx])
    N_ang, Ny, Nx = phase.shape
    Nz = Nx
    slice_dz = Nz * pixel / N_SLICES

    halo = Y_HALO if Y_HALO is not None else auto_halo(Nz, wavelength, pixel)
    coupling = Nz*wavelength/(2*pixel)
    print(f"  data {phase.shape}  pixel={pixel*1e9:.2f} nm  Nz={Nz}")
    print(f"  Fresnel y-coupling ~{coupling:.1f} px  ->  HALO={halo} px")
    print(f"  Y_CHUNK={Y_CHUNK}  ({int(np.ceil(Ny/Y_CHUNK))} bands)  "
          f"OPTIMIZE_BETA={OPTIMIZE_BETA}  N_SLICES={N_SLICES}")

    # backend
    try:
        import torch  # noqa
        mst = _load_module("toupy.tomo.multislice_torch",
                           os.path.join(_TOMO, "multislice_torch.py"))
        dev = mst.select_device(None if DEVICE_PREF == "auto" else DEVICE_PREF)
        print(f"  backend: {mst.device_info(dev)}")
        backend = ("torch", mst, dev)
    except Exception as e:
        ms_mod = _load_module("toupy.tomo.multislice",
                              os.path.join(_TOMO, "multislice.py"))
        print(f"  backend: NumPy CPU ({e})")
        backend = ("numpy", ms_mod)

    t0 = time.time()
    delta = reconstruct_chunked(phase, theta, k0, pixel, wavelength, slice_dz,
                                backend, N_SLICES, N_ITER, LR, LAMBDA_TV,
                                WARMUP_ITERS, OPTIMIZE_BETA, Y_CHUNK, halo)
    print(f"\n  Chunked reconstruction done in {(time.time()-t0)/60:.1f} min")

    out_dir = os.path.join(_HERE, "twopass_chunked_figures")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "twopass_chunked.npz"),
                        delta_tp=delta.astype(np.float32),
                        wavelength=wavelength, psize=pixel, theta=theta)
    print(f"  Saved: {out_dir}/twopass_chunked.npz   shape={delta.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run the chunked-vs-monolithic equivalence check")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    main()
