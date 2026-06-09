#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NUFFT (gridding) direct-Fourier back-projector for parallel-beam tomography.
============================================================================

Why
---
Standard FBP (skimage.iradon, or the grid_sample-based _fbp_gpu) back-projects
by *real-space interpolation*, whose transfer function rolls off near Nyquist
and quietly low-passes the finest detail.  A gridding / direct-Fourier
reconstruction instead inverts in Fourier space using a Kaiser-Bessel NUFFT
kernel with density compensation, which has smaller high-frequency error.

This module provides that as an OPTIONAL Pass-1 backend for
twopass_real_data.py (FBP_METHOD='gridding'), kept fully separate from the
validated iradon / _fbp_gpu paths.

IMPORTANT — self-validation
---------------------------
A wrong reconstructor (e.g. an orientation flip) is worse than a slightly
blurry one.  This module therefore SELF-TESTS against skimage.iradon on a
phantom and exposes the measured correlation; the pipeline refuses to use it
unless the test passes (corr >= SELFTEST_MIN_CORR).  Run standalone to check:

    python tutorial/nufft_gridding.py

Dependency
----------
Requires `finufft`  (pip install finufft).  A correct NUFFT needs the
Kaiser-Bessel gridding kernel + density compensation that finufft implements;
a hand-rolled scipy.griddata version breaks Hermitian symmetry and is NOT a
valid substitute.  If finufft is absent the functions raise a clear error and
the pipeline falls back to iradon.
"""

import numpy as np

SELFTEST_MIN_CORR = 0.95

# CPU NUFFT (finufft) and optional GPU NUFFT (cufinufft + cupy).
try:
    import finufft
    _HAVE_FINUFFT = True
except ImportError:
    _HAVE_FINUFFT = False

_CUFINUFFT_ERR = None
try:
    import cupy as _cp
    import cufinufft as _cufinufft
    _HAVE_CUFINUFFT = True
except Exception as _e:    # ImportError, missing CUDA runtime lib, driver init
    _HAVE_CUFINUFFT = False
    _CUFINUFFT_ERR = f"{type(_e).__name__}: {_e}"


def _resolve_grid_device(device):
    """
    Resolve the gridding compute device.

    device : 'auto' | 'gpu' | 'cpu' (or None == 'auto')
      'auto' -> GPU (cufinufft) if available, else CPU (finufft).
      'gpu'  -> require cufinufft+cupy (raises if absent).
      'cpu'  -> require finufft.
    """
    d = (device or 'auto').lower()
    if d == 'gpu':
        if not _HAVE_CUFINUFFT:
            raise RuntimeError("device='gpu' but cufinufft/cupy not available "
                               "(pip install cufinufft cupy).")
        return 'gpu'
    if d == 'cpu':
        if not _HAVE_FINUFFT:
            raise RuntimeError("device='cpu' but finufft not available "
                               "(pip install finufft).")
        return 'cpu'
    # auto
    if _HAVE_CUFINUFFT:
        return 'gpu'
    if _HAVE_FINUFFT:
        return 'cpu'
    raise RuntimeError("neither cufinufft (GPU) nor finufft (CPU) is available.")


# ---------------------------------------------------------------------------
# Core: single-slice gridding reconstruction
# ---------------------------------------------------------------------------
def gridding_reconstruct_slice(sino, theta_deg, device='auto'):
    """
    Direct-Fourier (gridding) reconstruction of one 2-D slice.

    Parameters
    ----------
    sino : (n_det, n_ang) real
        Sinogram; column i is the projection at theta_deg[i].
    theta_deg : (n_ang,) projection angles in degrees.
    device : 'auto' | 'gpu' | 'cpu'
        GPU path uses cufinufft+cupy; CPU path uses finufft.  The math is
        identical; both pin modeord=0 so the output is centre-ordered.

    Returns
    -------
    (N, N) real reconstruction (NumPy array), N = n_det.
    """
    dev = _resolve_grid_device(device)
    xp = _cp if dev == 'gpu' else np

    n_det, n_ang = sino.shape
    N = n_det

    sino_x = xp.asarray(sino)
    # 1-D FFT along the detector -> samples of the object FT on radial lines
    # (Fourier Slice Theorem).  Centre the detector origin first.
    rho = xp.fft.fftshift(xp.fft.fftfreq(N))                       # [-0.5, 0.5)
    S = xp.fft.fftshift(
            xp.fft.fft(xp.fft.ifftshift(sino_x, axes=0), axis=0), axes=0)

    th = xp.asarray(np.deg2rad(np.asarray(theta_deg, dtype=np.float64)))
    KX = xp.outer(rho, xp.cos(th))                                 # (n_rho, n_ang)
    KY = xp.outer(rho, xp.sin(th))
    ramp = xp.abs(rho)[:, None]                                    # density comp.
    c = (S * ramp).astype(xp.complex128).ravel()

    # type-1 NUFFT: f[k1,k2] = sum_j c_j exp(i (x_j k1 + y_j k2)),  x in [-pi,pi).
    # modeord=0 => output modes are centre-ordered (k = -N/2 .. N/2-1), so f IS
    # the centred image directly (do NOT fftshift; an earlier fftshift here
    # scrambled the result and the self-test caught it).  We pin modeord=0
    # explicitly because cufinufft has historically defaulted differently.
    x = (2.0 * np.pi * KX).ravel()
    y = (2.0 * np.pi * KY).ravel()

    if dev == 'gpu':
        f = _cufinufft.nufft2d1(x, y, c, (N, N), isign=+1, eps=1e-9, modeord=0)
        img = _cp.asnumpy(f.real) / N
    else:
        f = finufft.nufft2d1(x, y, c, (N, N), isign=+1, eps=1e-9, modeord=0)
        img = f.real / N                                           # scale fixed by calib

    return img


# ---------------------------------------------------------------------------
# Volume driver: loop over y-slices  (matches twopass _fbp orientation needs)
# ---------------------------------------------------------------------------
def gridding_reconstruct_volume(phase_use, theta_use, calib=None):
    """
    Reconstruct a (Nz, Ny, Nx) volume from phase projections.

    Parameters
    ----------
    phase_use : (N_ang, Ny, Nx) phase projections.
    theta_use : (N_ang,) angles [deg].
    calib : dict or None
        Calibration from self_test():
          {'op': (k, flip), 'scale': s, 'device': 'gpu'|'cpu'}
        `op` orients each slice to the iradon convention; `scale` matches the
        amplitude to iradon (correlation alone is scale-blind, so without this
        the absolute delta values would be wrong); `device` is the backend the
        self-test validated.  None => no correction, auto device.

    Returns
    -------
    (Nz=Nx, Ny, Nx) reconstruction, scaled to match iradon.
    """
    op = calib['op'] if calib else None
    scale = calib['scale'] if calib else 1.0
    device = calib.get('device', 'auto') if calib else 'auto'

    N_ang, Ny, Nx = phase_use.shape
    Nz = Nx
    vol = np.zeros((Nz, Ny, Nx), dtype=np.float64)
    for iy in range(Ny):
        sino = phase_use[:, iy, :].T                # (n_det=Nx, n_ang)
        rec = gridding_reconstruct_slice(sino, theta_use, device=device)
        if op is not None:
            k, flip = op
            rec = np.rot90(rec, k)
            if flip:
                rec = np.fliplr(rec)
        vol[:, iy, :] = scale * rec
    return vol


# ---------------------------------------------------------------------------
# Forward projector (re-projection): type-2 NUFFT, the matched adjoint of the
# type-1 back-projector above.  Use in consistency-based alignment so the
# forward and inverse operators are a matched pair (no operator mismatch).
# ---------------------------------------------------------------------------
def gridding_reproject_slice(obj, theta_deg, device='auto'):
    """
    Forward Radon (re-projection) of one 2-D slice via the Fourier Slice
    Theorem: P_theta(rho) = FT2(obj)|_(rho cos, rho sin), then 1-D IFFT over rho.

    Parameters
    ----------
    obj : (N, N) real   square slice (Nz == Nx).
    theta_deg : (n_ang,) angles [deg].

    Returns
    -------
    (N, n_ang) real sinogram (detector x angle).
    """
    dev = _resolve_grid_device(device)
    xp = _cp if dev == 'gpu' else np
    N = obj.shape[0]
    n_ang = len(theta_deg)

    rho = xp.fft.fftshift(xp.fft.fftfreq(N))
    th = xp.asarray(np.deg2rad(np.asarray(theta_deg, dtype=np.float64)))
    KX = xp.outer(rho, xp.cos(th))
    KY = xp.outer(rho, xp.sin(th))
    x = (2.0 * np.pi * KX).ravel()
    y = (2.0 * np.pi * KY).ravel()
    f = xp.asarray(obj).astype(xp.complex128)

    # type-2 NUFFT (uniform object -> nonuniform polar Fourier samples).
    # modeord=0 so f is interpreted centre-ordered (object centred in array).
    if dev == 'gpu':
        P = _cufinufft.nufft2d2(x, y, f, isign=+1, eps=1e-9, modeord=0)
    else:
        P = finufft.nufft2d2(x, y, f, isign=+1, eps=1e-9, modeord=0)
    P = P.reshape(N, n_ang)

    # radial samples -> projection: 1-D inverse FFT along rho
    sino = xp.fft.fftshift(
        xp.fft.ifft(xp.fft.ifftshift(P, axes=0), axis=0), axes=0).real
    return _cp.asnumpy(sino) if dev == 'gpu' else sino


def gridding_reproject_volume(vol, theta, calib=None):
    """
    Re-project a (Nz, Ny, Nx) volume to phase projections (N_ang, Ny, Nx),
    matching the phase_use layout used for alignment / self-consistency.

    calib : dict from self_test_forward(): {'flip': bool, 'scale': s, 'device'}.
    """
    flip = calib['flip'] if calib else False
    scale = calib['scale'] if calib else 1.0
    device = calib.get('device', 'auto') if calib else 'auto'

    Nz, Ny, Nx = vol.shape
    n_ang = len(theta)
    out = np.zeros((n_ang, Ny, Nx), dtype=np.float64)
    for iy in range(Ny):
        s = gridding_reproject_slice(vol[:, iy, :], theta, device=device)  # (Nx,n_ang)
        if flip:
            s = s[::-1]
        out[:, iy, :] = (scale * s).T
    return out


# ---------------------------------------------------------------------------
# Self-test against skimage.iradon
# ---------------------------------------------------------------------------
def _make_phantom(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    c = (n - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    ph = np.zeros((n, n))
    ph[r < 0.42 * n] = 1.0
    ph[int(0.20 * n):int(0.32 * n), int(0.30 * n):int(0.70 * n)] += 0.8
    ph[int(0.62 * n):int(0.78 * n), int(0.40 * n):int(0.55 * n)] += 0.6
    ph[r >= 0.42 * n] = 0.0
    return ph


def self_test(N=128, n_ang=180, device='auto', verbose=True):
    """
    Validate the gridding reconstructor against skimage.iradon on a phantom,
    on the SAME device that will be used for the real reconstruction.  This
    gates the (CPU or GPU) backend: the pipeline only uses gridding if the
    test passes, so an untested convention (e.g. a GPU modeord difference)
    fails safely back to iradon rather than corrupting the volume.

    Returns
    -------
    (passed: bool, corr: float, calib: dict or None)
        calib = {'op': (k, flip), 'scale': s, 'device': 'gpu'|'cpu'} maps the
        gridding output onto the iradon convention (orientation AND amplitude)
        and records the validated device; pass it to
        gridding_reconstruct_volume.
    """
    from skimage.transform import radon, iradon

    try:
        dev = _resolve_grid_device(device)
    except RuntimeError as e:
        if verbose:
            print(f"  [nufft_gridding] {e}")
        return False, float("nan"), None

    ph = _make_phantom(N)
    theta = np.linspace(0.0, 180.0, n_ang, endpoint=False)
    sino = radon(ph, theta=theta, circle=True)
    ref = iradon(sino, theta=theta, filter_name="ramp", circle=True)
    try:
        rec = gridding_reconstruct_slice(sino, theta, device=dev)
    except Exception as e:
        if verbose:
            print(f"  [nufft_gridding] {dev} reconstruction failed ({e}); "
                  f"backend unusable.")
        return False, float("nan"), None

    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    best_c, best_op = -2.0, (0, False)
    for k in range(4):
        for flip in (False, True):
            r = np.rot90(rec, k)
            r = np.fliplr(r) if flip else r
            cc = corr(r, ref)
            if cc > best_c:
                best_c, best_op = cc, (k, flip)

    # Least-squares amplitude scale to match iradon (correlation is scale-blind)
    r = np.rot90(rec, best_op[0])
    r = np.fliplr(r) if best_op[1] else r
    denom = float(np.sum(r * r))
    scale = float(np.sum(ref * r) / denom) if denom > 0 else 1.0

    passed = best_c >= SELFTEST_MIN_CORR
    calib = {'op': best_op, 'scale': scale, 'device': dev}
    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [nufft_gridding] self-test vs iradon ({dev}): "
              f"corr={best_c:.4f} (orient rot90^{best_op[0]}, "
              f"flip={best_op[1]}, scale={scale:.3e})  -> {status}")
    return passed, best_c, (calib if passed else None)


def self_test_forward(N=128, n_ang=180, device='auto', verbose=True):
    """
    Validate the type-2 forward projector (re-projection) against
    skimage.radon on the same device.  Returns (passed, corr, calib) with
    calib = {'flip': bool, 'scale': s, 'device': dev}.  Sinogram orientation
    only admits a detector-axis flip (the angle axis is fixed), so the search
    is simpler than the back-projector's dihedral one.
    """
    from skimage.transform import radon

    try:
        dev = _resolve_grid_device(device)
    except RuntimeError as e:
        if verbose:
            print(f"  [nufft_gridding] {e}")
        return False, float("nan"), None

    ph = _make_phantom(N)
    theta = np.linspace(0.0, 180.0, n_ang, endpoint=False)
    sino_ref = radon(ph, theta=theta, circle=True)             # (N, n_ang)
    try:
        sino = gridding_reproject_slice(ph, theta, device=dev)
    except Exception as e:
        if verbose:
            print(f"  [nufft_gridding] {dev} re-projection failed ({e}).")
        return False, float("nan"), None

    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    best_c, best_flip = -2.0, False
    for fl in (False, True):
        s = sino[::-1] if fl else sino
        cc = corr(s, sino_ref)
        if cc > best_c:
            best_c, best_flip = cc, fl
    s = sino[::-1] if best_flip else sino
    denom = float(np.sum(s * s))
    scale = float(np.sum(sino_ref * s) / denom) if denom > 0 else 1.0

    passed = best_c >= SELFTEST_MIN_CORR
    calib = {'flip': best_flip, 'scale': scale, 'device': dev}
    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  [nufft_gridding] forward self-test vs radon ({dev}): "
              f"corr={best_c:.4f} (flip={best_flip}, scale={scale:.3e})  "
              f"-> {status}")
    return passed, best_c, (calib if passed else None)


if __name__ == "__main__":
    print("=" * 60)
    print("NUFFT gridding back-projector — self-test")
    print("=" * 60)
    print(f"  finufft (CPU)   available: {_HAVE_FINUFFT}")
    print(f"  cufinufft (GPU) available: {_HAVE_CUFINUFFT}")
    if not _HAVE_CUFINUFFT and _CUFINUFFT_ERR:
        print(f"    GPU NUFFT disabled — {_CUFINUFFT_ERR}")
        print("    (this is harmless: the backend falls back to CPU finufft / "
              "iradon)")
    if not (_HAVE_FINUFFT or _HAVE_CUFINUFFT):
        raise SystemExit("Install a NUFFT backend: pip install finufft "
                         "(CPU) and/or cufinufft cupy (GPU).")
    print("\n-- back-projection (reconstruction) --")
    ok, c, calib = self_test(device='auto')
    print("\n-- forward projection (re-projection, for alignment) --")
    okf, cf, calibf = self_test_forward(device='auto')

    print(f"\nBack-projector : {'USABLE' if ok else 'NOT USABLE'}  "
          f"(corr={c:.4f}, threshold={SELFTEST_MIN_CORR})")
    if ok:
        print(f"  calib: {calib}")
    print(f"Forward proj.  : {'USABLE' if okf else 'NOT USABLE'}  "
          f"(corr={cf:.4f}, threshold={SELFTEST_MIN_CORR})")
    if okf:
        print(f"  calib: {calibf}")
    if not (ok and okf):
        print("\nDo NOT use a backend whose self-test did not pass.")
