#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignment-by-synthesis for local (truncated) tomography
=======================================================

Problem
-------
Standard projection alignment (centre-of-mass / cross-correlation consistency)
breaks for local tomography: each truncated projection sees DIFFERENT external
material as the sample rotates, mass enters and leaves the FOV, and the exterior
is non-rigid with respect to the ROI.  A reconstruction built only from the
truncated FOV cannot predict that moving external contribution, so registering a
measured projection against its re-projection gives the WRONG shift.

Idea (alignment-by-synthesis with a halo)
-----------------------------------------
Once we can estimate the exterior (the IRR halo, tutorial/halo_estimator.py),
the EXTENDED reconstruction CAN predict the moving external material.  So:

  1. apply the current shift estimate to the measured truncated projections;
  2. reconstruct the EXTENDED grid (FOV + halo) via IRR;
  3. re-project the extended volume and crop to the FOV detector window
     -> a model truncated projection that INCLUDES the external contribution;
  4. register each measured projection against its model -> residual (dy, dx);
  5. accumulate, repeat.

The halo is what makes step 3 predictive; without it the exterior mismatch
masquerades as a shift and the loop chases its own tail.  This module lets you
run BOTH (HALO vs FOV-only) to demonstrate the difference.

Conventions
-----------
Jitter (tutorial/local_tomo_simulator.py / perturb_alignment.py) is applied as
``fourier_shift(FFT(proj), (dy, dx))``.  To UNDO an estimated jitter we apply
``(-dy, -dx)``.  We work on the phase projections directly (the geometry is the
same as for δ).

Usage
-----
  python tutorial/local_tomo_align.py --selftest
  # programmatic: align_by_synthesis(...) -> estimated shifts
"""

import os
import sys
import importlib.util
import numpy as np

try:
    from scipy.ndimage import fourier_shift
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

try:
    from skimage.transform import radon, iradon
    from skimage.registration import phase_cross_correlation
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, fpath):
    spec = importlib.util.spec_from_file_location(name, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


halo_mod = _load_module("halo_estimator",
                        os.path.join(_HERE, "halo_estimator.py"))
estimate_halo_volume = halo_mod.estimate_halo_volume


# ---------------------------------------------------------------------------
# Shift application
# ---------------------------------------------------------------------------
def apply_shifts(proj, dy, dx):
    """Apply per-projection sub-pixel (dy, dx) via Fourier shift.

    proj : (n_ang, ny, nx).  dy, dx : (n_ang,).  Returns shifted copy.
    """
    if not _HAVE_SCIPY:
        sys.exit("scipy required (pip install scipy).")
    out = np.empty_like(proj, dtype=np.float32)
    for a in range(proj.shape[0]):
        if dy[a] == 0.0 and dx[a] == 0.0:
            out[a] = proj[a]
        else:
            ft = np.fft.fft2(proj[a])
            out[a] = np.fft.ifft2(fourier_shift(ft, (dy[a], dx[a]))).real
    return out


# ---------------------------------------------------------------------------
# Re-projection of an extended volume, cropped to the FOV detector window
# ---------------------------------------------------------------------------
def reproject_truncated(delta_full, theta_deg, x0, x1, k0, pixel):
    """
    Project a full-grid δ volume and crop the detector to the FOV window,
    returning model PHASE projections matching the measured truncated set.

    delta_full : (Nz, ny, Nx)  δ volume (z, y, x), Nz == Nx.
    Returns    : (n_ang, ny, nx) model phase [rad], nx = x1 - x0.
    """
    Nz, ny, Nx = delta_full.shape
    n_ang = len(theta_deg)
    model = np.empty((n_ang, ny, x1 - x0), dtype=np.float32)
    for iy in range(ny):
        # δ line integral (detector, angles); crop detector to FOV columns
        sino = radon(delta_full[:, iy, :], theta=theta_deg, circle=True)
        model[:, iy, :] = (-k0 * pixel) * sino[x0:x1, :].T
    return model


# ---------------------------------------------------------------------------
# Per-angle registration (measured vs model), FOV region
# ---------------------------------------------------------------------------
def _hann2d(shape):
    wy = np.hanning(shape[0]); wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def _remove_rigid_modes(theta_deg, d):
    """
    Project out the tomographic null-space of a per-angle shift sequence:
    a constant (recentring convention) and the cosθ / sinθ components (a rigid
    object translation).  These are pure coordinate ambiguities — they are not
    real mis-alignment, and if left in they accumulate and make the
    alignment-by-synthesis loop drift.  Returns the residual (true jitter).
    """
    th = np.deg2rad(theta_deg)
    A = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], axis=1)
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    return d - A @ coef


def estimate_shifts(measured, model, upsample=50):
    """
    Per-angle residual shift that best aligns `measured` to `model`.

    Returns (dy, dx, cc): arrays (n_ang,).  The convention is such that
    apply_shifts(measured, dy, dx) moves measured onto model.
    """
    n_ang = measured.shape[0]
    win = _hann2d(measured.shape[1:])
    dy = np.zeros(n_ang); dx = np.zeros(n_ang); cc = np.zeros(n_ang)
    for a in range(n_ang):
        ref = model[a] * win
        mov = measured[a] * win
        # shift returned moves `mov` (=measured) onto `ref` (=model)
        shift, err, _ = phase_cross_correlation(
            ref, mov, upsample_factor=upsample, normalization=None)
        dy[a], dx[a] = float(shift[0]), float(shift[1])
        cc[a] = 1.0 - err
    return dy, dx, cc


# ---------------------------------------------------------------------------
# Alignment-by-synthesis loop
# ---------------------------------------------------------------------------
def align_by_synthesis(phase_trunc, theta_deg, fov_x0, fov_x1, full_Nx,
                       k0, pixel, n_outer=5, halo=True, halo_n_iter=4,
                       upsample=50, damp=0.7, n_jobs=None, verbose=True):
    """
    Estimate per-projection (dy, dx) for truncated projections by synthesis.

    halo=True  -> reconstruct the EXTENDED grid (FOV + IRR halo) and reproject
                  it (predicts moving external material).
    halo=False -> reconstruct only the FOV (FBP of the truncated set) and
                  reproject that (no exterior) — the baseline that should FAIL.

    Returns
    -------
    dy_tot, dx_tot : (n_ang,) cumulative estimated shifts to APPLY to the input
                     `phase_trunc` to align it (i.e. correction = these values).
    history        : list of per-iteration residual-shift RMS (dx).
    """
    if not _HAVE_SKIMAGE:
        sys.exit("scikit-image required (pip install scikit-image).")
    n_ang, ny, nx = phase_trunc.shape
    assert (fov_x1 - fov_x0) == nx

    dy_tot = np.zeros(n_ang); dx_tot = np.zeros(n_ang)
    history = []

    for it in range(n_outer):
        cur = apply_shifts(phase_trunc, dy_tot, dx_tot)

        if halo:
            delta_full = estimate_halo_volume(
                cur, theta_deg, fov_x0, fov_x1, full_Nx, k0, pixel,
                n_iter=halo_n_iter, n_jobs=n_jobs, verbose=False)
        else:
            # FOV-only reconstruction embedded in the full grid (zero halo)
            delta_full = np.zeros((full_Nx, ny, full_Nx), dtype=np.float32)
            for iy in range(ny):
                f = iradon((-cur[:, iy, :].T) / (k0 * pixel), theta=theta_deg,
                           filter_name="ramp", circle=True, output_size=nx)
                delta_full[fov_x0:fov_x1, iy, fov_x0:fov_x1] = np.clip(f, 0, None)

        model = reproject_truncated(delta_full, theta_deg, fov_x0, fov_x1,
                                    k0, pixel)
        ddy, ddx, _ = estimate_shifts(cur, model, upsample=upsample)

        # Project out the tomographic null-space (recentring + rigid
        # translation) so the loop tracks only real per-angle jitter and does
        # not drift along the coordinate ambiguities.
        ddx = _remove_rigid_modes(theta_deg, ddx)
        ddy = _remove_rigid_modes(theta_deg, ddy)

        # damped accumulation (residual shift moves measured onto model)
        dy_tot += damp * ddy
        dx_tot += damp * ddx
        rms = float(np.sqrt(np.mean((damp * ddx) ** 2)))
        history.append(rms)
        if verbose:
            print(f"  [{'halo' if halo else 'fov '}] outer {it+1}/{n_outer}  "
                  f"residual dx RMS={rms:.3f}px  "
                  f"|dx_tot|<{np.abs(dx_tot).max():.2f}px", flush=True)

    return dy_tot, dx_tot, history


# ---------------------------------------------------------------------------
# Self-test: recover KNOWN injected shifts; halo must beat FOV-only
# ---------------------------------------------------------------------------
def _selftest():
    print("local_tomo_align self-test")
    if not (_HAVE_SKIMAGE and _HAVE_SCIPY):
        print("  needs scikit-image + scipy."); return False

    rng = np.random.default_rng(1)
    N, NY = 128, 48
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    r = np.sqrt((xx - N / 2) ** 2 + (yy - N / 2) ** 2)
    support = r < 0.46 * N
    # 3-D phantom with REAL y-varying structure (different blobs per slice) so
    # the per-projection 2-D registration has genuine information — a stack of
    # identical rows starves it and is not representative.
    vol = np.zeros((N, NY, N), dtype=float)
    for iy in range(NY):
        sl = np.where(support, 1.0, 0.0)
        for _ in range(8):
            bx, by = rng.uniform(0.28, 0.72, 2) * N
            sl[((xx - bx) ** 2 + (yy - by) ** 2) < (0.06 * N) ** 2] += 0.6
        for _ in range(6):
            bx, by = rng.uniform(0.28, 0.72, 2) * N
            sl[((xx - bx) ** 2 + (yy - by) ** 2) < (0.05 * N) ** 2] = 0.0
        vol[:, iy, :] = sl * support

    theta = np.linspace(0.0, 180.0, 150, endpoint=False)
    k0, pixel = 1.0, 1.0
    n_ang = len(theta)

    # truncate to 40 % FOV; build truncated phase projections (n_ang, NY, w)
    w = int(round(0.40 * N)); x0 = (N - w) // 2; x1 = x0 + w
    clean = np.empty((n_ang, NY, w), dtype=np.float32)
    for iy in range(NY):
        s = radon(vol[:, iy, :], theta=theta, circle=True)
        clean[:, iy, :] = (-(s[x0:x1, :]) * k0 * pixel).T

    # inject KNOWN per-angle horizontal jitter (x only); compare against its
    # random part (rigid modes are coordinate ambiguities, removed both sides).
    sigma = 0.5
    true_dx = rng.normal(0.0, sigma, n_ang)
    true_res = _remove_rigid_modes(theta, true_dx)
    jittered = apply_shifts(clean, np.zeros(n_ang), true_dx)

    dyh, dxh, _ = align_by_synthesis(jittered, theta, x0, x1, N, k0, pixel,
                                     n_outer=6, halo=True, halo_n_iter=4,
                                     n_jobs=1, verbose=False)
    dyf, dxf, _ = align_by_synthesis(jittered, theta, x0, x1, N, k0, pixel,
                                     n_outer=6, halo=False, n_jobs=1,
                                     verbose=False)

    # ideal correction = -true_res; residual jitter after applying correction
    res_halo = np.std(true_res + dxh)
    res_fov  = np.std(true_res + dxf)
    corr_halo = np.corrcoef(dxh, -true_res)[0, 1]
    corr_fov  = np.corrcoef(dxf, -true_res)[0, 1]

    checks = {
        "halo recovers jitter (corr>0.85)": corr_halo > 0.85,
        "halo residual < input sigma":      res_halo < np.std(true_res),
        "halo beats FOV-only (corr)":       corr_halo > corr_fov + 0.1,
        "halo residual < FOV-only":         res_halo < res_fov,
    }
    print(f"  input random sigma_dx : {np.std(true_res):.3f} px")
    print(f"  HALO   corr={corr_halo:+.3f}  residual sigma={res_halo:.3f} px")
    print(f"  FOV    corr={corr_fov:+.3f}  residual sigma={res_fov:.3f} px")
    for k, v in checks.items():
        print(f"  {k:34s}: {v}")
    ok = all(checks.values())
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    print(__doc__)
