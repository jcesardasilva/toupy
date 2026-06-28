#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative-reprojection halo (exterior) estimator for local tomography
=====================================================================

Purpose
-------
Before running the two-pass multislice refinement on truncated (interior /
ROI) data, estimate the material OUTSIDE the field of view (the "halo") so the
extended grid is initialised with a data-consistent exterior instead of zeros
or a constant.  A good exterior removes most of the truncation bias inside the
FOV *before* two-pass starts, leaving only the small irreducible interior
null-space for the DC / overview anchor to handle.

Why this works (and where the theorem bites)
--------------------------------------------
Each measured FOV ray integrates through exterior material that, at other
angles, lies outside the detector.  That exterior contribution is therefore
PRESENT in the data (recoverable), it is just entangled along the rays.  The
classic iterative reconstruction-reprojection (IRR) loop disentangles it:

    1. extend the truncated sinogram into the exterior (smooth taper init);
    2. FBP-reconstruct the full-width slice and impose positivity (object
       constraint);
    3. re-project to a synthetic full sinogram;
    4. overwrite the FOV columns with the MEASURED data (data constraint),
       keep the synthetic exterior columns;
    5. iterate.

This is alternating projection onto two convex-ish sets (measured-data-in-FOV;
range-of-Radon-of-a-positive-object), i.e. a Gerchberg–Papoulis / POCS
extrapolation specialised to truncation (cf. Ohnesorge 2000).  It converges to
a full sinogram consistent with the measured FOV and with being the projection
of a real positive object.

What it CANNOT do: the smooth, low-order additive interior null-space
(Courdurier/Kudyakov) is invisible to every measurement, so IRR cannot pin it.
That residual is left to the air anchor / overview prior in twopass_local_data.

Per-slice, parallel
-------------------
In parallel-beam geometry each detector row (constant y) is an independent 2-D
slice problem in the x–z plane, so the IRR runs per y-slice and parallelises
across slices.

Sign convention
---------------
Works in δ-sinogram units.  φ = −k₀·pixel·Σδ ⇒ the δ line integral is
g = −φ/(k₀·pixel).  Reconstructing g gives δ directly, so positivity (δ ≥ 0)
is applied in the correct space.

Usage
-----
  # standalone self-test
  python tutorial/halo_estimator.py --selftest

  # used programmatically by twopass_local_data.py (HALO_INIT='irr')
"""

import sys
import numpy as np

try:
    from skimage.transform import radon, iradon
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False

try:
    from joblib import Parallel, delayed as _delayed
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False


# ---------------------------------------------------------------------------
# Sinogram exterior taper (initial guess for the missing columns)
# ---------------------------------------------------------------------------
def _cosine_taper_exterior(p, x0, x1):
    """
    In-place: fill the exterior detector columns of a full-width δ-sinogram
    ``p`` (shape (n_det, n_ang)) with a half-cosine decay from the measured
    boundary value (at the FOV edge) to 0 at the detector edge.  This is only
    the IRR initial guess; the loop refines it.
    """
    n_det, n_ang = p.shape
    nL, nR = x0, n_det - x1
    if nL > 0:
        v = p[x0, :]                                   # inner-edge value (n_ang,)
        i = np.arange(nL)                              # 0 (far) .. nL-1 (near)
        wL = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / nL))  # 0 .. ~1
        p[:x0, :] = v[None, :] * wL[:, None]
    if nR > 0:
        v = p[x1 - 1, :]
        j = np.arange(nR)                              # 0 (near) .. nR-1 (far)
        wR = 0.5 * (1.0 - np.cos(np.pi * (nR - j) / nR))  # ~1 .. 0
        p[x1:, :] = v[None, :] * wR[:, None]


# ---------------------------------------------------------------------------
# One slice IRR
# ---------------------------------------------------------------------------
def irr_slice(phase_row, theta_deg, x0, x1, full_Nx, k0, pixel,
              n_iter=5, relax=1.0):
    """
    Iterative reconstruction-reprojection for one y-slice.

    Parameters
    ----------
    phase_row : (n_ang, nx) measured phase for this slice (FOV columns only) [rad].
    theta_deg : (n_ang,) angles [deg].
    x0, x1    : FOV detector window in the full-width detector (centred crop).
    full_Nx   : full detector / reconstruction width.
    k0, pixel : wavenumber [1/m] and pixel size [m] for the δ conversion.
    n_iter    : IRR iterations.
    relax     : exterior update relaxation in [0,1]; 1.0 = take the synthetic
                exterior fully each iteration, <1 blends with the previous one
                for extra stability.

    Returns
    -------
    delta_slice : (full_Nx, full_Nx) reconstructed δ slice (z, x), δ ≥ 0.
    """
    # measured δ-sinogram (detector, angles)
    m = (-phase_row.T) / (k0 * pixel)                  # (nx, n_ang)
    n_ang = m.shape[1]

    p = np.zeros((full_Nx, n_ang), dtype=np.float64)
    p[x0:x1, :] = m
    _cosine_taper_exterior(p, x0, x1)

    for _ in range(n_iter):
        f = iradon(p, theta=theta_deg, filter_name="ramp",
                   circle=True, output_size=full_Nx)
        np.clip(f, 0.0, None, out=f)                   # positivity (object set)
        p_syn = radon(f, theta=theta_deg, circle=True)  # (full_Nx, n_ang)
        if relax >= 1.0:
            p = p_syn
        else:
            p = relax * p_syn + (1.0 - relax) * p
        p[x0:x1, :] = m                                # data set: re-impose FOV

    f = iradon(p, theta=theta_deg, filter_name="ramp",
               circle=True, output_size=full_Nx)
    np.clip(f, 0.0, None, out=f)
    return f


# ---------------------------------------------------------------------------
# Volume driver
# ---------------------------------------------------------------------------
def estimate_halo_volume(phase_trunc, theta_deg, fov_x0, fov_x1, full_Nx,
                         k0, pixel, n_iter=5, relax=1.0, n_jobs=None,
                         verbose=True):
    """
    Run the IRR halo estimator on every y-slice of a truncated projection set.

    Parameters
    ----------
    phase_trunc : (n_ang, ny, nx) truncated phase projections [rad].
    theta_deg   : (n_ang,) angles [deg].
    fov_x0,fov_x1 : FOV detector window in the full-width detector.
    full_Nx     : full detector / reconstruction width.
    k0, pixel   : δ conversion constants.
    n_iter,relax: IRR controls (see irr_slice).
    n_jobs      : joblib jobs across slices (None → all cores).

    Returns
    -------
    delta_full : (full_Nx, ny, full_Nx) δ volume (z, y, x), δ ≥ 0.
                 Caller embeds it into the full extended grid (row offset).
    """
    if not _HAVE_SKIMAGE:
        sys.exit("halo_estimator requires scikit-image (pip install scikit-image).")

    n_ang, ny, nx = phase_trunc.shape
    assert (fov_x1 - fov_x0) == nx, \
        f"FOV width {fov_x1-fov_x0} != truncated nx {nx}"

    if verbose:
        print(f"  IRR halo estimator: {ny} slices, full width {full_Nx}, "
              f"n_iter={n_iter}, relax={relax}", flush=True)

    def _one(iy):
        return irr_slice(phase_trunc[:, iy, :], theta_deg, fov_x0, fov_x1,
                         full_Nx, k0, pixel, n_iter=n_iter, relax=relax)

    if _HAVE_JOBLIB and (n_jobs is None or n_jobs != 1):
        nj = n_jobs if n_jobs is not None else -1
        slices = Parallel(n_jobs=nj, prefer="threads")(
            _delayed(_one)(iy) for iy in range(ny))
        delta_full = np.stack(slices, axis=1)          # (full_Nx, ny, full_Nx)
    else:
        delta_full = np.empty((full_Nx, ny, full_Nx), dtype=np.float64)
        for iy in range(ny):
            delta_full[:, iy, :] = _one(iy)

    return delta_full.astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest():
    print("halo_estimator self-test")
    if not _HAVE_SKIMAGE:
        print("  scikit-image absent — cannot run.")
        return False

    rng = np.random.default_rng(0)
    N = 128
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    cx = cy = N / 2.0

    # Phantom: a CONTINUOUS object (realistic — material crosses the FOV
    # boundary), with internal blobs and voids, inside a circular support.
    # This is the regime where the near-boundary exterior is recoverable;
    # an isolated ring fully outside the FOV would be adversarial (its
    # projections never enter the truncated detector).
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    phantom = np.where(r < 0.46 * N, 1.0, 0.0)          # solid disk (continuous)
    for _ in range(10):                                 # denser blobs
        bx, by = rng.uniform(0.30, 0.70, 2) * N
        phantom[((xx - bx)**2 + (yy - by)**2) < (0.06 * N)**2] += 0.6
    for _ in range(8):                                  # voids
        bx, by = rng.uniform(0.30, 0.70, 2) * N
        phantom[((xx - bx)**2 + (yy - by)**2) < (0.05 * N)**2] = 0.0
    phantom *= (r < 0.46 * N)                           # inside the support

    theta = np.linspace(0.0, 180.0, 180, endpoint=False)
    k0, pixel = 1.0, 1.0                                # identity δ conversion

    # Full sinogram, then truncate the detector to the central FOV (40 %)
    full_sino = radon(phantom, theta=theta, circle=True)   # (N, n_ang)
    fov_frac = 0.40
    w = int(round(fov_frac * N))
    x0 = (N - w) // 2
    x1 = x0 + w
    # phase = -k0 pixel * (delta line integral); here delta=phantom
    phase_trunc = (-(full_sino[x0:x1, :]) * k0 * pixel).T   # (n_ang, nx)
    phase_trunc = phase_trunc[:, None, :]                    # (n_ang, 1, nx)

    # Recover
    delta = estimate_halo_volume(phase_trunc, theta, x0, x1, N, k0, pixel,
                                 n_iter=8, verbose=False)
    rec = delta[:, 0, :]                                # (N, N)

    # Reference FBP reconstructions of the SAME slice for comparison:
    #   (a) naive: truncated sinogram embedded in zeros (no exterior)
    naive_sino = np.zeros((N, len(theta)))
    naive_sino[x0:x1, :] = full_sino[x0:x1, :]
    naive = np.clip(iradon(naive_sino, theta=theta, filter_name="ramp",
                           circle=True, output_size=N), 0, None)

    # Masks: interior (FOV) disk and NEAR-BOUNDARY exterior annulus.
    # The near-boundary halo is what affects the FOV rays (and what two-pass
    # needs); the far exterior is weakly determined and not the target here.
    a = (x1 - x0) / 2.0
    interior = r < 0.9 * a
    exterior = (r > 1.05 * a) & (r < 1.8 * a) & (r < 0.46 * N)

    def _corr(u, v, mask):
        u, v = u[mask].ravel(), v[mask].ravel()
        return float(np.corrcoef(u, v)[0, 1])

    corr_int_irr   = _corr(rec,   phantom, interior)
    corr_int_naive = _corr(naive, phantom, interior)
    corr_ext_irr   = _corr(rec,   phantom, exterior)

    # FOV bias (mean error) — IRR should reduce it vs naive truncation
    bias_irr   = float(np.mean((rec   - phantom)[interior]))
    bias_naive = float(np.mean((naive - phantom)[interior]))

    checks = {
        "interior corr improved (IRR>naive)": corr_int_irr > corr_int_naive,
        "exterior recovered (corr>0.3)":      corr_ext_irr > 0.30,
        "FOV bias reduced (|IRR|<|naive|)":   abs(bias_irr) < abs(bias_naive),
        "output positive":                    rec.min() >= 0.0,
        "output shape":                       rec.shape == (N, N),
    }
    print(f"  interior corr : IRR={corr_int_irr:.3f}  naive={corr_int_naive:.3f}")
    print(f"  exterior corr : IRR={corr_ext_irr:.3f}")
    print(f"  FOV bias      : IRR={bias_irr:+.3e}  naive={bias_naive:+.3e}")
    for k, v in checks.items():
        print(f"  {k:38s}: {v}")
    ok = all(checks.values())
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    print(__doc__)
