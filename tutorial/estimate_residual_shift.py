#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate the residual alignment of PXCTalignedprojections via re-projection.
============================================================================

The alignment-jitter sweep (perturb_alignment.py + collect_jitter_fsc.py) tells
you *how sensitive* the FSC is to a known mis-alignment sigma, but not *where*
the real, already-aligned data sits on that curve.  This script measures that
missing number: the residual per-projection shift left in the aligned set.

Method (matched re-projection / consistency residual)
-----------------------------------------------------
A self-consistent reconstruction is the "consensus" of all projections.  If the
data were perfectly aligned, re-projecting that volume would reproduce every
measured projection exactly (up to noise); the residual is zero.  If residual
mis-alignment remains, the projections are mutually inconsistent, no single
volume can match them all, and each measured projection deviates from the
re-projection by roughly its own residual shift.  So:

  1. Re-project the two-pass volume with the *matched* type-2 NUFFT forward
     operator (the adjoint of the type-1 back-projector that built it).
  2. Register each measured projection against its re-projection (sub-pixel
     phase cross-correlation) -> per-angle (dy, dx).
  3. The std of dx about its mean (a constant offset is just a centring
     convention, not mis-alignment) is the residual jitter sigma_x.  Drop it
     onto the jitter-sweep curve to read off the head-room.

Interpretation
--------------
  * sigma_x >~ 0.25-0.5 px  -> on the steep part of the sweep: alignment is a
    live lever; matched-reprojection re-registration could buy resolution.
  * sigma_x << 0.25 px      -> on the flat foot: NOT alignment-limited; the
    ceiling is dose / phase-retrieval.

Usage
-----
  python tutorial/estimate_residual_shift.py
  python tutorial/estimate_residual_shift.py \\
      --recon tutorial/twopass_real_figures/twopass_reconstruction.npz \\
      --data  tutorial/PXCTalignedprojections.npz \\
      --crop-x 55 --crop-y 55 --use tp --device auto --upsample 50

Self-test (registration loop, on injected shifts):
  python tutorial/estimate_residual_shift.py --selftest
"""

import os
import sys
import argparse
import importlib.util
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(dotted_name, fpath):
    spec = importlib.util.spec_from_file_location(dotted_name, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------
def _hann2d(shape):
    """Separable 2-D Hann window to suppress projection-edge leakage in the
    phase correlation (a hard boundary fakes a high-frequency 'feature' that
    biases the shift estimate)."""
    wy = np.hanning(shape[0]) if shape[0] > 1 else np.ones(1)
    wx = np.hanning(shape[1]) if shape[1] > 1 else np.ones(1)
    return np.outer(wy, wx)


def _register_pair(ref, mov, upsample, win):
    """Sub-pixel shift (dy, dx) that moves `mov` onto `ref`, plus a normalised
    correlation as a sanity score.  Uses skimage phase_cross_correlation."""
    from skimage.registration import phase_cross_correlation
    r = (ref - ref.mean()) * win
    m = (mov - mov.mean()) * win
    shift, _err, _phase = phase_cross_correlation(
        r, m, upsample_factor=upsample, normalization=None)
    # normalised cross-correlation at the recovered integer offset is overkill;
    # a plain Pearson on the windowed pair is a robust orientation/scale sanity.
    corr = float(np.corrcoef(r.ravel(), m.ravel())[0, 1])
    return float(shift[0]), float(shift[1]), corr   # dy, dx, corr


def estimate_shifts(measured, model, upsample=50):
    """
    Per-angle (dy, dx, corr) between measured and re-projected projections.

    measured, model : (N_ang, Ny, Nx) real, same layout/angles.
    Returns dy, dx, corr  each (N_ang,).
    """
    N_ang = measured.shape[0]
    win = _hann2d(measured.shape[1:])
    dy = np.zeros(N_ang); dx = np.zeros(N_ang); cc = np.zeros(N_ang)
    for a in range(N_ang):
        dy[a], dx[a], cc[a] = _register_pair(
            measured[a], model[a], upsample, win)
    return dy, dx, cc


# ---------------------------------------------------------------------------
# Angle matching
# ---------------------------------------------------------------------------
def _fit_geometry(theta_deg, d):
    """
    Split a per-angle shift into a coherent rigid-geometry term and random
    residual jitter.

    A rigid transverse displacement of the object/rotation-axis (a centring or
    centre-of-rotation error, or a coordinate-convention offset between the
    re-projection and the back-projector that built the volume) projects onto
    the detector as  d(theta) = a0 + a1 cos(theta) + a2 sin(theta).  This is a
    smooth, fully *correctable* global term -- NOT the independent per-angle
    jitter that the FSC sweep modelled.  The residual after removing it is the
    genuine random mis-alignment, directly comparable to the sweep's sigma.

    Returns
    -------
    fit   : (N,) the a0 + a1 cos + a2 sin model
    resid : (N,) d - fit
    amp   : float  sqrt(a1^2 + a2^2), the rigid-offset amplitude [px]
    """
    th = np.deg2rad(theta_deg)
    A = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    fit = A @ coef
    amp = float(np.hypot(coef[1], coef[2]))
    return fit, d - fit, amp


def _angle_to_angle_jitter(theta_deg, d):
    """
    Trend-immune estimate of the *random* per-angle jitter.

    The first difference of the angle-sorted shifts cancels ANY smooth angular
    trend (centring, axis tilt, higher harmonics, slow drift -- all coherent and
    correctable), leaving only the white, angle-to-angle component, which is what
    the FSC jitter sweep modelled.  For independent jitter with std sigma the
    successive difference has std sqrt(2)*sigma, so sigma = MAD(diff)/sqrt(2)
    (robust 1.4826*MAD, immune to the occasional registration outlier).

    Returns sigma_random [px].
    """
    order = np.argsort(theta_deg)
    dd = np.diff(d[order])
    mad = np.median(np.abs(dd - np.median(dd)))
    return float(1.4826 * mad / np.sqrt(2.0))


def _match_angles(theta_recon, theta_data, tol=1e-3):
    """Index into theta_data for each theta_recon (nearest within tol deg).
    Lets you point at the full data file even when the recon used a subset."""
    idx = np.zeros(len(theta_recon), dtype=int)
    for i, t in enumerate(theta_recon):
        j = int(np.argmin(np.abs(theta_data - t)))
        if abs(theta_data[j] - t) > tol:
            raise SystemExit(
                f"angle {t:.4f} deg in recon not found in data "
                f"(closest {theta_data[j]:.4f}).  Use the same DATA_FILE the "
                f"reconstruction was built from.")
        idx[i] = j
    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default=os.path.join(
        _HERE, "twopass_real_figures", "twopass_reconstruction.npz"),
        help="twopass_reconstruction.npz (has delta_tp/delta_fbp, theta, psize)")
    ap.add_argument("--data", default=os.path.join(
        _HERE, "PXCTalignedprojections.npz"),
        help="aligned projections the recon was built from")
    ap.add_argument("--use", choices=["tp", "fbp"], default="tp",
                    help="which volume to re-project (default: two-pass)")
    ap.add_argument("--crop-x", type=int, default=55,
                    help="match the CROP_X used in twopass_real_data.py")
    ap.add_argument("--crop-y", type=int, default=55,
                    help="match the CROP_Y used in twopass_real_data.py")
    ap.add_argument("--device", default="auto", help="auto|gpu|cpu (NUFFT)")
    ap.add_argument("--upsample", type=int, default=50,
                    help="sub-pixel registration precision (1/upsample px)")
    ap.add_argument("--max-angles", type=int, default=0,
                    help="subsample to this many angles for speed (0 = all)")
    ap.add_argument("--corr-min", type=float, default=0.3,
                    help="warn if median model/data corr falls below this "
                         "(orientation/scale mismatch).")
    ap.add_argument("--out", default=None, help="output dir (default: recon dir)")
    args = ap.parse_args()

    grid = _load_module("nufft_gridding",
                        os.path.join(_HERE, "nufft_gridding.py"))

    if not os.path.isfile(args.recon):
        raise SystemExit(f"recon not found: {args.recon}")
    if not os.path.isfile(args.data):
        raise SystemExit(f"data not found: {args.data}")

    # ---- load reconstruction ----------------------------------------------
    R = np.load(args.recon, allow_pickle=True)
    key = "delta_tp" if args.use == "tp" else "delta_fbp"
    if key not in R.files:
        raise SystemExit(f"{key} not in {args.recon} (have: {list(R.files)})")
    vol = R[key].astype(np.float64)
    theta_recon = R["theta"].astype(np.float64)
    psize = float(R["psize"]) if "psize" in R.files else float("nan")
    wavelength = float(R["wavelength"]) if "wavelength" in R.files else float("nan")
    Nz, Ny, Nx = vol.shape
    print("=" * 64)
    print("Residual-alignment estimate via matched re-projection")
    print("=" * 64)
    print(f"  recon  : {args.recon}")
    print(f"  volume : {key}  shape {vol.shape}  psize {psize*1e9:.2f} nm")
    print(f"  angles : {len(theta_recon)}")

    # ---- load + crop measured projections to match the volume -------------
    D = np.load(args.data, allow_pickle=True)
    proj = D["projections"].astype(np.float64)            # (N_data, Ny0, Nx0)
    theta_data = D["theta"].astype(np.float64)
    cy, cx = args.crop_y, args.crop_x
    sy = slice(cy or None, -cy if cy else None)
    sx = slice(cx or None, -cx if cx else None)
    proj = proj[:, sy, sx]
    if proj.shape[1:] != (Ny, Nx):
        raise SystemExit(
            f"cropped projection {proj.shape[1:]} != volume (Ny,Nx)=({Ny},{Nx}). "
            f"Adjust --crop-x/--crop-y to the values used at reconstruction.")
    idx = _match_angles(theta_recon, theta_data)
    measured = proj[idx]                                  # (N_ang, Ny, Nx)

    # optional angle subsample for speed
    if args.max_angles and args.max_angles < len(theta_recon):
        sel = np.linspace(0, len(theta_recon) - 1, args.max_angles).astype(int)
        measured = measured[sel]; theta_recon = theta_recon[sel]
        print(f"  (subsampled to {len(theta_recon)} angles for speed)")

    # ---- validate + run the matched forward projector ---------------------
    okf, cf, calibf = grid.self_test_forward(device=args.device, verbose=True)
    if not okf:
        raise SystemExit("type-2 forward projector self-test FAILED; refusing "
                         "to trust the residual estimate (see message above).")
    print("  re-projecting volume (type-2 NUFFT) …", flush=True)
    model = grid.gridding_reproject_volume(vol, theta_recon, calib=calibf)

    # Put the model in measured-phase units (phi = -k0 * psize * sum(delta)).
    # Scale is irrelevant to the shift, but the sign flip makes the sanity
    # correlation positive and directly readable.
    if np.isfinite(wavelength) and np.isfinite(psize):
        model = (-2.0 * np.pi / wavelength * psize) * model

    # ---- register each angle ----------------------------------------------
    print("  registering measured vs re-projection …", flush=True)
    dy, dx, cc = estimate_shifts(measured, model, upsample=args.upsample)

    med_cc = float(np.median(cc))
    if med_cc < args.corr_min:
        print(f"\n  [WARN] median model/data correlation {med_cc:.2f} < "
              f"{args.corr_min}: orientation/scale may be off; the shift "
              f"estimate may be unreliable.")

    # ---- report -----------------------------------------------------------
    # A constant offset is just a frame-centring convention, not mis-alignment;
    # the residual *jitter* is the scatter about the mean (matches the zero-mean
    # Gaussian the sweep injected).
    dx_c = dx - dx.mean(); dy_c = dy - dy.mean()
    sx = float(np.std(dx_c)); sy_ = float(np.std(dy_c))

    # Decompose into rigid geometry (a0 + a1 cos + a2 sin) + random jitter.
    # The geometry term is a single correctable global registration; only the
    # detrended residual is comparable to the FSC sweep's independent jitter.
    fit_x, res_x, amp_x = _fit_geometry(theta_recon, dx)
    fit_y, res_y, amp_y = _fit_geometry(theta_recon, dy)
    jit_x = float(np.std(res_x)); jit_y = float(np.std(res_y))
    # trend-immune random jitter (cancels ALL smooth angular structure, not just
    # the first cos/sin harmonic) -- the honest sweep-comparable number.
    rnd_x = _angle_to_angle_jitter(theta_recon, dx)
    rnd_y = _angle_to_angle_jitter(theta_recon, dy)

    print("\n" + "=" * 64)
    print("Residual per-projection shift (pixels)")
    print("=" * 64)
    print(f"  median model/data corr : {med_cc:.3f}  "
          f"(low => estimate unreliable)")
    print(f"  x (horizontal, tomographically critical):")
    print(f"      raw std {sx:.3f}   max|.| {np.abs(dx_c).max():.3f}")
    print(f"      rigid 1st-harmonic amp {amp_x:.3f}  |  1st-harm residual "
          f"std {jit_x:.3f}  |  random (trend-immune) {rnd_x:.3f}")
    print(f"  y (along rotation axis):")
    print(f"      raw std {sy_:.3f}   max|.| {np.abs(dy_c).max():.3f}")
    print(f"      rigid 1st-harmonic amp {amp_y:.3f}  |  1st-harm residual "
          f"std {jit_y:.3f}  |  random (trend-immune) {rnd_y:.3f}")
    print("\n  Three layers, from most to least correctable:")
    print("   - rigid 1st-harmonic amp : centre-of-rotation / lateral-offset (or "
          "a NUFFT-vs-\n     iradon centre-convention) error; one global "
          "registration removes it.")
    print("   - 1st-harm residual std  : still contains higher-harmonic coherent "
          "structure\n     (axis tilt, operator non-ideality) -> OVER-estimates "
          "the random part.")
    print("   - random (trend-immune)  : MAD of angle-to-angle differences; "
          "cancels ALL smooth\n     structure -> the honest number to compare "
          "with the FSC jitter sweep.")

    # ---- verdict against the jitter sweep (use the TREND-IMMUNE jitter) ----
    print("\nVerdict (trend-immune random jitter vs the jitter-sweep curve):")
    print(f"  random sigma_x = {rnd_x:.3f} px   "
          f"(coherent/correctable part = {np.hypot(amp_x, jit_x):.2f} px, "
          f"separable)")
    if rnd_x >= 0.4:
        print("  => RANDOM JITTER ON THE STEEP PART (>~0.4 px): alignment is a "
              "live lever.\n     Matched-reprojection re-registration should "
              "recover resolution\n     (the sweep showed ~+14% FSC loss at 0.5 "
              "px).")
    elif rnd_x >= 0.2:
        print("  => MARGINAL random jitter (0.2-0.4 px): some head-room; one "
              "consistency-\n     alignment pass to confirm.")
    else:
        print("  => RANDOM JITTER ON THE FLAT FOOT (<0.2 px): the per-angle "
              "scatter is small.\n     The random part is NOT the ceiling "
              "(dose / phase-retrieval is); any\n     coherent term is a separate "
              "global re-centre, not a resolution lever.")
    if rnd_x < 0.4 <= jit_x:
        print("  (Note: the 1st-harmonic residual looked large, but it was "
              "coherent\n   structure, not random jitter -- the trend-immune "
              "estimate is the real one.)")
    print("  (Sweep reference: 0.25 px ~ noise, 0.5 px ~ +14% FSC loss.)")

    # ---- plot -------------------------------------------------------------
    out_dir = args.out or os.path.dirname(os.path.abspath(args.recon))
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(theta_recon)
        th_s = theta_recon[order]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        # left: raw dx with the fitted rigid-geometry curve overlaid
        ax[0].plot(theta_recon, dx_c, ".", ms=3, color="C1", label="dx (x)")
        ax[0].plot(th_s, (fit_x - dx.mean())[order], "-", color="C3", lw=1.6,
                   label=f"rigid fit (amp {amp_x:.2f} px)")
        ax[0].axhline(0, color="k", lw=0.6)
        ax[0].set_xlabel("projection angle [deg]")
        ax[0].set_ylabel("residual shift [px] (mean removed)")
        ax[0].set_title(f"dx: rigid geometry + jitter (jitter std {jit_x:.3f} px)")
        ax[0].legend(); ax[0].grid(alpha=0.3)
        # right: histogram of the DETRENDED random jitter (sweep-comparable)
        ax[1].hist(res_x, bins=31, color="C1", alpha=0.8,
                   label=f"dx  1st-harm s={jit_x:.2f}, rand {rnd_x:.2f}")
        ax[1].hist(res_y, bins=31, color="C0", alpha=0.5,
                   label=f"dy  1st-harm s={jit_y:.2f}, rand {rnd_y:.2f}")
        ax[1].set_xlabel("1st-harmonic residual shift [px]")
        ax[1].set_ylabel("count")
        ax[1].set_title(f"Residual (trend-immune random sigma_x={rnd_x:.2f} px)")
        ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fpath = os.path.join(out_dir, f"residual_shift_{args.use}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {fpath}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    # ---- save the per-angle shifts (feed a re-registration if wanted) -----
    npz = os.path.join(out_dir, f"residual_shift_{args.use}.npz")
    np.savez_compressed(npz, theta=theta_recon, dx=dx, dy=dy, corr=cc,
                        sigma_x=sx, sigma_y=sy_,
                        jitter_x=jit_x, jitter_y=jit_y,
                        random_x=rnd_x, random_y=rnd_y,
                        geom_amp_x=amp_x, geom_amp_y=amp_y,
                        geom_fit_x=fit_x, geom_fit_y=fit_y)
    print(f"Saved: {npz}")


# ---------------------------------------------------------------------------
# Self-test: inject known per-angle shifts, recover them.
# ---------------------------------------------------------------------------
def selftest():
    from scipy.ndimage import fourier_shift
    print("estimate_residual_shift self-test (registration loop)")
    rng = np.random.default_rng(0)
    N_ang, Ny, Nx = 40, 64, 64
    # a feature-rich "model" projection stack, with comparable texture in BOTH
    # axes so x and y register equally well.
    yy, xx = np.mgrid[0:Ny, 0:Nx].astype(float)
    base = (np.sin(xx / 5.0) * np.sin(yy / 4.0)
            + 0.7 * np.cos(xx / 3.0 + yy / 6.0)
            + np.exp(-((xx - 0.4*Nx)**2 + (yy - 0.6*Ny)**2) / (2*7.0**2))
            + 0.8 * np.exp(-((xx - 0.7*Nx)**2 + (yy - 0.3*Ny)**2) / (2*5.0**2)))
    model = np.stack([base * (1.0 + 0.1*np.sin(a/3.0)) for a in range(N_ang)])

    sigma_true = 0.5
    dx_true = rng.normal(0, sigma_true, N_ang)
    dy_true = rng.normal(0, sigma_true, N_ang)
    measured = np.empty_like(model)
    for a in range(N_ang):
        ft = np.fft.fft2(model[a])
        measured[a] = np.fft.ifft2(fourier_shift(ft, (dy_true[a], dx_true[a]))).real

    dy, dx, cc = estimate_shifts(measured, model, upsample=50)
    # phase_cross_correlation(ref=measured, mov=model) returns the shift moving
    # `model` onto `measured`, i.e. it should recover (dy_true, dx_true).
    err_x = np.std(dx - dx_true)
    err_y = np.std(dy - dy_true)
    rec_sx = np.std(dx - dx.mean())
    sample_sx = np.std(dx_true - dx_true.mean())   # sample std actually drawn
    # per-shift error ~0.05-0.07 px is registration noise at this tiny size;
    # the real measurement targets a ~0.3-0.5 px signal, so this is plenty.
    ok_recover = err_x < 0.08 and err_y < 0.08
    ok_sigma = abs(rec_sx - sample_sx) < 0.05      # vs SAMPLE std, not population
    ok_corr = np.median(cc) > 0.9

    # trend-immune estimator: on pure white jitter it must match the sample std
    rnd = _angle_to_angle_jitter(np.arange(N_ang, dtype=float), dx)
    ok_rnd = abs(rnd - sample_sx) < 0.12
    print(f"  trend-immune random sx : {rnd:.3f}  (sample {sample_sx:.3f})")

    print(f"  recovered dx err (std) : {err_x:.4f} px   (<0.08 ok)")
    print(f"  recovered dy err (std) : {err_y:.4f} px   (<0.08 ok)")
    print(f"  recovered sigma_x      : {rec_sx:.3f}  "
          f"(sample {sample_sx:.3f}, population {sigma_true})")
    print(f"  median corr            : {np.median(cc):.3f}")
    ok = ok_recover and ok_sigma and ok_corr and ok_rnd
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main()
