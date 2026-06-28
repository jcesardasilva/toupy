#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver: align truncated+jittered local-tomo data, then write aligned set
========================================================================

End-to-end stage-3 pipeline on a real testbed file:

  local_tomo_simulator.py --fov-frac-x 0.2 --jitter-px 0.5
        │  (writes PXCTalignedprojections_localfov20_jitter0.50x.npz)
        ▼
  local_tomo_align_run.py   ← THIS SCRIPT
        │  align-by-synthesis (halo) → per-angle shift estimate
        │  validate against the KNOWN injected jitter (jitter_dx/dy)
        │  apply the correction → write *_aligned.npz
        ▼
  twopass_local_data.py  (point DATA_FILE at the *_aligned.npz)

What it does
------------
1. Loads a jittered truncated projection set (+ its fov_* metadata, and the
   known jitter_dx/jitter_dy if present).
2. Runs alignment-by-synthesis with the IRR halo (and optionally the FOV-only
   baseline for comparison) from tutorial/local_tomo_align.py.
3. If the true jitter is in the file, reports how well the random part was
   recovered (rigid recentring/translation modes are coordinate ambiguities and
   are excluded from the score) and the residual jitter before/after.
4. Applies the estimated correction and saves an aligned .npz with the SAME
   layout as the simulator output, so twopass_local_data.py can reconstruct it.

Usage
-----
  python tutorial/local_tomo_align_run.py \\
      --data tutorial/PXCTalignedprojections_localfov20_jitter0.50x.npz
  python tutorial/local_tomo_align_run.py --data <...> --compare   # + FOV-only
  python tutorial/local_tomo_align_run.py --selftest
"""

import os
import sys
import argparse
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, fpath):
    spec = importlib.util.spec_from_file_location(name, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


align_mod = _load_module("local_tomo_align",
                         os.path.join(_HERE, "local_tomo_align.py"))
align_by_synthesis  = align_mod.align_by_synthesis
apply_shifts        = align_mod.apply_shifts
_remove_rigid_modes = align_mod._remove_rigid_modes


def _fov_key(d, name):
    for k in (name, "fov_" + name):
        if k in (d.files if hasattr(d, "files") else d):
            return int(d[k])
    raise KeyError(f"FOV key '{name}' not in npz (have {list(d.files)})")


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------
def run(data_file, n_outer=6, halo_n_iter=4, upsample=50, damp=0.7,
        compare=False, out_file=None, n_jobs=None, make_figs=True):
    data = np.load(data_file, allow_pickle=True)
    proj  = data["projections"].astype(np.float32)        # (n_ang, ny, nx)
    theta = data["theta"].astype(np.float64)
    wavelen = float(data["wavelen"]); psize = float(data["psize"])
    k0 = 2.0 * np.pi / wavelen

    x0 = _fov_key(data, "fov_x0"); x1 = _fov_key(data, "fov_x1")
    full_Nx = _fov_key(data, "full_Nx")

    have_truth = ("jitter_dx" in data.files)
    true_dx = data["jitter_dx"].astype(np.float64) if have_truth else None
    true_dy = data["jitter_dy"].astype(np.float64) if have_truth else None

    print("=" * 66)
    print("Local-tomo alignment-by-synthesis")
    print("=" * 66)
    print(f"  data    : {data_file}")
    print(f"  proj    : {proj.shape}   FOV x[{x0}:{x1}] of {full_Nx}")
    print(f"  truth   : {'present' if have_truth else 'absent'}")
    print()

    out = {}
    print("  [halo] aligning …", flush=True)
    dyh, dxh, hist_h = align_by_synthesis(
        proj, theta, x0, x1, full_Nx, k0, psize, n_outer=n_outer,
        halo=True, halo_n_iter=halo_n_iter, upsample=upsample, damp=damp,
        n_jobs=n_jobs)
    out["halo"] = (dyh, dxh, hist_h)

    if compare:
        print("  [fov ] aligning (baseline) …", flush=True)
        dyf, dxf, hist_f = align_by_synthesis(
            proj, theta, x0, x1, full_Nx, k0, psize, n_outer=n_outer,
            halo=False, upsample=upsample, damp=damp, n_jobs=n_jobs)
        out["fov"] = (dyf, dxf, hist_f)

    # ── validation against known jitter ───────────────────────────────────
    if have_truth:
        # injected jitter applied as shift(+dx); ideal correction = -dx.
        # Score only the random part (rigid modes are coordinate ambiguities).
        tr = _remove_rigid_modes(theta, true_dx)
        rec_h = -_remove_rigid_modes(theta, dxh)          # recovered jitter
        corr_h = float(np.corrcoef(rec_h, tr)[0, 1])
        res_before = float(np.std(tr))
        res_after_h = float(np.std(tr + _remove_rigid_modes(theta, dxh)))
        print()
        print("  Validation (random part of x-jitter):")
        print(f"    input residual jitter sigma : {res_before:.3f} px")
        print(f"    HALO  recovered corr={corr_h:+.3f}  "
              f"residual after={res_after_h:.3f} px")
        if compare:
            rec_f = -_remove_rigid_modes(theta, dxf)
            corr_f = float(np.corrcoef(rec_f, tr)[0, 1])
            res_after_f = float(np.std(tr + _remove_rigid_modes(theta, dxf)))
            print(f"    FOV   recovered corr={corr_f:+.3f}  "
                  f"residual after={res_after_f:.3f} px")

    # ── write aligned projections (halo result) ───────────────────────────
    aligned = apply_shifts(proj, dyh, dxh)
    out_file = out_file or data_file.replace(".npz", "_aligned.npz")
    save = {k: data[k] for k in data.files if k != "projections"}
    save["projections"] = aligned.astype(np.float32)
    save["align_dx"] = dxh
    save["align_dy"] = dyh
    np.savez_compressed(out_file, **save)
    print(f"\n  aligned set written: {out_file}")
    print(f"  → reconstruct with twopass_local_data.py (DATA_FILE = above)")

    # ── figures ───────────────────────────────────────────────────────────
    if make_figs:
        fig_dir = os.path.dirname(os.path.abspath(out_file))
        _figures(theta, out, true_dx if have_truth else None, fig_dir)

    return out


def _figures(theta, out, true_dx, fig_dir):
    dyh, dxh, hist_h = out["halo"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2),
                             gridspec_kw={"wspace": 0.30})
    # (a) recovered jitter per angle
    rec_h = -_remove_rigid_modes(theta, dxh)
    order = np.argsort(theta)
    axes[0].plot(theta[order], rec_h[order], "b.-", ms=3, lw=0.8,
                 label="recovered (halo)")
    if true_dx is not None:
        tr = _remove_rigid_modes(theta, true_dx)
        axes[0].plot(theta[order], tr[order], "k.-", ms=3, lw=0.8,
                     label="true jitter")
    if "fov" in out:
        rec_f = -_remove_rigid_modes(theta, out["fov"][1])
        axes[0].plot(theta[order], rec_f[order], "r.", ms=3, alpha=0.5,
                     label="recovered (FOV-only)")
    axes[0].set_xlabel("θ [deg]"); axes[0].set_ylabel("x-jitter [px]")
    axes[0].set_title("Recovered vs true per-angle jitter", fontsize=9)
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    # (b) recovered vs true scatter, or convergence if no truth
    if true_dx is not None:
        tr = _remove_rigid_modes(theta, true_dx)
        lim = max(np.abs(tr).max(), np.abs(rec_h).max()) * 1.1
        axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="ideal")
        axes[1].scatter(tr, rec_h, s=10, c="b", label=f"halo r={np.corrcoef(rec_h,tr)[0,1]:.3f}")
        if "fov" in out:
            rec_f = -_remove_rigid_modes(theta, out["fov"][1])
            axes[1].scatter(tr, rec_f, s=10, c="r", alpha=0.5,
                            label=f"FOV r={np.corrcoef(rec_f,tr)[0,1]:.3f}")
        axes[1].set_xlabel("true jitter [px]"); axes[1].set_ylabel("recovered [px]")
        axes[1].set_title("Recovered vs true", fontsize=9)
        axes[1].set_aspect("equal"); axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].plot(range(1, len(hist_h) + 1), hist_h, "b.-")
        axes[1].set_xlabel("outer iteration"); axes[1].set_ylabel("residual dx RMS [px]")
        axes[1].set_title("Convergence", fontsize=9); axes[1].grid(True, alpha=0.3)

    p = os.path.join(fig_dir, "fig_align.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {p}")


# ---------------------------------------------------------------------------
# Self-test: synthesise a tiny jittered npz, run the driver end-to-end
# ---------------------------------------------------------------------------
def _selftest():
    print("local_tomo_align_run self-test")
    try:
        from skimage.transform import radon
    except ImportError:
        print("  needs scikit-image."); return False

    rng = np.random.default_rng(2)
    N, NY = 96, 32
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    r = np.sqrt((xx - N / 2) ** 2 + (yy - N / 2) ** 2); sup = r < 0.46 * N
    vol = np.zeros((N, NY, N))
    for iy in range(NY):
        sl = np.where(sup, 1.0, 0.0)
        for _ in range(6):
            bx, by = rng.uniform(0.28, 0.72, 2) * N
            sl[((xx - bx) ** 2 + (yy - by) ** 2) < (0.06 * N) ** 2] += 0.6
        vol[:, iy, :] = sl * sup
    theta = np.linspace(0, 180, 120, endpoint=False); n = len(theta)
    w = int(0.40 * N); x0 = (N - w) // 2; x1 = x0 + w
    wavelen, psize = 2e-10, 2.86e-8; k0 = 2 * np.pi / wavelen
    clean = np.empty((n, NY, w), dtype=np.float32)
    for iy in range(NY):
        s = radon(vol[:, iy, :], theta=theta, circle=True)
        clean[:, iy, :] = (-k0 * psize) * s[x0:x1, :].T
    true_dx = rng.normal(0, 0.5, n)
    jit = apply_shifts(clean, np.zeros(n), true_dx)

    scratch = os.environ.get("TMPDIR", "/tmp")
    fpath = os.path.join(scratch, "_lta_selftest.npz")
    np.savez_compressed(fpath, projections=jit, theta=theta, wavelen=wavelen,
                        psize=psize, fov_x0=x0, fov_x1=x1, fov_y0=0, fov_y1=NY,
                        full_Nx=N, full_Ny=NY, jitter_dx=true_dx,
                        jitter_dy=np.zeros(n))
    out = run(fpath, n_outer=6, halo_n_iter=4, n_jobs=1, make_figs=False)
    tr = _remove_rigid_modes(theta, true_dx)
    rec = -_remove_rigid_modes(theta, out["halo"][1])
    corr = float(np.corrcoef(rec, tr)[0, 1])
    wrote = os.path.isfile(fpath.replace(".npz", "_aligned.npz"))
    checks = {"recovered corr>0.85": corr > 0.85,
              "aligned file written": wrote}
    for k, v in checks.items():
        print(f"  {k:24s}: {v}")
    ok = all(checks.values())
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="jittered truncated .npz from local_tomo_simulator")
    ap.add_argument("--n-outer", type=int, default=6)
    ap.add_argument("--halo-n-iter", type=int, default=4)
    ap.add_argument("--upsample", type=int, default=50)
    ap.add_argument("--damp", type=float, default=0.7)
    ap.add_argument("--compare", action="store_true",
                    help="also run the FOV-only baseline for comparison")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not args.data:
        ap.error("--data is required (or use --selftest)")
    _slurm = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None
    run(args.data, n_outer=args.n_outer, halo_n_iter=args.halo_n_iter,
        upsample=args.upsample, damp=args.damp, compare=args.compare,
        out_file=args.out, n_jobs=_slurm)


if __name__ == "__main__":
    main()
