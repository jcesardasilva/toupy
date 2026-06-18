#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect the alignment-jitter sweep into one summary: 1/2-bit resolution vs sigma.
=================================================================================

Reads the per-jitter ``fsc_data_normfreq.npz`` files produced by
``fsc_analysis_normfreq.py`` and plots two-pass (and FBP) 1/2-bit resolution
against the jitter amplitude sigma, answering: is residual mis-alignment
limiting the resolution?

  * Steep rise of the two-pass curve with sigma  -> alignment-limited
    (better registration, e.g. the NUFFT re-projection, would help).
  * Flat curve up to ~1 px                        -> NOT alignment-limited
    (dose / phase-retrieval floor; alignment won't help).

Expected inputs (the default layout from perturb_alignment.py +
fsc_analysis_normfreq.py):
  sigma=0  (reference) : twopass_real_figures_fsc_normfreq/fsc_data_normfreq.npz
  sigma>0              : twopass_real_figures_jitter<sigma>px_<axes>_half0/
                             fsc_normfreq/fsc_data_normfreq.npz

Usage
-----
  python tutorial/collect_jitter_fsc.py
  python tutorial/collect_jitter_fsc.py --root /path/to/tutorial --axes x
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGMA_LIST = [0.0, 0.25, 0.5, 1.0]     # px;  0.0 = unperturbed reference
JITTER_AXES = "x"


def _ref_path(root):
    return os.path.join(root, "twopass_real_figures_fsc_normfreq",
                        "fsc_data_normfreq.npz")


def _jitter_path(root, sigma, axes):
    return os.path.join(
        root, f"twopass_real_figures_jitter{sigma:.2f}px_{axes}_half0",
        "fsc_normfreq", "fsc_data_normfreq.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory containing the twopass_real_figures* folders")
    ap.add_argument("--axes", default=JITTER_AXES, help="jitter axis tag (x|y|xy)")
    ap.add_argument("--out", default=None, help="output dir (default: <root>)")
    args = ap.parse_args()
    out_dir = args.out or args.root

    rows = []   # (sigma, fbp_hb, tp_hb, fbp_ob, tp_ob)
    for s in SIGMA_LIST:
        path = _ref_path(args.root) if s == 0.0 else _jitter_path(args.root, s, args.axes)
        if not os.path.isfile(path):
            print(f"  [skip] sigma={s:.2f}: not found -> {path}")
            continue
        d = np.load(path, allow_pickle=True)
        rows.append((s,
                     float(d["res_hb_fbp"]), float(d["res_hb_tp"]),
                     float(d["res_ob_fbp"]), float(d["res_ob_tp"])))
        print(f"  [ok]   sigma={s:.2f}: TP 1/2-bit = {float(d['res_hb_tp']):.1f} nm")

    if len(rows) < 2:
        raise SystemExit("Need at least the sigma=0 reference and one jittered "
                         "point.  Run the reconstructions + fsc_analysis_normfreq "
                         "first (see perturb_alignment.py).")

    rows.sort(key=lambda r: r[0])
    sig = np.array([r[0] for r in rows])
    fbp_hb = np.array([r[1] for r in rows]); tp_hb = np.array([r[2] for r in rows])
    fbp_ob = np.array([r[3] for r in rows]); tp_ob = np.array([r[4] for r in rows])

    # ---- table + verdict ---------------------------------------------------
    base_tp = tp_hb[0]   # sigma = 0
    print("\n" + "=" * 60)
    print("Alignment-jitter sensitivity (1/2-bit resolution, nm)")
    print("=" * 60)
    print(f"  {'sigma[px]':>9}  {'FBP':>8}  {'Two-pass':>9}  {'TP Δ vs σ=0':>14}")
    for i in range(len(sig)):
        d_tp = tp_hb[i] - base_tp
        pct = 100.0 * d_tp / base_tp
        print(f"  {sig[i]:>9.2f}  {fbp_hb[i]:>8.1f}  {tp_hb[i]:>9.1f}  "
              f"{d_tp:>+7.1f} nm ({pct:>+5.1f}%)")

    # sensitivity at sigma ~ 0.5 px (typical residual scale) if present
    idx05 = np.argmin(np.abs(sig - 0.5))
    deg = 100.0 * (tp_hb[idx05] - base_tp) / base_tp if sig[idx05] > 0 else 0.0
    print("\nVerdict (heuristic):")
    print(f"  Two-pass 1/2-bit degrades {deg:+.1f}% at sigma={sig[idx05]:.2f} px.")
    if abs(deg) >= 5.0:
        print("  => ALIGNMENT-SENSITIVE: a residual misalignment of this scale "
              "noticeably\n     worsens the FSC, so better registration could "
              "buy resolution.")
    else:
        print("  => NOT alignment-limited at this scale: the FSC barely moves, "
              "so the\n     ceiling is dose / phase-retrieval, not alignment.")
    print("  (Compare 'deg' to your run-to-run FSC scatter; treat <~3% as noise.)")

    # ---- plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sig, tp_hb, "o-", color="C1", lw=2, label="Two-pass (½-bit)")
    ax.plot(sig, fbp_hb, "s--", color="C0", lw=1.5, label="FBP (½-bit)")
    ax.set_xlabel("Added alignment jitter  σ  [pixels]", fontsize=12)
    ax.set_ylabel("½-bit resolution [nm]  (smaller = better)", fontsize=12)
    ax.set_title("Alignment-sensitivity test", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.invert_yaxis()   # better resolution (smaller nm) plotted upward
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"jitter_sensitivity_{args.axes}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_path}")


if __name__ == "__main__":
    main()
