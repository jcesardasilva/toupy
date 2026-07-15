#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect the TV sweep: is the two-pass gain physics, or TV denoising?
===================================================================

Reads the per-LAMBDA_TV ``fsc_data_normfreq.npz`` files written by
``slurm_tv_sweep.sh`` and answers two questions.

1. THE DECISIVE ONE (LAMBDA_TV = 0).  FSC is blind to smoothing -- a
   deterministic filter cancels in the normalised correlation -- so an FSC gain
   measured with TV on cannot, by itself, be attributed to the multislice
   physics.  At TV = 0 there is no smoothing to hide behind:

     * two-pass still beats FBP  -> the gain is physics.  Report the TV=0 number.
     * the gain collapses        -> the "two-pass gain" is a TV-denoising gain:
                                    a different, weaker claim.

2. THE SHAPE of two-pass resolution vs LAMBDA_TV.  If it keeps improving as TV
   grows, FSC is being gamed by smoothing and cannot justify the TV choice.  It
   must turn over eventually (at extreme TV the volume is a blob); where it
   turns over is the honest optimum.

The FBP curve is the built-in control: FBP does not depend on LAMBDA_TV, so its
FSC must be flat across the sweep.  If it is not, the scatter you see there is
run-to-run noise, and any two-pass difference smaller than that is not real.

Usage
-----
  python collect_tv_sweep.py --root /path/to/scratch
  python collect_tv_sweep.py --point 0=/path/fsc_tv0/fsc_data_normfreq.npz ...
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TV_VALUES = [0.0, 1e-6, 1e-5, 5e-5, 5e-4]


def _fmt_tv(tv):
    """Match the shell's formatting of the TV value in paths (e.g. '0', '1e-6')."""
    if tv == 0:
        return "0"
    s = f"{tv:g}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd(),
                    help="directory holding the fsc_tv<TV>/ folders")
    ap.add_argument("--point", action="append", default=[], metavar="TV=PATH",
                    help="explicit 'tv=path/to/fsc_data_normfreq.npz' "
                         "(repeatable); bypasses the folder convention")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or args.root

    if args.point:
        pairs = []
        for p in args.point:
            if "=" not in p:
                raise SystemExit(f"--point must be 'tv=path', got {p!r}")
            k, v = p.split("=", 1)
            pairs.append((float(k), os.path.abspath(os.path.expanduser(v))))
    else:
        pairs = [(tv, os.path.join(args.root, f"fsc_tv{_fmt_tv(tv)}",
                                   "fsc_data_normfreq.npz"))
                 for tv in TV_VALUES]

    rows = []
    for tv, path in pairs:
        if not os.path.isfile(path):
            print(f"  [skip] TV={tv:g}: not found -> {path}")
            continue
        d = np.load(path, allow_pickle=True)
        rows.append((tv, float(d["res_hb_fbp"]), float(d["res_hb_tp"])))
        print(f"  [ok]   TV={tv:<8g} FBP {rows[-1][1]:6.1f} nm   "
              f"two-pass {rows[-1][2]:6.1f} nm")

    if not rows:
        raise SystemExit("No FSC results found. Run slurm_tv_sweep.sh first.")
    rows.sort(key=lambda r: r[0])
    tvs = np.array([r[0] for r in rows])
    fbp = np.array([r[1] for r in rows])
    tp = np.array([r[2] for r in rows])
    gain = 100.0 * (fbp - tp) / fbp

    print("\n" + "=" * 66)
    print("TV sweep — two-pass gain vs LAMBDA_TV (1/2-bit resolution, nm)")
    print("=" * 66)
    print(f"  {'LAMBDA_TV':>10} {'FBP':>8} {'two-pass':>9} {'gain':>8}")
    for tv, f, t, g in zip(tvs, fbp, tp, gain):
        print(f"  {tv:>10g} {f:>8.1f} {t:>9.1f} {g:>+7.1f} %")

    # --- control: FBP must not depend on TV -------------------------------
    fbp_spread = float(fbp.max() - fbp.min())
    print(f"\n  Control — FBP does not depend on LAMBDA_TV, so its spread is "
          f"pure\n  run-to-run scatter: {fbp_spread:.1f} nm "
          f"({100*fbp_spread/fbp.mean():.1f} %). Any two-pass difference "
          f"smaller\n  than this is not meaningful.")

    # --- the decisive point ------------------------------------------------
    print("\nVerdict:")
    if 0.0 in set(tvs):
        i0 = int(np.where(tvs == 0.0)[0][0])
        g0 = gain[i0]
        print(f"  LAMBDA_TV = 0 (no smoothing at all): two-pass "
              f"{g0:+.1f} % vs FBP "
              f"({fbp[i0]:.1f} -> {tp[i0]:.1f} nm)")
        if g0 > 5.0 and g0 > 100 * fbp_spread / fbp.mean():
            print("  => THE GAIN IS PHYSICS. It survives with TV switched off,\n"
                  "     so it cannot be a denoising artefact. Report the TV=0\n"
                  "     number as the honest two-pass result.")
        elif g0 > 0:
            print("  => WEAK/AMBIGUOUS at TV=0: the gain is within (or close to)\n"
                  "     the FBP run-to-run scatter. The published gain is then\n"
                  "     mostly TV denoising, not the multislice model.")
        else:
            print("  => THE GAIN DOES NOT SURVIVE TV=0. The 'two-pass gain' is a\n"
                  "     TV-denoising gain. Do not attribute it to the multislice\n"
                  "     physics without a further control.")
    else:
        print("  LAMBDA_TV = 0 missing — that is the decisive point. Run it.")

    # --- is FSC being gamed by smoothing? ---------------------------------
    if len(tvs) >= 3:
        best = int(np.argmin(tp))
        if best == len(tvs) - 1:
            print(f"\n  WARNING: two-pass resolution is still improving at the\n"
                  f"  largest TV tried ({tvs[-1]:g}). FSC is being gamed by\n"
                  f"  smoothing -- it cannot justify the TV choice. Extend the\n"
                  f"  sweep until it turns over.")
        else:
            print(f"\n  two-pass optimum at LAMBDA_TV = {tvs[best]:g} "
                  f"({tp[best]:.1f} nm); the curve turns over, so FSC is not\n"
                  f"  simply rewarding smoothing.")

    # --- plot ---------------------------------------------------------------
    x = np.arange(len(tvs))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, fbp, "s--", color="C0", label="FBP (TV-independent control)")
    ax[0].plot(x, tp, "o-", color="C1", lw=2, label="two-pass")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{t:g}" for t in tvs])
    ax[0].set_xlabel(r"$\lambda_{TV}$")
    ax[0].set_ylabel("½-bit resolution [nm]  (smaller = better)")
    ax[0].invert_yaxis(); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_title("Resolution vs TV")

    ax[1].bar(x, gain, color=["C2" if g > 0 else "C3" for g in gain])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{t:g}" for t in tvs])
    ax[1].set_xlabel(r"$\lambda_{TV}$"); ax[1].set_ylabel("two-pass gain [%]")
    ax[1].grid(alpha=0.3, axis="y")
    ax[1].set_title("Gain vs FBP  (TV=0 is the decisive point)")
    fig.tight_layout()
    p = os.path.join(out_dir, "tv_sweep.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
