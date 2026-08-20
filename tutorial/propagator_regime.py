#!/usr/bin/env python
"""
propagator_regime.py — beamtime-planning calculator: for a given wavelength,
voxel size and sample thickness, decide which propagation model you need.

It answers two DIFFERENT thickness questions that are often conflated:

  (1) Do I need the two-pass multislice at all (vs plain FBP)?
      -> the PROJECTION approximation (straight rays) breaks when a ray
         diffracts sideways by more than one voxel over the sample depth:
             lambda * f_Nyq * T  >  pixel     <=>     T > T_proj = 2*pixel^2/lambda
         Onset scales as (lambda*f)^2 * T.

  (2) Once I use multislice, does the propagator choice matter
      (exact angular-spectrum vs paraxial Fresnel)?
      -> the PARAXIAL approximation breaks when the accumulated wide-angle
         phase error over the thickness becomes appreciable:
             dphi(T) = (2*pi*T/lambda) * [ sqrt(1-s) - 1 + s/2 ],   s=(lambda*f_Nyq)^2
         Onset scales as (lambda*f)^4 * T  — one factor of s = (lambda*f)^2
         WEAKER than (1).  So the same thickness that turns on the two-pass
         gain does almost nothing to separate Fresnel from ASM.

The critical thickness for the propagator difference scales as
    T_crit(ASM) ∝ pixel^4 / lambda^3
so PIXEL SIZE (numerical aperture), not thickness, is the real lever: halving
the voxel cuts T_crit by 16x.

Usage
-----
    python propagator_regime.py --lambda-nm 0.20 --pixel-nm 28.6 --thickness-um 14
    python propagator_regime.py --energy-kev 12.4 --pixel-nm 5 --thickness-um 30
    python propagator_regime.py ... --table      # show T_crit vs a range of voxels
"""

import argparse
import math


def _lambda_from_energy_kev(e_kev):
    # lambda[nm] = h c / E ; hc = 1.23984193 keV*nm
    return 1.23984193 / e_kev


def analyse(lambda_m, pixel_m, thickness_m):
    """Return a dict of regime quantities (all SI unless noted)."""
    f_nyq = 1.0 / (2.0 * pixel_m)             # cycles / m
    theta_max = lambda_m * f_nyq              # rad (paraxial: sin ~ theta)
    s = theta_max ** 2                        # (lambda*f)^2 at Nyquist

    # (1) projection-approximation onset thickness (two-pass vs FBP)
    T_proj = 2.0 * pixel_m ** 2 / lambda_m

    # (2) exact angular-spectrum vs Fresnel accumulated phase at Nyquist
    #     bracket = sqrt(1-s) - 1 + s/2  (>=0; ~ s^2/8 for small s)
    bracket = math.sqrt(max(0.0, 1.0 - s)) - 1.0 + 0.5 * s
    bracket = abs(bracket)
    dphi = (2.0 * math.pi * thickness_m / lambda_m) * bracket   # rad, at Nyquist

    def T_for_phase(phi):
        if bracket <= 0:
            return math.inf
        return phi * lambda_m / (2.0 * math.pi * bracket)

    return dict(
        f_nyq=f_nyq, theta_max=theta_max, s=s,
        T_proj=T_proj, bracket=bracket, dphi=dphi,
        T_crit_1rad=T_for_phase(1.0),
        T_crit_0p1rad=T_for_phase(0.1),
    )


def _fmt_len(m):
    """Human length with sensible unit."""
    if not math.isfinite(m):
        return "inf"
    a = abs(m)
    if a >= 1.0:
        return f"{m:.3g} m"
    if a >= 1e-3:
        return f"{m*1e3:.3g} mm"
    if a >= 1e-6:
        return f"{m*1e6:.3g} um"
    return f"{m*1e9:.3g} nm"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lambda-nm", type=float, help="X-ray wavelength [nm]")
    g.add_argument("--energy-kev", type=float, help="X-ray photon energy [keV]")
    ap.add_argument("--pixel-nm", type=float, required=True,
                    help="voxel / pixel size [nm]")
    ap.add_argument("--thickness-um", type=float, required=True,
                    help="sample thickness along the beam [um]")
    ap.add_argument("--table", action="store_true",
                    help="also tabulate the ASM critical thickness vs voxel size")
    args = ap.parse_args()

    lam_nm = args.lambda_nm if args.lambda_nm else _lambda_from_energy_kev(args.energy_kev)
    lam = lam_nm * 1e-9
    px = args.pixel_nm * 1e-9
    T = args.thickness_um * 1e-6

    r = analyse(lam, px, T)

    print("=" * 68)
    print("Propagator regime — planning calculator")
    print("=" * 68)
    e_kev = _lambda_from_energy_kev(1.0) / lam_nm  # hc/lambda -> keV
    print(f"  wavelength      = {lam_nm:.4f} nm   ({e_kev:.2f} keV)")
    print(f"  voxel           = {args.pixel_nm:.3f} nm   "
          f"(Nyquist resolution 2*pixel = {2*args.pixel_nm:.1f} nm)")
    print(f"  sample thickness= {args.thickness_um:.3f} um")
    print(f"  Nyquist half-angle  theta_max = lambda/(2*pixel) = "
          f"{r['theta_max']*1e3:.3f} mrad")
    print(f"  paraxial parameter  (lambda*f)^2_max = {r['s']:.3e}")

    # --- (1) projection approximation: two-pass vs FBP --------------------
    ratio_proj = T / r["T_proj"]
    print("\n  (1) PROJECTION approximation  (two-pass multislice vs plain FBP)")
    print(f"      onset thickness T_proj = 2*pixel^2/lambda = {_fmt_len(r['T_proj'])}")
    print(f"      T / T_proj = {ratio_proj:.2f}")
    if ratio_proj < 0.3:
        print("      -> thin: straight-ray FBP is adequate; two-pass ~ FBP.")
    elif ratio_proj < 1.0:
        print("      -> approaching the limit: a modest two-pass gain is expected.")
    else:
        print("      -> thick: FBP blurs along the beam; the two-pass multislice")
        print("         forward model is needed (this is where its gain comes from).")

    # --- (2) paraxial approximation: ASM vs Fresnel ----------------------
    print("\n  (2) PARAXIAL approximation  (exact angular-spectrum vs Fresnel)")
    print(f"      accumulated ASM-vs-Fresnel phase at Nyquist over T:")
    print(f"          dphi(T) = {r['dphi']:.3e} rad  ({r['dphi']*1e6:.3g} urad)")
    print(f"      critical thickness for dphi = 0.1 rad (detectable): "
          f"{_fmt_len(r['T_crit_0p1rad'])}")
    print(f"      critical thickness for dphi = 1.0 rad (large)     : "
          f"{_fmt_len(r['T_crit_1rad'])}")

    # --- verdict ----------------------------------------------------------
    print("\n  " + "-" * 64)
    if r["dphi"] < 0.1:
        print("  VERDICT: Fresnel and ASM are INDISTINGUISHABLE for this sample.")
        print("           Use the default 'fresnel' propagator. To reach even a")
        print(f"           0.1 rad difference you would need T ~ "
              f"{_fmt_len(r['T_crit_0p1rad'])}")
        headroom = r["T_crit_0p1rad"] / T if T > 0 else math.inf
        print(f"           ({headroom:.0f}x thicker than this sample).")
    elif r["dphi"] < 1.0:
        print("  VERDICT: ASM difference is becoming DETECTABLE (0.1-1 rad at")
        print("           Nyquist). Worth running both and comparing with")
        print("           compare_propagators.py; 'asm' is the physically exact one.")
    else:
        print("  VERDICT: ASM is NEEDED — the paraxial (Fresnel) approximation")
        print("           accumulates >~1 rad of wide-angle phase error over this")
        print("           thickness. Use propagator='asm' (and check that the")
        print("           thin-slice/transmission approximations still hold too).")
    print("  " + "-" * 64)

    # --- sensitivity: the pixel^4 lever ----------------------------------
    print(f"\n  Note: T_crit(ASM) ∝ pixel^4 / lambda^3 — pixel size, not thickness,")
    print(f"        is the real lever. Halving the voxel cuts T_crit by ~16x.")

    if args.table:
        print("\n  ASM critical thickness (dphi=1 rad) vs voxel size "
              f"(lambda fixed at {lam_nm:.3f} nm):")
        print(f"    {'voxel [nm]':>12} {'(lambda*f)^2':>14} {'T_crit(1rad)':>16}")
        for pnm in (50, 28.6, 20, 10, 5, 2, 1):
            rr = analyse(lam, pnm * 1e-9, T)
            print(f"    {pnm:>12.1f} {rr['s']:>14.2e} {_fmt_len(rr['T_crit_1rad']):>16}")


if __name__ == "__main__":
    main()
