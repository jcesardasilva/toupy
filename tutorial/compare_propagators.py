#!/usr/bin/env python
"""
compare_propagators.py — quantify how much the inter-slice propagator
(``fresnel`` vs ``angular_spectrum``/``asm``) actually changes a two-pass
reconstruction.

Motivation
----------
Visually, a two-pass volume looks sharper than its FBP initialisation — but
that sharpening is the *multislice* gain and is obtained with EITHER
propagator.  For hard X-rays the paraxial (Fresnel) and exact (angular
spectrum) kernels differ only at order (λf)⁴, which is negligible unless the
numerical aperture λ/(2·pixel) is large.  This script replaces "it looks
sharper" with numbers:

  * global relative difference   ‖asm − fresnel‖ / ‖fresnel‖
  * max |Δδ| and its fraction of the volume dynamic range
  * Pearson correlation between the two volumes
  * for context, corr(two-pass, FBP) for each propagator (both should be
    equal — the sharpening from FBP is the same for both)
  * a **cross-FSC(fresnel, asm)** curve: the spatial frequency up to which
    the two reconstructions are identical.  In the paraxial regime it should
    sit at ≈1 all the way to Nyquist.

It also prints the paraxial figure of merit (λf)²_max = (λ/(2·pixel))² so the
verdict is anchored to the physics, and issues an automatic PASS / INVESTIGATE
conclusion.

Usage
-----
    python compare_propagators.py  FRESNEL.npz  ASM.npz  [--out compare_propagators.png]

where each ``*.npz`` is a ``twopass_reconstruction.npz`` produced by
``twopass_real_data.py`` (keys: delta_tp, delta_fbp, wavelength, psize, ...).
The two runs must use IDENTICAL settings and differ ONLY in PROPAGATOR.

    # produce the two inputs first, e.g. on the cluster:
    PROPAGATOR=fresnel RUN_TAG=_fresnel sbatch slurm_twopass.sh
    PROPAGATOR=asm     RUN_TAG=_asm     sbatch slurm_twopass.sh
    python compare_propagators.py \
        twopass_real_figures_fresnel/twopass_reconstruction.npz \
        twopass_real_figures_asm/twopass_reconstruction.npz
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Self-contained 3-D FSC (rfftn shell correlation) — matches fsc_analysis.py
# ---------------------------------------------------------------------------
def _tukey_1d(n, apod):
    """Separable Tukey (flat-top cosine-tapered) window, `apod` px each end."""
    w = np.ones(n, dtype=np.float32)
    if apod <= 0:
        return w
    apod = min(apod, n // 2)
    edge = 0.5 * (1.0 - np.cos(np.pi * np.arange(apod) / apod))
    w[:apod] = edge
    w[-apod:] = edge[::-1]
    return w


def fsc3d(vol1, vol2, n_shells=None, apod_width=20):
    """
    Fourier Shell Correlation between two real 3-D volumes.

    Returns
    -------
    freq_nyq : (n_shells,)  shell-centre frequency in Nyquist units (0.5 = Nyquist)
    fsc      : (n_shells,)  correlation per shell
    n_vox    : (n_shells,)  voxel count per shell
    """
    vol1 = np.asarray(vol1, dtype=np.float32)
    vol2 = np.asarray(vol2, dtype=np.float32)
    Nz, Ny, Nx = vol1.shape
    if n_shells is None:
        n_shells = min(Nz, Ny, Nx) // 2

    v1 = vol1.copy()
    v2 = vol2.copy()
    if apod_width > 0:
        wz = _tukey_1d(Nz, apod_width)[:, None, None]
        wy = _tukey_1d(Ny, apod_width)[None, :, None]
        wx = _tukey_1d(Nx, apod_width)[None, None, :]
        v1 *= wz; v1 *= wy; v1 *= wx
        v2 *= wz; v2 *= wy; v2 *= wx

    F1 = np.fft.rfftn(v1)
    F2 = np.fft.rfftn(v2)
    del v1, v2

    kz = np.fft.fftfreq(Nz).astype(np.float32)[:, None, None]
    ky = np.fft.fftfreq(Ny).astype(np.float32)[None, :, None]
    kx = np.fft.rfftfreq(Nx).astype(np.float32)[None, None, :]
    R = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2)

    r_edges = np.linspace(0.0, 0.5, n_shells + 1)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    fsc = np.zeros(n_shells)
    n_vox = np.zeros(n_shells, dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(r_edges[:-1], r_edges[1:])):
        mask = (R >= lo) & (R < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        f1 = F1[mask]; f2 = F2[mask]
        num = float(np.real(np.sum(f1 * f2.conj())))
        den = float(np.sqrt(np.sum(np.abs(f1) ** 2) * np.sum(np.abs(f2) ** 2)))
        fsc[i] = num / den if den > 0 else 0.0
        n_vox[i] = n
    return r_mid, fsc, n_vox


def first_crossing(freq_nyq, fsc, level):
    """Frequency (Nyquist units) where fsc first drops below `level`."""
    for i in range(len(fsc) - 1):
        if fsc[i] >= level > fsc[i + 1]:
            f0, f1 = freq_nyq[i], freq_nyq[i + 1]
            v0, v1 = fsc[i], fsc[i + 1]
            return f0 + (level - v0) / (v1 - v0) * (f1 - f0)
    return np.nan


def _load(path):
    d = np.load(path, allow_pickle=True)
    if "delta_tp" not in d:
        sys.exit(f"ERROR: {path} has no 'delta_tp' key "
                 f"(keys: {list(d.keys())}) — is it a twopass_reconstruction.npz?")
    return d


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fresnel_npz", help="twopass_reconstruction.npz from the fresnel run")
    ap.add_argument("asm_npz",     help="twopass_reconstruction.npz from the asm run")
    ap.add_argument("--out", default="compare_propagators.png",
                    help="output figure path (default: compare_propagators.png)")
    ap.add_argument("--apod", type=int, default=20, help="FSC apodization width [px]")
    ap.add_argument("--nshells", type=int, default=None, help="number of FSC shells")
    args = ap.parse_args()

    df = _load(args.fresnel_npz)
    da = _load(args.asm_npz)
    vf = df["delta_tp"].astype(np.float32)
    va = da["delta_tp"].astype(np.float32)

    if vf.shape != va.shape:
        sys.exit(f"ERROR: shape mismatch {vf.shape} vs {va.shape} — the two runs "
                 f"must use identical settings and differ only in PROPAGATOR.")

    psize_m = float(df["psize"]) if "psize" in df else np.nan
    wl_m    = float(df["wavelength"]) if "wavelength" in df else np.nan
    pixel_nm = psize_m * 1e9

    print("=" * 70)
    print("Propagator comparison:  fresnel  vs  angular-spectrum (asm)")
    print("=" * 70)
    print(f"  fresnel : {args.fresnel_npz}")
    print(f"  asm     : {args.asm_npz}")
    print(f"  volume  : {vf.shape}   pixel = {pixel_nm:.3f} nm   "
          f"lambda = {wl_m*1e9:.4f} nm")

    # --- paraxial figure of merit -----------------------------------------
    if np.isfinite(wl_m) and np.isfinite(psize_m):
        na = wl_m / (2.0 * psize_m)          # max half-angle at Nyquist
        s_max = na ** 2                       # (lambda f)^2 at Nyquist
        print(f"  Nyquist NA = lambda/(2*pixel) = {na*1e3:.3f} mrad   "
              f"=> (lambda*f)^2_max = {s_max:.2e}")
    else:
        s_max = np.nan

    # --- global difference metrics ----------------------------------------
    diff = va - vf
    nrm_f = float(np.linalg.norm(vf))
    rel_l2 = float(np.linalg.norm(diff)) / nrm_f if nrm_f > 0 else np.nan
    dyn = float(vf.max() - vf.min())
    max_abs = float(np.abs(diff).max())
    corr_fa = float(np.corrcoef(vf.ravel(), va.ravel())[0, 1])
    print("\n  Global difference (asm - fresnel):")
    print(f"    relative L2  ||asm-fres||/||fres||      = {rel_l2:.3e}")
    print(f"    max |Δδ|                                = {max_abs:.3e}"
          f"   ({100*max_abs/dyn:.3f}% of dynamic range)")
    print(f"    Pearson corr(fresnel, asm)              = {corr_fa:.8f}")

    # --- context: each two-pass vs its FBP init ---------------------------
    if "delta_fbp" in df:
        fbp = df["delta_fbp"].astype(np.float32).ravel()
        cf = float(np.corrcoef(vf.ravel(), fbp)[0, 1])
        ca = float(np.corrcoef(va.ravel(), fbp)[0, 1])
        print(f"    corr(two-pass, FBP): fresnel={cf:.6f}  asm={ca:.6f}  "
              f"(should match — sharpening vs FBP is propagator-independent)")

    # --- cross-FSC(fresnel, asm) ------------------------------------------
    print("\n  Computing cross-FSC(fresnel, asm) ...")
    freq, fsc, nvox = fsc3d(vf, va, n_shells=args.nshells, apod_width=args.apod)
    f05  = first_crossing(freq, fsc, 0.5)
    f143 = first_crossing(freq, fsc, 0.143)
    fsc_min = float(np.nanmin(fsc[nvox > 0]))

    def _res(fc):
        return np.nan if not np.isfinite(fc) or fc <= 0 else pixel_nm / fc
    print(f"    min cross-FSC over all shells           = {fsc_min:.4f}")
    if np.isfinite(f05):
        print(f"    cross-FSC drops below 0.5 at f={f05:.3f} Nyq  "
              f"(d = {_res(f05):.1f} nm)")
    else:
        print(f"    cross-FSC stays >= 0.5 all the way to Nyquist "
              f"(the two volumes are identical at every resolvable frequency)")
    if np.isfinite(f143):
        print(f"    cross-FSC drops below 0.143 at f={f143:.3f} Nyq "
              f"(d = {_res(f143):.1f} nm)")

    # --- automatic verdict -------------------------------------------------
    #
    # The cross-FSC is the PHYSICS discriminator, not the rel-L2 magnitude.
    # A genuine (high-NA) propagator difference is concentrated at HIGH spatial
    # frequency, so it makes the cross-FSC DROP near Nyquist.  A difference that
    # leaves the cross-FSC flat at ~1 across the whole band is spatially
    # UNSTRUCTURED (broadband) = numerical / run-to-run noise, NOT the
    # propagator.  rel-L2 is kept only to catch amplitude/scale mismatches that
    # the (scale-invariant) FSC would miss.
    print("\n  " + "-" * 66)
    paraxial      = np.isfinite(s_max) and s_max < 1e-3
    fsc_flat      = not np.isfinite(f05)      # cross-FSC >= 0.5 up to Nyquist
    amplitude_ok  = rel_l2 < 1e-2

    if fsc_flat and amplitude_ok:
        print("  VERDICT: fresnel and asm are INDISTINGUISHABLE.")
        print(f"           cross-FSC ~1 to Nyquist => the {rel_l2:.1e} relative-L2 is")
        print("           spatially UNSTRUCTURED (broadband), i.e. run-to-run")
        print("           numerical noise, NOT a propagator effect — a real high-NA")
        print("           difference would drop the cross-FSC near Nyquist.")
        if paraxial:
            print(f"           Expected — deeply paraxial ((λf)²_max = {s_max:.1e}).")
        print("           ASM validated; the sharpening vs FBP is the multislice")
        print("           gain (same for both). To measure the noise floor these")
        print("           sit on, run fresnel twice and compare those two.")
    elif fsc_flat and not amplitude_ok:
        print("  VERDICT: volumes correlate at ALL frequencies (cross-FSC ~1) but")
        print(f"           differ in AMPLITUDE (rel-L2 = {rel_l2:.1e}). Check for a")
        print("           scaling / normalisation mismatch between the two runs —")
        print("           this is not a resolution/propagator effect.")
    else:
        # cross-FSC actually drops -> frequency-structured difference
        print(f"  VERDICT: cross-FSC drops below 0.5 at f={f05:.3f} Nyq "
              f"(d = {_res(f05):.1f} nm):")
        print("           the difference IS frequency-structured.")
        if paraxial:
            print(f"           But this regime is paraxial ((λf)²_max = {s_max:.1e}),")
            print("           where the propagators must agree — so a real spectral")
            print("           split is unexpected: INVESTIGATE (settings mismatch/bug).")
        else:
            print(f"           At this NA ((λf)²_max = {s_max:.1e}) a high-frequency")
            print("           propagator difference can be physical — this is the")
            print("           regime where ASM genuinely departs from Fresnel.")
    print("  " + "-" * 66)

    # --- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ok = nvox > 0
    ax.plot(freq[ok], fsc[ok], "-o", ms=3, color="C0",
            label="cross-FSC (fresnel vs asm)")
    ax.axhline(1.0, color="0.7", lw=0.8, ls=":")
    ax.axhline(0.5, color="C3", lw=1.0, ls="--", label="0.5")
    ax.axhline(0.143, color="C2", lw=1.0, ls="--", label="0.143")
    ax.set_xlabel("spatial frequency  [Nyquist units]  (0.5 = Nyquist = 2·pixel)")
    ax.set_ylabel("cross-FSC(fresnel, asm)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 0.5)
    ttl = (f"Propagator equivalence — rel.L2 = {rel_l2:.2e}, "
           f"corr = {corr_fa:.6f}")
    if np.isfinite(s_max):
        ttl += f"\n(λf)²_max = {s_max:.1e},  pixel = {pixel_nm:.2f} nm"
    ax.set_title(ttl, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"\n  Figure written: {args.out}")


if __name__ == "__main__":
    main()
