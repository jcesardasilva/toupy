#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-data checks for toupy.resolution.ImageDecorr, on PXCT phase projections.

Complements ``decorr_test.py`` (synthetic).  The point of running on real data
is that synthetic phantoms have convenient statistics; real projections have a
steep, low-frequency-dominated spectrum, an air/sample boundary and real noise.

Ground truth on real data is obtained by *imposing* a known band-limit on a real
projection: everything above a cutoff kc is zeroed, so the correct answer is kc
by construction while the image keeps realistic statistics below it.

IMPORTANT -- ptychographic border crop
--------------------------------------
These projections are ptychographic reconstructions: near the edges of the
scanned field the probe overlap is poor and the retrieved phase is dominated by
noise.  Those borders MUST be discarded before any resolution analysis
(``BORDER_CROP`` px from every side).  They are not a cosmetic nuisance: the
low-overlap noise is broadband, so leaving it in flattens the spectrum toward
white noise, the decorrelation curve loses its interior maximum, and the
analysis degenerates to the low-frequency end of the sweep -- returning a
confident-looking number that is really just a grid index.  ``report_border_crop_
caveat`` below demonstrates this directly.

The data file is large (~2 GB) and is not in the repository, so this script
SKIPS (exit 0) when it is absent.

Run standalone::

    python test/decorr_realdata_test.py
    python test/decorr_realdata_test.py /path/to/PXCTalignedprojections_big.npz
"""

import os
import sys
import warnings

import numpy as np

from toupy.resolution import ImageDecorr

DEFAULT_PATHS = [
    "tutorial/PXCTalignedprojections_big.npz",
    "tutorial/PXCTalignedprojections.npz",
]

# Pixels to discard from every border: the ptychographic low-overlap region.
# 110 px for the unbinned (14.32 nm) set; the 2x-binned (28.64 nm) set uses 55.
BORDER_CROP = 110

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def decorr(im, psize, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ImageDecorr(im, pixel_size=psize, **kw)


def lowpass(im, kc):
    """Zero everything above `kc` (cycles/pixel): imposes a known band-limit."""
    F = np.fft.fft2(im)
    fy = np.fft.fftfreq(im.shape[0])
    fx = np.fft.fftfreq(im.shape[1])
    R = np.hypot(*np.meshgrid(fy, fx, indexing="ij"))
    F[R > kc] = 0.0
    return np.real(np.fft.ifft2(F))


def load(path):
    d = np.load(path, allow_pickle=True)
    psize_m = float(d["psize"])
    proj = d["projections"]
    n = proj.shape[0]
    idx = [0, n // 3, 2 * n // 3]
    few = np.stack([proj[i].astype(np.float64) for i in idx])
    del proj, d
    return few, psize_m * 1e9, idx   # psize in nm


# ---------------------------------------------------------------------------
def test_known_bandlimit_on_real_image(img, psize):
    """Core test: impose a known cutoff on a REAL projection and recover it."""
    print("\n[1] recover an imposed band-limit on a real projection")
    for kc in (0.05, 0.10, 0.15, 0.20):
        d = decorr(lowpass(img, kc), psize)
        ratio = d.r_res / kc
        check(f"kc={kc:.2f} -> r_res={d.r_res:.4f} (ratio {ratio:.2f}, "
              f"{d.resolution:.1f} nm)", 0.9 <= ratio <= 1.15)


def test_dc_pedestal_invariance_real(img, psize):
    """Regression guard with real statistics: a DC pedestal must not matter."""
    print("\n[2] DC-pedestal invariance on real data")
    ref = decorr(img, psize).resolution
    for ped in (10.0, 1000.0):
        got = decorr(img + ped, psize).resolution
        check(f"pedestal={ped:<7g} -> {got:.2f} nm", abs(got - ref) < 0.05 * ref,
              f"ref={ref:.2f} nm")


def test_apod_width_independence_real(img, psize):
    """
    The answer must not track the apodization width on real data either.

    Tolerance is 10 % here rather than the 5 % used on synthetic images: a real
    projection genuinely shifts by a few percent as the taper eats more of the
    field.  That is ample to catch the failure this guards against -- the window
    artefact made the result *proportional* to apod_width (a 2-4x spread).
    """
    print("\n[3] independence from apod_width on real data")
    vals = [decorr(img + 1000.0, psize, apod_width=aw).resolution
            for aw in (10, 20, 40)]
    spread = max(vals) - min(vals)
    check(f"stable across apod_width 10/20/40: {['%.1f' % v for v in vals]} nm",
          spread < 0.10 * float(np.mean(vals)), f"spread={spread:.2f} nm")


def test_physical_sanity(few, psize, crop):
    """
    Across projections: a genuine interior peak, and never better than Nyquist.

    On border-cropped ptychographic projections the analysis must find a real
    peak -- well clear of the low-frequency end of the sweep and with a healthy
    amplitude -- not degenerate to a grid index.
    """
    print("\n[4] genuine peak + Nyquist floor across projections")
    for i, p in enumerate(few):
        img = p[crop:-crop, crop:-crop] if crop else p
        d = decorr(img, psize)
        kc_max = float(np.max(d.kc_candidates))
        snr_max = float(np.max(d.snr_candidates))
        check(f"projection {i}: {d.resolution:.1f} nm ({d.resolution_px:.2f} px)"
              f" >= Nyquist (2 px)", d.resolution_px >= 2.0 - 1e-9)
        check(f"projection {i}: genuine peak (kc={kc_max:.3f} cyc/px, "
              f"snr={snr_max:.2f})", kc_max > 0.05 and snr_max > 0.3,
              "degenerate result would pin kc to the sweep floor")


def report_border_crop_caveat(p, psize, crop):
    """
    NOT an assertion -- a demonstration of why BORDER_CROP is mandatory.

    Ptychographic projections are noise-dominated near the edges of the scan,
    where probe overlap is poor.  That noise is BROADBAND, so leaving it in
    flattens the spectrum toward white noise; a flat spectrum makes the
    decorrelation curve rise monotonically, it loses its interior maximum,
    `getDcorrLocalMax` walks back to the start of the sweep, and the result is
    pinned to a low-frequency grid index -- the same value for every
    projection, and it passes the SNR filter, so it looks like a measurement.
    Cropping the borders restores a genuine peak.
    """
    print("\n[i] why the ptychographic border crop matters "
          "(informational, not asserted)")
    for tag, im in (("uncropped (WRONG)", p),
                    (f"cropped {crop}px (correct)", p[crop:-crop, crop:-crop])):
        d = decorr(im, psize)
        print(f"      {tag:26s} {str(im.shape):>12}  {d.resolution:8.1f} nm  "
              f"(max kc={np.max(d.kc_candidates):.4f}, "
              f"snr={np.max(d.snr_candidates):.2f})")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        for c in DEFAULT_PATHS:
            if os.path.isfile(c):
                path = c
                break
    if path is None or not os.path.isfile(path):
        print("SKIP: no projection data found "
              f"(looked for: {', '.join(DEFAULT_PATHS)}).")
        print("Pass a path explicitly to run these checks.")
        sys.exit(0)

    print("=" * 70)
    print(f"ImageDecorr — real-data checks  ({path})")
    print("=" * 70)
    few, psize, idx = load(path)
    print(f"  loaded projections {idx} of shape {few.shape[1:]}, "
          f"pixel size {psize:.3f} nm")

    # Discard the ptychographic low-overlap borders, then use the FULL valid
    # region -- no arbitrary sub-crop needed once the noisy edges are gone.
    crop = BORDER_CROP
    if min(few.shape[1:]) <= 2 * crop + 64:
        crop = 0
        print("  [warn] image too small for the border crop; using it as-is")
    p = few[len(few) // 2]
    img = p[crop:-crop, crop:-crop] if crop else p
    print(f"  border crop  : {crop} px/side (ptychographic probe overlap)")
    print(f"  working image: {img.shape}, |mean|/std = "
          f"{abs(img.mean())/img.std():.3f}")

    test_known_bandlimit_on_real_image(img, psize)
    test_dc_pedestal_invariance_real(img, psize)
    test_apod_width_independence_real(img, psize)
    test_physical_sanity(few, psize, crop)
    if crop:
        report_border_crop_caveat(p, psize, crop)

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n" + "=" * 70)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    print("=" * 70)
    sys.exit(1 if n_fail else 0)
