#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correctness checks for toupy.resolution.ImageDecorr (decorrelation analysis).

Run standalone::

    python test/decorr_test.py

Exits non-zero if any check fails, so it is usable as a CI gate.

The central check is that the algorithm recovers a *known* band-limit: images
are built by low-passing white noise at a hard cutoff kc, so the true answer is
known by construction and the reported frequency must track it.  The remaining
checks are regression guards for failure modes that return a confident but
meaningless number.
"""

import sys
import warnings

import numpy as np

from toupy.resolution import ImageDecorr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def bandlimited(n, kc, seed=0, noise=0.0):
    """White noise low-passed to a hard cutoff `kc` (cycles/pixel)."""
    rng = np.random.default_rng(seed)
    im = rng.normal(0.0, 1.0, (n, n))
    F = np.fft.fft2(im)
    f = np.fft.fftfreq(n)
    R = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    F[R > kc] = 0.0
    out = np.real(np.fft.ifft2(F))
    out /= out.std()
    if noise:
        out += rng.normal(0.0, noise, out.shape)
    return out


def decorr(im, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ImageDecorr(im, pixel_size=1.0, **kw)


RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def test_recovers_known_bandlimit():
    """The reported frequency must track a known cutoff over a wide range."""
    print("\n[1] recovery of a known band-limit")
    for kc in (0.10, 0.15, 0.20, 0.30):
        d = decorr(bandlimited(256, kc))
        ratio = d.r_res / kc
        check(f"kc={kc:.2f} recovered (r_res={d.r_res:.4f}, ratio={ratio:.2f})",
              0.9 <= ratio <= 1.15, f"res_px={d.resolution_px:.2f}")


def test_dc_pedestal_invariance():
    """
    Regression: adding a DC pedestal must not change the answer.

    _apodize must remove the mean BEFORE windowing; otherwise the window taper
    becomes the dominant spectral structure for a low-contrast image and the
    reported resolution collapses onto ~apod_width.
    """
    print("\n[2] DC-pedestal invariance (apodization-order regression)")
    base = bandlimited(256, 0.15)
    ref = decorr(base).resolution_px
    for pedestal in (0.0, 10.0, 1000.0):
        got = decorr(base + pedestal).resolution_px
        check(f"pedestal={pedestal:<8g} -> res_px={got:.2f}",
              abs(got - ref) < 0.05 * ref, f"ref={ref:.2f}")


def test_apod_width_independence():
    """The result must not track the apodization width for a real signal."""
    print("\n[3] independence from apod_width")
    base = bandlimited(256, 0.15) + 1000.0      # pedestal makes this sensitive
    vals = [decorr(base, apod_width=aw).resolution_px for aw in (10, 20, 40)]
    spread = max(vals) - min(vals)
    check(f"res_px stable across apod_width 10/20/40: "
          f"{['%.2f' % v for v in vals]}", spread < 0.05 * np.mean(vals),
          f"spread={spread:.3f}")


def test_blank_image_is_degenerate():
    """A constant image has nothing to resolve: report the floor, not Nyquist."""
    print("\n[4] blank image -> degenerate floor")
    n = 128
    d = decorr(np.ones((n, n)))
    check(f"constant image res_px={d.resolution_px:.1f} is worse than image size",
          d.resolution_px >= n, "must not claim a fine resolution")
    check(f"constant image max_snr={np.max(d.snr_candidates):.3f} ~ 0",
          float(np.max(d.snr_candidates)) < 0.05)


def test_pure_noise_reports_floor():
    """Structureless noise: no candidate passes SNR, so fall back to the floor."""
    print("\n[5] pure noise -> floor (no valid peak)")
    rng = np.random.default_rng(0)
    n = 256
    d = decorr(rng.normal(0.0, 1.0, (n, n)))
    npass = int(np.sum(np.asarray(d.snr_candidates) >= 0.05))
    check(f"no candidate passes SNR filter (npass={npass})", npass == 0)
    check(f"reports floor res_px={d.resolution_px:.1f} (>= image size {n})",
          d.resolution_px >= n, "optimistic fallback would give a small value")


def test_noise_robustness():
    """A real band-limit must survive substantial additive noise."""
    print("\n[6] robustness to additive noise")
    for lvl in (0.5, 1.0, 1.5):
        d = decorr(bandlimited(256, 0.15, noise=lvl))
        check(f"noise={lvl}x -> r_res={d.r_res:.4f}",
              0.85 <= d.r_res / 0.15 <= 1.25, f"res_px={d.resolution_px:.2f}")


def test_extreme_noise_fails_safe():
    """
    At SNR ~ 0.5 this phantom (band-limited *noise*, so no structural
    redundancy) is beyond the method.  It must then FAIL SAFE -- no candidate
    passes the SNR filter and the floor is reported -- rather than return a
    confident but wrong resolution.  This is what the non-optimistic fallback
    buys; taking max() over rejected candidates would invent an answer here.
    """
    print("\n[6b] extreme noise -> fails safe (no invented answer)")
    d = decorr(bandlimited(256, 0.15, noise=2.0))
    npass = int(np.sum(np.asarray(d.snr_candidates) >= 0.05))
    check(f"noise=2.0x: no candidate passes SNR (npass={npass})", npass == 0)
    check(f"noise=2.0x: reports floor res_px={d.resolution_px:.1f}, not a "
          f"plausible-looking value", d.resolution_px >= 256)


def test_no_threshold_parameter():
    """`threshold` must be deprecated and have no effect on the result."""
    print("\n[7] parameter-free: threshold is inert")
    base = bandlimited(256, 0.15)
    ref = decorr(base).resolution_px
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = ImageDecorr(base, pixel_size=1.0, threshold=0.5).resolution_px
        warned = any(issubclass(x.category, DeprecationWarning) for x in w)
    check("passing threshold= raises DeprecationWarning", warned)
    check(f"threshold does not change the result ({got:.2f} == {ref:.2f})",
          abs(got - ref) < 1e-9)


def test_full_period_convention():
    """resolution_px == 1/r_res == 2/kc_norm (full-period, not half-period)."""
    print("\n[8] full-period convention")
    d = decorr(bandlimited(256, 0.15))
    check(f"resolution_px == 1/r_res ({d.resolution_px:.4f})",
          abs(d.resolution_px - 1.0 / d.r_res) < 1e-9)
    check("resolution scales with pixel_size",
          abs(decorr(bandlimited(256, 0.15)).resolution_px * 2.5
              - ImageDecorr(bandlimited(256, 0.15), pixel_size=2.5).resolution) < 1e-6)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 68)
    print("ImageDecorr — decorrelation analysis checks")
    print("=" * 68)

    test_recovers_known_bandlimit()
    test_dc_pedestal_invariance()
    test_apod_width_independence()
    test_blank_image_is_degenerate()
    test_pure_noise_reports_floor()
    test_noise_robustness()
    test_extreme_noise_fails_safe()
    test_no_threshold_parameter()
    test_full_period_convention()

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n" + "=" * 68)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    print("=" * 68)
    sys.exit(1 if n_fail else 0)
