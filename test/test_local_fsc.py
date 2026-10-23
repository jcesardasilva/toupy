#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression tests for the block-wise local FSC/FRC map (LocalFSC).

Covers the sign guarantee of the resolution map (no extrapolated
negative "resolutions" in the border band), the NaN returned when the
local FSC never crosses the threshold, the NaN-aware summary statistics
and ``n_valid`` count, and the optional support mask.

Run with::

    pytest test/test_local_fsc.py
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from toupy.resolution.localres import LocalFSC


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_halves(seed=0, n=96, ndim=2, noise=0.3, smooth=2.0):
    """Two half-reconstructions sharing a smooth object plus independent noise."""
    rng = np.random.default_rng(seed)
    shape = (n,) * ndim
    obj = gaussian_filter(rng.normal(size=shape), smooth)
    obj /= obj.std()
    return (obj + noise * rng.normal(size=shape),
            obj + noise * rng.normal(size=shape))


def make_disc_object(seed=3, n=128, radius=40, noise=0.3, smooth=2.0):
    """A disc of material on empty background, with its support."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    support = ((yy - n / 2) ** 2 + (xx - n / 2) ** 2) < radius ** 2
    obj = gaussian_filter(rng.normal(size=(n, n)), smooth)
    obj = obj / obj.std() * support
    img1 = obj + noise * rng.normal(size=(n, n))
    img2 = obj + noise * rng.normal(size=(n, n))
    return img1, img2, support


# ---------------------------------------------------------------------------
# sign of the resolution map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim, n, box", [(2, 96, 32), (3, 48, 16)])
def test_resolution_map_is_never_negative(ndim, n, box):
    """No entry of the map may be <= 0: a resolution is a length.

    The border band is not covered by the grid of box centres, and linear
    extrapolation there used to hand back negative values.
    """
    v1, v2 = make_halves(n=n, ndim=ndim, smooth=1.5)
    lfsc = LocalFSC(v1, v2, box_size=box, step=box // 2)

    finite = lfsc.resolution_map[np.isfinite(lfsc.resolution_map)]
    assert finite.size > 0
    assert np.all(finite > 0.0)
    for attr in ("resolution_map_phys", "resolution_map_half",
                 "resolution_map_phys_half"):
        arr = getattr(lfsc, attr)
        assert np.all(arr[np.isfinite(arr)] > 0.0)


def test_map_stays_within_the_measured_range():
    """Border voxels hold the nearest estimate rather than extrapolating."""
    v1, v2 = make_halves()
    lfsc = LocalFSC(v1, v2, box_size=32, step=16)

    lo = np.nanmin(lfsc.coarse_map)
    hi = np.nanmax(lfsc.coarse_map)
    finite = lfsc.resolution_map[np.isfinite(lfsc.resolution_map)]
    assert finite.min() >= lo - 1e-9
    assert finite.max() <= hi + 1e-9


# ---------------------------------------------------------------------------
# first crossing of the threshold
# ---------------------------------------------------------------------------

def band_limited_volume(L, k_true, rng):
    """A 3-D field containing no spatial frequency above shell *k_true*."""
    k = np.fft.fftfreq(L) * L
    R = np.rint(np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2
                        + k[None, None, :] ** 2)).astype(int)
    field = np.fft.fftn(rng.standard_normal((L,) * 3))
    field[R > k_true] = 0.0
    out = np.real(np.fft.ifftn(field))
    return out / out.std()


@pytest.mark.parametrize("L, k_true", [(32, 12), (64, 24)])
def test_known_band_limit_is_recovered(L, k_true):
    """A box with a known band limit must report that band limit.

    The resolution is read at the *highest* shell at or above the
    threshold.  Ending the run at the first shell to drop below instead
    fails here: the shell-size-corrected threshold peaks exactly where
    such a run starts (T = 0.462 at shell 1 for L = 64), so the run
    terminates in the first few shells whatever the signal above it.
    """
    rng = np.random.default_rng(5)
    pad = np.random.default_rng(0).normal(size=(L + 32,) * 3)
    lfsc = LocalFSC(pad, pad.copy(), box_size=L, step=L // 2,
                    threshold="halfbit")
    window = lfsc._make_window()

    true_px = L / k_true
    for _ in range(10):
        obj = band_limited_volume(L, k_true, rng)
        res = lfsc._fsc_box(obj + rng.standard_normal((L,) * 3),
                            obj + rng.standard_normal((L,) * 3),
                            window)
        assert np.isfinite(res)
        # the Hann window smears the band edge by about one shell
        assert abs(res - true_px) < 0.5, (res, true_px)


def test_uncorrelated_noise_is_not_reported_as_high_resolution():
    """Pure 3-D noise must not yield near-Nyquist estimates.

    The per-shell threshold has a finite false-positive rate, so single
    boxes do survive; what matters is that none of them claims a fine
    resolution.  Note this margin is much thinner for 2-D FRC with a
    fixed threshold, where the shells hold far fewer Fourier voxels — see
    the class docstring.
    """
    for seed in range(4):
        rng = np.random.default_rng(seed)
        w1 = rng.normal(size=(64,) * 3)
        w2 = rng.normal(size=(64,) * 3)
        lfsc = LocalFSC(w1, w2, box_size=32, step=16, threshold="halfbit")

        estimates = lfsc.coarse_map[np.isfinite(lfsc.coarse_map)]
        if estimates.size:
            # whatever survives must be coarse, not Nyquist-scale
            assert estimates.min() >= 4.0, (seed, estimates)


# ---------------------------------------------------------------------------
# non-crossing FSC -> NaN
# ---------------------------------------------------------------------------

def test_non_crossing_fsc_gives_nan_not_a_stand_in():
    """An unreachable threshold on uncorrelated halves leaves no estimate."""
    rng = np.random.default_rng(7)
    v1 = rng.normal(size=(96, 96))
    v2 = rng.normal(size=(96, 96))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lfsc = LocalFSC(v1, v2, box_size=32, step=16, threshold=0.999)

    assert lfsc.n_valid == 0
    assert np.all(np.isnan(lfsc.resolution_map))
    assert np.isnan(lfsc.resolution_median)
    assert np.isnan(lfsc.resolution_mean)
    assert np.isnan(lfsc.resolution_std)
    assert any("no analysis box produced" in str(c.message) for c in caught)


def test_fsc_box_returns_nan_when_nothing_is_above_threshold():
    v1, v2 = make_halves(n=64)
    lfsc = LocalFSC(v1, v2, box_size=32, step=16)
    window = lfsc._make_window()
    box = np.zeros((32, 32))
    assert np.isnan(lfsc._fsc_box(box, box, window))


# ---------------------------------------------------------------------------
# statistics and counts
# ---------------------------------------------------------------------------

def test_statistics_ignore_nan_and_n_valid_counts_estimates():
    v1, v2, support = make_disc_object()
    lfsc = LocalFSC(v1, v2, box_size=32, step=16, mask=support)

    rmap = lfsc.resolution_map
    assert np.any(np.isnan(rmap))          # background is NaN
    assert np.any(np.isfinite(rmap))
    finite = rmap[np.isfinite(rmap)]
    assert lfsc.resolution_median == pytest.approx(np.median(finite))
    assert lfsc.resolution_mean == pytest.approx(np.mean(finite))
    assert lfsc.resolution_std == pytest.approx(np.std(finite))
    assert lfsc.resolution_median_half == pytest.approx(
        lfsc.resolution_median / 2.0)

    assert lfsc.n_boxes == lfsc.coarse_map.size
    assert lfsc.n_valid == int(np.count_nonzero(np.isfinite(lfsc.coarse_map)))
    assert 0 < lfsc.n_valid <= lfsc.n_boxes


# ---------------------------------------------------------------------------
# support mask
# ---------------------------------------------------------------------------

def test_mask_is_eroded_and_map_is_nan_outside_it():
    v1, v2, support = make_disc_object()
    lfsc = LocalFSC(v1, v2, box_size=32, step=16, mask=support)

    assert lfsc.mask is not None
    # eroded support is strictly inside the support
    assert lfsc.mask_eroded.sum() < support.sum()
    assert np.all(support[lfsc.mask_eroded])

    rmap = lfsc.resolution_map
    assert np.all(np.isnan(rmap[~lfsc.mask_eroded]))
    # and continuous inside it — skipped boxes must not punch holes
    assert np.all(np.isfinite(rmap[lfsc.mask_eroded]))


def test_no_mask_means_no_masking():
    """The support is never guessed: without a mask no region is excluded.

    Background boxes are still *attempted* — some of them fail the FSC on
    their own merits, which is a measurement, not an inferred support.
    """
    v1, v2, support = make_disc_object()
    lfsc = LocalFSC(v1, v2, box_size=32, step=16)

    assert lfsc.mask is None
    assert lfsc.mask_eroded is None
    assert np.any(np.isfinite(lfsc.resolution_map[~support]))


def disc_support(n=256, radius=90):
    yy, xx = np.mgrid[0:n, 0:n]
    c = n / 2
    return np.hypot(yy - c, xx - c) <= radius, c


def test_default_erosion_is_the_cube_the_box_occupies():
    """The default is the exact reading of 'the box fits in the support'."""
    support, _ = disc_support()
    v1, v2 = make_halves(n=256)
    lfsc = LocalFSC(v1, v2, box_size=32, step=16, mask=support)

    assert lfsc.erosion == "cube"
    assert lfsc.erosion_radius == 16.0

    # every retained centre carries a full box inside the support
    ey, ex = np.nonzero(lfsc.mask_eroded)
    h = 16
    for y, x in zip(ey[::37], ex[::37]):
        assert support[y - h:y + h, x - h:x + h].all()


def test_ball_erosion_is_round_and_still_guarantees_containment():
    support, c = disc_support()
    v1, v2 = make_halves(n=256)
    cube = LocalFSC(v1, v2, box_size=32, step=16, mask=support)
    ball = LocalFSC(v1, v2, box_size=32, step=16, mask=support,
                    mask_eroded="ball")

    assert ball.erosion == "ball"
    assert ball.erosion_radius == pytest.approx(32 * np.sqrt(2) / 2)

    # round boundary
    ey, ex = np.nonzero(ball.mask_eroded)
    r = np.hypot(ey - c, ex - c)
    ang = np.arctan2(ey - c, ex - c)
    bins = np.digitize(ang, np.linspace(-np.pi, np.pi, 37))
    edge = np.array([r[bins == k].max() for k in range(1, 37)
                     if np.any(bins == k)])
    assert edge.max() - edge.min() < 1.5

    # the circumradius ball contains the cube, so it erodes at least as
    # much: never more permissive than the exact answer
    assert ball.mask_eroded.sum() < cube.mask_eroded.sum()


def test_numeric_erosion_radius_is_a_ball_of_that_radius():
    support, _ = disc_support()
    v1, v2 = make_halves(n=256)
    # in-plane circumradius for a cylindrical specimen, z unconstrained
    lfsc = LocalFSC(v1, v2, box_size=32, step=16, mask=support,
                    mask_eroded=32 / np.sqrt(2))

    assert lfsc.erosion == "ball"
    assert lfsc.erosion_radius == pytest.approx(32 / np.sqrt(2))


def test_permissive_ball_radius_admits_boxes_leaving_the_support():
    """Documented behaviour: a radius below the circumradius is permissive.

    A ball of box_size/2 does not contain the box, so this is not a
    silent alternative to the cube.
    """
    support, _ = disc_support()
    v1, v2 = make_halves(n=256)
    cube = LocalFSC(v1, v2, box_size=32, step=16, mask=support)
    loose = LocalFSC(v1, v2, box_size=32, step=16, mask=support,
                     mask_eroded=16.0)

    assert loose.mask_eroded.sum() > cube.mask_eroded.sum()


def test_mask_eroded_none_uses_the_mask_verbatim():
    v1, v2, support = make_disc_object()
    auto = LocalFSC(v1, v2, box_size=32, step=16, mask=support)
    asis = LocalFSC(v1, v2, box_size=32, step=16, mask=support,
                    mask_eroded=None)

    assert np.array_equal(asis.mask_eroded, support)
    assert np.array_equal(asis.mask, support)
    # no erosion means a strictly larger evaluated region, hence more boxes
    assert asis.mask_eroded.sum() > auto.mask_eroded.sum()
    assert asis.n_valid > auto.n_valid
    assert np.all(np.isnan(asis.resolution_map[~support]))


def test_mask_eroded_none_allows_a_support_smaller_than_the_box():
    """The erosion guard must not fire when the caller opts out of it."""
    v1, v2, _ = make_disc_object()
    tiny = np.zeros((128, 128), dtype=bool)
    tiny[48:80, 48:80] = True

    with pytest.raises(ValueError, match="vanishes when eroded"):
        LocalFSC(v1, v2, box_size=64, mask=tiny)

    lfsc = LocalFSC(v1, v2, box_size=64, step=16, mask=tiny, mask_eroded=None)
    assert np.array_equal(lfsc.mask_eroded, tiny)


@pytest.mark.parametrize("bad", ["ellipse", True, -4.0, 0.0, [1, 2]])
def test_invalid_mask_eroded_raises(bad):
    v1, v2, support = make_disc_object()
    with pytest.raises(ValueError, match="mask_eroded|must be positive"):
        LocalFSC(v1, v2, box_size=32, mask=support, mask_eroded=bad)


def test_mask_eroded_none_without_a_mask_warns():
    v1, v2, _ = make_disc_object()
    with pytest.warns(UserWarning, match="no effect without a mask"):
        lfsc = LocalFSC(v1, v2, box_size=32, step=16, mask_eroded=None)
    assert lfsc.mask_eroded is None


@pytest.mark.parametrize("bad, match", [
    (np.ones((10, 10), dtype=bool), "does not match"),
    (np.zeros((128, 128), dtype=bool), "no True entries"),
])
def test_invalid_mask_raises(bad, match):
    v1, v2, _ = make_disc_object()
    with pytest.raises(ValueError, match=match):
        LocalFSC(v1, v2, box_size=32, mask=bad)


def test_support_smaller_than_the_box_raises():
    v1, v2, _ = make_disc_object()
    tiny = np.zeros((128, 128), dtype=bool)
    tiny[62:66, 62:66] = True
    with pytest.raises(ValueError, match="vanishes when eroded"):
        LocalFSC(v1, v2, box_size=32, mask=tiny)
