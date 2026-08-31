#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression tests for the phase-randomization FSC test (RandomFSC).

Covers the conventions of the corrected curve (Chen et al., 2013), the
threshold-keyed transition mask and its safety cap, and the Hermitian
symmetry of the random phase field.

Run with::

    pytest test/test_random_fsc.py
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from toupy.resolution.FSC import RandomFSC


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_halves(seed=0, n=64, ndim=2, noise=0.6, smooth=3.0):
    """Two half-reconstructions sharing a smooth object plus independent noise."""
    rng = np.random.default_rng(seed)
    shape = (n,) * ndim
    obj = gaussian_filter(rng.normal(size=shape), smooth)
    obj /= obj.std()
    img1 = obj + noise * gaussian_filter(rng.normal(size=shape), 0.8)
    img2 = obj + noise * gaussian_filter(rng.normal(size=shape), 0.8)
    return img1, img2


@pytest.fixture(scope="module")
def rfsc():
    img1, img2 = make_halves()
    return RandomFSC(img1, img2, ring_thick=1, apod_width=10,
                     fsc_cutoff=0.8, random_seed=1)


# ---------------------------------------------------------------------------
# corrected-curve conventions
# ---------------------------------------------------------------------------

def test_no_correction_at_or_below_cutoff(rfsc):
    """Nothing was randomized there, so FSC_corr is FSC_obs, not 0/0 noise."""
    stop = rfsc.cutoff_shell + 1
    obs = np.asarray(rfsc.FSC_obs, dtype=np.float64).real
    assert np.allclose(rfsc.FSC_corr[:stop], obs[:stop])
    assert np.all(np.isfinite(rfsc.FSC_corr[:stop]))


def test_transition_band_is_nan_and_contiguous(rfsc):
    """The masked shells sit immediately above the cutoff, with no strays."""
    nan_idx = np.flatnonzero(np.isnan(rfsc.FSC_corr))
    assert len(nan_idx) == rfsc.transition_shells
    if rfsc.transition_shells > 0:
        expected = np.arange(rfsc.cutoff_shell + 1,
                             rfsc.cutoff_shell + 1 + rfsc.transition_shells)
        assert np.array_equal(nan_idx, expected)


def test_transition_ends_at_last_shell_above_threshold(rfsc):
    """Mask reaches the last shell with FSC_rand > T inside the search window."""
    start = rfsc.cutoff_shell + 1
    end = start + rfsc.transition_shells
    rand = np.asarray(rfsc.FSC_rand, dtype=np.float64).real
    T = np.asarray(rfsc.T, dtype=np.float64)
    if rfsc.transition_shells > 0:
        # the last masked shell is above T ...
        assert rand[end - 1] > T[end - 1]
    cap = max(5, int(round(0.05 * len(rfsc.f))))
    # ... and nothing further inside the window is
    window = slice(end, start + cap)
    assert not np.any(rand[window] > T[window])


def test_negative_corr_is_preserved_and_harmless(rfsc):
    """FSC_rand > FSC_obs is diagnostic; it is kept, and cannot fake a crossing."""
    finite = rfsc.FSC_corr[np.isfinite(rfsc.FSC_corr)]
    assert finite.min() >= -1.0
    T = np.asarray(rfsc.T, dtype=np.float64)
    neg = np.isfinite(rfsc.FSC_corr) & (rfsc.FSC_corr < 0)
    assert not np.any(rfsc.FSC_corr[neg] > T[neg])


def test_resolution_ignores_masked_shells(rfsc):
    """The crossing search must not resolve on a NaN shell."""
    if rfsc.fn_res is not None:
        idx = int(round(rfsc.fn_res * rfsc.fnyquist))
        assert np.isfinite(rfsc.FSC_corr[idx])


# ---------------------------------------------------------------------------
# transition-mask cap
# ---------------------------------------------------------------------------

def test_cap_bounds_mask_and_warns_when_rand_never_decays(rfsc):
    """
    Pathological case: FSC_rand stays above T all the way to Nyquist.

    Without the cap the threshold rule would blank every shell above the
    cutoff. The cap must bound the mask and say so.
    """
    obj = RandomFSC.__new__(RandomFSC)
    nshells = 200
    obj.f = np.arange(nshells)
    obj.cutoff_shell = 80
    obj.max_transition_shells = None
    obj.T = np.full(nshells, 0.2)
    obj.FSC_rand = np.full(nshells, 0.9)      # never falls back to T

    with pytest.warns(UserWarning, match="still above the threshold"):
        n = obj._find_transition_shells()

    cap = max(5, int(round(0.05 * nshells)))
    assert n == cap
    assert n < nshells - obj.cutoff_shell - 1   # curve is not blanked


def test_cap_is_configurable():
    obj = RandomFSC.__new__(RandomFSC)
    nshells = 200
    obj.f = np.arange(nshells)
    obj.cutoff_shell = 80
    obj.max_transition_shells = 3
    obj.T = np.full(nshells, 0.2)
    obj.FSC_rand = np.full(nshells, 0.9)
    with pytest.warns(UserWarning):
        assert obj._find_transition_shells() == 3


def test_no_warning_when_rand_decays_normally():
    obj = RandomFSC.__new__(RandomFSC)
    nshells = 200
    obj.f = np.arange(nshells)
    obj.cutoff_shell = 80
    obj.max_transition_shells = None
    obj.T = np.full(nshells, 0.2)
    rand = np.full(nshells, 0.01)
    rand[81:83] = 0.9                          # two-shell transition
    obj.FSC_rand = rand
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert obj._find_transition_shells() == 2


def test_ragged_transition_masks_through_last_excursion():
    """A one-shell dip below T must not end the mask early."""
    obj = RandomFSC.__new__(RandomFSC)
    nshells = 200
    obj.f = np.arange(nshells)
    obj.cutoff_shell = 80
    obj.max_transition_shells = None
    obj.T = np.full(nshells, 0.3)
    rand = np.full(nshells, 0.01)
    rand[81] = 0.5
    rand[82] = 0.1                             # dip
    rand[83] = 0.4                             # back above T
    obj.FSC_rand = rand
    assert obj._find_transition_shells() == 3


# ---------------------------------------------------------------------------
# Hermitian symmetry of the random phase field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(32, 32), (31, 32), (16, 16, 16)])
def test_randomization_preserves_amplitude_spectrum(shape):
    """
    The amplitude spectrum must survive randomization exactly.

    Independent phases at +k and -k followed by np.real() would rescale it by
    a random factor (~0.64 +- 0.31), leaving FSC_rand wrong by an amount whose
    sign depends on the data.
    """
    rng = np.random.default_rng(5)
    vol = gaussian_filter(rng.normal(size=shape), 2.0)

    obj = RandomFSC.__new__(RandomFSC)
    out = obj._randomize_phases_above(vol, 0.15, np.random.default_rng(2))

    F, G = np.fft.fftn(vol), np.fft.fftn(out)
    freqs = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    mask = np.sqrt(sum(g ** 2 for g in freqs)) > 0.15

    ratio = np.abs(G)[mask] / np.maximum(np.abs(F)[mask], 1e-30)
    assert np.allclose(ratio, 1.0, atol=1e-8)


@pytest.mark.parametrize("shape", [(32, 32), (31, 32), (16, 16, 16)])
def test_randomization_leaves_low_frequencies_untouched(shape):
    rng = np.random.default_rng(5)
    vol = gaussian_filter(rng.normal(size=shape), 2.0)

    obj = RandomFSC.__new__(RandomFSC)
    out = obj._randomize_phases_above(vol, 0.15, np.random.default_rng(2))

    F, G = np.fft.fftn(vol), np.fft.fftn(out)
    freqs = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    keep = np.sqrt(sum(g ** 2 for g in freqs)) <= 0.15

    assert np.allclose(np.angle(F)[keep], np.angle(G)[keep], atol=1e-8)


def test_negate_frequency_is_an_involution():
    rng = np.random.default_rng(0)
    for shape in [(8, 8), (7, 8), (5, 6, 7)]:
        a = rng.normal(size=shape)
        neg = RandomFSC._negate_frequency
        assert np.array_equal(neg(neg(a)), a)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def test_plot_returns_five_tuple(rfsc):
    """tutorial/PXCT_pipeline_extra.ipynb unpacks exactly five values."""
    out = rfsc.plot()
    assert len(out) == 5


def test_cutoff_shell_is_public(rfsc):
    assert isinstance(rfsc.cutoff_shell, int)
    assert isinstance(rfsc.transition_shells, int)


def test_warns_when_crossing_lands_below_cutoff():
    """
    FSC_corr equals FSC_obs at and below the cutoff, so a crossing there
    reports an uncorrected resolution and must say so.
    """
    img1, img2 = make_halves(seed=2, n=64, noise=1.4, smooth=6.0)
    with pytest.warns(UserWarning, match="carries no phase-randomization"):
        RandomFSC(img1, img2, ring_thick=1, apod_width=10,
                  fsc_cutoff=0.3, random_seed=1)
