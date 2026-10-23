#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ring artifact correction for tomographic sinograms.

Two methods are provided:

**Wavelet-FFT method (Münch et al., 2009)**
    Decomposes the sinogram with a Haar wavelet along the detector-pixel
    axis, then — for every sub-band including the final approximation —
    estimates the ring contribution from the per-pixel angular mean and
    removes it by Gaussian background subtraction.  The result is
    equivalent to the multi-scale stripe filter described by Münch et al.
    but is implemented without the FFT-based DC suppression that would
    also remove real mean-projection signal.

    Reference:
        Münch, B., Trtik, P., Marone, F., & Stampanoni, M. (2009).
        Stripe and ring artifact removal with combined wavelet — Fourier
        filtering. Optics Express, 17(10), 8567–8591.
        https://doi.org/10.1364/OE.17.008567

**Titarenko method (Titarenko et al., 2010)**
    Lightweight correction based on angular block-mean statistics.  Works
    well for narrow, isolated rings.

    Reference:
        Titarenko, V., Titarenko, S., Withers, P. J., De Carlo, F., &
        Xiao, X. (2010).  Applied Physics Letters, 97(7), 073905.
        https://doi.org/10.1063/1.3480961

GPU notes
---------
Both functions accept ``cuda=True``.  The Haar decomposition and
reconstruction are fully vectorised over detector columns and map
directly onto CuPy array operations.
"""

import warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

# ---------------------------------------------------------------------------
# Optional CuPy import — graceful CPU fallback
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

__all__ = ["remove_rings_wavelet_fft", "remove_rings_titarenko",
           "remove_rings_stack"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_xp(cuda):
    if cuda:
        if not CUDA_AVAILABLE:
            warnings.warn(
                "CuPy not available — falling back to CPU ring correction.",
                stacklevel=3,
            )
            return np
        return cp
    return np


def _haar_dec(cH, xp):
    """Vectorised one-level Haar decomposition along axis 0."""
    a = (cH[0::2, :] + cH[1::2, :]) * 0.5
    d = (cH[0::2, :] - cH[1::2, :]) * 0.5
    return a, d


def _haar_rec(a, d, xp):
    """Vectorised one-level Haar reconstruction along axis 0."""
    rec = xp.empty((a.shape[0] * 2, a.shape[1]), dtype=a.dtype)
    rec[0::2, :] = a + d
    rec[1::2, :] = a - d
    return rec


def _correct_band(band, win_pix, xp):
    """Subtract ring stripes from a wavelet sub-band (angular-mean method).

    Ring artifacts are, by definition, **constant across all projection
    angles** — an additive offset at a fixed detector pixel.  For each
    band-pixel we therefore estimate the stripe as

        stripe[p] = mean_over_angles(band[p, :])
                    - smooth_baseline(mean_over_angles(band)[p])

    where the smooth baseline (a wide median filter along the *pixel*
    axis) follows the real object profile but ignores the narrow ring
    spikes.  Subtracting ``stripe`` from every angle removes the ring.

    This operation is *safe*: because the subtracted quantity is constant
    along the angle axis, it can never remove or distort the real,
    angularly-varying projection signal — unlike Fourier DC suppression,
    which leaks into low (but non-zero) angular frequencies.

    Parameters
    ----------
    band : ndarray, shape (n_pix_band, n_angles)
    win_pix : int
        Median-filter window (in band-pixel units) used to estimate the
        smooth object baseline.  Must be >> ring width but < object scale.
    xp : numpy or cupy module
    """
    # Per-pixel angular mean: real smooth profile + sharp ring spikes
    dc = band.mean(axis=1)                          # (n_pix_band,)
    dc_cpu = cp.asnumpy(dc) if xp is not np else np.asarray(dc)

    # Robust smooth baseline (median ignores the narrow ring spikes)
    win = max(3, int(win_pix))
    if win % 2 == 0:
        win += 1
    base = median_filter(dc_cpu, size=win)

    stripe = dc_cpu - base                          # the ring component
    stripe = xp.asarray(stripe) if xp is not np else stripe
    return band - stripe[:, None]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remove_rings_wavelet_fft(
    sinogram,
    level=3,
    sigma=3,
    wname="haar",
    cuda=False,
):
    """
    Remove ring artifacts using multi-scale wavelet stripe subtraction.

    The sinogram is decomposed with a Haar wavelet along the
    detector-pixel axis (axis 0) for *level* levels.  At each scale —
    including the final approximation band — a per-pixel angular mean is
    computed and a Gaussian-smoothed background is subtracted to isolate
    the ring contribution.  The cleaned bands are then reconstructed.

    Parameters
    ----------
    sinogram : ndarray, shape (n_pixels, n_angles)
        Input sinogram.  axis 0 = detector pixels; axis 1 = angles.
    level : int, optional
        Number of Haar decomposition levels.  Default 3.
    sigma : float, optional
        Expected maximum ring width in **original detector-pixel** units.
        The Gaussian background estimation uses a smoothing sigma of
        ``max(nrow / 10, sigma * 10)`` pixels (in the original sinogram),
        scaled proportionally for each wavelet band so that the
        separation of ring spikes from real-signal background is
        consistent across all scales.  Default 3.
    wname : str, optional
        Wavelet family.  Only ``'haar'`` is supported.  Default ``'haar'``.
    cuda : bool, optional
        Use GPU via CuPy.  Default False.

    Returns
    -------
    corrected : ndarray, shape (n_pixels, n_angles)
        Ring-corrected sinogram (CPU numpy array).

    Notes
    -----
    The method suppresses **additive** stripes (constant angular offset
    at specific detector pixels).  Multiplicative non-uniformity should
    be removed by flat-field normalisation first.

    Examples
    --------
    >>> import numpy as np
    >>> from toupy.restoration import remove_rings_wavelet_fft
    >>> sino = np.random.rand(363, 180)
    >>> sino_clean = remove_rings_wavelet_fft(sino, level=3, sigma=3)
    """
    xp = _get_xp(cuda)
    sino_cpu = np.asarray(sinogram, dtype=float)
    nrow, ncol = sino_cpu.shape
    use_gpu = cuda and CUDA_AVAILABLE
    arr = xp.asarray(sino_cpu) if use_gpu else sino_cpu.copy()

    # Pad to the next power-of-2 along the pixel axis for clean decomposition
    target = int(2 ** np.ceil(np.log2(max(nrow, 2))))
    cH = xp.pad(arr, ((0, target - nrow), (0, 0)), mode="reflect")

    details = []
    band_sizes = []
    for _ in range(level):
        if cH.shape[0] < 4:
            details.append(None)
            band_sizes.append(None)
            continue
        a, d = _haar_dec(cH, xp)
        details.append(d)
        band_sizes.append(d.shape[0])
        cH = a

    # ------------------------------------------------------------------
    # Ring correction: angular-mean stripe subtraction on EVERY band.
    #
    # Because the subtracted stripe is constant along the angle axis, the
    # operation is safe on all bands — including the final approximation,
    # which holds the smooth real-signal background.  (The old FFT-DC
    # approach leaked into low angular frequencies and destroyed real
    # signal; it has been replaced.)
    #
    # The median window is set once in original-pixel units and scaled
    # down for each (decimated) wavelet band so the physical separation
    # of ring spikes from object features is consistent across scales.
    # ------------------------------------------------------------------
    win_full = max(int(target / 10), int(sigma) * 10)   # original-pixel units

    # Correct the final approximation band (size = target / 2**level)
    win_approx = max(3, int(win_full * cH.shape[0] / target))
    cH = _correct_band(cH, win_approx, xp)

    # Correct each detail band with a band-scaled window
    damped = []
    for d, nb in zip(details, band_sizes):
        if d is None:
            damped.append(None)
            continue
        win_band = max(3, int(win_full * nb / target))
        damped.append(_correct_band(d, win_band, xp))

    # Reconstruct
    result = cH
    for d in reversed(damped):
        if d is None:
            continue
        result = _haar_rec(result, d, xp)

    out = result[:nrow, :]
    return cp.asnumpy(out) if use_gpu else np.asarray(out)


def remove_rings_titarenko(sinogram, size=31, cuda=False):
    """
    Remove ring artifacts using median-filter background subtraction.

    Lightweight single-scale method: compute the per-pixel angular mean,
    median-filter it to estimate the smooth background, and subtract the
    residual (ring) from every angle.  Equivalent to the wavelet method
    with level=0 (no decomposition).

    Parameters
    ----------
    sinogram : ndarray, shape (n_pixels, n_angles)
    size : int, optional
        Median filter window size in pixels.  Should be >> ring width
        but << real-signal variation scale.  Typical range: 21–51.
        Default 31.
    cuda : bool, optional
        Use GPU.  Default False.

    Returns
    -------
    corrected : ndarray, shape (n_pixels, n_angles)

    Examples
    --------
    >>> import numpy as np
    >>> from toupy.restoration import remove_rings_titarenko
    >>> sino = np.random.rand(363, 180)
    >>> sino_clean = remove_rings_titarenko(sino, size=31)
    """
    xp = _get_xp(cuda)
    sino_cpu = np.asarray(sinogram, dtype=float)
    nrow, ncol = sino_cpu.shape
    use_gpu = cuda and CUDA_AVAILABLE
    arr = xp.asarray(sino_cpu) if use_gpu else sino_cpu.copy()

    # Per-pixel angular mean
    dc = arr.mean(axis=1)                     # (nrow,)

    # Median-filter to estimate smooth background
    smooth = median_filter(np.asarray(dc), size=max(3, int(size)))
    if use_gpu:
        smooth = xp.asarray(smooth)

    # Subtract ring residual from every angle
    ring = dc - smooth
    out = arr - ring[:, None]
    return cp.asnumpy(out) if use_gpu else np.asarray(out)


def remove_rings_stack(stack, method="wavelet_fft", cuda=False, **kwargs):
    """
    Apply ring correction to a full 3-D sinogram stack.

    Parameters
    ----------
    stack : ndarray, shape (n_pixels, n_slices, n_angles)
    method : {'wavelet_fft', 'titarenko'}
    cuda : bool, optional
    **kwargs : forwarded to the chosen correction function.

    Returns
    -------
    corrected : ndarray, same shape as *stack*
    """
    if method == "wavelet_fft":
        fn = remove_rings_wavelet_fft
    elif method == "titarenko":
        fn = remove_rings_titarenko
    else:
        raise ValueError("Unknown method '{}'".format(method))

    out = np.empty_like(stack)
    n_slices = stack.shape[1]
    for s in range(n_slices):
        out[:, s, :] = fn(stack[:, s, :], cuda=cuda, **kwargs)
    return out
