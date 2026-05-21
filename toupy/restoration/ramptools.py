#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np

# local packages
from ..utils.funcutils import deprecated

__all__ = ["rmphaseramp", "rmlinearphase", "rmair"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _otsu_threshold(arr):
    """
    Pure-NumPy Otsu threshold for a real-valued array.

    Maximises the inter-class variance between two populations (here:
    object and air/background pixels).

    Parameters
    ----------
    arr : ndarray
        1-D or N-D real array.

    Returns
    -------
    float
        Threshold value separating the two populations.
    """
    flat = arr.ravel().astype(np.float64)
    hist, edges = np.histogram(flat, bins=256)
    centers = (edges[:-1] + edges[1:]) * 0.5

    # Class probabilities (normalise by total pixel count, not bin width)
    total = float(hist.sum())
    w0 = np.cumsum(hist, dtype=np.float64) / total   # fraction of pixels in class 0
    w1 = 1.0 - w0                                     # fraction in class 1

    # Class means
    cum_mean = np.cumsum(hist * centers, dtype=np.float64) / total
    total_mean = cum_mean[-1]
    mu0 = cum_mean / np.maximum(w0, 1e-12)
    mu1 = (total_mean - cum_mean) / np.maximum(w1, 1e-12)

    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    return centers[np.argmax(sigma_b)]


def _estimate_ramp(ph, weight):
    """
    Estimate the mean phase-gradient slopes (agx, agy) from the unit
    phasor ``ph``, optionally weighted.

    Parameters
    ----------
    ph : ndarray, complex
        Unit phasor array (|ph| == 1 everywhere).
    weight : ndarray or None
        * ``None`` — global median over all pixels (robust to < 50 %
          object coverage).
        * Binary (bool or 0/1) array — median over selected pixels.
          Using median rather than masked mean avoids contamination by
          large gradient values at the object/air boundary, which have
          air-level amplitude but are not representative of the smooth
          ramp.
        * Continuous float array — modulus-weighted mean (appropriate for
          ``weight='abs'`` where values vary smoothly).

    Returns
    -------
    agx, agy : float
        Mean phase gradient along rows and columns respectively.
    """
    gx_c, gy_c = np.gradient(ph)
    ph_conj = ph.conj()
    gx = np.imag(gx_c * ph_conj)
    gy = np.imag(gy_c * ph_conj)

    if weight is None:
        # Global median — robust to up to 50 % object contamination
        agx = float(np.median(gx))
        agy = float(np.median(gy))
    elif np.array_equal(weight, weight.astype(bool)):
        # Binary mask: median within selected region — handles boundary
        # gradient artefacts that a simple masked mean cannot reject
        sel = weight.astype(bool)
        agx = float(np.median(gx[sel]))
        agy = float(np.median(gy[sel]))
    else:
        # Continuous weights (e.g. amplitude): weighted mean
        nrm = weight.sum()
        agx = float((gx * weight).sum() / nrm)
        agy = float((gy * weight).sum() / nrm)

    return agx, agy


def _build_phasor(shape, agx, agy):
    """Return the conjugate ramp phasor exp(-i*(agx*row + agy*col))."""
    xx, yy = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-1j * (agx * xx + agy * yy))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def rmphaseramp(a, weight=None, return_phaseramp=False, n_iter=1,
                zero_air_phase=False, border=0):
    """
    Remove the linear phase ramp from a complex ptychographic image.

    Parameters
    ----------
    a : array_like
        Input complex 2-D image.
    weight : {None, 'median', 'abs', 'auto'} or array_like, optional
        Strategy for estimating the ramp slope.

        ``None``
            Unweighted — uses the **median** phase gradient across all
            pixels.  Robust to object contamination as long as air covers
            more than 50 % of the image (replaces the old biased mean).

        ``'median'``
            Explicit alias for ``None``.

        ``'abs'``
            Modulus-weighted mean gradient (original behaviour).

        ``'auto'``
            Automatically detect the air region from the amplitude image
            using Otsu thresholding (air = high amplitude).  Reliable even
            when the air region is very small, provided the amplitude
            contrast between air and object is detectable.

        array_like
            Custom weight array (same shape as ``a``).

    return_phaseramp : bool, optional
        If ``True``, also return the ramp phasor ``p``.  Default ``False``.
    n_iter : int, optional
        Number of self-consistent refinement iterations.  After the first
        correction, pixels whose corrected phase is close to zero are
        identified as air and used to refine the ramp estimate.  Useful
        when the initial estimate is coarse (e.g. ``weight=None`` with a
        large object).  Default ``1`` (single pass, no refinement).
    zero_air_phase : bool, optional
        If ``True``, apply a global phase offset after ramp removal so that
        the mean phase over the detected air/vacuum region is exactly zero.
        The air region is determined automatically:

        * ``weight='auto'`` or a custom binary mask — the already-computed
          mask is reused (no extra cost).
        * All other cases — Otsu thresholding is applied to the corrected
          amplitude to detect the air region.

        This is equivalent to what :func:`rmlinearphase` does when a mask
        is supplied, but without requiring the user to draw or supply one.
        Default ``False``.
    border : int, optional
        Number of pixels to exclude from each edge before computing the
        Otsu threshold, gradient estimation, and phase-offset correction.
        In ptychography, scan positions at the image boundary often lack
        proper overlap, producing strong noise artefacts that corrupt the
        ramp estimate.  Setting ``border`` to the beam-diameter / scan-step
        ratio (in pixels) removes these artefacts entirely.  Default ``0``
        (no exclusion).

    Returns
    -------
    out : ndarray, complex
        Ramp-corrected image ``a * p``.
    p : ndarray, complex
        Ramp phasor (only when ``return_phaseramp=True``).

    Notes
    -----
    Forked from Ptypy (https://github.com/ptycho/ptypy), ported to
    Python 3, and extended with robust estimation strategies.

    Examples
    --------
    >>> b = rmphaseramp(image)
    >>> b = rmphaseramp(image, weight='auto')
    >>> b = rmphaseramp(image, weight='auto', zero_air_phase=True)
    >>> b = rmphaseramp(image, weight='auto', zero_air_phase=True, border=30)
    >>> b, p = rmphaseramp(image, return_phaseramp=True)
    >>> b = rmphaseramp(image, weight='auto', n_iter=3, zero_air_phase=True)
    """
    a = np.asarray(a)

    # Build an interior mask that excludes noisy scan-boundary pixels.
    # All subsequent weight/mask computations are restricted to this region.
    if border > 0:
        interior = np.zeros(a.shape, dtype=bool)
        interior[border:-border, border:-border] = True
    else:
        interior = np.ones(a.shape, dtype=bool)

    # Resolve weight into an array or None (→ median)
    # Also track whether we already have a binary air mask to reuse later.
    air_mask_for_offset = None   # set below when a binary mask is available

    if weight is None or weight == 'median':
        # Restrict global median to interior pixels when border > 0
        w = interior.astype(np.float64) if border > 0 else None
        if border > 0:
            air_mask_for_offset = interior
    elif weight == 'abs':
        w = np.abs(a) * interior
    elif weight == 'auto':
        amp = np.abs(a)
        # Compute Otsu threshold on interior pixels only
        thresh = _otsu_threshold(amp[interior])
        w = ((amp > thresh) & interior).astype(np.float64)
        air_mask_for_offset = w.astype(bool)
        n_air = w.sum()
        n_total = interior.sum()
        if n_air < 0.01 * n_total:
            import warnings
            warnings.warn(
                "rmphaseramp 'auto': Otsu found < 1 % air pixels "
                f"({int(n_air)}/{int(n_total)}). "
                "Consider passing an explicit mask or using weight='median'.",
                UserWarning, stacklevel=2,
            )
    else:
        w = np.asarray(weight, dtype=np.float64)
        if border > 0:
            w = w * interior          # zero out border in custom mask too
        if np.array_equal(w, w.astype(bool)):
            air_mask_for_offset = w.astype(bool)

    # Unit phasor
    absval = np.abs(a)
    ph = np.where(absval > 0, a / absval, np.ones_like(a))

    # Iterative self-consistent ramp estimation
    a_work = a.copy()
    p_total = np.ones_like(a)

    for iteration in range(n_iter):
        absval_w = np.abs(a_work)
        ph_w = np.where(absval_w > 0, a_work / absval_w, np.ones_like(a_work))

        # On iterations after the first, refine the air mask from the
        # corrected-phase residual — but only when no prior mask is available.
        # When weight='auto' or a custom mask was supplied, the original mask
        # is already reliable; replacing it with a residual-derived mask can
        # introduce instability (e.g. tilt the ramp estimate).
        w_iter = w
        if iteration > 0 and w is None:
            phase_residual = np.angle(a_work)
            residual_std = phase_residual.std()
            air_mask = np.abs(phase_residual) < residual_std
            if border > 0:
                air_mask &= interior
            if air_mask.sum() > 0.01 * air_mask.size:
                w_iter = air_mask.astype(np.float64)
            # else keep original w (None → global median)

        agx, agy = _estimate_ramp(ph_w, w_iter)
        p_iter = _build_phasor(a.shape, agx, agy)
        a_work = a_work * p_iter
        p_total = p_total * p_iter

    # Optional global phase offset so that the mean air phase is zero.
    # Uses the circular mean of unit phasors (rotation-invariant, matches
    # rmlinearphase) rather than the median of scalar angles.
    if zero_air_phase:
        if air_mask_for_offset is None:
            # Detect air from the corrected amplitude via Otsu on interior
            amp_corr = np.abs(a_work)
            thresh = _otsu_threshold(amp_corr[interior])
            air_mask_for_offset = (amp_corr > thresh) & interior
        ph_air = a_work[air_mask_for_offset]
        ph_air = ph_air / np.abs(ph_air)          # unit phasors
        mean_phasor = ph_air.mean()
        mean_phasor /= np.abs(mean_phasor)        # normalise to unit circle
        offset = mean_phasor.conj()               # rotate so mean → 1 (phase=0)
        a_work = a_work * offset
        p_total = p_total * offset

    if return_phaseramp:
        return a_work, p_total
    return a_work


def rmlinearphase(image, mask=None, weight='auto'):
    """
    Remove the linear phase ramp from a complex ptychographic image
    using a flat-region (air/vacuum) mask.

    Parameters
    ----------
    image : array_like
        Input complex image.
    mask : array_like of bool, optional
        Boolean array marking the air/vacuum region (``True`` = air).
        If ``None`` (default), the mask is generated automatically via
        Otsu thresholding of the amplitude — equivalent to passing
        ``weight='auto'`` to :func:`rmphaseramp`.
    weight : str, optional
        Only used when ``mask=None``.  Passed as the ``weight`` argument
        to :func:`rmphaseramp`.  Default ``'auto'``.

    Returns
    -------
    im_output : ndarray, complex
        Linear-ramp-corrected image.

    Notes
    -----
    When ``mask`` is supplied the function behaves as before: the ramp
    slope is estimated from the masked (air) pixels only, and the output
    phase is further corrected so that the mean phase over the mask is
    zero.  Passing ``mask=None`` is a convenience shortcut that delegates
    to :func:`rmphaseramp` with automatic air detection.
    """
    image = np.asarray(image)

    if mask is None:
        # Delegate to rmphaseramp with automatic air detection
        return rmphaseramp(image, weight=weight)

    mask = np.asarray(mask, dtype=bool)

    absval = np.abs(image)
    ph = np.where(absval > 0, image / absval, np.ones_like(image))
    ph_conj = ph.conj()
    gx_c, gy_c = np.gradient(ph)
    gx = np.imag(gx_c * ph_conj)
    gy = np.imag(gy_c * ph_conj)

    nrm = mask.sum()
    agx = float((gx * mask).sum() / nrm)
    agy = float((gy * mask).sum() / nrm)

    xx, yy = np.ogrid[:image.shape[0], :image.shape[1]]
    p = np.exp(-1j * (agx * xx + agy * yy))
    ph_corr = ph * p
    # Zero-mean phase over the air mask
    ph_corr *= np.conj((ph_corr * mask).sum() / nrm)

    return np.abs(image) * ph_corr


def rmair(image, mask):
    """
    Normalise an amplitude image so that the air/vacuum region has mean 1.

    Parameters
    ----------
    image : array_like
        Amplitude-contrast image.
    mask : array_like of bool
        Boolean array indicating the air/vacuum region.

    Returns
    -------
    normalizedimage : ndarray
        Image divided by the mean air value.
    """
    mask = np.asarray(mask, dtype=bool)
    norm_val = image[mask].mean()
    print("Normalization value: {}".format(norm_val))
    return image / norm_val
