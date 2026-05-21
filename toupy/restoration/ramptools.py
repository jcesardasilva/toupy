#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np

# local packages
from ..utils.funcutils import deprecated

__all__ = ["rmphaseramp", "rmlinearphase", "rmair", "normalize_projections_phase"]


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

    This is the single entry point for all phase-ramp correction tasks,
    including the role previously filled by :func:`rmlinearphase`.  Pass a
    boolean mask as ``weight`` together with ``zero_air_phase=True`` to
    replicate the old ``rmlinearphase(image, mask=mask)`` call exactly.

    Parameters
    ----------
    a : array_like
        Input complex 2-D image.
    weight : {None, 'median', 'abs', 'auto'} or array_like, optional
        Strategy for estimating the ramp slope.

        ``None`` / ``'median'``
            Global **median** phase gradient.  Robust to object
            contamination as long as air covers more than 50 % of the
            image.

        ``'abs'``
            Amplitude-weighted mean gradient (original Ptypy behaviour).

        ``'auto'``
            Automatically detect the air region from the amplitude image
            using Otsu thresholding (air = high amplitude).  Reliable even
            when the air region is small, provided amplitude contrast
            between air and object is detectable.

        array_like (boolean or float)
            **Custom mask / weight array** (same shape as ``a``).  A
            boolean or 0/1 integer array is treated as a binary air mask:
            ramp slope is estimated via the median within the mask, and
            ``zero_air_phase=True`` uses the circular mean over those
            pixels to zero the air phase.  A continuous float array uses a
            modulus-weighted mean instead.

            *Replacing* :func:`rmlinearphase` *with a mask*:
            ``rmlinearphase(image, mask=mask)``
            → ``rmphaseramp(image, weight=mask, zero_air_phase=True)``

    return_phaseramp : bool, optional
        If ``True``, also return the total correction phasor ``p``.
        Default ``False``.
    n_iter : int, optional
        Number of self-consistent refinement iterations.  Only active
        when ``weight`` is ``None`` / ``'median'`` (no prior mask
        supplied).  After the first correction, pixels whose corrected
        phase is close to zero are identified as air and used to refine
        the ramp estimate.  Default ``1`` (single pass, no refinement).
    zero_air_phase : bool, optional
        If ``True``, apply a global phase offset after ramp removal so
        that the mean phase over the air/vacuum region is exactly zero.

        The air phase offset is estimated by two strategies, chosen
        automatically:

        * **Circular mean** (used when ``weight='auto'`` or a binary mask
          is supplied): the already-computed air mask is reused.  Fast
          and precise when the mask correctly identifies the air region.
        * **Phase histogram mode** (fallback for ``weight=None/'median'/
          'abs'``): finds the dominant peak in the interior phase
          histogram via a two-step smoothed + raw argmax + parabolic
          interpolation.  Robust even without an explicit mask, and
          tolerant of bright reconstruction artefacts inside the image
          border.

        Default ``False``.
    border : int, optional
        Number of pixels to exclude from each edge before computing
        the Otsu threshold, gradient estimation, and phase-offset
        correction.  In ptychography, scan positions at the image
        boundary often lack proper overlap, producing strong noise
        artefacts that corrupt the ramp estimate.  Set this to
        approximately (beam diameter) / (scan step) pixels to remove
        these artefacts entirely.  Default ``0`` (no exclusion).

    Returns
    -------
    out : ndarray, complex
        Ramp-corrected image.
    p : ndarray, complex
        Total correction phasor (only returned when
        ``return_phaseramp=True``).

    Notes
    -----
    Forked from Ptypy (https://github.com/ptycho/ptypy), ported to
    Python 3, and extended with robust estimation strategies.

    :func:`rmlinearphase` is now a deprecated wrapper around this
    function and will be removed in a future version.

    Examples
    --------
    **Fully automatic (no mask needed):**

    >>> b = rmphaseramp(image, weight='auto', zero_air_phase=True)

    **With border exclusion (ptychography scan-boundary noise):**

    >>> b = rmphaseramp(image, weight='auto', zero_air_phase=True, border=55)

    **With a hand-drawn air mask (replaces rmlinearphase):**

    >>> b = rmphaseramp(image, weight=air_mask, zero_air_phase=True)

    **Also return the correction phasor:**

    >>> b, p = rmphaseramp(image, weight='auto', zero_air_phase=True,
    ...                    return_phaseramp=True)

    **Iterative refinement (coarse initial estimate):**

    >>> b = rmphaseramp(image, n_iter=3)
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

    # Guard every string comparison with isinstance so that passing a
    # numpy array as weight does not trigger "truth value of array is
    # ambiguous" from broadcasted == comparisons.
    _w_is_str = isinstance(weight, str)

    if weight is None or (_w_is_str and weight == 'median'):
        # Restrict global median to interior pixels when border > 0
        w = interior.astype(np.float64) if border > 0 else None
        if border > 0:
            air_mask_for_offset = interior
    elif _w_is_str and weight == 'abs':
        w = np.abs(a) * interior
    elif _w_is_str and weight == 'auto':
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

    # Optional global phase offset so that the air phase is zero.
    #
    # Two strategies depending on whether a reliable air mask is available:
    #
    # A) Binary mask known (weight='auto' or custom binary array):
    #    Circular mean of phasors over the mask pixels — wrap-safe and
    #    precise.  The mask already identifies the air region, so no
    #    further detection is needed.
    #
    # B) No mask (weight=None/'median'/'abs'):
    #    Phase histogram mode over interior pixels.  Air pixels form a
    #    sharp dominant peak; object and artefact pixels spread broadly.
    #    The peak is located via a two-step smoothed-then-raw argmax with
    #    parabolic sub-bin interpolation, using a circular (wrap-around)
    #    boxcar to handle the ±π boundary.
    if zero_air_phase:
        if air_mask_for_offset is not None:
            # --- Strategy A: circular mean over the known air mask ---
            phases_air = np.angle(a_work[air_mask_for_offset])
            peak_phase = float(np.angle(np.exp(1j * phases_air).mean()))
        else:
            # --- Strategy B: histogram mode over interior pixels ---
            phase_int = np.angle(a_work)[interior]
            n_bins = 512
            bin_width = 2.0 * np.pi / n_bins
            hist, edges = np.histogram(phase_int, bins=n_bins,
                                       range=(-np.pi, np.pi))
            centers = (edges[:-1] + edges[1:]) * 0.5
            # Step 1: smooth with a circular (wrap-around) boxcar to
            # suppress isolated noise spikes and locate the approximate
            # peak region.
            sw = 9
            padded = np.concatenate([hist[-sw // 2:], hist, hist[:sw // 2]])
            smoothed = np.convolve(padded, np.ones(sw) / sw, mode='valid')
            pk_approx = int(np.argmax(smoothed))
            # Step 2: refine — find the bin with the highest RAW count
            # within ±sw of the smoothed peak.  Parabolic interpolation
            # on the raw histogram gives sub-bin precision even when the
            # smoothed peak is flat-topped.
            lo = max(0, pk_approx - sw)
            hi = min(n_bins, pk_approx + sw + 1)
            pk = lo + int(np.argmax(hist[lo:hi]))
            if 0 < pk < n_bins - 1:
                dl = float(hist[pk - 1])
                dc = float(hist[pk])
                dr = float(hist[pk + 1])
                denom = dl - 2.0 * dc + dr
                frac = 0.5 * (dl - dr) / denom if abs(denom) > 1e-12 else 0.0
                peak_phase = float(centers[pk]) + frac * bin_width
            else:
                peak_phase = float(centers[pk])
        offset = np.exp(-1j * peak_phase)
        a_work = a_work * offset
        p_total = p_total * offset

    if return_phaseramp:
        return a_work, p_total
    return a_work


def rmlinearphase(image, mask=None, weight='auto'):
    """
    .. deprecated::
        Use :func:`rmphaseramp` instead — it covers all cases and has
        additional features (``border``, ``n_iter``, ``return_phaseramp``).

        Migration guide:

        +-----------------------------------------+--------------------------------------------------+
        | Old call                                | New equivalent                                   |
        +=========================================+==================================================+
        | ``rmlinearphase(image)``                | ``rmphaseramp(image, weight='auto')``            |
        +-----------------------------------------+--------------------------------------------------+
        | ``rmlinearphase(image, mask=mask)``     | ``rmphaseramp(image, weight=mask,                |
        |                                         | zero_air_phase=True)``                           |
        +-----------------------------------------+--------------------------------------------------+

    Remove the linear phase ramp from a complex ptychographic image.

    Parameters
    ----------
    image : array_like
        Input complex image.
    mask : array_like of bool, optional
        Boolean array marking the air/vacuum region (``True`` = air).
        If ``None``, air is detected automatically via Otsu thresholding.
    weight : str, optional
        Only used when ``mask=None``.  Passed to :func:`rmphaseramp`.
        Default ``'auto'``.

    Returns
    -------
    im_output : ndarray, complex
        Linear-ramp-corrected image.
    """
    import warnings
    warnings.warn(
        "rmlinearphase() is deprecated and will be removed in a future "
        "version.  Use rmphaseramp() instead:\n"
        "  rmlinearphase(image)           "
        "→ rmphaseramp(image, weight='auto')\n"
        "  rmlinearphase(image, mask=mask)"
        "→ rmphaseramp(image, weight=mask, zero_air_phase=True)",
        DeprecationWarning,
        stacklevel=2,
    )
    if mask is None:
        return rmphaseramp(image, weight=weight)
    return rmphaseramp(image, weight=mask, zero_air_phase=True)


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


def normalize_projections_phase(stack, border=0):
    """
    Remove residual per-projection phase offsets from a stack of projections.

    After :func:`rmphaseramp`, individual projections may still carry
    small projection-to-projection phase variations that appear as
    vertical stripes in sinograms.  This function corrects them in a
    second pass using the sinogram itself:

    * Rows that are **consistently air at all angles** have low phase
      variance across the stack (object rows vary strongly with angle).
    * Otsu thresholding on the per-row standard-deviation separates
      air rows (low std) from object rows (high std) automatically.
    * For each projection the residual offset is the circular mean phase
      over the detected air rows; this is subtracted so that all
      projections share the same air phase reference.

    Parameters
    ----------
    stack : array_like, shape (nprojs, nr, nc)
        Stack of ramp-corrected projections.  Accepted types:

        * **complex** — direct output of :func:`rmphaseramp`.  The
          function returns a complex array; extract the phase sinogram
          with ``np.angle(normalized)``.
        * **real (float)** — phase values already extracted via
          ``np.angle()``.  The function returns a float array of the same
          dtype; use it directly as the sinogram.

        .. warning::
            Do **not** pass ``np.real(rmphaseramp(...))``.  That gives
            the real part of the complex projection (``amplitude·cos φ``),
            not the phase.  Use ``np.angle(rmphaseramp(...))`` or pass
            the complex array directly.

    border : int, optional
        Number of pixels to exclude from each edge — should match the
        ``border`` value used in :func:`rmphaseramp`.  Default ``0``.

    Returns
    -------
    normalized : ndarray, shape (nprojs, nr, nc)
        Stack with per-projection offsets removed.  Same dtype as
        ``stack`` (complex if complex in, float if float in).
    offsets : ndarray, float, shape (nprojs,)
        Per-projection phase offsets that were subtracted (radians).
        Relative to the stack circular mean, so they sum to approximately
        zero.

    Examples
    --------
    **Option A — complex stack (direct output of rmphaseramp):**

    >>> corrected = np.array([
    ...     rmphaseramp(proj, weight='auto', zero_air_phase=True, border=30)
    ...     for proj in stack
    ... ])                                         # complex, shape (N, nr, nc)
    >>> corrected, offsets = normalize_projections_phase(corrected, border=30)
    >>> sinogram = np.angle(corrected)             # float sinogram

    **Option B — float phase stack:**

    >>> phase_stack = np.array([
    ...     np.angle(rmphaseramp(proj, weight='auto', zero_air_phase=True, border=30))
    ...     for proj in stack
    ... ])                                         # float, shape (N, nr, nc)
    >>> phase_stack, offsets = normalize_projections_phase(phase_stack, border=30)
    >>> sinogram = phase_stack                     # already float
    """
    stack = np.asarray(stack)
    nprojs, nr, nc = stack.shape
    b = border

    # Interior slice — exclude noisy scan-boundary pixels
    rslice = slice(b, nr - b) if b > 0 else slice(None)
    cslice = slice(b, nc - b) if b > 0 else slice(None)

    # Extract phase values for the interior region.
    #
    # CRITICAL: np.angle() applied to a real float array does NOT return
    # the phase — it returns 0.0 for positive values and π for negative
    # values (it interprets the real number as a complex number with zero
    # imaginary part).  This binary 0/π array makes the offset estimator
    # produce spurious ±π corrections that destroy the sinogram.
    #
    # Detection: if the input is already real-valued, use it directly as
    # phase; only call np.angle() for genuine complex arrays.
    is_complex = np.iscomplexobj(stack)
    interior = stack[:, rslice, cslice]
    if is_complex:
        phase_int = np.angle(interior)             # (nprojs, nr_int, nc_int)
    else:
        phase_int = interior.real.astype(np.float64)

    # Detect air rows: those with small phase variance across projections.
    # Object rows vary strongly with angle; air rows stay approximately
    # constant (any residual is the per-projection offset we want to find).
    row_std = phase_int.std(axis=(0, 2))           # shape (nr_int,)
    thresh = _otsu_threshold(row_std)
    air_row_mask = row_std < thresh                # True = consistently air

    if air_row_mask.sum() < 3:
        # Fallback: use top and bottom 10 % of interior rows
        n_fallback = max(3, (nr - 2 * b) // 10)
        air_row_mask[:] = False
        air_row_mask[:n_fallback] = True
        air_row_mask[-n_fallback:] = True

    # Per-projection offset: circular mean of phase over detected air rows.
    # Arithmetic median/mean is NOT wrap-safe: projections whose air phase
    # happens to be near ±π receive a spurious offset of ≈ ±π (sign flip).
    # Circular mean operates on unit phasors and is always wrap-safe.
    # shape: (nprojs, n_air_rows * nc_int)
    air_phases = phase_int[:, air_row_mask, :].reshape(nprojs, -1)
    mean_phasors = np.exp(1j * air_phases).mean(axis=1)    # complex, (nprojs,)
    offsets = np.angle(mean_phasors)                        # wrap-safe, (nprojs,)

    # Make offsets relative to the stack circular mean so the absolute
    # phase level is preserved (only projection-to-projection variation
    # removed).  Use circular subtraction to stay wrap-safe.
    ref_phasor = np.exp(1j * offsets).mean()                # scalar complex
    offsets = np.angle(np.exp(1j * offsets) * ref_phasor.conj())

    # Apply correction in the same domain as the input
    if is_complex:
        normalized = stack * np.exp(-1j * offsets[:, np.newaxis, np.newaxis])
    else:
        # Wrap-safe phase subtraction — stays in (−π, π]
        normalized = np.angle(
            np.exp(1j * (stack - offsets[:, np.newaxis, np.newaxis]))
        ).astype(stack.dtype)

    return normalized, offsets
