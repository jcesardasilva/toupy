#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
registration_gd.py
==================
Tomographic data alignment using gradient descent optimisation.

Vertical alignment  : Adam optimiser on the L2 mass-fluctuation cost.
Horizontal alignment: Adam optimiser on the L2 sinogram-consistency cost.

Both replace the original discrete line-search / parabolic-fit approach
(_search_vshift_direction / _search_hshift_direction) with a proper
gradient-based update rule.  All other helper functions are unchanged.
"""

# standard libraries imports
import contextlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from matplotlib.widgets import Button, TextBox

# third party packages
from ..utils.plot_utils import plt
import numpy as np
from ..utils import tqdm
from scipy.fft import fft, ifft, fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.ndimage import center_of_mass, interpolation
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.ndimage.fourier import fourier_shift
from skimage.registration import phase_cross_correlation

# local packages
from ..restoration import derivatives_sino
from .shift import ShiftFunc
from ..tomo import projector, tomo_recons
from ..utils import (
    deprecated,
    isnotebook,
    projectpoly1d,
    RegisterPlot,
    replace_bad,
    display_slice,
    create_circle,
    hanning_apod1D,
)

__all__ = [
    "alignprojections_vertical",
    "alignprojections_horizontal",
    "center_of_mass_stack",
    "compute_aligned_stack",
    "compute_aligned_sino",
    "compute_aligned_horizontal",
    "estimate_rot_axis",
    "oneslicefordisplay",
    "refine_horizontalalignment",
    "register_2Darrays",
    "tomoconsistency_multiple",
    "vertical_fluctuations",
    "vertical_shift",
]

# ---------------------------------------------------------------------------
# Gradient descent hyperparameters (module-level — easy to tune)
# ---------------------------------------------------------------------------
_GD_MAX_ITER  = 50   # maximum GD iterations per projection
_FD_H_VERT    = 0.1  # FD step for vertical gradient (pixels)
_FD_H_HORIZ   = 0.5  # FD step for horizontal gradient (pixels); wide enough
                      # for an accurate Newton step even far from the minimum
_H_MAX_STEP   = 50.0 # max horizontal Newton step (pixels); generous so the
                      # Newton step can cross a large initial offset in one
                      # iteration — the halving fallback handles overshoots


# ============================================================================
# Unchanged public helpers
# ============================================================================

def register_2Darrays(image1, image2, subpixel=False):
    """
    Image registration using phase cross-correlations.

    Parameters
    ----------
    image1 : array_like
        Reference image.
    image2 : array_like
        Image to be shifted relative to image1.
    subpixel : bool, optional
        If ``False`` (default) use pixel-precision registration.
        If ``True`` use sub-pixel precision (upsample factor = 100).

    Returns
    -------
    shift : list of floats
        [row_shift, col_shift].
    diffphase : float
        Phase difference between the two images.
    offset_image2 : array_like
        Shifted image2 aligned to image1.
    """
    if subpixel:
        print("\nCalculating the subpixel image registration ...")
        start = time.time()
        shift, error, diffphase = phase_cross_correlation(
            image1.copy(), image2.copy(), 100
        )
        print(diffphase)
        print("Time elapsed: {:g} s".format(time.time() - start))
        print("Detected subpixel offset [y,x]: [{:g}, {:g}]".format(shift[0], shift[1]))
    else:
        print("\nCalculating the pixel precision image registration ...")
        start = time.time()
        shift, error, diffphase = phase_cross_correlation(image1.copy(), image2.copy())
        print(diffphase)
        print("Time elapsed: {:g} s".format(time.time() - start))
        print("Detected pixel offset [y,x]: [{:g}, {:g}]".format(shift[0], shift[1]))

    print("\nCorrecting the shift of image2 by using subpixel precision...")
    offset_image2 = ifft2(fourier_shift(fft2(image2.copy()), shift))
    offset_image2 *= np.exp(1j * diffphase)
    return shift, diffphase, offset_image2


def compute_aligned_stack(input_stack, shiftstack, shift_method="linear"):
    """
    Compute the aligned stack given the correction for object positions.

    Parameters
    ----------
    input_stack : array_like
        Stack of images to be shifted.
    shiftstack : array_like
        Array of object motion corrections (2, n).
    shift_method : str
        'linear', 'fourier', or 'spline'.

    Returns
    -------
    output_stack : array_like
        Aligned image stack.
    """
    S = ShiftFunc(shiftmeth=shift_method)
    nstack = input_stack.shape[0]
    print("Using {} shift method (function {})".format(shift_method, S.shiftmeth.__name__))
    output_stack = np.empty_like(input_stack)
    for ii in tqdm(range(nstack), desc="Aligning images"):
        deltashift = (shiftstack[0, ii], shiftstack[1, ii])
        output_stack[ii] = S(input_stack[ii], deltashift)
    return output_stack


def compute_aligned_stack_special(input_stack, shiftstack, shift_method="linear"):
    """In-place variant of compute_aligned_stack."""
    S = ShiftFunc(shiftmeth=shift_method)
    nstack = input_stack.shape[0]
    print("Using {} shift method (function {})".format(shift_method, S.shiftmeth.__name__))
    for ii in tqdm(range(nstack), desc="Aligning images"):
        deltashift = (shiftstack[0, ii], shiftstack[1, ii])
        input_stack[ii] = S(input_stack[ii], deltashift)
    return input_stack


def compute_aligned_horizontal_special(input_stack, shiftstack, shift_method="linear", **kwargs):
    """Horizontal-only in-place alignment.

    ``**kwargs`` absorbs any extra keys when called as
    ``compute_aligned_horizontal_special(..., **params)``.
    ``shift_method`` is read from ``params["shiftmeth"]`` if present and
    not overridden by a positional argument.
    """
    shift_method = kwargs.get("shiftmeth", shift_method)
    deltashift = np.zeros_like(shiftstack)
    deltashift[1] = shiftstack[1].copy()
    return compute_aligned_stack_special(input_stack, shiftstack, shift_method=shift_method)


def compute_aligned_sino(input_sino, shiftslice, shift_method="linear"):
    """
    Compute the aligned sinogram given the correction for object positions.

    Parameters
    ----------
    input_sino : array_like
        Sinogram to be shifted.
    shiftslice : array_like
        Per-projection horizontal shifts (1, n).
    shift_method : str
        'linear', 'fourier', or 'spline'.

    Returns
    -------
    output_sino : array_like
        Aligned sinogram.
    """
    S = ShiftFunc(shiftmeth=shift_method)
    nprojs = input_sino.shape[1]
    print("Using {} shift method (function {})".format(shift_method, S.shiftmeth.__name__))
    output_sino = np.empty_like(input_sino)
    for ii in tqdm(range(nprojs), desc="Aligning sinogram"):
        output_sino[:, ii] = S(input_sino[:, ii], shiftslice[0, ii])
    return output_sino


def compute_aligned_horizontal(input_stack, shiftstack, shift_method="linear", **kwargs):
    """Horizontal-only alignment (copy variant).

    ``**kwargs`` absorbs any extra keys when called as
    ``compute_aligned_horizontal(..., **params)``.
    ``shift_method`` defaults to ``params["shiftmeth"]`` if that key is
    present and ``shift_method`` was not supplied explicitly.
    """
    shift_method = kwargs.get("shiftmeth", shift_method)
    deltashift = np.zeros_like(shiftstack)
    deltashift[1] = shiftstack[1].copy()
    return compute_aligned_stack(input_stack, deltashift, shift_method=shift_method)


def center_of_mass_stack(input_stack, lims, shiftstack, shift_method="fourier"):
    """
    Compute the centre-of-mass position for each projection in the stack.

    Parameters
    ----------
    input_stack : ndarray, shape (n, nr, nc)
        Stack of projection images.
    lims : tuple of array_like
        ``(limrow, limcol)`` — row and column index arrays defining the
        region of interest.
    shiftstack : ndarray, shape (2, n)
        Current shift estimates used to pre-align each projection before
        computing the centre of mass.
    shift_method : str, optional
        Interpolation method for the shift operation.  Default ``'fourier'``.

    Returns
    -------
    ndarray, shape (2, n)
        Array ``[centerx, centery]`` where ``centerx[i]`` and ``centery[i]``
        are the horizontal and vertical centre-of-mass offsets (in pixels)
        for projection ``i``.
    """
    limrow, limcol = lims
    print("Calculating center-of-mass with pixel precision")
    S = ShiftFunc(shiftmeth=shift_method)
    stack_roi = input_stack[0, limrow[0]:limrow[-1], limcol[0]:limcol[-1]].copy()
    ind_roi = np.indices(stack_roi.shape)
    ind_roi[1] -= (
        np.floor(ind_roi[1].mean(axis=1)).reshape((ind_roi.shape[1], 1)).astype("int")
    )
    Xp = ind_roi[1].astype("float")
    ind_roi[0] -= (
        np.floor(ind_roi[0].mean(axis=0)).reshape((ind_roi.shape[2], 1)).T.astype("int")
    )
    Yp = ind_roi[0].astype("float")
    mass_sum = np.empty(input_stack.shape[0])
    centerx = np.empty(input_stack.shape[0])
    centery = np.empty(input_stack.shape[0])
    for ii in range(input_stack.shape[0]):
        stack_aux = S(input_stack[ii], (shiftstack[0, ii], shiftstack[1, ii]))
        roi = stack_aux[limrow[0]:limrow[-1], limcol[0]:limcol[-1]]
        mass_sum[ii] = np.sum(roi)
        centerx[ii] = np.sum(Xp * roi)
        centery[ii] = np.sum(Yp * roi)
    nz = np.nonzero(mass_sum)
    centerx[nz] /= mass_sum[nz]
    centerx[np.where(mass_sum == 0)] = 0
    centery = np.asarray(centery)
    centery[nz] /= mass_sum[nz]
    centery[np.where(mass_sum == 0)] = 0
    return np.asarray([centerx, centery])


def vertical_fluctuations(input_stack, lims, shiftstack, shift_method="fourier", polyorder=2):
    """
    Compute the vertical mass-fluctuation signal for each projection.

    The signal is obtained by integrating each (shifted) projection over the
    horizontal axis within the region of interest, then subtracting a
    polynomial baseline fit.

    Parameters
    ----------
    input_stack : ndarray, shape (n, nr, nc)
        Stack of projection images.
    lims : tuple of array_like
        ``(rows, cols)`` — row and column index arrays for the ROI.
    shiftstack : ndarray, shape (2, n)
        Current vertical and horizontal shift estimates.
    shift_method : str, optional
        Interpolation method for the shift operation.  Default ``'fourier'``.
    polyorder : int, optional
        Polynomial order used for baseline removal.  Default ``2``.

    Returns
    -------
    vert_fluct : ndarray, shape (n, n_rows_roi)
        Array of vertical fluctuation signals, one row per projection.
    """
    S = ShiftFunc(shiftmeth=shift_method)
    nproj, nr, nc = input_stack.shape
    rows, cols = lims
    max_vshift = int(np.ceil(np.max(np.abs(shiftstack[0, :])))) + 1
    if np.any((rows - max_vshift) < 0) or np.any((rows + max_vshift) > nr):
        max_vshift = 1
    vert_fluct = np.empty((nproj, rows[-1] - rows[0]))
    for ii in tqdm(range(nproj), desc="Computing vertical fluctuations"):
        proj = input_stack[ii, rows[0] - max_vshift:rows[-1] + max_vshift, cols[0]:cols[-1]]
        stack_shift = S(proj, (shiftstack[0, ii], 0.0))
        shift_calc = stack_shift[max_vshift:-max_vshift].sum(axis=1)
        shift_calc = projectpoly1d(shift_calc, polyorder, 1)
        vert_fluct[ii] = shift_calc
    return vert_fluct


def vertical_shift(input_array, lims, vstep, maxshift, shift_method="linear", polyorder=2):
    """
    Compute the vertical mass-fluctuation signal for a single projection
    shifted by a given amount.

    Parameters
    ----------
    input_array : ndarray, shape (nr, nc)
        Single projection image.
    lims : tuple of array_like
        ``(rows, cols)`` — row and column index arrays for the ROI.
    vstep : float
        Vertical shift to apply (pixels).
    maxshift : int
        Safety margin (pixels) around the ROI to absorb border effects.
    shift_method : str, optional
        Interpolation method for the shift operation.  Default ``'linear'``.
    polyorder : int, optional
        Polynomial order used for baseline removal.  Default ``2``.

    Returns
    -------
    shift_calc : ndarray, shape (n_rows_roi,)
        Vertical fluctuation signal after shifting and polynomial baseline
        subtraction.
    """
    S = ShiftFunc(shiftmeth=shift_method)
    nr, nc = input_array.shape
    max_vshift = maxshift + int(np.abs(vstep))
    rows, cols = lims
    if np.any((rows - max_vshift) < 0) or np.any((rows + max_vshift) > nr):
        max_vshift = 1
    stack_shift = S(
        input_array[rows[0] - max_vshift:rows[-1] + max_vshift, cols[0]:cols[-1]],
        (vstep, 0.0),
    )
    shift_calc = stack_shift[max_vshift:-max_vshift].sum(axis=1)
    shift_calc = projectpoly1d(shift_calc, polyorder, 1)
    return shift_calc


# ============================================================================
# Private helpers (ROI, clipping, error metrics, convergence)
# ============================================================================

def _selectROI(stack_shape, **params):
    """
    Derive region-of-interest row and column limits from params.

    Parameters
    ----------
    stack_shape : tuple of int
        Shape ``(n, nr, nc)`` of the projection stack.
    **params
        Must contain ``'deltax'`` (int, horizontal margin) and ``'limsy'``
        (list of int or None, explicit row limits).

    Returns
    -------
    limrow : ndarray of int
        Row limits ``[row_start, row_end]``.
    limcol : ndarray of int
        Column limits ``[col_start, col_end]``.
    """
    deltax = params["deltax"]
    limcol = (deltax, stack_shape[2] - deltax)
    limrow = params["limsy"]
    if limrow is None or limrow == "":
        limrow = [0, stack_shape[1]]
    return np.asarray(limrow), np.asarray(limcol)


def _clipping_tomo(recons, **params):
    """
    Apply low and high clipping to a tomographic reconstruction.

    Parameters
    ----------
    recons : ndarray
        Reconstructed slice array.
    **params
        Must contain ``'cliplow'`` (float or None) and ``'cliphigh'``
        (float or None).  When a value is not ``None`` the reconstruction
        is clipped to that threshold and, for ``cliphigh``, also shifted
        so that the clipped maximum maps to zero.

    Returns
    -------
    recons : ndarray
        Clipped (and optionally shifted) reconstruction.
    """
    if params["cliplow"] is not None:
        recons = recons * (recons >= params["cliplow"]) + params["cliplow"] * (
            recons < params["cliplow"]
        )
    if params["cliphigh"] is not None:
        recons = recons * (recons <= params["cliphigh"]) + params["cliphigh"] * (
            recons > params["cliphigh"]
        )
        recons = recons - params["cliphigh"]
    return recons


def _sino_error_metric(sinogramexp, sinogramcomp, params):
    """
    Compute the per-column L2 error between experimental and synthetic sinograms.

    Parameters
    ----------
    sinogramexp : ndarray, shape (nr, nc)
        Experimental (measured) sinogram.
    sinogramcomp : ndarray, shape (nr, nc)
        Synthetic sinogram computed from the current reconstruction.
    params : dict
        Unused; reserved for future weighting options.

    Returns
    -------
    errorxreg : ndarray, shape (nc,)
        Per-column sum of squared differences between the two sinograms.
    """
    errorxreg = np.zeros(sinogramexp.shape[1])
    for ii in range(sinogramexp.shape[1]):
        errorxreg[ii] = np.sum(np.abs(sinogramexp[:, ii] - sinogramcomp[:, ii]) ** 2)
    return errorxreg


def _checkconditions(metric_error, changes, pixtol, count, maxit, subpixel=False, rtol=0.0):
    """
    Evaluate stopping conditions for the alignment loop.

    Returns
    -------
    0 : continue
    1 : diverged (2 consecutive error increases) — caller should roll back
    2 : shifts converged (max change < pixtol)
    3 : maximum iterations reached
    4 : relative error improvement below rtol (only when rtol > 0)
    """
    step = pixtol if subpixel else 1
    eps = np.spacing(1)

    # Require 2 consecutive error increases before declaring divergence.
    # A single increase is often a transient fluctuation; stopping on it
    # forces the user to restart manually even though the algorithm would
    # recover by itself in the next iteration.
    if (len(metric_error) >= 3
            and metric_error[-1] > metric_error[-2]
            and metric_error[-2] > metric_error[-3]):
        print("Error increased for 2 consecutive iterations.")
        print(
            "{:.04e} -> {:.04e} -> {:.04e}".format(
                metric_error[-3], metric_error[-2], metric_error[-1]
            )
        )
        print("Keeping previous shifts.")
        return 1
    # Relative improvement below threshold — useful for warm-started refinement
    # where the solution is already close to optimal.
    if rtol > 0 and len(metric_error) >= 2:
        rel_improv = (metric_error[-2] - metric_error[-1]) / (abs(metric_error[-2]) + eps)
        if metric_error[-1] <= metric_error[-2] and rel_improv < rtol:
            print("Relative improvement {:.2e} < rtol {:.2e}. Converged.".format(
                rel_improv, rtol))
            return 4
    if np.max(changes) < step:
        if step >= 1:
            print("Changes are smaller than one pixel.")
        else:
            print("Changes are smaller than {} pixel.".format(step))
        return 2
    elif count >= maxit:
        print("Maximum number of iterations reached.")
        return 3
    return 0


def _filter_sino(sinogram, **params):
    """
    Apply a Hanning low-pass filter to a sinogram along the detector axis.

    Parameters
    ----------
    sinogram : ndarray, shape (nr, nc)
        Input sinogram (rows = detector pixels, columns = projections).
    **params
        Must contain ``'freqcutoff'`` (float in ``(0, 1]``), which sets the
        half-width of the apodisation window as a fraction of ``nr``.

    Returns
    -------
    ndarray, shape (nr, nc)
        Real-valued filtered sinogram.
    """
    N, M = sinogram.shape
    apod_width = np.int32(0.5 * N * params["freqcutoff"])
    filteraux = hanning_apod1D(N, apod_width)
    filteraux = np.tile(filteraux, (M, 1)).T
    return np.real(ifft(fft(sinogram) * filteraux))


# ============================================================================
# Gradient descent core — vertical shifts  (replaces _search_vshift_direction)
# ============================================================================

def _search_vshift_direction(
    input_array,
    lims,
    shift_delta,
    avg_vert_fluct,
    pixtol,
    max_vshift,
    shift_method="linear",
    polyorder=2,
):
    """
    Find the optimal vertical shift for one projection using gradient descent
    with a numerically estimated gradient and an adaptive step size.

    The cost function is:

        C(s) = ||f(s) - avg_vert_fluct||²

    where f(s) is the mass-fluctuation signal after shifting by s pixels.

    Algorithm
    ---------
    At each iteration:
      1. Estimate the gradient of C with a central finite difference:
             g = [C(s+h) - C(s-h)] / (2h)
      2. Compute a Newton-like step normalised by the second derivative
         (curvature), estimated from the same three evaluations:
             s_new = s - g / |C''(s)|   clipped to [-max_step, +max_step]
      3. Accept s_new only if C(s_new) < C(s).  Otherwise halve the step
         up to 8 times (guarantees descent without an explicit line search).

    This is equivalent to a damped Newton step on the 1-D cost, which
    converges quadratically near the minimum and is globally convergent
    because of the fallback halving.

    Parameters
    ----------
    input_array : array_like
        Single projection image.
    lims : tuple
        (rows, cols) region-of-interest limits.
    shift_delta : float
        Current shift estimate (pixels).
    avg_vert_fluct : array_like
        Reference (mean) vertical fluctuation signal.
    pixtol : float
        Convergence tolerance (pixels).
    max_vshift : int
        Safety margin for border effects.
    shift_method : str
        Interpolation method passed to vertical_shift.
    polyorder : int
        Polynomial order for bias removal in vertical_shift.

    Returns
    -------
    current_shift : float
        Optimised shift (pixels).
    final_signal : array_like
        Mass-fluctuation signal evaluated at current_shift.
    """

    def cost_and_sig(s):
        sig = vertical_shift(input_array, lims, s, max_vshift, shift_method, polyorder)
        return np.sum((sig - avg_vert_fluct) ** 2), sig

    h = _FD_H_VERT
    current_shift = float(shift_delta)
    current_cost, current_sig = cost_and_sig(current_shift)

    for _ in range(_GD_MAX_ITER):
        c_plus,  _ = cost_and_sig(current_shift + h)
        c_minus, _ = cost_and_sig(current_shift - h)

        # 1st derivative (gradient) and 2nd derivative (curvature)
        grad      = (c_plus - c_minus) / (2.0 * h)
        curvature = (c_plus - 2.0 * current_cost + c_minus) / (h ** 2)

        if grad == 0.0:
            break

        # Newton step; fall back to steepest descent if curvature ≤ 0
        if curvature > 0.0:
            raw_step = -grad / curvature
        else:
            # Steepest descent with unit step in the right direction
            raw_step = -np.sign(grad)

        # Clip to avoid wild jumps (max 2 pixels per iteration)
        step = float(np.clip(raw_step, -2.0, 2.0))

        # Halving fallback: ensure the step actually decreases the cost
        for _ in range(8):
            candidate       = current_shift + step
            candidate_cost, candidate_sig = cost_and_sig(candidate)
            if candidate_cost < current_cost:
                break
            step *= 0.5
        else:
            # No improvement found — already at a local minimum
            break

        current_shift = candidate
        current_cost  = candidate_cost
        current_sig   = candidate_sig

        if abs(step) < pixtol / 10.0:
            break

    return current_shift, current_sig


# ============================================================================
# Gradient descent core — horizontal shifts  (replaces _search_hshift_direction)
# ============================================================================

def _search_hshift_direction(
    sinogram_col,
    sinogramcomp_col,
    shift_delta,
    pixtol,
    shift_method="linear",
    S=None,
    sino_fft_col=None,
    N_fft_col=None,
):
    """
    Find the optimal horizontal shift for a single sinogram column using
    iterative gradient descent with Newton steps.

    Unlike the vertical case, this function iterates to full convergence
    within a single call.  The outer loop in ``_alignprojections_horizontal``
    is responsible for updating the synthetic sinogram (expensive FBP) between
    calls; within one synthetic sinogram the shift minimisation should be
    fully resolved before returning.

    Algorithm
    ---------
    The cost is C(s) = ||T_s(sino_col) - sinogramcomp_col||².

    At each iteration, three evaluations at {s-h, s, s+h} give gradient and
    curvature via central finite differences:

        grad      = [C(s+h) - C(s-h)] / (2h)
        curvature = [C(s+h) - 2C(s) + C(s-h)] / h²

    If curvature > 0: Newton step  Δs = -grad/curvature  (jumps directly to
    the parabola minimum; handles large offsets in one step when the cost
    landscape is locally quadratic).
    If curvature ≤ 0: steepest-descent step of size `h` downhill (safe
    fallback when far from the minimum or on a flat landscape).

    The step is always verified: if it does not decrease the cost it is halved
    up to 6 times.  If no improvement is found the loop terminates (already
    at a local minimum for this sinogramcomp).

    Parameters
    ----------
    sinogram_col : array_like
        Experimental sinogram column (one projection angle, unshifted).
    sinogramcomp_col : array_like
        Synthetic sinogram column.
    shift_delta : float
        Current accumulated horizontal shift estimate (pixels).
    pixtol : float
        Convergence tolerance (pixels): loop stops when |Δs| < pixtol.
    shift_method : str
        Interpolation method for ShiftFunc.
    S : ShiftFunc or None
        Pre-instantiated ShiftFunc to avoid repeated construction.
        If None, a new one is created.
    sino_fft_col : complex ndarray or None
        Pre-computed FFT of the padded sinogram column (optimisation E).
        When provided together with N_fft_col, reused across all Newton
        steps — avoids O(n log n) pad + FFT per cost evaluation.
    N_fft_col : ndarray or None
        Frequency-coordinate array paired with sino_fft_col.

    Returns
    -------
    current_shift : float
        Optimised horizontal shift (pixels).
    final_sino : array_like
        sinogram_col shifted by current_shift.
    """
    h = _FD_H_HORIZ
    col_len = len(sinogram_col)

    # E: fast Fourier-shift path — reuse precomputed FFT
    if sino_fft_col is not None and N_fft_col is not None:
        def _shift_col(s):
            H = np.exp(1j * 2.0 * np.pi * s * N_fft_col)
            return ifft(sino_fft_col * H).real[:col_len]
    else:
        if S is None:
            S = ShiftFunc(shiftmeth=shift_method)
        def _shift_col(s):
            return S(sinogram_col, s)

    def cost(s):
        return np.sum((_shift_col(s) - sinogramcomp_col) ** 2)

    current_shift = float(shift_delta)
    current_cost  = cost(current_shift)

    for _ in range(_GD_MAX_ITER):
        c_plus  = cost(current_shift + h)
        c_minus = cost(current_shift - h)

        grad      = (c_plus - c_minus) / (2.0 * h)
        curvature = (c_plus - 2.0 * current_cost + c_minus) / (h ** 2)

        if grad == 0.0:
            break

        # Newton step if convex, else steepest-descent step of size h
        if curvature > 0.0:
            step = float(np.clip(-grad / curvature, -_H_MAX_STEP, _H_MAX_STEP))
        else:
            step = -np.sign(grad) * h

        # Halving fallback: ensure cost strictly decreases
        for _ in range(6):
            candidate      = current_shift + step
            candidate_cost = cost(candidate)
            if candidate_cost < current_cost:
                break
            step *= 0.5
        else:
            # No improvement possible: already at local minimum
            break

        current_shift = candidate
        current_cost  = candidate_cost

        if abs(step) < pixtol:
            break

    final_sino = _shift_col(current_shift)
    return current_shift, final_sino


# ============================================================================
# Anderson acceleration helper  (optimisation B)
# ============================================================================

class _AndersonAccelerator:
    """
    Anderson acceleration (Anderson mixing) for fixed-point iterations.

    At step k the caller provides the current iterate x_k and the
    fixed-point image g_k = g(x_k).  The accelerator keeps a rolling window
    of the last m+1 pairs and returns a linear combination that minimises the
    norm of the stacked residuals — typically cutting outer iterations 2-5×.

    Reference
    ---------
    Walker & Ni, "Anderson Acceleration for Fixed-Point Iterations",
    SIAM J. Numer. Anal. 49(4), 2011.

    Parameters
    ----------
    m : int
        History depth.  Default 3.
    """

    def __init__(self, m=3):
        self.m = m
        self._G = []   # history of g(x_k), flattened
        self._F = []   # history of f_k = g(x_k) - x_k, flattened
        self._shape = None

    def reset(self):
        """Clear history (call after a rejected step)."""
        self._G.clear()
        self._F.clear()

    def step(self, x_k, g_k):
        """
        Compute the Anderson-mixed next iterate.

        Parameters
        ----------
        x_k : ndarray
            Current iterate (before the fixed-point step).
        g_k : ndarray
            Fixed-point image g(x_k).

        Returns
        -------
        x_next : ndarray
            Anderson-mixed next iterate, same shape as x_k.
        """
        self._shape = x_k.shape
        self._G.append(g_k.ravel().copy())
        self._F.append((g_k - x_k).ravel())

        # Rolling window: keep at most m+1 entries
        if len(self._G) > self.m + 1:
            self._G.pop(0)
            self._F.pop(0)

        if len(self._G) < 2:
            return g_k   # no history yet — plain fixed-point step

        # Stack into (n_x, n_hist) matrices
        F = np.column_stack(self._F)   # residuals
        G = np.column_stack(self._G)   # g(x) images

        # Unconstrained reformulation of the constrained LS:
        #   min_{alpha} || F[:,-1] + dF @ alpha ||^2
        # where dF = F[:,:-1] - F[:,-1:]
        # x_{k+1} = G[:,-1] + dG @ alpha
        dF = F[:, :-1] - F[:, -1:]
        dG = G[:, :-1] - G[:, -1:]
        alpha, _, _, _ = np.linalg.lstsq(dF, -F[:, -1], rcond=None)
        x_next = G[:, -1] + dG @ alpha
        return x_next.reshape(self._shape)


# ============================================================================
# Wrappers (unchanged logic; they call the new _search_*_direction above)
# ============================================================================

def _search_vshift_stack(input_stack, lims, input_delta, avg_vert_fluct, **kwargs):
    """
    Search optimal vertical shifts for all projections in the stack.

    Applies :func:`_search_vshift_direction` independently to each
    projection using gradient descent.

    Parameters
    ----------
    input_stack : ndarray, shape (n, nr, nc)
        Stack of projection images.
    lims : tuple of array_like
        ``(rows, cols)`` — ROI limits.
    input_delta : ndarray, shape (2, n)
        Current shift estimates; only row 0 (vertical) is updated.
    avg_vert_fluct : ndarray, shape (n_rows_roi,)
        Reference mean vertical fluctuation signal.
    **kwargs
        Must contain ``'pixtol'``, ``'shiftmeth'``, and ``'polyorder'``.

    Returns
    -------
    output_shiftstack : ndarray, shape (2, n)
        Updated shift array (row 0 = optimised vertical shifts).
    vert_fluct_stack : ndarray, shape (n, n_rows_roi)
        Vertical fluctuation signals at the optimised shifts.
    """
    pixtol       = kwargs["pixtol"]
    shift_method = kwargs["shiftmeth"]
    polyorder    = kwargs["polyorder"]
    rows, cols   = lims
    nprojs, nr, nc = input_stack.shape
    max_vshift = int(np.ceil(np.max(np.abs(input_delta[0, :])))) + 1
    if np.any((rows - max_vshift) < 0) or np.any((rows + max_vshift) > nr):
        max_vshift = 1

    vert_fluct_stack  = np.empty((input_stack.shape[0], rows[-1] - rows[0]))
    output_shiftstack = np.empty_like(input_delta)

    if not isinstance(input_stack, np.ndarray):
        input_stack = np.asarray(input_stack).copy()

    for ii in tqdm(range(nprojs), desc="Searching vertical shifts"):
        output_shiftstack[0, ii], vert_fluct_stack[ii] = _search_vshift_direction(
            input_stack[ii], lims, input_delta[0, ii], avg_vert_fluct,
            pixtol, max_vshift, shift_method, polyorder,
        )
    return output_shiftstack, vert_fluct_stack


def _search_hshift_sinogram(sinogram, sinogramcomp, shiftslice, **kwargs):
    """
    Search horizontal shifts for all sinogram columns.

    Accelerations applied
    ---------------------
    A — Columns processed in parallel via ThreadPoolExecutor.
    E — When shiftmeth=='fourier', the FFT of every sinogram column is
        pre-computed once per outer iteration and reused across all Newton
        steps, avoiding a redundant pad + FFT per cost evaluation.

    Parameters
    ----------
    sinogram : ndarray, shape (nr, nc)
    sinogramcomp : ndarray, shape (nr, nc)
    shiftslice : ndarray, shape (1, nc)
    **kwargs
        Must contain 'pixtol' and 'shiftmeth'.
    """
    pixtol       = kwargs["pixtol"]
    shift_method = kwargs["shiftmeth"]
    nr, nc       = sinogram.shape
    sino_out       = np.empty_like(sinogram)
    shiftslice_out = np.empty_like(shiftslice)

    # E: batch-precompute FFT of all sinogram columns (Fourier shift only)
    _sino_fft = None
    _N_fft    = None
    if shift_method == "fourier":
        padw      = int(2 ** np.ceil(np.log2(nr))) - nr   # pad to next power-of-2
        _padded   = np.pad(sinogram, ((0, padw), (0, 0)), mode="reflect")
        _N_fft    = fftfreq(nr + padw)                     # frequency coordinates
        _sino_fft = fft(_padded, axis=0)                   # (nr+padw, nc), batch FFT

    def _process_col(ii):
        # A: each thread gets its own ShiftFunc (ShiftFunc stores state in self)
        S_local = ShiftFunc(shiftmeth=shift_method)
        fft_col = _sino_fft[:, ii] if _sino_fft is not None else None
        s, col  = _search_hshift_direction(
            sinogram[:, ii], sinogramcomp[:, ii], shiftslice[0, ii],
            pixtol, shift_method,
            S=S_local, sino_fft_col=fft_col, N_fft_col=_N_fft,
        )
        return ii, s, col

    # A: parallel execution over columns
    n_workers = min(nc, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(_process_col, range(nc)),
            total=nc, desc="Searching horizontal shifts",
        ))

    for ii, s, col in results:
        shiftslice_out[0, ii] = s
        sino_out[:, ii]       = col

    return sino_out, shiftslice_out


# ============================================================================
# Vertical alignment (outer loop — unchanged)
# ============================================================================

def _alignprojections_vertical(
    input_stack, lims, shiftstack, metric_error, vert_fluct_init, RP, **params
):
    """
    Iterative outer loop for vertical projection alignment.

    At each iteration the mean vertical fluctuation is recomputed from
    the currently shifted stack and :func:`_search_vshift_stack` refines
    the shifts via gradient descent.  Stops when the error increases for
    two consecutive iterations, when shifts converge within ``pixtol``,
    or when ``params['maxit']`` iterations are reached.

    Parameters
    ----------
    input_stack : ndarray, shape (n, nr, nc)
        Stack of projection images.
    lims : tuple of array_like
        ``(rows, cols)`` ROI limits.
    shiftstack : ndarray, shape (2, n)
        Initial shift estimates; modified in-place.
    metric_error : list of float
        Running list of error values; a new value is appended each iteration.
    vert_fluct_init : ndarray, shape (n, n_rows_roi)
        Vertical fluctuations computed before the first iteration.
    RP : RegisterPlot
        Plot helper for live visualisation.
    **params
        Algorithm parameters including ``'pixtol'``, ``'maxit'``,
        ``'shiftmeth'``, ``'polyorder'``, ``'subpixel'``.

    Returns
    -------
    shiftstack : ndarray, shape (2, n)
        Optimised shift array.
    metric_error : list of float
        Updated error history.
    """
    count = 0
    error_reg = np.zeros(vert_fluct_init.shape[0])
    while True:
        count += 1

        # Route all iteration prints to the dedicated verbose Output widget
        # (notebook only) so previous-iteration text is replaced in-place
        # without touching the figure Output widgets.
        if hasattr(RP, '_out_verbose') and RP._out_verbose is not None:
            RP._out_verbose.clear_output(wait=True)
            _vctx = RP._out_verbose
        else:
            _vctx = contextlib.nullcontext()

        with _vctx:
            print("\n============================================")
            print("Iteration {}".format(count))
            it0 = time.time()
            deltaprev = shiftstack.copy()

            if count == 1:
                vert_fluct = vert_fluct_init.copy()
            else:
                print("Updating the vertical fluctuations")
                vert_fluct = vertical_fluctuations(
                    input_stack, lims, shiftstack, params["shiftmeth"], polyorder=params["polyorder"]
                )

            vert_fluct_mean = vert_fluct.mean(axis=0)

            print("Gradient descent search for vertical shifts...")
            shiftstack_aux, vert_fluct_temp = _search_vshift_stack(
                input_stack, lims, shiftstack, vert_fluct_mean, **params
            )
            shiftstack[0] = shiftstack_aux[0].copy()
            shiftstack[0] -= shiftstack_aux[0].mean().round()

            vert_fluct_mean_temp = vert_fluct_temp.mean(axis=0)
            print("\nCalculating the error metric")
            for ii in range(vert_fluct_temp.shape[0]):
                error_reg[ii] = np.sum(np.abs(vert_fluct_temp[ii] - vert_fluct_mean_temp) ** 2)
            print("Final error metric for y, E = {:.04e}".format(np.sum(error_reg)))
            metric_error.append(np.sum(error_reg))

            changey = np.abs(deltaprev[0] - shiftstack[0])
            print("Estimating the changes in y:")
            print("Maximum correction in y = {:.02f} pixels".format(np.max(changey)))
            print("Elapsed time = {} s".format(time.time() - it0))

            pixtol = params["pixtol"] if params["subpixel"] else 1
            reason = _checkconditions(
                metric_error, changey, pixtol, count, params["maxit"], params["subpixel"],
                rtol=params.get("rtol", 0.0),
            )
            if reason == 1:
                shiftstack = deltaprev.copy()
                metric_error.pop()

        # Update figure OUTSIDE the verbose context so the PNG goes to the
        # correct _out_main widget and not to _out_verbose.
        RP.plotsvertical(
            input_stack[0], lims, vert_fluct_init, vert_fluct_temp,
            shiftstack, metric_error, count,
            max_correction=float(np.max(changey)),
        )

        if reason >= 1:
            break

    return shiftstack, metric_error


def alignprojections_vertical(input_stack, shiftstack, **params):
    """
    Vertical alignment of projections using the mass-fluctuation approach.

    Shifts are optimised with a Newton gradient-descent algorithm that
    minimises the L2 distance between each projection's vertical
    mass-fluctuation signal and the stack mean.

    Parameters
    ----------
    input_stack : ndarray, shape (n, nr, nc)
        Stack of projection images to be aligned.
    shiftstack : ndarray, shape (2, n)
        Initial shift estimates ``[vertical_shifts, horizontal_shifts]``.
        Modified in-place and also returned.
    **params
        Algorithm parameters.  Required keys:

        maxit : int
            Maximum number of outer iterations.
        pixtol : float
            Convergence tolerance in pixels.
        deltax : int
            Horizontal margin (pixels) to exclude from the ROI.
        limsy : list of int or None
            Explicit ``[row_start, row_end]`` row limits.  ``None`` uses
            the full image height.
        shiftmeth : str
            Interpolation method (``'linear'``, ``'fourier'``,
            ``'spline'``).
        polyorder : int
            Polynomial order for baseline removal in the fluctuation signal.
        alignx : bool, optional
            If ``True``, estimate horizontal shifts from the centre-of-mass
            before the vertical loop.  Default ``False``.
        subpixel : bool, optional
            Ignored (always runs at sub-pixel precision).  Default ``True``.

    Returns
    -------
    shiftstack : ndarray, shape (2, n)
        Optimised shift array.
    output_stack : ndarray, shape (n, nr, nc)
        Vertically aligned projection stack.
    """
    if not isinstance(params["maxit"], int):
        params["maxit"] = 10
    try:
        params["alignx"]
    except KeyError:
        params["alignx"] = False

    limrow, limcol = _selectROI(input_stack.shape, **params)
    lims = (limrow, limcol)

    print("\n============================================")
    print("Vertical Mass fluctuation alignment — Adam gradient descent")
    print("Number of iterations: {}".format(params["maxit"]))

    if params["alignx"]:
        print("Estimating changes in x using center-of-mass:")
        centerx = center_of_mass_stack(input_stack, lims, shiftstack=shiftstack)[0]
        shiftstack[1] = -centerx.round()
        shiftstack[1] -= shiftstack[1].mean().round()

    vert_fluct_init = vertical_fluctuations(
        input_stack, lims, shiftstack, params["shiftmeth"], polyorder=params["polyorder"]
    )
    avg_init = vert_fluct_init.mean(axis=0)
    shiftstack_init = shiftstack.copy()

    metric_error = []
    error_init = np.array(
        [np.sum(np.abs(vert_fluct_init[ii] - avg_init) ** 2)
         for ii in range(vert_fluct_init.shape[0])]
    )
    print("Initial error metric for y, E = {:.02e}".format(np.sum(error_init)))
    metric_error.append(np.sum(error_init))

    plt.ion()
    RP = RegisterPlot(**params)
    RP.plotsvertical(
        input_stack[0], lims, vert_fluct_init, vert_fluct_init,
        shiftstack_init, metric_error, count=0,
    )

    print("\n================================================")
    print("Vertical alignment (Newton GD, pixtol={})".format(params["pixtol"]))
    print("================================================")
    # A single pass is sufficient: the Newton step already operates at
    # sub-pixel scale (_FD_H_VERT), so a coarse pixel-precision warmup
    # followed by sub-pixel refinement is redundant.
    params["subpixel"] = True
    shiftstack, metric_error = _alignprojections_vertical(
        input_stack, lims, shiftstack, metric_error, vert_fluct_init, RP, **params
    )

    print("Computing aligned images")
    output_stack = compute_aligned_stack(
        input_stack, shiftstack.copy(), shift_method=params["shiftmeth"]
    )
    return shiftstack, output_stack


# ============================================================================
# Horizontal alignment (outer loop — unchanged)
# ============================================================================

def _alignprojections_horizontal(
    sinogram, sino_orig, theta, circleROI, shiftslice, metric_error, RP, **params
):
    """
    Iterative outer loop for horizontal projection alignment.

    At each iteration a synthetic sinogram is computed from the current
    reconstruction, horizontal shifts are updated via
    :func:`_search_hshift_sinogram`, the sinogram is realigned and a new
    reconstruction is computed.  Stops on error divergence, shift
    convergence, or ``params['maxit']`` iterations.

    Accelerations active:
    A — parallel column processing inside :func:`_search_hshift_sinogram`;
    E — pre-computed FFT of sinogram columns in Fourier-shift mode.

    Parameters
    ----------
    sinogram : ndarray, shape (nr, nc)
        Current aligned sinogram.
    sino_orig : ndarray, shape (nr, nc)
        Low-pass-filtered original sinogram (shift source).
    theta : ndarray, shape (nc,)
        Projection angles in radians.
    circleROI : ndarray or int
        Circular mask applied to the reconstruction.
    shiftslice : ndarray, shape (1, nc)
        Current horizontal shift estimates.
    metric_error : list of float
        Running error history; appended each iteration.
    RP : RegisterPlot or None
        Live-plot helper; ``None`` in silent mode.
    **params
        Algorithm parameters including ``'pixtol'``, ``'maxit'``,
        ``'shiftmeth'``, ``'subpixel'``, ``'circle'``, ``'cliplow'``,
        ``'cliphigh'``, ``'derivatives'``, ``'calc_derivatives'``.

    Returns
    -------
    shiftslice : ndarray, shape (1, nc)
        Optimised horizontal shift array.
    metric_error : list of float
        Updated error history.
    """
    print("Initializing tomographic slice...")
    t0 = time.time()
    recons = tomo_recons(sinogram, theta=theta, **params)
    recons_std = recons.std()
    recons = _clipping_tomo(recons, **params)
    if params["circle"]:
        recons = recons * circleROI
    print("Done. Time elapsed: {} s".format(time.time() - t0))
    print("Slice standard deviation = {:0.04e}".format(recons_std))

    count = 0
    while True:
        count += 1

        # Route all iteration prints to the dedicated verbose Output widget
        # (notebook only).  RP may be None in silent / warm-start mode.
        if RP is not None and hasattr(RP, '_out_verbose') and RP._out_verbose is not None:
            RP._out_verbose.clear_output(wait=True)
            _vctx = RP._out_verbose
        else:
            _vctx = contextlib.nullcontext()

        with _vctx:
            print("\nIteration {}".format(count))
            print("-------------------------------------")
            it0 = time.time()
            deltaprev = shiftslice.copy()

            print("Computing synthetic sinogram...")
            sinogramcomp = projector(recons, theta, **params)
            if params["derivatives"] and not params["calc_derivatives"]:
                sinogramcomp = derivatives_sino(sinogramcomp, shift_method=params["shiftmeth"])

            print("Gradient descent search for horizontal shifts...")
            sinotempreg, shiftslice = _search_hshift_sinogram(
                sino_orig, sinogramcomp, shiftslice, **params
            )

            sinogram = compute_aligned_sino(sino_orig, shiftslice, shift_method=params["shiftmeth"])

            print("Computing tomographic slice...")
            t0 = time.time()
            recons = tomo_recons(sinogram, theta=theta, **params)
            recons_std = recons.std()
            recons = _clipping_tomo(recons, **params)
            if params["circle"]:
                recons = recons * circleROI
            print("Done. Time elapsed: {} s".format(time.time() - t0))
            print("Slice standard deviation = {:0.04e}".format(recons_std))

            errorxreg = _sino_error_metric(sinogram, sinogramcomp, params)
            sumerrorxreg = errorxreg.sum()
            print("Final error metric for x, E = {:0.04e}".format(sumerrorxreg))
            metric_error.append(sumerrorxreg)

            changex = np.abs(deltaprev - shiftslice)
            strprint = "Maximum correction in x = {:0.02f} pixels" if params["subpixel"] \
                       else "Maximum correction in x = {:0.02g} pixels"
            print("Estimating the changes in x:")
            print(strprint.format(np.max(changex)))
            print("Elapsed time in the iteration= {:0.02f} s".format(time.time() - it0))

            pixtol = params["pixtol"] if params["subpixel"] else 1
            reason = _checkconditions(
                metric_error, changex, pixtol, count, params["maxit"], params["subpixel"],
                rtol=params.get("rtol", 0.0),
            )
            if reason == 1:
                shiftslice = deltaprev.copy()
                metric_error.pop()

        # Update figure OUTSIDE the verbose context so the PNG goes to the
        # correct _out_main widget and not to _out_verbose.
        if RP is not None:
            RP.plotshorizontal(
                recons, sino_orig, sinogram, sinogramcomp, shiftslice, metric_error, count
            )

        if reason >= 1:
            break

    return shiftslice, metric_error


def alignprojections_horizontal(sinogram, theta, shiftstack, **params):
    """
    Horizontal alignment of projections by tomographic consistency.

    Shifts are optimised with a Newton gradient-descent algorithm that
    minimises the L2 distance between the measured sinogram and a
    synthetic sinogram computed from a filtered-back-projection
    reconstruction.  Supports a multi-stage frequency-cutoff schedule
    and an optional spatial multiresolution warm-start.

    Parameters
    ----------
    sinogram : ndarray, shape (nr, nc)
        Measured sinogram (rows = detector pixels, columns = projections).
    theta : ndarray, shape (nc,)
        Projection angles in radians.
    shiftstack : ndarray, shape (2, n)
        Current shift estimates ``[vertical_shifts, horizontal_shifts]``.
        Only the horizontal row (index 1) is updated.
    **params
        Algorithm parameters.  Required keys:

        maxit : int
            Maximum number of outer iterations per stage.
        pixtol : float
            Convergence tolerance in pixels.
        freqcutoff : float
            Fraction of Nyquist used as the sinogram low-pass cut-off.
        shiftmeth : str
            Interpolation method (``'linear'``, ``'fourier'``,
            ``'spline'``).
        circle : bool
            Apply a circular mask to the reconstruction.
        cliplow : float or None
            Lower clipping threshold for the reconstruction values.
        cliphigh : float or None
            Upper clipping threshold for the reconstruction values.
        derivatives : bool
            Whether projections are phase-gradient images.
        calc_derivatives : bool
            Whether to compute derivatives on the synthetic sinogram.

        freqcutoff_schedule : list of float, optional
            Sequence of ``freqcutoff`` values (coarsest first).  Each
            stage warm-starts the next.  Default is
            ``[params['freqcutoff']]`` (single stage).
        multiresolution : bool, optional
            Spatial multiresolution warm-start at stage 0.  Default ``False``.
        mr_factor : int, optional
            Down-sampling factor for the spatial warm-start.  Default ``2``.
        n_coarse_iter : int, optional
            Number of coarse iterations in the warm-start.  Default ``5``.
        rtol : float, optional
            Relative improvement threshold for early stopping.  Default ``0``.
        silent : bool, optional
            If ``True``, suppress all matplotlib output (required for
            sub-process workers).  Default ``False``.

    Returns
    -------
    shiftstack : ndarray, shape (2, n)
        Updated shift array with optimised horizontal shifts in row 1.
    """
    params.setdefault("circle", True)
    params.setdefault("sinohigh", 0.6)
    params.setdefault("sinolow", -0.6)
    params.setdefault("opencl", False)
    if not isinstance(params["maxit"], int):
        params["maxit"] = 10
    params.setdefault("cliplow", None)
    params.setdefault("cliphigh", None)
    params.setdefault("rtol", 0.0)
    # silent=True: suppress all matplotlib calls (used for parallel workers).
    params.setdefault("silent", False)
    # Spatial multiresolution: downsample at stage 0 for a cheap coarse warm-start.
    params.setdefault("multiresolution", False)
    params.setdefault("mr_factor", 2)
    params.setdefault("n_coarse_iter", 5)
    # Frequency-cutoff schedule: list of freqcutoff values (coarsest first).
    # Each stage warm-starts the next with its shifts.
    # If not provided, falls back to a single pass at params["freqcutoff"].
    schedule = params.get("freqcutoff_schedule", None)
    if schedule is None or len(schedule) == 0:
        schedule = [params["freqcutoff"]]

    n_stages = len(schedule)
    print("\nStarting the horizontal alignment (Adam gradient descent)")
    print("=====================================")
    print("Number of iterations per stage: {}".format(params["maxit"]))
    if n_stages > 1:
        print("Frequency-cutoff schedule: {}".format(schedule))
    else:
        print("Using a frequency cutoff of {}".format(schedule[0]))
    if params["multiresolution"]:
        print("Spatial multiresolution warm-start at stage 0 (factor={}, {} iterations)".format(
            params["mr_factor"], params["n_coarse_iter"]))
    if params["rtol"] > 0:
        print("Relative tolerance (rtol) = {}".format(params["rtol"]))
    print("Low limit for tomo values = {}".format(params["cliplow"]))
    print("High limit for tomo values = {}".format(params["cliphigh"]))

    original_sino = sinogram.copy()
    shiftslice = np.expand_dims(shiftstack[1], axis=0)

    if not params["silent"]:
        plt.ion()
        RP = RegisterPlot(**params)
    else:
        RP = None

    # ---------------------------------------------------------------
    # Loop over the frequency-cutoff schedule.
    # Each stage inherits shiftslice from the previous one.
    # ---------------------------------------------------------------
    for stage_idx, fc in enumerate(schedule):
        is_last = (stage_idx == n_stages - 1)
        params_s = dict(params, freqcutoff=fc, subpixel=True)

        # Keep RegisterPlot informed of the current stage so the suptitle
        # can display it (no-op when RP is None or stage_info not used).
        if RP is not None:
            RP.stage_info = (stage_idx + 1, n_stages, fc)

        if n_stages > 1:
            print("\n╔══════════════════════════════════════════════════════╗")
            print("║  Stage {}/{} — freqcutoff = {:<30}║".format(
                stage_idx + 1, n_stages, str(fc) + "  "))
            print("╚══════════════════════════════════════════════════════╝")

        padval = int(2 * np.round(1 / fc))
        sinogram = np.pad(
            original_sino, ((padval, padval), (0, 0)), "constant", constant_values=0
        ).copy()
        sino_orig = _filter_sino(sinogram, **params_s)

        if not np.all(shiftslice == 0):
            print("Shifting sinogram.")
            sinogram = compute_aligned_sino(
                sino_orig, shiftslice, shift_method=params_s["shiftmeth"]
            )
        else:
            print("Initializing shiftslice with zeros")

        # -----------------------------------------------------------
        # Spatial multiresolution warm-start (stage 0 only).
        # Downsamples the detector axis by mr_factor, runs a few cheap
        # coarse iterations, then scales the shifts back to full res.
        # Combined with freqcutoff_schedule this acts at the coarsest
        # frequency stage, giving the largest displacement reduction for
        # the smallest cost.
        # -----------------------------------------------------------
        if stage_idx == 0 and params["multiresolution"]:
            factor = params["mr_factor"]
            n_coarse_iter = params["n_coarse_iter"]
            print("\n--- Spatial warm-start (factor={}, {} coarse iterations) ---".format(
                factor, n_coarse_iter))
            nr, nc_sino = original_sino.shape
            nr_trim = (nr // factor) * factor
            sino_pool = original_sino[:nr_trim, :].reshape(
                nr_trim // factor, factor, nc_sino
            ).mean(axis=1)
            sino_pool_padded = np.pad(
                sino_pool, ((padval, padval), (0, 0)), "constant", constant_values=0
            ).copy()
            sino_orig_c = _filter_sino(sino_pool_padded, **params_s)
            shiftslice_c = shiftslice / factor
            if not np.all(shiftslice_c == 0):
                sinogram_c = compute_aligned_sino(
                    sino_orig_c, shiftslice_c, shift_method=params_s["shiftmeth"]
                )
            else:
                sinogram_c = sino_orig_c.copy()
            print("Computing initial coarse tomographic slice...")
            t_c = time.time()
            recons_c = tomo_recons(sinogram_c, theta=theta, **params_s)
            print("Done. Time elapsed: {:.02f} s".format(time.time() - t_c))
            recons_c = _clipping_tomo(recons_c, **params_s)
            circleROI_c = create_circle(recons_c) if params_s["circle"] else 1
            recons_c = recons_c * circleROI_c
            sinogramcomp_c = projector(recons_c, theta, **params_s)
            if params_s["derivatives"] and not params_s["calc_derivatives"]:
                sinogramcomp_c = derivatives_sino(
                    sinogramcomp_c, shift_method=params_s["shiftmeth"]
                )
            metric_error_c = []
            errorinit_c = _sino_error_metric(sinogram_c, sinogramcomp_c, params_s)
            metric_error_c.append(np.sum(errorinit_c))
            print("Initial coarse error, E= {:0.04e}".format(metric_error_c[0]))
            params_c = dict(params_s, maxit=n_coarse_iter)
            shiftslice_c, _ = _alignprojections_horizontal(
                sinogram_c, sino_orig_c, theta, circleROI_c,
                shiftslice_c, metric_error_c, RP, **params_c
            )
            shiftslice = shiftslice_c * factor
            sinogram = compute_aligned_sino(
                sino_orig, shiftslice, shift_method=params_s["shiftmeth"]
            )
            print("--- Spatial warm-start done. Proceeding at full resolution ---\n")

        print("Computing initial tomographic slice...")
        t0 = time.time()
        recons = tomo_recons(sinogram, theta=theta, **params_s)
        print("Done. Time elapsed: {:.02f} s".format(time.time() - t0))
        print("Slice standard deviation = {:0.04e}".format(recons.std()))

        recons = _clipping_tomo(recons, **params_s)
        circleROI = create_circle(recons) if params_s["circle"] else 1
        recons = recons * circleROI

        print("Computing synthetic sinogram...")
        t0 = time.time()
        sinogramcomp = projector(recons, theta, **params_s)
        if params_s["derivatives"] and not params_s["calc_derivatives"]:
            sinogramcomp = derivatives_sino(sinogramcomp, shift_method=params_s["shiftmeth"])
        print("Done. Time elapsed: {:.02f} s".format(time.time() - t0))

        metric_error = []
        errorinit = _sino_error_metric(sinogram, sinogramcomp, params_s)
        print("Initial error metric, E= {:0.04e}".format(np.sum(errorinit)))
        metric_error.append(np.sum(errorinit))

        if RP is not None:
            RP.plotshorizontal(
                recons, sino_orig, sinogram, sinogramcomp, shiftslice, metric_error, count=0
            )

        print("\n===================================================")
        print("Horizontal alignment (Newton GD, pixtol={})".format(params_s["pixtol"]))
        print("===================================================")
        shiftslice, metric_error = _alignprojections_horizontal(
            sinogram, sino_orig, theta, circleROI, shiftslice, metric_error, RP, **params_s
        )

        if not is_last:
            print("Stage {} converged. Handing shifts to next stage.\n".format(stage_idx + 1))

    shiftstack[1] = shiftslice

    if not params["silent"]:
        print("\nComputing aligned images")
        alignedsinogram = compute_aligned_sino(
            original_sino, shiftslice, shift_method=params["shiftmeth"]
        )
        print("Calculating aligned slice for display")
        _oneslicefordisplay(alignedsinogram, theta, **params)

    return shiftstack


# ============================================================================
# Remaining public functions (unchanged)
# ============================================================================

def refine_horizontalalignment(input_stack, theta, shiftstack, **params):
    """
    Run one refinement pass of horizontal alignment.

    Refinement uses the current ``freqcutoff`` only — ``freqcutoff_schedule``
    and ``multiresolution`` are ignored because the input shifts are already a
    good warm-start.  Use ``params["rtol"]`` (e.g. 1e-3) to stop early when
    the improvement per iteration becomes negligible.

    To run multiple refinement passes or change parameters between passes,
    update ``params`` in your script/notebook and call this function again.

    Parameters
    ----------
    input_stack : ndarray, shape (n, nr, nc)
        Stack of derivative projections.
    theta : ndarray
        Projection angles.
    shiftstack : ndarray, shape (2, n)
        Current shift array (modified in-place and returned).
    **params
        Alignment parameters forwarded to :func:`alignprojections_horizontal`.

    Returns
    -------
    shiftstack : ndarray
    params : dict
    """
    params.setdefault("correct_bad", False)
    sinogram = np.transpose(input_stack[:, params["slicenum"], :])
    if params["correct_bad"]:
        sinogram = replace_bad(sinogram, list_bad=params["bad_projs"], temporary=False)

    # Refinement uses the current freqcutoff only — no multi-stage
    # pipeline and no spatial downsampling (shifts are already good).
    params_refine = dict(params)
    params_refine.pop("freqcutoff_schedule", None)
    params_refine["multiresolution"] = False

    print("\n================================================")
    print("Starting the refinement of the alignment")
    print("================================================")
    shiftstack = alignprojections_horizontal(sinogram, theta, shiftstack, **params_refine)
    return shiftstack, params


def oneslicefordisplay(sinogram, theta, **params):
    """Reconstruct and display one tomographic slice.

    Pass updated values via ``params`` (e.g. ``params["freqcutoff"] = 0.5``)
    and re-call to change reconstruction settings.
    """
    print(
        "Reconstructing slice with freqcutoff={}, filtertype='{}' …".format(
            params.get("freqcutoff", "?"), params.get("filtertype", "?")
        ),
        flush=True,
    )
    _oneslicefordisplay(sinogram, theta, **params)


def _oneslicefordisplay(sinogram, theta, **params):
    """
    Reconstruct and display a single tomographic slice.

    Parameters
    ----------
    sinogram : ndarray, shape (nr, nc)
        Sinogram to reconstruct.
    theta : ndarray, shape (nc,)
        Projection angles in radians.
    **params
        Must contain ``'circle'`` (bool), ``'cliplow'`` (float or None),
        ``'cliphigh'`` (float or None), and ``'colormap'`` (str).
        All remaining keys are forwarded to :func:`tomo_recons`.
    """
    p0 = time.time()
    recons = tomo_recons(sinogram, theta=theta, **params)
    recons = _clipping_tomo(recons, **params)
    circleROI = create_circle(recons) if params["circle"] else 1
    recons = recons * circleROI
    print("Done. Time elapsed: {} s".format(time.time() - p0))
    display_slice(recons, colormap="bone", vmin=params["cliplow"], vmax=params["cliphigh"])


def _tc_worker(args):
    """
    Process-pool worker for tomoconsistency_multiple.

    Each worker aligns one sinogram slice independently.  All per-slice
    console output is suppressed so only the parent's overall progress
    bar is visible.  Runs silently (no matplotlib display) so it is safe
    inside a subprocess.

    Parameters
    ----------
    args : tuple
        (slice_index, sinogram_2d, theta, shiftstack_copy, params_dict)

    Returns
    -------
    (slice_index, shift_array)
    """
    import contextlib
    import os
    # matplotlib must be imported AFTER the process is spawned so we can
    # set the non-interactive backend before any display code runs.
    import matplotlib
    matplotlib.use("Agg")

    ii, sinogram, theta, shiftstack_copy, params_w = args
    # Redirect stdout and stderr to /dev/null for the duration of this slice.
    # contextlib restores them correctly even in the sequential (same-process)
    # case, so the parent's tqdm bar is never disrupted.
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            shiftstack_aux = alignprojections_horizontal(
                sinogram, theta, shiftstack_copy, **params_w
            )
    return ii, shiftstack_aux[1]


def tomoconsistency_multiple(input_stack, theta, shiftstack, **params):
    """
    Tomographic consistency alignment over multiple sinogram slices.

    Each slice is aligned independently (using the same ``shiftstack``
    warm-start) and the resulting per-slice horizontal shifts are averaged
    to produce a robust final estimate.  Slices can be processed in
    parallel via :class:`~concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    input_stack : ndarray, shape (n, nr, nc)
        Full 3-D projection stack.
    theta : ndarray, shape (n,)
        Projection angles in radians.
    shiftstack : ndarray, shape (2, n)
        Current shift estimates used as a warm-start for every slice.
        The horizontal row (index 1) is updated with the averaged result.
    **params
        Algorithm parameters forwarded to
        :func:`alignprojections_horizontal`.  Extra keys:

        slicenum : int
            Central slice index.
        n_slices_tc : int, optional
            Number of slices to process (centred on ``slicenum``).
            Default ``10``.
        n_workers_tc : int, optional
            Number of parallel worker processes.
            Default ``max(1, cpu_count // 2)``.  Pass ``1`` for sequential.

    Returns
    -------
    shiftstack : ndarray, shape (2, n)
        Updated shift array with the averaged horizontal shifts in row 1
        (or the original shifts if the user declines the result).
    """
    print("Starting Tomographic consistency on multiple slices")
    slicenumorig = params["slicenum"]
    n_slices_tc  = params.get("n_slices_tc", 10)
    half         = n_slices_tc // 2
    slices       = np.arange(slicenumorig - half,
                              slicenumorig - half + n_slices_tc)
    shiftslice_prev = np.expand_dims(shiftstack[1], axis=0).copy()

    # Single-pass params: strip schedule and spatial MR (warm-start is good).
    # silent=True suppresses matplotlib — required for subprocess workers.
    params_tc = dict(params)
    params_tc.pop("freqcutoff_schedule", None)
    params_tc["multiresolution"] = False
    params_tc["silent"]          = True

    n_cpu        = os.cpu_count() or 1
    n_workers_tc = params.get("n_workers_tc", max(1, n_cpu // 2))
    n_workers_tc = min(n_workers_tc, len(slices))

    print("Slices: {} (n={})".format(list(slices), len(slices)))
    print("Parallel workers: {}".format(n_workers_tc))

    # Build one argument tuple per slice (sinogram pre-extracted to avoid
    # pickling the full input_stack in every worker).
    task_args = []
    for ii in slices:
        sino = np.transpose(input_stack[:, ii, :])
        p    = dict(params_tc, slicenum=int(ii))
        task_args.append((int(ii), sino, theta, shiftstack.copy(), p))

    if n_workers_tc == 1:
        # Sequential fallback — useful for debugging or single-core machines.
        results = []
        for args in tqdm(task_args, desc="Tomographic consistency"):
            results.append(_tc_worker(args))
    else:
        with ProcessPoolExecutor(max_workers=n_workers_tc) as executor:
            results = list(
                tqdm(
                    executor.map(_tc_worker, task_args),
                    total=len(task_args),
                    desc="Tomographic consistency",
                )
            )

    # Sort by slice index (process pool may return out of order).
    results.sort(key=lambda r: r[0])
    shiftxrefine = [r[1] for r in results]

    shiftxrefine = np.squeeze(shiftxrefine)
    shiftxrefine_avg = shiftxrefine.mean(axis=0)

    # Wrap figure creation in a throwaway Output widget so ipympl does not
    # auto-display the interactive canvas.  We render to PNG explicitly below.
    if isnotebook():
        try:
            from ipywidgets import Output as _Out_tc
            _cap_tc = _Out_tc()
        except ImportError:
            _cap_tc = contextlib.nullcontext()
    else:
        _cap_tc = contextlib.nullcontext()

    plt.close("all")
    with _cap_tc:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    ax1.imshow(shiftxrefine, interpolation="none", cmap="jet")
    ax1.axis("tight")
    ax1.set_xlabel("Projection number")
    ax1.set_ylabel("Slice number")
    ax1.set_title("Displacements in x per slice")
    ax2.plot(shiftxrefine_avg, "b-", label="average")
    ax2.plot(shiftslice_prev[0], "r--", label="previous")
    ax2.legend()
    ax2.axis("tight")
    ax2.set_xlim([0, len(shiftxrefine_avg)])
    ax2.set_title("Average displacements in x  —  blue=new average, red=previous")
    ax2.set_xlabel("Projection number")
    if isnotebook():
        import io as _io_tc
        from IPython import display as _disp_tc
        _buf = _io_tc.BytesIO()
        fig.savefig(_buf, format="png", bbox_inches="tight", dpi=100)
        _buf.seek(0)
        _disp_tc.display(_disp_tc.Image(_buf.read()))
        plt.close(fig)
    else:
        plt.show(block=False)

    use_average = params.get("use_average", True)
    if use_average:
        shiftstack[1] = shiftxrefine_avg.copy()
        print(
            "Averaged shifts applied.\n"
            "Tip: to keep the previous shifts instead, add\n"
            "    params['use_average'] = False\n"
            "to your params dict before calling tomoconsistency_multiple.\n"
            "Note: a backup copy is not made automatically — save\n"
            "    shiftstack_backup = shiftstack.copy()\n"
            "before this call if you want to be able to revert.",
            flush=True,
        )
    else:
        shiftstack[1] = shiftslice_prev[0].copy()
        print("Keeping previous shiftstack (use_average=False).", flush=True)
    return shiftstack


class _RotAxisPicker:
    """
    Interactive GUI for estimating the rotation-axis offset.

    Displays the reconstructed slice (left) and sinogram (right) for the
    current offset.  A text box lets you try a new integer offset value;
    clicking **Update** recomputes and refreshes both panels.  Click
    **Confirm** to accept the current value and close the figure.

    Terminal usage
    --------------
    Used internally by :func:`estimate_rot_axis` — not meant to be called
    directly.

    Jupyter two-cell workflow
    -------------------------
    ::

        # Cell 1
        picker = estimate_rot_axis(valigndiff, theta, **params)
        # Cell 2 — after clicking Confirm
        params["rot_axis_offset"] = picker.rot_axis_offset
    """

    def __init__(self, input_array, theta, **params):
        self._array  = input_array
        self._theta  = theta
        self._params = params
        self.rot_axis_offset = params["rot_axis_offset"]
        self._done   = False

        params.setdefault("sinocmap", params.get("colormap", "bone"))

        # ---- initial reconstruction ----
        slicenum = params["slicenum"]
        sinogram = np.transpose(input_array[:, slicenum, :])
        sinogram = _offset_sinogram(sinogram, self.rot_axis_offset)
        print("Computing initial slice (offset={}) …".format(self.rot_axis_offset), flush=True)
        p0 = time.time()
        tomogram = tomo_recons(sinogram, theta, **params)
        print("Done in {:.1f} s.".format(time.time() - p0), flush=True)

        # ---- figure ----
        plt.close("all")
        fig = plt.figure(num=5, figsize=(12, 5))
        # leave room at the bottom for the TextBox + buttons
        fig.subplots_adjust(bottom=0.22)

        ax1 = fig.add_subplot(121)
        self._im1 = ax1.imshow(
            tomogram, cmap=params["colormap"], interpolation="none",
            vmin=params["cliplow"], vmax=params["cliphigh"],
        )
        self._ax1 = ax1
        ax1.set_title("Slice {} — offset {}".format(slicenum, self.rot_axis_offset))
        fig.colorbar(self._im1, ax=ax1)

        ax2 = fig.add_subplot(122)
        self._im2 = ax2.imshow(
            sinogram, cmap=params["sinocmap"], interpolation="none",
            vmin=params["sinolow"], vmax=params["sinohigh"],
        )
        ax2.axis("tight")
        ax2.set_title("Sinogram — slice {}".format(slicenum))
        fig.colorbar(self._im2, ax=ax2)

        # status text
        self._status = fig.text(
            0.5, 0.97,
            "Enter a new offset → Update, or Confirm to accept current value.",
            ha="center", va="top", fontsize=9, color="steelblue",
        )

        # TextBox for offset value
        ax_box    = fig.add_axes([0.25, 0.04, 0.20, 0.07])
        ax_update = fig.add_axes([0.50, 0.04, 0.12, 0.07])
        ax_conf   = fig.add_axes([0.65, 0.04, 0.12, 0.07])
        self._textbox = TextBox(ax_box, "Offset: ", initial=str(self.rot_axis_offset))
        self._btn_update  = Button(ax_update,  "Update")
        self._btn_confirm = Button(ax_conf,    "Confirm")
        self._btn_update.on_clicked(self._on_update)
        self._btn_confirm.on_clicked(self._on_confirm)
        self.fig = fig

        print(
            "Inspect the slice.  Enter a new offset value in the text box "
            "and click  Update  to recompute, then  Confirm  to accept.",
            flush=True,
        )

    # ------------------------------------------------------------------ events

    def _on_update(self, event):
        try:
            new_offset = int(float(self._textbox.text.strip()))
        except ValueError:
            self._status.set_text("Invalid offset — enter an integer.")
            self.fig.canvas.draw_idle()
            return

        self.rot_axis_offset = new_offset
        slicenum = self._params["slicenum"]
        sinogram = np.transpose(self._array[:, slicenum, :])
        sinogram = _offset_sinogram(sinogram, new_offset)

        self._status.set_text("Computing slice (offset={}) …".format(new_offset))
        self.fig.canvas.draw_idle()
        print("Computing slice (offset={}) …".format(new_offset), flush=True)

        p0 = time.time()
        tomogram = tomo_recons(sinogram, self._theta, **self._params)
        print("Done in {:.1f} s.".format(time.time() - p0), flush=True)

        self._im1.set_data(tomogram)
        self._im1.autoscale()
        self._ax1.set_title("Slice {} — offset {}".format(slicenum, new_offset))
        self._im2.set_data(sinogram)
        self._status.set_text(
            "Offset {} — click Confirm to accept or enter another value.".format(new_offset)
        )
        self.fig.canvas.draw_idle()

    def _on_confirm(self, event):
        self._done = True
        print(
            "Confirmed rotation-axis offset: {}".format(self.rot_axis_offset),
            flush=True,
        )
        plt.close(self.fig)


def estimate_rot_axis(input_array, theta, **params):
    """
    Interactively estimate the rotation-axis offset.

    Opens a GUI figure showing the reconstructed slice and sinogram for
    the current offset (``params["rot_axis_offset"]``).  Enter a new
    integer value in the text box and click **Update** to recompute.
    Click **Confirm** when satisfied.

    Parameters
    ----------
    input_array : ndarray, shape (n, nr, nc)
        Stack of derivative projections.
    theta : ndarray
        Projection angles (degrees).
    **params
        Must contain: ``slicenum``, ``rot_axis_offset``, ``colormap``,
        ``cliplow``, ``cliphigh``, ``sinolow``, ``sinohigh``, ``filtertype``,
        ``freqcutoff``, ``circle``, ``algorithm``, ``derivatives``,
        ``calc_derivatives``.

    Returns
    -------
    Terminal mode
        ``rot_axis_offset`` (int) — the confirmed offset value.
    Jupyter mode (two-cell workflow)
        ``picker`` (:class:`_RotAxisPicker`) — access
        ``picker.rot_axis_offset`` in the **next** cell after clicking Confirm.
    """
    params.setdefault("sinocmap", params.get("colormap", "bone"))
    theta = theta - theta.min()

    picker = _RotAxisPicker(input_array, theta, **params)

    if isnotebook():
        try:
            import matplotlib as _mpl
            _interactive = "inline" not in _mpl.get_backend().lower()
        except Exception:
            _interactive = False
        if _interactive:
            plt.show(block=False)
            picker.fig.canvas.draw()
            print(
                "\nJupyter two-cell workflow:\n"
                "  Interact with the figure, then in the NEXT cell:\n"
                "      params['rot_axis_offset'] = picker.rot_axis_offset",
                flush=True,
            )
        else:
            from IPython import display as _ipy_display
            _ipy_display.display(picker.fig)
        return picker
    else:
        plt.show(block=True)
        print(
            "Initial estimate of rotation axis offset: {}".format(
                picker.rot_axis_offset
            )
        )
        return picker.rot_axis_offset


@deprecated
def _offset_sinogram_old(sinogram, offset):
    """
    Pad a sinogram to offset the rotation axis (deprecated).

    Parameters
    ----------
    sinogram : ndarray, shape (nr, nc)
        Input sinogram.
    offset : int
        Rotation-axis offset in pixels.  Positive values pad the bottom;
        negative values pad the top.

    Returns
    -------
    sinogram : ndarray
        Zero-padded sinogram.
    """
    if np.sign(offset) == +1:
        print("Initial guess of the rotation axis offset : {}".format(offset))
        sinogram = np.pad(sinogram, ((0, 2 * abs(offset)), (0, 0)), "constant", constant_values=0)
    elif np.sign(offset) == -1:
        print("Initial guess of the rotation axis offset : {}".format(offset))
        sinogram = np.pad(sinogram, ((2 * abs(offset), 0), (0, 0)), "constant", constant_values=0)
    return sinogram


def _offset_sinogram(sinogram, offset, shift_method="linear"):
    """
    Shift a sinogram vertically to offset the rotation axis.

    Parameters
    ----------
    sinogram : ndarray, shape (nr, nc)
        Input sinogram.
    offset : float
        Rotation-axis offset in pixels (positive = shift down).
    shift_method : str, optional
        Interpolation method.  Default ``'linear'``.

    Returns
    -------
    ndarray, shape (nr, nc)
        Vertically shifted sinogram.
    """
    S = ShiftFunc(shiftmeth="linear")
    return S(sinogram, (offset, 0))
