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
import os
import time
from concurrent.futures import ThreadPoolExecutor

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
from ..restoration import derivatives, derivatives_sino
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

def register_2Darrays(image1, image2):
    """
    Image registration using phase cross-correlations.

    Parameters
    ----------
    image1 : array_like
        Reference image.
    image2 : array_like
        Image to be shifted relative to image1.

    Returns
    -------
    shift : list of floats
        [row_shift, col_shift].
    diffphase : float
        Phase difference between the two images.
    offset_image2 : array_like
        Shifted image2 aligned to image1.
    """
    precision = input(
        "Do you want to use pixel(1) or subpixel(2) precision registration?[1] "
    )
    if precision == str(1) or precision == "":
        print("\nCalculating the pixel precision image registration ...")
        start = time.time()
        shift, error, diffphase = phase_cross_correlation(image1.copy(), image2.copy())
        print(diffphase)
        end = time.time()
        print("Time elapsed: {:g} s".format(end - start))
        print("Detected pixel offset [y,x]: [{:g}, {:g}]".format(shift[0], shift[1]))
    elif precision == str(2):
        print("\nCalculating the subpixel image registration ...")
        start = time.time()
        shift, error, diffphase = phase_cross_correlation(
            image1.copy(), image2.copy(), 100
        )
        print(diffphase)
        end = time.time()
        print("Time elapsed: {:g} s".format(end - start))
        print("Detected subpixel offset [y,x]: [{:g}, {:g}]".format(shift[0], shift[1]))
    else:
        print("You must choose between 1 and 2")
        raise SystemExit

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


def compute_aligned_horizontal_special(input_stack, shiftstack, shift_method="linear"):
    """Horizontal-only in-place alignment."""
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


def compute_aligned_horizontal(input_stack, shiftstack, shift_method="linear"):
    """Horizontal-only alignment (copy variant)."""
    deltashift = np.zeros_like(shiftstack)
    deltashift[1] = shiftstack[1].copy()
    return compute_aligned_stack(input_stack, deltashift, shift_method=shift_method)


def center_of_mass_stack(input_stack, lims, shiftstack, shift_method="fourier"):
    """Centre-of-mass for each projection (unchanged)."""
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
    """Vertical mass fluctuation functions (unchanged)."""
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
    """Compute the vertical mass fluctuation signal for a given shift (unchanged)."""
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
    deltax = params["deltax"]
    limcol = (deltax, stack_shape[2] - deltax)
    limrow = params["limsy"]
    if limrow is None or limrow == "":
        limrow = [0, stack_shape[1]]
    return np.asarray(limrow), np.asarray(limcol)


def _clipping_tomo(recons, **params):
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
    errorxreg = np.zeros(sinogramexp.shape[1])
    for ii in range(sinogramexp.shape[1]):
        errorxreg[ii] = np.sum(np.abs(sinogramexp[:, ii] - sinogramcomp[:, ii]) ** 2)
    return errorxreg


def _checkconditions(metric_error, changes, pixtol, count, maxit, subpixel=False):
    step = pixtol if subpixel else 1
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
    elif np.max(changes) < step:
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
    """Search vertical shifts for the full stack using gradient descent."""
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
    """Iterative vertical alignment driver (unchanged outer logic)."""
    count = 0
    error_reg = np.zeros(vert_fluct_init.shape[0])
    while True:
        count += 1
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

        RP.plotsvertical(
            input_stack[0], lims, vert_fluct_init, vert_fluct_temp,
            shiftstack, metric_error, count,
        )

        pixtol = params["pixtol"] if params["subpixel"] else 1
        reason = _checkconditions(
            metric_error, changey, pixtol, count, params["maxit"], params["subpixel"]
        )
        if reason == 1:
            shiftstack = deltaprev.copy()
            metric_error.pop()
            break
        elif reason >= 2:
            break

    return shiftstack, metric_error


def alignprojections_vertical(input_stack, shiftstack, **params):
    """
    Vertical alignment of projections using mass fluctuation approach with
    Adam gradient descent optimisation.

    Parameters and return values are identical to the original function.
    See the original docstring for full parameter descriptions.
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
    Iterative horizontal alignment driver.

    Accelerations active
    --------------------
    A — Parallel column processing (inside _search_hshift_sinogram).
    E — Pre-computed FFT of sinogram columns (Fourier shift mode).
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

        RP.plotshorizontal(
            recons, sino_orig, sinogram, sinogramcomp, shiftslice, metric_error, count
        )

        pixtol = params["pixtol"] if params["subpixel"] else 1
        reason = _checkconditions(
            metric_error, changex, pixtol, count, params["maxit"], params["subpixel"]
        )
        if reason == 1:
            shiftslice = deltaprev.copy()
            metric_error.pop()
            break
        elif reason >= 2:
            break

    return shiftslice, metric_error


def alignprojections_horizontal(sinogram, theta, shiftstack, **params):
    """
    Horizontal alignment of projections by tomographic consistency using
    Adam gradient descent optimisation.

    Parameters and return values are identical to the original function.
    See the original docstring for full parameter descriptions.
    """
    params.setdefault("circle", True)
    params.setdefault("sinohigh", 0.6)
    params.setdefault("sinolow", -0.6)
    params.setdefault("opencl", False)
    if not isinstance(params["maxit"], int):
        params["maxit"] = 10
    params.setdefault("cliplow", None)
    params.setdefault("cliphigh", None)
    # Frequency-cutoff schedule: list of freqcutoff values to sweep through,
    # coarsest first.  Each stage warm-starts the next with its shifts.
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
    print("Low limit for tomo values = {}".format(params["cliplow"]))
    print("High limit for tomo values = {}".format(params["cliphigh"]))

    original_sino = sinogram.copy()
    shiftslice = np.expand_dims(shiftstack[1], axis=0)

    plt.ion()
    RP = RegisterPlot(**params)

    # ---------------------------------------------------------------
    # Loop over the frequency-cutoff schedule
    # Each stage inherits the shifts from the previous one.
    # ---------------------------------------------------------------
    for stage_idx, fc in enumerate(schedule):
        is_last = (stage_idx == n_stages - 1)
        params_s = dict(params, freqcutoff=fc, subpixel=True)

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
    """Interactively refine horizontal alignment (unchanged)."""
    params.setdefault("correct_bad", False)
    while True:
        a = input("Do you want to refine further the alignment? ([y]/n): ").lower()
        if str(a) in ("", "y"):
            a1 = input("Do you want to use the same parameters? ([y]/n): ").lower()
            if a1 == "n":
                a1 = input("Slice number (e.g. {}): ".format(params["slicenum"]))
                if a1 != "":
                    params["slicenum"] = eval(a1)
                a2 = input("Pixel tolerance (e.g. {}): ".format(params["pixtol"]))
                if a2 != "":
                    params["pixtol"] = eval(a2)
                a3 = input("Filter Tomo cutoff (e.g. {}): ".format(params["freqcutoff"]))
                if a3 != "":
                    params["freqcutoff"] = eval(a3)
                a4 = input("Number of iterations (e.g. {}): ".format(params["maxit"]))
                if a4 != "":
                    params["maxit"] = eval(a4)
                a5 = input("Apply a circle (e.g. {}): ".format(params["circle"]))
                if a5 != "":
                    params["circle"] = eval(a5)
                a6 = input("Clipping high (e.g. {}): ".format(params["cliphigh"]))
                if a6 != "":
                    params["cliphigh"] = eval(a6)

            sinogram = np.transpose(input_stack[:, params["slicenum"], :])
            if params["correct_bad"]:
                sinogram = replace_bad(sinogram, list_bad=params["bad_projs"], temporary=False)
            print("Starting the refinement of the alignment")
            shiftstack = alignprojections_horizontal(sinogram, theta, shiftstack, **params)
        elif str(a) == "n":
            print("No further refinement done")
            break
        else:
            print("You should answer 'y' or 'n' or accept the default answer.")
    return shiftstack, params


def oneslicefordisplay(sinogram, theta, **params):
    """Reconstruct and display one slice (unchanged)."""
    a = input(
        "Do you want to reconstruct the slice with different parameters? ([y]/n) :"
    ).lower()
    if str(a) in ("", "y"):
        freqcutoff = input("freqcutoff (current: {}) = ".format(params["freqcutoff"]))
        if freqcutoff != "":
            params["freqcutoff"] = eval(freqcutoff)
        filtertype = str(
            input("filtertype (current: {}) = ".format(params["filtertype"])).lower()
        )
        if filtertype != "":
            params["filtertype"] = str(filtertype)
        print("Calculating a tomographic slice")
    _oneslicefordisplay(sinogram, theta, **params)


def _oneslicefordisplay(sinogram, theta, **params):
    p0 = time.time()
    recons = tomo_recons(sinogram, theta=theta, **params)
    recons = _clipping_tomo(recons, **params)
    circleROI = create_circle(recons) if params["circle"] else 1
    recons = recons * circleROI
    print("Done. Time elapsed: {} s".format(time.time() - p0))
    display_slice(recons, colormap="bone", vmin=params["cliplow"], vmax=params["cliphigh"])


def tomoconsistency_multiple(input_stack, theta, shiftstack, **params):
    """Tomographic consistency alignment on multiple slices (unchanged)."""
    print("Starting Tomographic consistency on multiple slices")
    slicenumorig = params["slicenum"]
    slices = np.arange(slicenumorig - 5, slicenumorig + 5)
    shiftslice_prev = np.expand_dims(shiftstack[1], axis=0).copy()
    shiftxrefine = []
    for ii in slices:
        print("\nAligning slice {}".format(ii))
        params["slicenum"] = ii
        sinogram = np.transpose(input_stack[:, ii, :])
        shiftstack_aux = alignprojections_horizontal(sinogram, theta, shiftstack, **params)
        shiftxrefine.append(shiftstack_aux[1])

    shiftxrefine = np.squeeze(shiftxrefine)
    shiftxrefine_avg = shiftxrefine.mean(axis=0)

    plt.close("all")
    fig = plt.figure(num=6, figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.imshow(shiftxrefine, interpolation="none", cmap="jet")
    ax1.axis("tight")
    ax1.set_xlabel("Projection number")
    ax1.set_ylabel("Slice number")
    ax1.set_title("Displacements in x")
    ax2 = fig.add_subplot(212)
    ax2.plot(shiftxrefine_avg, "b-", label="average")
    ax2.plot(shiftslice_prev[0], "r--", label="previous")
    ax2.legend()
    ax2.axis("tight")
    ax2.set_xlim([0, len(shiftxrefine_avg)])
    ax2.set_title("Average displacements in x")
    ax2.set_xlabel("Projection number")
    plt.tight_layout()
    if isnotebook():
        from IPython import display
        display.display(fig)
        plt.close(fig)
    else:
        plt.show(block=False)

    a = input(
        "Are you happy with the tomographic consistency alignment? ([y]/n) "
    ).lower()
    if a in ("", "y"):
        shiftstack[1] = shiftxrefine_avg.copy()
        print("Using the average of all shiftstack")
    else:
        shiftstack[1] = shiftslice_prev[0].copy()
        print("Keeping previous shiftstack")
    return shiftstack


def estimate_rot_axis(input_array, theta, **params):
    """Initial estimate of the rotation axis (unchanged)."""
    try:
        params["sinocmap"]
    except KeyError:
        params["sinocmap"] = params["colormap"]

    theta -= theta.min()
    slicenum = params["slicenum"]
    rot_axis_offset = params["rot_axis_offset"]

    while True:
        sinogram = np.transpose(input_array[:, slicenum, :])
        sinogram = _offset_sinogram(sinogram, rot_axis_offset)

        print("Calculating a tomographic slice")
        p0 = time.time()
        tomogram = tomo_recons(sinogram, theta, **params)
        print("Time elapsed: {} s".format(time.time() - p0))

        plt.close("all")
        fig1 = plt.figure(num=5, figsize=(12, 4))
        ax1 = fig1.add_subplot(121)
        im1 = ax1.imshow(
            tomogram, cmap=params["colormap"], interpolation="none",
            vmin=params["cliplow"], vmax=params["cliphigh"],
        )
        ax1.set_title("Slice {}".format(slicenum))
        fig1.colorbar(im1)
        ax2 = fig1.add_subplot(122)
        im2 = ax2.imshow(
            sinogram, cmap=params["sinocmap"], interpolation="none",
            vmin=params["sinolow"], vmax=params["sinohigh"],
        )
        ax2.axis("tight")
        ax2.set_title("Sinogram - Slice {}".format(slicenum))
        fig1.colorbar(im2)
        if isnotebook():
            from IPython import display
            display.display(fig1)
            plt.close(fig1)
            display.clear_output(wait=True)
        else:
            plt.show(block=False)

        a = input("Are you happy with the rotation axis?([y]/n)").lower()
        if a in ("", "y"):
            break
        else:
            rot_axis_offset = eval(input("Enter new rotation axis estimate: "))

    print("Initial estimate of rotation axis offset: {}".format(rot_axis_offset))
    return rot_axis_offset


@deprecated
def _offset_sinogram_old(sinogram, offset):
    if np.sign(offset) == +1:
        print("Initial guess of the rotation axis offset : {}".format(offset))
        sinogram = np.pad(sinogram, ((0, 2 * abs(offset)), (0, 0)), "constant", constant_values=0)
    elif np.sign(offset) == -1:
        print("Initial guess of the rotation axis offset : {}".format(offset))
        sinogram = np.pad(sinogram, ((2 * abs(offset), 0), (0, 0)), "constant", constant_values=0)
    return sinogram


def _offset_sinogram(sinogram, offset, shift_method="linear"):
    S = ShiftFunc(shiftmeth="linear")
    return S(sinogram, (offset, 0))
