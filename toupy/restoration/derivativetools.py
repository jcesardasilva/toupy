#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard packages
import os
import warnings

# third party packages
from ..utils.plot_utils import plt
import numpy as np
from scipy.fft import fftfreq, fft, ifft
from ..utils import tqdm

# local packages
from ..registration.shift import ShiftFunc
from ..utils.plot_utils import _plotdelimiters
from ..utils import isnotebook

__all__ = [
    "calculate_derivatives",
    "calculate_derivatives_fft",   # deprecated alias — use calculate_derivatives
    "chooseregiontoderivatives",
    "derivatives",
    "derivatives_fft",             # deprecated alias — use derivatives
    "derivatives_sino",
    "gradient_axis",
]


def gradient_axis(x, axis=-1):
    """
    Compute the forward-difference gradient along one axis, preserving shape.

    Unlike :func:`numpy.gradient`, this function keeps all dimensions
    unchanged and sets the last slice along the chosen axis to zero.

    Parameters
    ----------
    x : ndarray, shape (..., M, N)
        Input 2-D (or higher-dimensional) array.
    axis : int, optional
        Axis along which to compute the difference.  ``-1`` (default)
        computes the difference along columns; ``0`` computes it along rows.

    Returns
    -------
    ndarray
        Array of the same shape as ``x`` containing the forward finite
        differences along ``axis``.
    """
    # Single output array: out[i] = x[i+1] - x[i], last slice = 0.
    # Avoids allocating two full temporaries of the same shape.
    out = np.empty_like(x)
    if axis != 0:
        out[:, :-1] = x[:, 1:] - x[:, :-1]
        out[:, -1] = 0
    else:
        out[:-1, :] = x[1:, :] - x[:-1, :]
        out[-1, :] = 0
    return out


def chooseregiontoderivatives(stack_array, **params):
    """
    Interactively choose the region of interest for derivative computation.

    Displays the first projection with the current ROI boundaries overlaid
    and lets the user refine the limits before returning.

    Parameters
    ----------
    stack_array : ndarray, shape (n, nr, nc)
        Stack of projection images.
    **params
        Must contain:

        deltax : int
            Horizontal margin in pixels to exclude from the left and right
            edges of the image.
        limsy : tuple of int
            ``(row_start, row_end)`` vertical limits passed to
            :func:`range` via tuple unpacking.

    Returns
    -------
    roix : range
        Horizontal index range selected by the user.
    roiy : range
        Vertical index range selected by the user.
    """
    # horizontal ROI
    deltax = params["deltax"]
    roix = range(deltax, stack_array.shape[2] - deltax)  # update roix
    roiy = range(*params["limsy"])  # tuple unpacking

    # Display the projections
    while True:
        plt.close("all")
        fig = plt.figure(5)
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(stack_array[0], cmap="bone")
        ax1 = _plotdelimiters(ax1, roiy, roix)
        ax1.axis("tight")
        if isnotebook():
            from IPython import display
            display.display(fig)
            plt.close(fig)
        else:
            plt.show(block=False)

        ans = input("Are you happy with the boundaries? ([y]/n)").lower()
        if str(ans) == "" or str(ans) == "y":
            break
        else:
            print(
                "The array dimensions are {} x {}".format(
                    stack_array[0].shape[0], stack_array[0].shape[1]
                )
            )
            while True:
                roiy = eval(input("Enter new range in y (top, bottom): "))
                if isinstance(roiy, tuple):
                    roiy = range(roiy[0], roiy[-1])
                    break
                else:
                    print("Wrong typing. Try it again.")
            while True:
                deltax = eval(
                    input("Enter new value from edge of region to edge of image in x: ")
                )
                if isinstance(deltax, int):
                    roix = range(deltax, stack_array.shape[2] - deltax)  # update roix
                    break
                else:
                    print("Wrong typing. Try it again.")

    return roix, roiy


def derivatives(input_array, shift_method="fourier", symmetric=True, n_cpus=-1):
    """
    Calculate the horizontal derivative of a 2-D image.

    Parameters
    ----------
    input_array : array_like
        Input 2-D image.
    shift_method : str, optional
        Shift / differentiation method.

        ``"fourier"`` (default)
            Pure FFT symmetric-difference filter.  Applied directly via
            :func:`scipy.fft.fft` — no ``ShiftFunc`` overhead.  Supports
            the ``symmetric`` and ``n_cpus`` parameters.

        ``"spline"``, ``"linear"``
            Sub-pixel shift via :class:`~toupy.registration.shift.ShiftFunc`
            (always symmetric ±0.5 px).  ``symmetric`` and ``n_cpus`` are
            ignored for these methods.

    symmetric : bool, optional
        Only used when ``shift_method="fourier"``.  If ``True`` (default),
        applies a symmetric ±½-pixel difference (filter
        ``2i·sin(π f)``).  If ``False``, applies a forward 1-pixel
        difference (filter ``exp(2πi f) − 1``).
    n_cpus : int, optional
        Number of threads passed to :func:`scipy.fft.fft`.
        Only used when ``shift_method="fourier"``.
        ``-1`` (default) uses all available cores.

    Returns
    -------
    diffimg : ndarray
        Derivative image (same shape as ``input_array``).
    """
    if shift_method == "fourier":
        if n_cpus < 0:
            n_cpus = os.cpu_count() or 1
        freqs = fftfreq(input_array.shape[1])
        if symmetric:
            rshift, lshift = 0.5, 0.5
        else:
            rshift, lshift = 1.0, 0.0
        kernel = (
            np.exp( 1j * 2.0 * np.pi * freqs * rshift)
            - np.exp(-1j * 2.0 * np.pi * freqs * lshift)
        )
        return ifft(kernel * fft(input_array, workers=n_cpus),
                    workers=n_cpus).real

    # Non-fourier methods use ShiftFunc (C extensions, GIL released)
    S = ShiftFunc(shiftmeth=shift_method)
    return S(input_array, [0, 0.5]) - S(input_array, [0, -0.5])


def derivatives_fft(input_img, symmetric=True, n_cpus=-1):
    """
    .. deprecated::
        Use :func:`derivatives` with ``shift_method="fourier"`` instead.
    """
    warnings.warn(
        "derivatives_fft() is deprecated; use derivatives(shift_method='fourier') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return derivatives(input_img, shift_method="fourier",
                       symmetric=symmetric, n_cpus=n_cpus)


def calculate_derivatives(stack_array, roiy, roix,
                           shift_method="fourier", symmetric=True, n_cpus=-1):
    """
    Compute projection derivatives over a stack of images.

    Parameters
    ----------
    stack_array : array_like, shape (nprojs, nr, nc)
        Input stack of projection images.
    roiy, roix : range or tuple
        Row and column limits of the ROI.
    shift_method : str, optional
        Passed to :func:`derivatives`.  Default ``"fourier"``.
    symmetric : bool, optional
        Passed to :func:`derivatives` (fourier path only).
        Default ``True``.
    n_cpus : int, optional
        Number of CPU cores / threads.

        * ``"fourier"`` — passed as ``workers`` to :func:`scipy.fft.fft`,
          which parallelises across **all** rows of all projections in a
          single call (no Python loop).
        * Other methods — ignored; each C extension manages its own
          threading internally.

        ``-1`` (default) uses all available cores.

    Returns
    -------
    aligned_diff : ndarray, shape (nprojs, roi_nr, roi_nc)
        Stack of derivative images.
    """
    roi_stack = stack_array[:, roiy[0]:roiy[-1], roix[0]:roix[-1]]

    if shift_method == "fourier":
        # Vectorise over the entire (nprojs, roi_rows, nc) array in one
        # scipy.fft call — no Python-level loop needed.
        if n_cpus < 0:
            n_cpus = os.cpu_count() or 1
        nc_roi = roi_stack.shape[2]
        freqs = fftfreq(nc_roi)
        if symmetric:
            rshift, lshift = 0.5, 0.5
        else:
            rshift, lshift = 1.0, 0.0
        kernel = (
            np.exp( 1j * 2.0 * np.pi * freqs * rshift)
            - np.exp(-1j * 2.0 * np.pi * freqs * lshift)
        )                                              # shape (nc_roi,)
        fft_all = fft(roi_stack, workers=n_cpus)       # axis=-1 by default
        fft_all *= kernel                              # broadcast over leading axes
        return ifft(fft_all, workers=n_cpus).real.astype(roi_stack.dtype)

    # Non-fourier: sequential loop; C extensions handle their own threading
    aligned_diff = np.empty_like(roi_stack)
    for ii in tqdm(range(stack_array.shape[0]), desc="Computing derivatives"):
        aligned_diff[ii] = derivatives(roi_stack[ii], shift_method)
    return aligned_diff


def calculate_derivatives_fft(stack_array, roiy, roix, n_cpus=-1):
    """
    .. deprecated::
        Use :func:`calculate_derivatives` with ``shift_method="fourier"`` instead.
    """
    warnings.warn(
        "calculate_derivatives_fft() is deprecated; "
        "use calculate_derivatives(shift_method='fourier') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calculate_derivatives(stack_array, roiy, roix,
                                 shift_method="fourier", n_cpus=n_cpus)


def derivatives_sino(input_sino, shift_method="fourier"):
    """
    Calculate the derivative of the sinogram along the radial direction.

    Parameters
    ----------
    input_sino : array_like
        Input sinogram.
    shift_method : str, optional
        Passed to :func:`derivatives`.  Default ``"fourier"``.

    Returns
    -------
    diffsino : array_like
        Derivative of the sinogram along the radial direction.
    """
    rollsino = np.rollaxis(input_sino, 1)  # (nc, nprojs) → derivative along rows
    rolldiff = derivatives(rollsino, shift_method)
    return np.rollaxis(rolldiff, 1)
