#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard packages
import os

# third party packages
from ..utils.plot_utils import plt
from joblib import Parallel, delayed, parallel_backend
import numpy as np
from scipy.fft import fftfreq, fft, ifft
from ..utils import tqdm

# local packages
from ..registration.shift import ShiftFunc
from ..utils.plot_utils import _plotdelimiters
from ..utils import isnotebook

__all__ = [
    "calculate_derivatives",
    "calculate_derivatives_fft",
    "chooseregiontoderivatives",
    "derivatives",
    "derivatives_fft",
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
    # Avoids allocating two full temporaries (t1, t2) of the same shape.
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
            # print(stack_array.shape)
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
                    # roix = range(valx, aligned.shape[2] - valx)  # update roix
                    break
                else:
                    print("Wrong typing. Try it again.")

    return roix, roiy


def calculate_derivatives(stack_array, roiy, roix, shift_method="fourier", n_cpus=1):
    """
    Compute projection derivatives

    Parameters
    ----------
    stack_array : array_like
        Input stack of arrays to calculate the derivatives
    roix, roiy : tuple
        Limits of the area on which to calculate the derivatives
    shift_method : str
        Name of the shift method to use. For the available options, please
        see :class:`ShiftFunc()` in :mod:`toupy.registration`
    n_cpus : int, optional
        Number of CPU cores for parallel computation via joblib.
        ``1`` (default) runs sequentially.  ``-1`` uses all available cores.

    Returns
    -------
    aligned_diff : array_like
        Stack of derivatives of the arrays along the horizontal direction
    """
    nprojs, nr, nc = stack_array.shape
    roi_stack = stack_array[:, roiy[0] : roiy[-1], roix[0] : roix[-1]]

    if n_cpus == 1:
        aligned_diff = np.empty_like(roi_stack)
        for ii in tqdm(range(nprojs), desc="Computing derivatives"):
            aligned_diff[ii] = derivatives(roi_stack[ii], shift_method)
    else:
        if n_cpus < 0:
            n_cpus = os.cpu_count()
        with parallel_backend("loky"):
            results = Parallel(n_jobs=n_cpus)(
                delayed(derivatives)(roi_stack[ii], shift_method)
                for ii in tqdm(range(nprojs), desc="Computing derivatives")
            )
        aligned_diff = np.array(results)

    return aligned_diff


def calculate_derivatives_fft(stack_array, roiy, roix, n_cpus=-1):
    """
    Compute projection derivatives using FFTs

    Parameters
    ----------
    stack_array : array_like
        Input stack of arrays to calculate the derivatives
    roix, roiy : tuple
        Limits of the area on which to calculate the derivatives
    n_cpus : int, optional
        Number of threads passed to ``scipy.fft`` via its ``workers``
        parameter.  ``-1`` (default) uses all available CPU cores.
        ``1`` runs single-threaded.

    Returns
    -------
    aligned_diff : array_like
        Stack of derivatives of the arrays along the horizontal direction

    Notes
    -----
    The entire ROI stack is processed in a single pair of
    :func:`scipy.fft.fft` / :func:`scipy.fft.ifft` calls.
    ``scipy.fft`` treats the leading ``(nprojs, roi_rows)`` axes as
    independent batch rows and distributes them across ``workers`` threads
    internally — no Python-level loop or process pool is needed.
    """
    if n_cpus < 0:
        n_cpus = os.cpu_count() or 1

    roi_stack = stack_array[:, roiy[0] : roiy[-1], roix[0] : roix[-1]]

    # Frequency axis for the horizontal (column) direction
    nc_roi = roi_stack.shape[2]
    freqs = fftfreq(nc_roi)                        # shape (nc_roi,)

    # Symmetric-difference filter kernel (same as derivatives_fft default)
    kernel = (
        np.exp( 1j * 2.0 * np.pi * freqs * 0.5)
        - np.exp(-1j * 2.0 * np.pi * freqs * 0.5)
    )                                              # shape (nc_roi,)

    # Single FFT call over the entire (nprojs, roi_rows, nc_roi) array.
    # scipy.fft parallelises across all rows of all projections at once
    # using C-level threads that release the GIL — no joblib needed.
    fft_all = fft(roi_stack, workers=n_cpus)       # FFT along axis=-1
    fft_all *= kernel                              # broadcast (nc_roi,)
    aligned_diff = ifft(fft_all, workers=n_cpus).real

    return aligned_diff.astype(roi_stack.dtype)


def derivatives(input_array, shift_method="fourier"):
    """
    Calculate the derivative of an image

    Parameters
    ----------
    input_array: array_like
        Input image to calculate the derivatives
    shift_method: str
        Name of the shift method to use. For the available options, please
        see :class:`ShiftFunc()` in :mod:`toupy.registration`

    Returns
    -------
    diffimg : array_like
        Derivatives of the images along the row direction
    """
    S = ShiftFunc(shiftmeth=shift_method)
    rshift = [0, 0.5]
    lshift = [0, -0.5]
    diffimg = S(input_array, rshift) - S(input_array, lshift)
    # ~ if shift_method == 'phasor':
    # ~ diffimg = np.angle(S(np.exp(1j*input_array),rshift,'reflect',True)*S(np.exp(-1j*input_array),lshift,'reflect',True))
    return diffimg


def derivatives_fft(input_img, symmetric=True, n_cpus=-1):
    """
    Calculate the derivative of an image using FFT along the horizontal direction

    Parameters
    ----------
    input_array: array_like
        Input image to calculate the derivatives
    symmetric: bool
        If `True`, symmetric difference is calculated
    n_cpus: int
        The number of cpus for parallel computing. If `n_cpus<0`, the number of cpus
        will be determined by `os.cpu_counts()`

    Returns
    -------
    diffimg : array_like
        Derivatives of the images along the row direction
    """
    if n_cpus < 0:
        n_cpus = os.cpu_count()
    freqs = fftfreq(input_img.shape[1])
    if symmetric:
        rshift, lshift = 0.5, 0.5
    else:
        rshift, lshift = 1.0, 0.0
    fftimg = (
        np.exp(1j * 2 * np.pi * freqs * rshift)
        - np.exp(-1j * 2 * np.pi * freqs * lshift)
    ) * fft(input_img, workers=n_cpus)
    return ifft(fftimg, workers=n_cpus).real


def derivatives_sino(input_sino, shift_method="fourier"):
    """
    Calculate the derivative of the sinogram

    Parameters
    ----------
    input_array : array_like
        Input sinogram to calculate the derivatives
    shift_method : str
        Name of the shift method to use. For the available options, please
        see :class:`ShiftFunc()` in :mod:`toupy.registration`

    Returns
    -------
    diffsino : array_like
        Derivatives of the sinogram along the radial direction
    """
    rollsino = np.rollaxis(input_sino, 1)  # same as np.transpose(input_sino,1)
    rolldiff = derivatives(rollsino, shift_method)
    diffsino = np.rollaxis(rolldiff, 1)  # same as np.transpose(rolldiff,1)
    return diffsino
