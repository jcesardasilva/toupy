#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import time

# third party packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, interpolation
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d


# local packages
from .register_translation_fast import register_translation
from .shift import ShiftFunc
from ..tomo.iradon import backprojector
from ..tomo.radon import projector
from ..utils.funcutils import deprecated
from ..utils.array_utils import projectpoly1d, fract_hanning_pad
from ..restoration import derivatives, derivatives_sino
from ..utils.plot_utils import RegisterPlot

__all__ = [
    "compute_aligned_stack",
    "compute_aligned_sino",
    "center_of_mass_stack",
    "vertical_fluctuations",
    "vertical_shift",
    "alignprojections_vertical",
    "alignprojections_horizontal",
    "cc_align",
]


def compute_aligned_stack(input_stack, shiftstack, shift_method="linear"):
    """
    Compute the aligned stack given the correction for object positions

    Parameters
    ----------
    input_array : ndarray
        Stack of images to be shifted
    shiftstack : ndarray
        Array of initial estimates for object motion (2,n)
    shift_method : str (default linear)
        Name of the shift method. Options: 'linear', 'fourier', 'spline'

    Returns
    -------
    output_stack : ndarray
        2D function containing the stack of aligned images
    """
    # Initialize shift class
    S = ShiftFunc(shiftmeth=shift_method)
    # array shape
    nstack = input_stack.shape[0]
    print(
        "Using {} shift method (function {})".format(shift_method, S.shiftmeth.__name__)
    )
    output_stack = np.empty_like(input_stack)
    for ii in range(nstack):
        deltashift = (shiftstack[0, ii], shiftstack[1, ii])
        output_stack[ii] = S(input_stack[ii], deltashift)
        print("Image {} of {}".format(ii + 1, nstack), end="\r")
    print("\r")
    return output_stack


def compute_aligned_sino(input_sino, shiftslice, shift_method="linear"):
    """
    Compute the aligned sinogram given the correction for object positions

    Parameters
    ----------
    input_sino : ndarray
        Input sinogram to be shifted
    shiftslice : ndarray
        Array of estimates for object motion (1,n)
    shift_method : str (default linear)
        Name of the shift method. Options: 'linear', 'fourier', 'spline'

    Returns
    -------
    output_sino: ndarray
        2D function containing the aligned sinogram
    """
    # Initialize shift class
    S = ShiftFunc(shiftmeth=shift_method)
    # array shape
    nprojs = input_sino.shape[1]
    print(
        "Using {} shift method (function {})".format(shift_method, S.shiftmeth.__name__)
    )
    output_sino = np.empty_like(input_sino)
    for ii in range(nprojs):
        deltashift = shiftslice[0, ii]
        output_sino[:, ii] = S(input_sino[:, ii], deltashift)
        print("Image {} of {}".format(ii + 1, nprojs), end="\r")
    print("\r")
    return output_sino


def center_of_mass_stack(input_stack, lims, shiftstack, shift_method="fourier"):
    """
    Calculates the center of the mass for each projection in the stack and
    returns a stack of centers of mass (row, col) i.e., returns shiftstack[1]
    If the array is zero, it return the center of mass at 0.
    """
    # separate lims
    limrow, limcol = lims

    print("Calculating center-of-mass with pixel precision")
    # initialize shift class
    S = ShiftFunc(shiftmeth=shift_method)

    # create array positions
    stack_roi = input_stack[0, limrow[0] : limrow[-1], limcol[0] : limcol[-1]].copy()
    ind_roi = np.indices(stack_roi.shape)
    # create array Xp of horizontal of positions
    ind_roi[1] -= (
        np.floor(ind_roi[1].mean(axis=1)).reshape((ind_roi.shape[1], 1)).astype("int")
    )
    Xp = ind_roi[1].astype("float")
    # create array Xp of horizontal of positions
    ind_roi[0] -= (
        np.floor(ind_roi[0].mean(axis=0)).reshape((ind_roi.shape[2], 1)).T.astype("int")
    )
    Yp = ind_roi[0].astype("float")

    # initializing the arrays
    mass_sum = np.empty(input_stack.shape[0])
    centerx = np.empty(input_stack.shape[0])
    centery = np.empty(input_stack.shape[0])

    for ii in range(input_stack.shape[0]):
        stack_aux = S(input_stack[ii], (shiftstack[0, ii], shiftstack[1, ii]))
        mass_sum[ii] = np.sum(stack_aux[limrow[0] : limrow[-1], limcol[0] : limcol[-1]])
        centerx[ii] = np.sum(
            Xp * stack_aux[limrow[0] : limrow[-1], limcol[0] : limcol[-1]]
        )
        centery[ii] = np.sum(
            Yp * stack_aux[limrow[0] : limrow[-1], limcol[0] : limcol[-1]]
        )
    centerx[np.nonzero(mass_sum)] = (
        centerx[np.nonzero(mass_sum)] / mass_sum[np.nonzero(mass_sum)]
    )
    centerx[np.where(mass_sum == 0)] = 0
    centery = np.asarray(centery)
    centery[np.nonzero(mass_sum)] = (
        centery[np.nonzero(mass_sum)] / mass_sum[np.nonzero(mass_sum)]
    )
    centery[np.where(mass_sum == 0)] = 0

    return np.asarray([centerx, centery])


def vertical_fluctuations(
    input_stack, lims, shiftstack, shift_method="fourier", polyorder=2
):
    """
    Calculate the vertical fluctuation functions of a stack

    Parameters
    ----------
    input_array : ndarray
        Stack of images to be shifted
    lims : list of ints
        Limits of rows and columns to be considered. lims=[limrow,limcol]
    shiftstack : ndarray
        Array of initial estimates for object motion (2,n)
    shift_method : str (default linear)
        Name of the shift method. Options: 'linear', 'fourier', 'spline'
    polyorder : int (default 2)
        Order of the polynomial to remove bias from the mass fluctuation
        function

    Returns
    -------
    vert_fluct : ndarray
        2D function containing the mass fluctuation after shift and bias
        removal for the stack of images
    """
    # Initialize shift class
    S = ShiftFunc(shiftmeth=shift_method)
    # array shape
    _, nr, nc = input_stack.shape
    # separate the lims
    rows, cols = lims
    # get the maximum shift value
    max_vshift = int(np.ceil(np.max(np.abs(shiftstack[0, :])))) + 1
    if np.any((rows - max_vshift) < 0) or np.any((rows + max_vshift) > nr):
        max_vshift = 1

    # initializing array
    # +2*max_vshift))
    vert_fluct = np.empty((input_stack.shape[0], rows[-1] - rows[0]))
    for ii in range(input_stack.shape[0]):
        print("Calculating for projection: {}".format(ii + 1), end="\r")
        proj = input_stack[
            ii, rows[0] - max_vshift : rows[-1] + max_vshift, cols[0] : cols[-1]
        ]
        stack_shift = S(proj, (shiftstack[0, ii], 0.0))
        # the max_vshift has to be subtracted
        shift_calc = stack_shift[max_vshift:-max_vshift].sum(axis=1)
        # to remove possible bias
        shift_calc = projectpoly1d(shift_calc, order, 1)
        vert_fluct[ii] = shift_calc
    return vert_fluct


def vertical_shift(
    input_array, lims, vstep, maxshift, shift_method="linear", polyorder=2
):
    """
    Calculate the vertical shift of an array

    Parameters
    ----------
    input_array : ndarray
        Image to be shifted
    lims : list of ints
        Limits of rows and columns to be considered. lims=[limrow,limcol]
    vstep : float
        Amount to shift the input_array vertically
    maxshift : float
        Maximum value of the shifts in order to avoid border problems
    shift_method : str (default linear)
        Name of the shift method. Options: 'linear', 'fourier', 'spline'
    polyorder : int (default 2)
        Order of the polynomial to remove bias from the mass fluctuation
        function

    Returns
    -------
    shift_cal : ndarray
        1D function containing the mass fluctuation after shift and bias
        removal
    """
    # Initialize shift class
    S = ShiftFunc(shiftmeth=shift_method)
    # array shape
    nr, nc = input_array.shape

    # Max vertical shift + 1. At least one for a margin. Had to take the int of vstep.
    max_vshift = maxshift + int(np.abs(vstep))  # +1

    # get the maximum shift value
    rows, cols = lims
    if np.any((rows - max_vshift) < 0) or np.any((rows + max_vshift) > nr):
        max_vshift = 1
    stack_shift = S(
        input_array[rows[0] - max_vshift : rows[-1] + max_vshift, cols[0] : cols[-1]],
        (vstep, 0.0),
    )
    # Integration because stack_shift is 2D
    shift_calc = stack_shift[max_vshift:-max_vshift].sum(axis=1)
    # to remove possible bias
    shift_calc = projectpoly1d(shift_calc, polyorder, 1)

    return shift_calc


def _alignprojections_vertical(
    input_stack, lims, shiftstack, metric_error, vert_fluct_init, RP, **params
):
    """
    Auxiliary function for align the projection vertically. It contains
    the wrapper for iteration during the alignement
    """
    # Initialize the counter
    count = 0
    error_reg = np.zeros(vert_fluct_init.shape[0])
    while True:
        count += 1
        print("\n============================================")
        print("Iteration {}".format(count))
        it0 = time.time()
        deltaprev = shiftstack.copy()

        # Mass distribution registration in y
        if count == 1:
            vert_fluct = vert_fluct_init.copy()
        else:
            print("Updating the vertical fluctuations")
            vert_fluct = vertical_fluctuations(
                input_stack,
                lims,
                shiftstack,
                params["shiftmeth"],
                polyorder=params["polyorder"],
            )

        # Average the vertical fluctuation functions
        print("Calculating the average of the vertical fluctuation function")
        vert_fluct_mean = vert_fluct.mean(axis=0)

        # Search for shifts with respect to mean
        print("Search for the shifts with respect to the mean vertical fluctuations...")
        shiftstack_aux, vert_fluct_temp = _search_vshift_stack(
            input_stack, lims, shiftstack, vert_fluct_mean, **params
        )
        shiftstack[0] = shiftstack_aux[0].copy()
        shiftstack[0] -= shiftstack_aux[0].mean().round()  # recentering

        # Error calculation
        # keep temporarily the vertical fluctuations
        vert_fluct_mean_temp = vert_fluct_temp.mean(axis=0)
        print("\nCalculating the error metric")
        for ii in range(vert_fluct_temp.shape[0]):
            error_reg[ii] = np.sum(
                np.abs(vert_fluct_temp[ii] - vert_fluct_mean_temp) ** 2
            )
        print("Final error metric for y, E = {:.04e}".format(np.sum(error_reg)))
        metric_error.append(np.sum(error_reg))

        # Maximum changes in y
        print("Estimating the changes in y:")
        changey = np.abs(deltaprev[0] - shiftstack[0])
        print("Maximum correction in y = {:.02f} pixels".format(np.max(changey)))

        print("Elapsed time = {} s".format(time.time() - it0))

        # update figures
        RP.plotsvertical(
            input_stack[0],
            lims,
            vert_fluct_init,
            vert_fluct_temp,
            shiftstack,
            metric_error,
            count,
        )

        if params["subpixel"]:
            pixtol = params["pixtol"]
        else:
            pixtol = 1
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


def alignprojections_vertical(input_stack, limrow, limcol, shiftstack, **params):
    """
    Vertical alignment of projections using mass fluctuation approach.
    It relies on having air on both sides of the sample (non local tomography).
    It performs a local search in y, so convergence issues can be addressed by
    giving an approximate initial guess for a possible drift via shiftstack

    Parameters
    ----------
    input_stack : ndarray
        Stack of projections
    limrow : list of ints
        Limits of window of interest in y
    limcol : list of ints
        Limits of window of interest in x
    shiftstack : ndarray
        Array of initial estimates for object motion (2,n)
    Extra parameters in the dictionary params:
    params['pixtol'] : float
        Tolerance for alignment, which is also used as a search step
    params['polyorder'] : int
        Specify the polynomial order of bias removal.
        For example: polyorder = 1 -> mean, polyorder = 2 -> linear).
    params['alignx'] : bool
        True or False to activate align x using center of mass
        (default= False, which means align y only)
    params['shiftmeth'] : str
        Shift images with sinc interpolation (default). The options are:
           'linear' - Shift images with linear interpolation (default)
           'fourier' - Fourier shift
           'spline' - Shift images with spline interpolation

    Returns
    -------
    shiftstack : ndarray
        Corrected bject positions
    input_stack : ndarray
        Aligned stack of the projections
    """
    if not isinstance(params["maxit"], int):
        params["maxit"] = 10
    if not isinstance(limrow, np.ndarray) or not isinstance(limcol, np.ndarray):
        limrow = np.asarray(limrow)
        limcol = np.asarray(limcol)
    lims = (limrow, limcol)

    print("\n============================================")
    print("Vertical Mass fluctuation pixel alignment")
    print("Number of iteration: 10".format(params["maxit"]))

    # horizontal alignement with center of mass if requested
    if params["alignx"] and count == 0:
        print("Estimating the changes in x using center-of-mass:")
        centerx = center_of_mass_stack(
            input_stack, params, limrow=limrow, limcol=limcol, shiftstack=shiftstack
        )[
            0
        ]  # [1]
        # Correction with mass center
        shiftstack[1] = -centerx.round()
        # ~ shiftstack[1] -= shiftstack[1].mean().round()
        changex = np.max(np.abs(deltaprev[1] - shiftstack[1]))
        print(
            "Maximum correction of center of mass in x = {:.02f} pixels".format(changex)
        )
    else:
        changex = 0

    # first iteration only correcting for the limrow and limcol and in case shiftstack is already no zero
    vert_fluct_init = vertical_fluctuations(
        input_stack,
        (limrow, limcol),
        shiftstack,
        params["shiftmeth"],
        polyorder=params["polyorder"],
    )
    avg_init = vert_fluct_init.mean(axis=0)
    shiftstack_init = shiftstack.copy()
    nr, nc = vert_fluct_init.shape  # for the image display

    # Store initial states
    metric_error = []  # initialize metrics
    error_init = np.zeros(vert_fluct_init.shape[0])
    error_reg = np.zeros_like(error_init)
    for ii in range(vert_fluct_init.shape[0]):
        error_init[ii] = np.sum(np.abs(vert_fluct_init[ii] - avg_init) ** 2)
    print("Initial error metric for y, E = {:.02e}".format(np.sum(error_init)))
    metric_error.append(np.sum(error_init))

    # initializing display canvas for the figures
    plt.ion()
    RP = RegisterPlot(**params)
    RP.plotsvertical(
        input_stack[0],
        lims,
        vert_fluct_init,
        vert_fluct_init,
        shiftstack_init,
        metric_error,
        count=0,
    )

    # Single pixel precision
    print("\n================================================")
    print("Registration of projections with pixel precision")
    print("================================================")
    params["subpixel"] = False

    shiftstack, metric_error = _alignprojections_vertical(
        input_stack, lims, shiftstack, metric_error, vert_fluct_init, RP, **params
    )

    # Subpixel precision
    print("\n================================================")
    print("Registration of projections with subpixel precision")
    print("================================================")

    params["subpixel"] = True
    shiftstack, metric_error = _alignprojections_vertical(
        input_stack, lims, shiftstack, metric_error, vert_fluct_init, RP, **params
    )

    # Compute the shifted images
    print("Computing aligned images")
    output_stack = compute_aligned_stack(
        input_stack, shiftstack.copy(), shift_method=params["shiftmeth"]
    )

    return shiftstack, output_stack


def _alignprojections_horizontal(
    sinogram, sino_orig, theta, circleROI, shiftslice, metric_error, RP, **params
):
    """
    Auxiliary function for align the projection horizontally. It contains
    the wrapper for iteration during the alignement
    """
    # Compute tomogram with current sinogram
    print("Initializing tomographic slice...")
    t0 = time.time()
    recons = backprojector(
        sinogram, theta=theta, output_size=sinogram.shape[0], **params
    )
    recons_std = recons.std()
    # clipping gray level if needed
    recons = _clipping_tomo(recons, **params)
    if params["circle"]:
        recons = recons * circleROI
    print("Done. Time elapsed: {} s".format(time.time() - t0))
    print("Slice standard deviation = {:0.04e}".format(recons_std))

    # Initialize the counter
    count = 0
    while True:
        count += 1
        print("\nIteration {}".format(count))
        print("-------------------------------------")
        it0 = time.time()
        sinoprev = sinogram.copy()
        # keep deltaprev in case the iteration does not decrease the error
        deltaprev = shiftslice.copy()

        # Compute synthetic sinogram
        print("Computing synthetic sinogram...")
        sinogramcomp = projector(recons, theta, **params)
        if params["derivatives"]:
            sinogramcomp = derivatives_sino(
                sinogramcomp, shift_method=params["shiftmeth"]
            )

        # Start searching for shift relative to synthetic sinogram
        sinotempreg, shiftslice = _search_hshift_sinogram(
            sino_orig, sinogramcomp, shiftslice, **params
        )
        # updating sinogram
        sinogram = compute_aligned_sino(
            sino_orig, shiftslice, shift_method=params["shiftmeth"]
        )

        # Compute tomogram with current sinogram
        print("Computing tomographic slice...")
        t0 = time.time()
        recons = backprojector(
            sinogram, theta=theta, output_size=sinogram.shape[0], **params
        )
        recons_std = recons.std()
        # clipping gray level if needed
        recons = _clipping_tomo(recons, **params)
        if params["circle"]:
            recons = recons * circleROI
        print("Done. Time elapsed: {} s".format(time.time() - t0))
        print("Slice standard deviation = {:0.04e}".format(recons_std))

        # Calculate the error:
        errorxreg = _sino_error_metric(sinogram, sinogramcomp, params)
        sumerrorxreg = errorxreg.sum()
        print("Final error metric for x, E = {:0.04e}".format(sumerrorxreg))
        metric_error.append(sumerrorxreg)

        # Estimate amount of changes
        print("Estimating the changes in x:")
        changex = np.abs(deltaprev - shiftslice)
        if params["subpixel"]:
            strprint = "Maximum correction in x = {:0.02f} pixels"
        else:
            strprint = "Maximum correction in x = {:0.02g} pixels"
        print(strprint.format(np.max(changex)))

        print("Elapsed time in the iteration= {:0.02f} s".format(time.time() - it0))

        # update figures
        RP.plotshorizontal(
            recons, sino_orig, sinogram, sinogramcomp, shiftslice, metric_error, count
        )

        if params["subpixel"]:
            pixtol = params["pixtol"]
        else:
            pixtol = 1
        reason = _checkconditions(
            metric_error, changex, pixtol, count, params["maxit"], params["subpixel"]
        )

        if reason == 1:
            shiftslice = deltaprev.copy()
            # ~ sinogram = sinoprev.copy()
            metric_error.pop()
            break
        elif reason >= 2:
            break

    return shiftslice, metric_error


def alignprojections_horizontal(sinogram, theta, shiftslice, **params):
    """
    Function to align projections. It relies on having already aligned the
    vertical direction. The code aligns using the consistency before and
    after tomographic combination of projections.

    Parameters
    ----------
    sinogram : ndarray
        Sinogram derivative, the second index should be the angle
    shiftslice : ndarray
        Row array with initial estimates of positions
    Extra parameters in the dictionary params:
    params['pixtol'] : float
        Tolerance for alignment, which is also used as a search step
    params['disp'] : int
        = 0 Display no images
        = 1 Final diagnostic images
        = 2 Diagnostic images per iteration
    params['alignx'] : bool
        True or False to activate align x using center of mass
        (default= False, which means align y only)
    params['shiftmeth'] : str
        Shift images with sinc interpolation (default). The options are:
           'linear' - Shift images with linear interpolation (default)
           'fourier' - Fourier shift
           'spline' - Shift images with spline interpolation
    params['circle'] : bool
        Use a circular mask to eliminate corners of the tomogram
    params['filtertomo'] : float
        Frequency cutoff for tomography filter
    params['cliplow'] : float
        Minimum value in tomogram
    params['cliphigh'] : float
        Maximum value in tomogram

    Returns
    -------
    shiftstack : ndarray
        Corrected object positions
    alinedsinogram : ndarray
        Array containting the aligned sinogram
    """
    # parsing of the parameters
    try:
        params["circle"]
    except KeyError:
        params["circle"] = True

    try:
        params["sinohigh"]
    except KeyError:
        params["sinohigh"] = 0.6

    try:
        params["sinolow"]
    except KeyError:
        params["sinolow"] = -0.6

    try:
        params["opencl"]
    except KeyError:
        params["opencl"] = False

    if not isinstance(params["maxit"], int):
        params["maxit"] = 10

    try:
        params["cliplow"]
    except:
        params["cliplow"] = None

    try:
        params["cliphigh"]
    except:
        params["cliphigh"] = None

    print("\nStarting the horizontal alignment")
    print("=====================================")
    print("Number of iteration: {}".format(params["maxit"]))
    print("Using a frequency cutoff of {}".format(params["filtertomo"]))
    print("Low limit for tomo values = {}".format(params["cliplow"]))
    print("High limit for tomo values = {}".format(params["cliphigh"]))

    # appropriate keeping of variable
    original_sino = sinogram.copy()

    # pad sinogram of derivatives
    # TODO: check if we only need this for derivative (if params['derivatives']:) or not!
    padval = int(2 * np.round(1 / params["filtertomo"]))
    sinogram = np.pad(
        sinogram, ((padval, padval), (0, 0)), "constant", constant_values=0
    ).copy()
    N = sinogram.shape[0]

    # applying a filter to the sinogram #TODO: improve this part
    filteraux = fract_hanning_pad(
        N, N, np.round(N * (1 - params["filtertomo"]))
    )  # 1- at the beginning
    filteraux = np.tile(np.fft.fftshift(filteraux[0, :]), (len(theta), 1))
    sino_orig = np.real(np.fft.ifft(np.fft.fft(sinogram) * filteraux.T))

    # Shifting projection according to the initial shiftslice
    if not np.all(shiftslice == 0):
        print("Shifting sinogram.")
        sinogram = compute_aligned_sino(
            sino_orig, shiftslice, shift_method=params["shiftmeth"]
        )
        print("Done.")
    else:
        print("Initializing shiftslice with zeros")

    # initial reconstruction
    print("Computing initial tomographic slice...")
    # Filtered back projection
    print("Backprojecting")
    t0 = time.time()
    recons = backprojector(
        sinogram, theta=theta, output_size=sinogram.shape[0], **params
    )
    print("Done. Time elapsed: {} s".format(time.time() - t0))
    print("Slice standard deviation = {:0.04e}".format(recons.std()))

    # clipping gray level if needed
    recons = _clipping_tomo(recons, **params)
    if params["circle"]:
        circleROI = _create_circle(recons)  # only need to calculate once
    else:
        circleROI = 1
    recons = recons * circleROI

    # initial synthetic sinogram
    print("Computing synthetic sinogram...")
    t0 = time.time()
    sinogramcomp = projector(recons, theta, **params)
    if params["derivatives"]:
        sinogramcomp = derivatives_sino(sinogramcomp, shift_method=params["shiftmeth"])
    print("Done. Time elapsed: {:0.02f} s".format(time.time() - t0))

    # store initial error metric
    metric_error = []
    print("Store initial error metric")
    errorinit = _sino_error_metric(sinogram, sinogramcomp, params)
    sumerrorinit = np.sum(errorinit)
    print("Initial error metric, E= {:0.04e}".format(sumerrorinit))
    metric_error.append(sumerrorinit)

    # initializing display canvas for the figures
    plt.ion()
    RP = RegisterPlot(**params)
    RP.plotshorizontal(
        recons, sino_orig, sinogram, sinogramcomp, shiftslice, metric_error, count=0
    )

    # Single pixel precision
    print("\n===================================================")
    print("Registration of projections with pixel precision")
    print("===================================================")
    params["subpixel"] = False
    shiftslice, metric_error = _alignprojections_horizontal(
        sinogram, sino_orig, theta, circleROI, shiftslice, metric_error, RP, **params
    )

    print("\n===================================================")
    print("Registration of projections with subpixel precision")
    print("===================================================")

    params["subpixel"] = True
    shiftslice, metric_error = _alignprojections_horizontal(
        sinogram, sino_orig, theta, circleROI, shiftslice, metric_error, RP, **params
    )

    # Compute the shifted images
    print("\nComputing aligned images")
    alignedsinogram = compute_aligned_sino(
        original_sino, shiftslice, shift_method=params["shiftmeth"]
    )

    print("Calculating aligned slice for display")
    p0 = time.time()
    recons = backprojector(
        alignedsinogram, theta=theta, output_size=alignedsinogram.shape[0], **params
    )
    # clipping gray level if needed
    recons = _clipping_tomo(recons, **params)
    if params["circle"]:
        circleROI = _create_circle(recons)
        recons = recons * circleROI
    print("Done. Time elapsed: {} s".format(time.time() - p0))

    fig = plt.figure(num=10)
    plt.clf()
    ax1 = fig.add_subplot(111)
    ax1.imshow(recons, cmap="bone")
    ax1.axis("image")
    ax1.set_title("Aligned tomographic slice")
    ax1.set_xlabel("x [pixels]")
    ax1.set_ylabel("y [pixels]")
    plt.show(block=False)
    plt.pause(0.01)

    return shiftslice, alignedsinogram


def tomoconsistency_multiple(input_stack, theta, shiftstack, **params):
    """
    Apply tomographic consistency alignement on multiple slices
    """
    params["apply_alignement"] = False
    print("Starting Tomographic consistency on multiple slices")
    # select the slices, which are typically +5 and -5 relative to slicenum
    slices = np.arange(params["slicenum"] - 5, params["slicenum"] + 5)
    plt.close("all")
    shiftslice = shiftstack[1].copy()
    shiftslice_prev = shiftslice.copy()
    deltaxrefine = []
    for ii in slices:
        print("\nAligning slice {}".format(ii + 1))
        sinogram = np.transpose(input_stack[:, ii, :])  # create the sinogram
        params[u"apply_alignement"] = False
        delta_aux, aligned_sino_aux = alignprojections_horizontal(
            sinogram, theta, shiftslice, **params
        )
        deltaxrefine.append(delta_aux)
        shiftslice = delta_aux.copy()  # updating shiftslice

    deltaxrefine = np.squeeze(deltaxrefine)
    deltaxrefine_avg = deltaxrefine.mean(axis=0)

    plt.close("all")
    fig = plt.figure(num=6, figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.imshow(deltaxrefine.astype(np.float), interpolation="none", cmap="jet")
    ax1.axis("tight")
    ax1.set_xlabel("Projection number")
    ax1.set_ylabel("Slice number")
    ax1.set_title("Displacements in x")
    ax2 = fig.add_subplot(212)
    ax2.plot(deltaxrefine_avg.astype(np.float), "b-", label="average")
    ax2.plot(shiftslice_prev[0], "r--", label="previous")
    ax2.legend()
    ax2.axis("tight")
    ax2.set_xlim([0, len(deltaxrefine_avg)])
    ax2.set_title("Average displacements in x")
    ax2.set_xlabel("Projection number")
    plt.tight_layout()
    plt.show()
    a = input(
        "Are you happy with the tomographic consistency alignment of the multiples slices? ([y]/n) "
    ).lower()
    if a == "" or a == "y":
        shiftstack[1] = deltaxrefine_avg.copy()
        print("Using the average of all shiftstack")
    else:
        shiftstack[1] = shiftslice_prev[0].copy()
        print("Using the shiftstack before tomographic consisteny in multiple slices")

    return shiftstack


def _search_vshift_stack(input_stack, lims, input_delta, avg_vert_fluct, **kwargs):
    """
    Search for the shifts directions for the stack
    """
    if isinstance(kwargs["pixtol"], int) or kwargs["subpixel"] == False:
        pixtol = 1
        shift_method = "linear"
    elif not isinstance(kwargs["pixtol"], int) or kwargs["subpixel"] == True:
        pixtol = kwargs["pixtol"]
        shift_method = kwargs["shiftmeth"]

    # polynomial order to remove bias
    polyorder = kwargs["polyorder"]

    # separate the lims
    rows, cols = lims
    # get array dimensions
    nprojs, nr, nc = input_stack.shape
    # get the maximum shift value from input_delta
    # plus 1 for a margin
    max_vshift = int(np.ceil(np.max(np.abs(input_delta[0, :])))) + 1
    if np.any((rows - max_vshift) < 0) or np.any((rows + max_vshift) > nr):
        max_vshift = 1  # at least one for a margin

    # initializing array
    vert_fluct_stack = np.empty((input_stack.shape[0], rows[-1] - rows[0]))
    output_shiftstack = np.empty_like(input_delta)  # np.zeros_like(input_delta)

    if not isinstance(input_stack, np.ndarray):
        input_stack = np.asarray(input_stack).copy()

    for ii in range(nprojs):
        print("Searching the shifts for projection: {}".format(ii + 1), end="\r")
        shift_delta = input_delta[0, ii]
        output_shiftstack[0, ii], vert_fluct_stack[ii] = _search_vshift_direction(
            input_stack[ii],
            lims,
            shift_delta,
            avg_vert_fluct,
            pixtol,
            max_vshift,
            shift_method,
            polyorder,
        )
    print("\r")
    return output_shiftstack, vert_fluct_stack


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
    Search for the shifts directions for each image
    """
    # Search for shifts with respect to mean
    dir_shift = dict()  # dictionary shift directions
    shifts = dict()  # dictionary shifts arrays

    # compute current shift error
    shifts["current"] = vertical_shift(
        input_array, lims, shift_delta - 0, max_vshift, shift_method, polyorder
    )
    # compute shift forward error
    shifts["forward"] = vertical_shift(
        input_array, lims, shift_delta + pixtol, max_vshift, shift_method, polyorder
    )
    # compute shift backward error
    shifts["backward"] = vertical_shift(
        input_array, lims, shift_delta - pixtol, max_vshift, shift_method, polyorder
    )

    # directional shift error calculation
    dir_shift["current"] = np.sum(np.abs(shifts["current"] - avg_vert_fluct) ** 2)
    dir_shift["forward"] = np.sum(np.abs(shifts["forward"] - avg_vert_fluct) ** 2)
    dir_shift["backward"] = np.sum(np.abs(shifts["backward"] - avg_vert_fluct) ** 2)

    # get the smallest shift error
    min_error = min(dir_shift, key=dir_shift.get)
    # calculate the increment to be shifted
    if min_error == u"current":
        dir_inc = 0
    elif min_error == u"backward":
        dir_inc = -1 * pixtol
    elif min_error == u"forward":
        dir_inc = 1 * pixtol
    # update shift_delta
    shift_delta += dir_inc

    # keep shifting in the direction that minimizes errors.
    shift = shift_delta.copy()  # will return this value if dir_inc = 0
    if dir_inc != 0:
        shift += dir_inc
        while True:
            # shift the stack once more in the same direction
            shifted_stack = vertical_shift(
                input_array, lims, shift, max_vshift, shift_method, polyorder
            )
            nexterror = np.sum(np.abs(shifted_stack - avg_vert_fluct) ** 2)
            if nexterror < dir_shift["current"]:  # if error is minimized
                dir_shift["current"] = nexterror
                shift += dir_inc
            else:
                shift -= (
                    dir_inc
                )  # subtract once dir_inc in case of no sucess in the previous iteraction
                break
    else:
        shifted_stack = shifts["current"]
    return shift, shifted_stack


def _search_hshift_sinogram(sinogram, sinogramcomp, shiftslice, **kwargs):
    """
    Wrapper to search for the shifts in the sinogram
    """
    if isinstance(kwargs["pixtol"], int) or kwargs["subpixel"] == False:
        pixtol = 1
        shift_method = "linear"
    elif not isinstance(kwargs["pixtol"], int) or kwargs["subpixel"] == True:
        pixtol = kwargs["pixtol"]
        shift_method = kwargs["shiftmeth"]

    # get array dimensions
    nr, nc = sinogram.shape

    # intializing arrays
    sino_out = np.zeros_like(sinogram)
    shiftslice_out = np.zeros_like(shiftslice)

    for ii in range(nc):
        print("Searching the shifts for projection: {}".format(ii + 1), end="\r")
        shift_delta = shiftslice[0, ii]
        shiftslice_out[0, ii], sino_out[:, ii] = _search_hshift_direction(
            sinogram[:, ii], sinogramcomp[:, ii], shift_delta, pixtol, shift_method
        )
    print("\r")
    return sino_out, shiftslice_out


def _search_hshift_direction(
    sinogram, sinogramcomp, shift_delta, pixtol, shift_method="linear"
):
    """
    Search for sinogram shift for each projection
    """
    shifts = dict()  # dictionary shifts arrays
    dir_shift = dict()  # dictionary shifts direction

    # Initialize shift class
    S = ShiftFunc(shiftmeth=shift_method)

    # looking both ways
    # compute current shift error
    shifts["current"] = S(sinogram, shift_delta - 0)
    # compute shift forward error
    shifts["forwards"] = S(sinogram, shift_delta + pixtol)
    # compute shift backward error
    shifts["backwards"] = S(sinogram, shift_delta - pixtol)

    # directional shift error calculation
    dir_shift["current"] = np.sum(np.abs(shifts["current"] - sinogramcomp) ** 2)
    dir_shift["forward"] = np.sum(np.abs(shifts["forwards"] - sinogramcomp) ** 2)
    dir_shift["backward"] = np.sum(np.abs(shifts["backwards"] - sinogramcomp) ** 2)

    # get the smallest shift error
    min_error = min(dir_shift, key=dir_shift.get)
    # calculate the increment to be shifted
    if min_error == u"current":
        dir_inc = 0
    elif min_error == u"backward":
        dir_inc = -1 * pixtol
    elif min_error == u"forward":
        dir_inc = 1 * pixtol
    # update shift delta
    shift_delta += dir_inc

    # keep shifting in the direction that minimizes errors.
    shift = shift_delta.copy()  # will return this value if dir_inc = 0
    if dir_inc != 0:
        shift += dir_inc
        while True:
            # shift the sino according to shift
            shifted_sino = S(sinogram, shift)
            nexterror = np.sum(np.abs(shifted_sino - sinogramcomp) ** 2)
            if nexterror < dir_shift["current"]:  # if error is minimized
                dir_shift["current"] = nexterror
                shift += dir_inc  # shift the sino once more in the same direction
            else:
                shift -= (
                    dir_inc
                )  # subtract once dir_inc in case of no sucess in the previous iteraction
                # errorxreg[ii] = dir_shift['current'].copy()#currenterror
                break
    else:
        shifted_sino = shifts["current"].copy()
    return shift, shifted_sino


def _clipping_tomo(recons, **params):
    """
    Clip gray level of tomographic images
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


def _create_circle(inputimg):
    """
    Create circle with apodized edges
    """
    bordercrop = 10
    nr, nc = inputimg.shape
    Y, X = np.indices((nr, nc))
    Y -= np.round(nr / 2).astype(int)
    X -= np.round(nc / 2).astype(int)
    R = np.sqrt(X ** 2 + Y ** 2)
    Rmax = np.round(np.max(R.shape) / 2.0)
    maskout = R < Rmax
    t = maskout * (1 - np.cos(np.pi * (R - Rmax - 2 * bordercrop) / bordercrop)) / 2.0
    t[np.where(R < (Rmax - bordercrop))] = 1
    return t


def _sino_error_metric(sinogramexp, sinogramcomp, params):
    """
    Estimate the error metric between the experimental sinogram and
    the synthetic one.
    @author: jdasilva
    """
    errorxreg = np.zeros(sinogramexp.shape[1])
    for ii in range(sinogramexp.shape[1]):
        errorxreg[ii] = np.sum(np.abs(sinogramexp[:, ii] - sinogramcomp[:, ii]) ** 2)
    return errorxreg


def _checkconditions(metric_error, changes, pixtol, count, maxit, subpixel=False):
    """
    Check if the registration conditions are satisfied
    """
    if subpixel:
        step = pixtol
    else:
        step = 1

    # We then check if the error increases
    # compare the last with the before last value
    if metric_error[-1] > metric_error[-2]:
        print("Last iteration increased error.")
        print(
            "Before -> {:.04e}, current -> {:.04e}".format(
                metric_error[-2], metric_error[-1]
            )
        )
        print("Keeping previous shifts.")
        reason = 1

    # We check if the changes is larger than 1 or pixtol
    elif np.max(changes) < step:
        if step >= 1:
            print("Changes are smaller than one pixel.")
        else:
            print("Changes are smaller than {} pixel.".format(step))
        reason = 2

    # we check if the number of iteration is reached
    elif count >= maxit:
        print("Maximum number of iterations reached.")
        reason = 3
    else:
        reason = 0

    return reason


@deprecated
def cc_align(input_stack, limrow, limcol, params):
    """
    Cross-correlation alignment (DEPRECATED)
    FIXME: IT IS NOT WORKING PROPERLY
    """
    shift_values = np.empty((len(input_stack), 2))
    # The cross-correlation compares to the first projections, which does not move
    shift_values[0] = np.array([0, 0])

    for ii in range(1, len(input_stack)):
        print("\nCalculating the subpixel image registration...")
        print("Projection: {}".format(ii - 1))
        image1 = input_stack[ii - 1, limrow[0] : limrow[-1], limcol[0] : limcol[-1]]
        print("Projection: {}".format(ii))
        image2 = input_stack[ii, limrow[0] : limrow[-1], limcol[0] : limcol[-1]]
        start = time.time()
        if params["gaussian_filter"]:
            image1 = gaussian_filter(image1, params["gaussian_sigma"])
            image2 = gaussian_filter(image2, params["gaussian_sigma"])
        shift, error, diffphase = register_translation(image1, image2, 100)
        shift_values[ii] = shift
        print(diffphase)
        end = time.time()
        print("Time elapsed: {} s".format(end - start))
        print("Detected subpixel offset [y,x]: [{}, {}]".format(shift[0], shift[1]))

    shift_vert_aux = np.array(shift_values)[:, 0]
    shift_hor_aux = np.array(shift_values)[:, 1]
    # Cumulative sum of the shifts minus the average
    shift_vert = np.cumsum(shift_vert_aux - shift_vert_aux.mean())
    shift_hor = np.cumsum(shift_hor_aux - shift_hor_aux.mean())

    # smoothing the shifts is needed
    if params["smooth_shifts"] is not None:
        shift_vert = snf.gaussian_filter1d(shift_vert, params["smooth_shifts"])
        shift_hor = snf.gaussian_filter1d(shift_hor, params["smooth_shifts"])

    # display shifts
    plt.close("all")
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(211)
    ax1.plot(np.array(shift_vert), "ro-")
    ax1.set_title("Vertical shifts")
    ax2 = fig1.add_subplot(212)
    ax2.plot(np.array(shift_hor), "ro-")
    ax2.set_title("Horizontal shifts")
    plt.show()

    # updating the shiftstack
    shiftstack = np.zeros((2, input_stack.shape[0]))
    shiftstack[0] = shift_vert
    shiftstack[1] = shift_hor

    # Compute the shifted images
    # print('Computing aligned images')
    # if not params['expshift']:
    # output_stack = compute_aligned_stack(input_stack,shiftstack.copy(),params)
    # else:
    # print('Computing aligned images in phase space')
    # output_stack = np.angle(compute_aligned_stack(np.exp(1j*input_stack),shiftstack.copy(),params))

    # return shiftstack,output_stack

    plt.close("all")
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)  # (ncols=1, figsize=(14, 6))
    im1 = ax1.imshow(
        stack_unwrap[1, limrow[0] : limrow[-1], limcol[0] : limcol[-1]],
        interpolation="none",
        cmap="bone",
    )
    ax1.set_axis_off()
    ax1.set_title("Offset corrected image2")

    # offset_stack_unwrap = np.empty_like(stack_unwrap[:,80:-80,80:-80])
    # aligned = np.empty_like(stack_unwrap[:,80:-80,80:-80])
    aligned = compute_aligned_stack(
        input_stack, shiftstack.copy(), shift_method=params["shiftmeth"]
    )
    plt.ion()
    for ii in range(0, len(stack_unwrap)):
        # img = stack_unwrap[ii,80:-80,80:-80]
        shift = np.array([shift_vert[ii], shift_hor[ii]])
        print(shift)
        print(
            "\nCorrecting the shift of projection {} by using subpixel precision.".format(
                ii
            )
        )
        # offset_stack_unwrap[ii] = np.fft.ifftn(fourier_shift(np.fft.fftn(img),shift))#
        # aligned[ii] = np.fft.ifftn(fourier_shift(np.fft.fftn(img),shift))#
        # im1.set_data(offset_stack_unwrap[ii])
        im1.set_data(aligned[ii])
        ax1.set_title(u"Projection {}".format(ii))
        fig1.canvas.draw()
        plt.pause(0.001)
    plt.ioff()

    # Display the images
    fig, (ax1, ax2, ax3) = plt.subplots(num=3, ncols=3, figsize=(14, 6))
    ax1.imshow(image1, interpolation="none", cmap="bone")
    ax1.set_axis_off()
    ax1.set_title("Image 1 (ref.)")
    ax2.imshow(image2, interpolation="none", cmap="bone")
    ax2.set_axis_off()
    ax2.set_title("Image 2")

    # View the output of a cross-correlation to show what the algorithm is
    # doing behind the scenes
    image_product = np.fft.fft2(image1) * np.fft.fft2(image2).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ax3.imshow(cc_image.real)
    # ax3.set_axis_off()
    ax3.set_title("Cross-correlation")

    plt.show(block=False)
    return shiftstack, aligned
