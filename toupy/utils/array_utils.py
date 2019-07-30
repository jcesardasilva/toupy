#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np
import scipy.constants as consts
from scipy.ndimage import filters

__all__ = [
    "crop",
    "fract_hanning",
    "fract_hanning_pad",
    "mask_borders",
    "mask_borders",
    "padarray_bothsides",
    "polynomial1d",
    "projectpoly1d",
    "radtap",
    "replace_bad",
    "round_to_even",
    "sharpening_image",
    "smooth_image",
    "sort_array",
]

def smooth_image(input_image, filter_size=3):
    """
    Smooth image with a median filter

    Parameters
    ----------
    input_image : nadarray
        Image to be smoothed
    filter_size : int
        Size of the filter
    Returns
    -------
    ndarray
        Smoothed image
    """
    return filters.median_filter(input_image,filter_size)

def sharpening_image(input_image, filter_size=3, alpha=30):
    """
    Sharpen image with a median filter

    Parameters
    ----------
    input_image : nadarray
        Image to be sharpened
    filter_size : int
        Size of the filter
    alpha : float
        Strength of the sharpening
    Returns
    -------
    ndarray
        Sharpened image
    """
    blurredimg = filters.median_filter(input_image, filter_size)
    filter_blurredimg = filters.median_filter(blurredimg, 1)
    sharpimg = blurredimg + alpha * (blurredimg - filter_blurredimg)
    return sharpimg


def sort_array(input_array, ref_array):
    """
    Sort array based on another array
    
    Parameters
    ----------
    input_array : ndarray
        Array to be sorted
    ref_array : ndarray
        Array on which the sorting will be based

    Returns
    -------
    sorted_input_array : ndarray
        Sorted input array
    sorted_ref_array : ndarray
        Sorted reference array
    """
    idxsort = np.argsort(ref_array)
    sorted_ref_array = ref_array[idxsort]
    sorted_input_array = input_array[idxsort]

    return sorted_input_array, sorted_ref_array


def replace_bad(input_stack, list_bad=[], temporary=False):
    """
    correcting bad projections before unwrapping
    """
    if list_bad == []:
        raise ValueError("List of bad projections is empty")
    else:
        for ii in list_bad:
            print("\rTemporary replacement of bad projection: {}".format(ii), end="")
            if temporary:
                input_stack[ii] = input_stack[ii - 1]
            else:
                input_stack[ii] = (input_stack[ii - 1] + input_stack[ii + 1]) / 2
        print('\r')
    return input_stack


def round_to_even(x):
    return int(2 * np.floor(x / 2))


def polynomial1d(x, order=1, w=1):
    """
    Generates a 1D orthonormal polynomial base.
    Inspired by legendrepoly1D_2.m created by Manuel Guizar in March 10,2009

    Parameters
    ----------
    x : ndarray
        Array containing the values of x for the polynomial
    order : int
        Order of the polynomial
    w : int
        Weights of the coefficients

    Returns
    -------
    polyseries : ndarray
        Orthonormal polymonial up to order
    """

    polyseries = []
    for ii in range(order + 1):
        polyseries.append(np.power(x, ii)[0])
    # Convenient convertion to numpy array
    polyseries = np.asarray(polyseries).astype(float)
    # Normalization
    for ii in range(len(polyseries)):
        polyseries[ii] /= np.sqrt(np.sum(w * np.abs(polyseries[ii]) ** 2))
    # Orthonormalization
    for ii in range(1, len(polyseries)):
        for jj in range(0, ii):
            polyseries[ii] -= (
                np.sum(polyseries[ii] * polyseries[jj] * w) * polyseries[jj]
            )
        # Re-normalization
        polyseries[ii] /= np.sqrt(np.sum(w * np.abs(polyseries[ii]) ** 2))
    return polyseries


def projectpoly1d(func1d, order=1, w=1):
    """
    Projects a 1D function onto orthonormalized base
    Inspired by projectleg1D_2.m created by Manuel Guizar in March 10,2009

    Parameters
    ----------
    func1d : ndarray
        Array containing the values of the 1D function
    order : int (default=1)
        Order of the polynomial
    w : int (default=1)
        Weights of the coefficients

    Returns
    -------
    projfunc1d : ndarray
        Projected 1D funtion on orthonormal base
    """
    x = np.indices(func1d.shape)
    x -= np.ceil(x.mean()).astype("int")
    polyseries = polynomial1d(x, order, w)
    # needs to be float for the subtraction below
    projfunc1d = func1d.astype("float").copy()
    for ii in range(len(polyseries)):
        coeff = np.sum(func1d * polyseries[ii] * w)
        projfunc1d -= polyseries[ii] * coeff  # all array needs to be float
    return projfunc1d


def crop(input_array, delcropx, delcropy):
    """
    Crop images
    Inputs:
        input_array: input image to be cropped
        **params: dict of parameters
    @author: Julio C. da Silva (jdasilva@esrf.fr)
    """
    if delcropx is not None or delcropy is not None:
        print("Cropping ROI of data")
        print("Before: " + input_array.shape)
        print(input_array[delcropy:-delcropy, delcropx:-delcropx].shape)
        if input_array.ndim == 2:
            return input_array[delcropy:-delcropy, delcropx:-delcropx]
        elif input_array.ndim == 3:
            return input_array[:, delcropy:-delcropy, delcropx:-delcropx]
        print("After: " + input_array.shape)
    else:
        print("No cropping of data")
        return input_array


def radtap(X, Y, tappix, zerorad):
    """
    Creates a central cosine tapering for beam.
    It receives the X and Y coordinates, tappix is the extent of
    tapering, zerorad is the radius with no data (zeros).
    @author: jdasilva
    """
    tau = 2 * tappix  # period of cosine function (only half a period is used)

    R = np.sqrt(X ** 2 + Y ** 2)
    taperfunc = 0.5 * (1 + np.cos(2 * np.pi * (R - zerorad - tau / 2.0) / tau))
    taperfunc = (R > zerorad + tau / 2.0) * 1.0 + taperfunc * (R <= zerorad + tau / 2)
    taperfunc = taperfunc * (R >= zerorad)
    return taperfunc


def fract_hanning(outputdim, unmodsize):
    """
    fract_hanning(outputdim,unmodsize)
    out = Square array containing a fractional separable Hanning window with
    DC in upper left corner.
    outputdim = size of the output array
    unmodsize = Size of the central array containing no modulation.
    Creates a square hanning window if unmodsize = 0 (or ommited), otherwise the output array
    will contain an array of ones in the center and cosine modulation on the
    edges, the array of ones will have DC in upper left corner.
    @author: jdasilva
    """
    if outputdim < unmodsize:
        raise SystemExit(
            "Output dimension must be smaller or equal to size of unmodulated window"
        )

    if unmodsize < 0:
        unmodsize = 0
        print("Specified unmodsize<0, setting unmodsize = 0")

    N = np.arange(0, outputdim)
    Nc, Nr = np.meshgrid(N, N)
    if unmodsize == 0:
        out = (
            (1.0 + np.cos(2 * np.pi * Nc / outputdim))
            * (1.0 + np.cos(2 * np.pi * Nr / outputdim))
            / 4.0
        )
    else:
        # columns modulation
        outc = (
            1.0
            + np.cos(
                2
                * np.pi
                * (Nc - np.floor((unmodsize - 1) / 2))
                / (outputdim + 1 - unmodsize)
            )
        ) / 2.0
        if np.floor((unmodsize - 1) / 2.0) > 0:
            outc[:, : int(np.floor((unmodsize - 1) / 2.0))] = 1
        outc[
            :, int(np.floor((unmodsize - 1) / 2) + outputdim + 3 - unmodsize) : len(N)
        ] = 1
        # row modulation
        outr = (
            1.0
            + np.cos(
                2
                * np.pi
                * (Nr - np.floor((unmodsize - 1) / 2))
                / (outputdim + 1 - unmodsize)
            )
        ) / 2.0
        if np.floor((unmodsize - 1) / 2.0) > 0:
            outr[: int(np.floor((unmodsize - 1) / 2.0)), :] = 1
        outr[
            int(np.floor((unmodsize - 1) / 2) + outputdim + 3 - unmodsize) : len(N), :
        ] = 1

        out = outc * outr

    return out


def fract_hanning_pad(outputdim, filterdim, unmodsize):
    """
    fract_hanning_pad(outputdim,filterdim,unmodsize)
    out = Square array containing a fractional separable Hanning window with
    DC in upper left corner.
    outputdim = size of the output array
    filterdim = size of filter (it will zero pad if filterdim<outputdim
    unmodsize = Size of the central array containing no modulation.
    Creates a square hanning window if unmodsize = 0 (or ommited), otherwise the output array
    will contain an array of ones in the center and cosine modulation on the
    edges, the array of ones will have DC in upper left corner.
    @author: jdasilva
    """
    if outputdim < unmodsize:
        raise SystemExit(
            "Output dimension must be smaller or equal to size of unmodulated window"
        )
    if outputdim < filterdim:
        raise SystemExit("Filter cannot be larger than output size")
    if unmodsize < 0:
        unmodsize = 0
        print("Specified unmodsize<0, setting unmodsize = 0")

    out = np.zeros((outputdim, outputdim))
    auxindini = int(np.round(outputdim / 2 - filterdim / 2))
    auxindend = int(np.round(outputdim / 2 + filterdim / 2))
    out[auxindini:auxindend, auxindini:auxindend] = np.fft.fftshift(
        fract_hanning(filterdim, unmodsize)
    )
    return np.fft.fftshift(out)


def mask_borders(imgarray, mask_array, threshold=4e-7):
    # mask borders
    gr, gc = np.gradient(imgarray)
    mask_border = np.sqrt(gr ** 2 + gc ** 2) > threshold
    mask_array *= ~mask_border
    return mask_array


def padarray_bothsides(input_array, newshape, padmode="edge"):
    """
    Pad array in both sides
    """
    nro, nco = input_array.shape
    nrn, ncn = newshape

    if np.abs(nrn - nro) % 2 == 0:
        padr = (int((nrn - nro) / 2),) * 2
    else:
        padr = (int((nrn - nro) / 2), int((nrn - nro) / 2) + 1)

    if np.abs(ncn - nco) % 2 == 0:
        padc = (int((ncn - nco) / 2),) * 2
    else:
        padc = (int((ncn - nco) / 2), int((ncn - nco) / 2) + 1)

    return np.pad(input_array, (padr, padc), mode=padmode)
