#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standar packages
import time

# third party package
import matplotlib.pyplot as plt
import numpy as np

# local packages
from ..utils import progbar, display_slice
from .iradon import backprojector, reconsSART
from ..restoration import derivatives_sino

__all__ = ["full_tomo_recons", "tomo_recons"]


def tomo_recons(sinogram, theta, **params):
    """
    Wrapper to select tomographic algorithm
    
    Parameters
    ----------
    sinogram : array_like
        Array containing the sinogram
    theta : array_like
        Array containing the theta
    params : dict
        Dictionary containing additional parameters
    params["algorithm"] : str
        Choice of algorithm. Two algorithm implemented: "FBP" and "SART"
    params["slicenum"] : int
        Slice number
    params["filtertype"] : str
        Filter to use for FBP
    params["freqcutoff"] : float
        Frequency cutoff (between 0 and 1)
    params["circle"] : bool
        Multiply the reconstructed slice by a circle to remove borders
    params["derivatives"] : bool
        If the projections are derivatives. Only for FBP.
    params["calc_derivatives"] : bool
        Calculate derivatives of the sinogram if not done yet.
    params["opencl"] : bool
        Implement the tomographic reconstruction in opencl as implemented
        in Silx
    params["autosave"] : bool
        Save the data at the end without asking
    params["vmin_plot"] : float
        Minimum value for the gray level at each display
    params["vmax_plot"] : float
        Maximum value for the gray level at each display
    params["colormap"] : str
        Colormap
    params["showrecons"] : bool
        If to show the reconstructed slices

    Return
    ------
    recons : array_like
        Reconstructed slice
    """
    if params["algorithm"] == "FBP":
        if params["calc_derivatives"]:
            print("Calculating the derivatives of the sinogram")
            sinogram = derivatives_sino(sinogram, shift_method="fourier")
        recons = backprojector(sinogram, theta, **params)
    elif params["algorithm"] == "SART":
        if params["calc_derivatives"]:
            raise ValueError("Reconstruction from derivatives only works with FBP")
        recons = reconsSART(sinogram, theta, **params)
    return recons


def full_tomo_recons(input_stack, theta, **params):
    """
    Full tomographic reconstruction

    Parameters
    ----------
    input_stack : array_like
        Stack of projections
    theta : array_like
        Array containing the theta
    params : dict
        Dictionary containing additional parameters
    params["algorithm"] : str
        Choice of algorithm. Two algorithm implemented: "FBP" and "SART"
    params["slicenum"] : int
        Slice number
    params["filtertype"] : str
        Filter to use for FBP
    params["freqcutoff"] : float
        Frequency cutoff (between 0 and 1)
    params["circle"] : bool
        Multiply the reconstructed slice by a circle to remove borders
    params["derivatives"] : bool
        If the projections are derivatives. Only for FBP.
    params["calc_derivatives"] : bool
        Calculate derivatives of the sinogram if not done yet.
    params["opencl"] : bool
        Implement the tomographic reconstruction in opencl as implemented
        in Silx
    params["autosave"] : bool
        Save the data at the end without asking
    params["vmin_plot"] : float
        Minimum value for the gray level at each display
    params["vmax_plot"] : float
        Maximum value for the gray level at each display
    params["colormap"] : str
        Colormap
    params["showrecons"] : bool
        If to show the reconstructed slices


    Return
    ------
    Tomogram : array_like
        Full reconstructed tomogram
    """

    print("Calculating a slice for display")
    slicenum = params["slicenum"]

    sinogram0 = np.transpose(input_stack[:, slicenum, :])

    # calculating one slice for estimating sizes
    t0 = time.time()
    tomogram0 = tomo_recons(sinogram0, theta, **params)
    nr, nc = tomogram0.shape  # size of the slices
    print("Calculation done. Time elapsed: {:.02f} s".format(time.time() - t0))

    # display one slice
    plt.close("all")
    display_slice(
        tomogram0,
        colormap=params["colormap"],
        vmin=params["cliplow"],
        vmax=params["cliphigh"],
    )

    # estimate the total number of slices
    nslices = input_stack.shape[1]
    print("The total number of slices is {}".format(nslices))

    # actual wrapper for the reconstruction
    a = input("Do you want to start the full reconstruction? ([y]/n): ").lower()
    if str(a) == "" or str(a) == "y":
        plt.close("all")
        tomogram = np.zeros((nslices, nr, nc), dtype=np.float32)
        for ii in range(nslices):  # num_projections):#sorted(frames):
            strbar = "Slice: {} out of {}".format(ii + 1, nslices)
            sinogram = np.transpose(input_stack[:, ii, :])
            tomogram[ii] = tomo_recons(sinogram, theta, **params)
            progbar(ii + 1, nslices, strbar)
        print("\r")
    elif str(a) == "n":
        raise ValueError("The full tomographic reconstrucion has not been done.")

    return tomogram
