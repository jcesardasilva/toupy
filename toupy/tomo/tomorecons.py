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
    """
    if params["algorithm"] == "FBP":
        if params["calc_derivatives"]:
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
    """
    try:
        calc_derivatives = params["calc_derivatives"]
        print("Calculating the derivatives of the sinogram")
    except KeyError:
        calc_derivatives = False

    print("Calculating a slice for display")
    slicenum = params["slicenum"]

    sinogram0 = np.transpose(input_stack[:, slicenum, :])

    # calculating one slice for estimating sizes
    t0 = time.time()
    tomogram0 = tomo_recons(sinogram0, theta, **params)
    nr, nc = tomogram0.shape  # size of the slices
    print("Calculation done. Time elapsed: {:.02f} s".format(time.time() - t0))

    # display one slice
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
        tomogram = np.zeros((nslices, nr, nc))
        for ii in range(nslices):  # num_projections):#sorted(frames):
            strbar = "Slice: {} out of {}".format(ii + 1, nslices)
            sinogram = np.transpose(input_stack[:, ii, :])
            tomogram[ii] = tomo_recons(sinogram, theta, **params)
            progbar(ii + 1, nslices, strbar)
        print("\r")
    elif str(a) == "n":
        raise ValueError("The full tomographic reconstrucion has not been done.")

    return tomogram
