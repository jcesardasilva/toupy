#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template for the horizontal alignment of the projections based on the
tomographic consistency conditions.

You will first align for one slice and we repeat the alignment at
multiple slices and average the shift function. If the average is
satisfactory, you can proceed and save the data. 
"""

# standard libraries imports
import re
import socket

### quick fix to avoid ImportError: dlopen: cannot load any more object with static TLS
### not used when not using GPUs
if re.search("gpu", socket.gethostname()) or re.search("gpid16a", socket.gethostname()):
    import pyfftw  # has to be imported first
###

# third party package
import numpy as np

# local packages
from toupy.io import LoadData, SaveData
from toupy.utils import sort_array, replace_bad, iterative_show
from toupy.registration import (
    alignprojections_horizontal,
    compute_aligned_horizontal,
    oneslicefordisplay,
    refine_horizontalalignment,
    tomoconsistency_multiple,
)

# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["slicenum"] = 550  # Choose the slice
params["filtertype"] = "hann"  # Filter to use for FBP
params["freqcutoff"] = 0.2  # Frequency cutoff (between 0 and 1)
params["circle"] = True
params["algorithm"] = "FBP"
# initial guess of the offset of the axis of rotation
params["rot_axis_offset"] = 0
params["pixtol"] = 0.01  # Tolerance of registration in pixels
params["shiftmeth"] = "fourier"  # 'sinc' or 'linear' better for noise
params["maxit"] = 20  # max of iterations
params["cliplow"] = None  # clip air threshold
params["cliphigh"] = -1e-4  # clip on sample threshold
params["colormap"] = "bone"
params["sinohigh"] = None
params["sinolow"] = None
params["derivatives"] = True
params["calc_derivatives"] = False  # Calculate derivatives if not done
params["opencl"] = True
params["autosave"] = False
params["load_previous_shiftstack"] = False
params["correct_bad"] = True
params["bad_projs"] = [156, 226, 363, 371, 673, 990]  # starting at zero
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # loading the data
    aligned_diff, theta, shiftstack, params = LoadData.load(
        "aligned_derivatives.h5", **params
    )

    # to start at zero
    theta -= theta.min()

    # if you want to sort theta, uncomment line below:
    # aligned_diff, theta = sort_array(aligned_diff, theta)

    # If you want to initialize shiftstack with previous alignment values
    if params["load_previous_shiftstack"]:
        shiftstack = LoadData.loadshiftstack("aligned_projections.h5", **params)
        print("Using previous estimate of shiftstack")
    else:
        # initializing shiftstack with zero plus rot_axis_offset
        shiftstack[1] = np.zeros(aligned_diff.shape[0]) + params["rot_axis_offset"]

    # calculate the sinogram needed for the alignment
    sinogram = np.transpose(aligned_diff[:, params["slicenum"], :]).copy()

    # actual alignement
    shiftstack = alignprojections_horizontal(sinogram, theta, shiftstack, **params)

    # alignment refinement with different parameters if necessary
    shiftstack, params = refine_horizontalalignment(
        aligned_diff, theta, shiftstack, **params
    )

    # tomographic consistency on multiples slices
    a = input(
        "Do you want to perform Tomographic consistency on multiples slices? (y/[n]):"
    ).lower()

    if str(a) == "y":
        shiftstack = tomoconsistency_multiple(aligned_diff, theta, shiftstack, **params)
    else:
        print("Tomo consistency on multiples slices not done")

    # Shift projections (Only shift the horizontal)
    print("Shifting the projections (only shifts in the horizontal direction)")
    aligned_projections = compute_aligned_horizontal(
        aligned_diff, shiftstack, shift_method=params["shiftmeth"]
    )
    # correcting bad projections after the alignment if needed
    if params["correct_bad"]:
        aligned_projections = replace_bad(
            aligned_projections, list_bad=params["bad_projs"], temporary=False
        )

    a = input("Do you want to display the aligned projections? (y/[n]) :").lower()
    if str(a) == "y":
        iterative_show(
            aligned_projections, vmin=-0.2, vmax=0.2
        )  # Show aligned projections derivatives

    # calculate one slice for display
    aligned_sinogram = np.transpose(aligned_projections[:, params["slicenum"], :])
    oneslicefordisplay(aligned_sinogram, theta, **params)

    # save horizontally aligned_projections
    SaveData.save(
        "aligned_projections.h5", aligned_projections, theta, shiftstack, **params
    )
    # next step
    print('You should run "tomographic_reconstruction.py" now')
    # =============================================================================#
