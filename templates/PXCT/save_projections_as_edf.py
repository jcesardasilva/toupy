#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Template to save the projections in edf files
"""

# standard libraries imports
import os
import time

# local packages
from toupy.io import LoadData, SaveData, write_edf
from toupy.registration import compute_aligned_stack
from toupy.utils import sort_array, replace_bad, progbar

# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["unwrappedphases"] = True
params["apply_alignment"] = True
params["shiftmeth"] = "linear"  # 'fourier' or 'linear'
params["sort_theta"] = True
params["tomo_type"] = "delta"
params["correct_bad"] = False
params["bad_projs"] = []  # starting at zero
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":

    shifts_file = "aligned_projections.h5"
    foldername = params["samplename"]  #'projs'
    pathfile = foldername + "/{}_{:04d}.edf"
    if params["tomo_type"] == "delta" and params["unwrappedphases"]:
        load_filename = "unwrapped_phases.h5"
    elif params["tomo_type"] == "delta" and not params["unwrappedphases"]:
        load_filename = "reconstructed_projections.h5"
    elif params["tomo_type"] == "beta":
        load_filename = "air_corrected_amplitude.h5"
        foldername = params["samplename"] + "_amp"  #'projs_amp'
        pathfile = foldername + "/{}_amp_{:04d}.edf"

    # load the projections before alignment and derivatives
    output_stack, theta, shiftstack, params = LoadData.load(load_file, **params)

    # size of the array
    nprojs, nr, nc = output_stack.shape
    if nprojs != len(theta):
        raise ValueError(
            "The number of projections is different from the number of angles"
        )
    print("The total number of projections is {}".format(nprojs))

    # Apply alignment if needed
    if params["apply_alignment"]:
        # load shiftstack from aligned phase projections
        shiftstack = LoadData.loadshiftstack(shifts_file, **params)
        print("The shiftstack length is {}".format(shiftstack.shape[1]))
        if shiftstack.shape[1] != output_stack.shape[0]:
            raise ValueError(
                "The array with shifts is not compatible with the number of projections"
            )
        print("Computing aligned images")
        output_stack = compute_aligned_stack(
            output_stack, shiftstack, shift_method=params["shiftmeth"]
        )

    # correcting bad projections after the alignment if needed
    if params["correct_bad"]:
        output_stack = replace_bad(
            output_stack, list_bad=params["bad_projs"], temporary=False
        )

    # to start at zero
    theta -= theta.min()
    if params["sort_theta"]:
        # sort theta
        output_stack, theta = sort_array(output_stack, theta)

    # check if directory already exists
    if not os.path.isdir(foldername):
        print("Creating the directory {}".format(foldername))
        os.makedirs(foldername)
    else:
        print("Directory {} already exists and will be used".format(foldername))

    # writing the edf files
    print("Saving projections as edf files.")
    for ii in range(nprojs):
        strbar = "Projection: {} out of {}".format(ii, nprojs)
        fname = pathfile.format(params["samplename"], ii)
        write_edf(fname, output_stack_sorted[ii], hd=params)
        progbar(ii + 1, nprojs, strbar)
        print("Time elapsed: {:4.02f} s\n".format(pf))
