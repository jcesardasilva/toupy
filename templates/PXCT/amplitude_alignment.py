#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Template for the alignment of the amplitude projections based on the
results of the alignment of the phase projections
"""

# local packages
from toupy.io import LoadData, SaveData
from toupy.utils import replace_bad, iterative_show
from toupy.registration import compute_aligned_stack

# initializing dictionaries
params = dict()

# Register (align) projections by vertical mass fluctuations and center of mass
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["deltax"] = 2  # From edge of region to edge of image in x
params["limsy"] = (1, 2329)  # (top, bottom)
params["shift_method"] = "fourier"
params["correct_bad"] = True
params["bad_projs"] = [156, 226, 363, 371, 673, 990]  # starting at zero
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # loading the data
    shiftstack = LoadData.loadshiftstack("aligned_projections.h5", **params)

    # loading the data
    stack_amp_corr, theta, shiftstack, params = LoadData.load(
        "air_corrected_amplitude.h5", **params
    )

    # checking array sizes
    if stack_amp_corr.shape[0] != shiftstack.shape[1]:
        raise ValueError(
            "The size of stack ({}) and deltastack ({}) are different.".format(
                stack_amp_corr.shape[0], shiftstack.shape[1]
            )
        )

    # aligning the projections
    aligned_amp_projections = compute_aligned_stack(stack_amp_corr, deltastack, params)
    previous_shape = aligned_amp_projections.shape

    # cropping projection
    print("Cropping projections to match phase projection size")
    print("Before: {} x {}".format(previous_shape[1], previous_shape[2]))
    # horizontal ROI
    deltax = params["deltax"]
    roix = range(deltax, stack_array.shape[2] - deltax)  # update roix
    # vertical ROI
    roiy = params["limsy"]
    aligned_amp_projections = aligned_amp_projections[
        :, roiy[0] : roiy[-1], roix[0] : roix[-1]
    ]
    new_shape = aligned_amp_projections.shape
    print("New: {} x {}".format(new_shape[1], new_shape[2]))

    # correcting bad projections after the alignment if needed
    if params["correct_bad"]:
        aligned_amp_projections = replace_bad(
            aligned_amp_projections, list_bad=params["bad_projs"], temporary=False
        )

    a = input("Do you want to display the aligned projections? (y/[n]) :").lower()
    if str(a) == "y":
        iterative_show(
            aligned_amp_projections, vmin=None, vmax=None
        )  # Show aligned projections derivatives

    # save horizontally aligned_projections
    SaveData.save(
        "aligned_amp_projections.h5",
        aligned_amp_projections,
        theta,
        shiftstack,
        **params
    )
    # next step
    print('You should run "tomographic_reconstruction_amp.py" now')
    # =============================================================================#
