#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template to vertical alignment of the projections based on the
vertical mass fluctuations and tomographic consistency.
"""

# third party packages
import numpy as np

# local packages
from toupy.io import LoadData, SaveData
from toupy.registration import alignprojections_vertical, compute_aligned_stack
from toupy.utils import iterative_show, replace_bad

# initializing dictionaries
params = dict()

# Register (align) projections by vertical mass fluctuations and center of mass
# =========================
params["samplename"] = "H2int_15000h_outlet"
params["regime"] = "holoct"
params["pixtol"] = 0.1  # Tolerance of registration in pixels
params["polyorder"] = 2  # Polynomial order to remove bias
params["shiftmeth"] = "linear"
params["alignx"] = False  # Align horizontally with center of mass
params["maxit"] = 20  # max of iterations
params["deltax"] = 20  # From edge of region to edge of image in x
params["limsy"] = (190, 800)
params["autosave"] = True
params["load_previous_shiftstack"] = False  # True
# =========================

# =============================================================================#
# Don't edit below this line, please                                           #
# =============================================================================#
if __name__ == "__main__":
    # loading the data
    stack_unwrap, theta, shiftstack, params = LoadData.load(
        "reconstructed_projections.h5", **params
    )

    # initializing shiftstack
    if params["load_previous_shiftstack"]:
        shiftstack = LoadData.loadshiftstack("vertical_alignment.h5", **params)
        print("Using previous estimate of shiftstack")
    else:
        # initializing shiftstack with zeros
        shiftstack[0] = np.zeros(stack_unwrap.shape[0])

    # Vertical alignment
    shiftstack = alignprojections_vertical(stack_unwrap, shiftstack, **params)

    a = input("Do you want to refine further the alignment? (y/[n]): ").lower()
    if str(a) == "y":
        shiftstack = alignprojections_vertical(stack_unwrap, shiftstack, **params)

    # computing the aligned images (separate to avoid memory issues)
    print("Computing aligned images")
    aligned = compute_aligned_stack(
        stack_unwrap, shiftstack, shift_method=params["shiftmeth"]
    )

    a = input("Do you want to display the aligned projections? (y/[n]) :").lower()
    if str(a) == "y":
        iterative_show(aligned)  # Show aligned projections

    # save vertically aligned_projections
    SaveData.save("vertical_alignment.h5", aligned, theta, shiftstack, **params)
    # next step
    print('You should run "projections_derivatives.py" now')
    # =============================================================================#
