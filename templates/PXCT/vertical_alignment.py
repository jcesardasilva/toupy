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
from toupy.registration import alignprojections_vertical
from toupy.utils import iterative_show, replace_bad

# initializing dictionaries
params = dict()

# Register (align) projections by vertical mass fluctuations and center of mass
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["phaseonly"] = True
params["pixtol"] = 0.1  # Tolerance of registration in pixels
params["polyorder"] = 2  # Polynomial order to remove bias
params["shiftmeth"] = "linear"
params["alignx"] = False  # Align horizontally with center of mass
params["maxit"] = 10  # max of iterations
params["deltax"] = 20  # From edge of region to edge of image in x
params["limsy"] = (190, 800)
params["autosave"] = False
params["load_previous_shiftstack"] = False  # True
params["correct_bad"] = True
params["bad_projs"] = [156, 226, 363, 371, 673, 990]  # starting at zero
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # loading the data
    stack_unwrap, theta, shiftstack, params = LoadData.load(
        "unwrapped_phases.h5", **params
    )

    # initializing shiftstack
    if params["load_previous_shiftstack"]:
        shiftstack = LoadData.loadshiftstack("vertical_alignment.h5", **params)
        print("Using previous estimate of shiftstack")
    else:
        # initializing shiftstack with zeros
        shiftstack[0] = np.zeros(stack_unwrap.shape[0])

    # Vertical alignment
    shiftstack, aligned = alignprojections_vertical(stack_unwrap, shiftstack, **params)

    a = input("Do you want to refine further the alignment? (y/[n]): ").lower()
    if str(a) == "y":
        shiftstack, aligned = alignprojections_vertical(
            stack_unwrap, shiftstack, **params
        )

    # correcting bad projections after unwrapping
    if params["correct_bad"]:
        aligned = replace_bad(aligned, list_bad=params["bad_projs"], temporary=False)

    a = input("Do you want to display the aligned projections? (y/[n]) :").lower()
    if str(a) == "y":
        iterative_show(aligned)  # Show aligned projections

    # save vertically aligned_projections
    SaveData.save("vertical_alignment.h5", aligned, theta, shiftstack, **params)
    # next step
    print('You should run "projections_derivatives.py" now')
    # =============================================================================#
