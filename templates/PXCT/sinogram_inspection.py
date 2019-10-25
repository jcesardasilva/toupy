#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template for inspecting the rotation axis

This routine will help you to estimate a first guess for the position of
the rotation axis. This will be useful to speed up the horizontal
alignment.
"""

# local packages
from toupy.io import LoadData, SaveData
from toupy.registration import estimate_rot_axis

# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["slicenum"] = 550  # Choose the slice
params["filtertype"] = "hann"  # Filter to use for FBP
params["freqcutoff"] = 0.9  # Frequency cutoff
params["circle"] = True
params["algorithm"] = "FBP"
# initial guess of the offset of the axis of rotation
params["rot_axis_offset"] = 0
params["cliplow"] = None  # clip on low threshold
params["cliphigh"] = -1e-4  # clip on high threshold
params["sinohigh"] = None  # -0.1
params["sinolow"] = None  # 0.1
params["sinocmap"] = "bone"
params["colormap"] = "bone"
params["derivatives"] = True
params["calc_derivatives"] = False  # Calculate derivatives if not done
params["opencl"] = True
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # loading the data
    aligned_diff, theta, shiftstack, params = LoadData.load(
        "aligned_derivatives.h5", **params
    )

    estimate_rot_axis(aligned_diff, theta, **params)

    # next step
    print('You should run "horizontal_alignment.py" now')
    # =============================================================================#
