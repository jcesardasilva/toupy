#!/usr/bin/env python
# -*- coding: utf-8 -*-

# local packages
from toupy.io import LoadData, SaveData
from toupy.registration import estimate_rot_axis

# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["phaseonly"] = True
params["slicenum"] = 550  # Choose the slice
params["filtertype"] = "hann"  # Filter to use for FBP
params["filtertomo"] = 0.9  # Frequency cutoff
params["circle"] = True
# initial guess of the offset of the axis of rotation
params["rot_axis_offset"] = 0
params["cliplow"] = None  # clip on low threshold
params["cliphigh"] = -1e-4  # clip on high threshold
params["colormap"] = "bone"
params["sinohigh"] = None  # -0.1
params["sinolow"] = None  # 0.1
params["sinocmap"] = "bone"
params["derivatives"] = True
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
