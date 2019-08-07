#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Template for the convertion of the tomographic volume to tiff files
"""

# local packages
from toupy.io import LoadTomogram, SaveTomogram

# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["tomo_type"] = "delta"  #'delta' or 'beta'
params["bits"] = 16  # 8 or 16 bits
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#

if __name__ == "__main__":

    # loading files
    if params["tomo_type"] == "delta":
        tomogram, theta, shiftstack, params = LoadTomogram.load("tomogram.h5", **params)
    elif params["tomo_type"] == "beta":
        tomogram, theta, shiftstack, params = LoadTomogram.load(
            "tomogram_amp.h5", **params
        )
    else:
        raise ValueError("Unrecognized tomography type")

    # Write the files
    SaveTomogram.convert_to_tiff(tomogram, **params)
