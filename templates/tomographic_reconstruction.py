#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template for tomographic reconstruction from the phase projections
"""

# standard packages
import time

# local packages
from toupy.io import LoadData, SaveTomogram
from toupy.utils import display_slice, iterative_show
from toupy.tomo import full_tomo_recons


# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["slicenum"] = 550  # Choose the slice
params["filtertype"] = "hann"  # Filter to use for FBP
params["freqcutoff"] = 1.0  # Frequency cutoff (between 0 and 1)
params["circle"] = True
params["algorithm"] = "FBP"  # FBP or SART
params["derivatives"] = True  # only for FBP
params["calc_derivatives"] = False  # Calculate derivatives if not done
params["opencl"] = True
params["autosave"] = False
params["vmin_plot"] = None  # 0.5e-5
params["vmax_plot"] = -1e-4  # None
params["colormap"] = "bone"
params["showrecons"] = False  # to display the reconstructed slice on the fly.
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#

if __name__ == "__main__":

    # loading the data
    aligned_projections, theta, shiftstack, params = LoadData.load(
        "aligned_projections.h5", **params
    )

    tomogram = full_tomo_recons(aligned_projections, theta, **params)

    a = input("Do you want to display the tomographic slices? (y/[n]) :").lower()
    if str(a) == "y":
        iterative_show(
            tomogram,
            vmin=params["vmin_plot"],
            vmax=params["vmax_plot"],
            colormap=params["colormap"],
        )  # Show aligned projections derivatives

    # save the tomograms
    SaveTomogram.save("tomogram.h5", tomogram, theta, shiftstack, **params)

    # next step
    print('You should run "tiff_convertion.py" now')
    # =============================================================================#
