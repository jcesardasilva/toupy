#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard packages
import time

# local packages
from toupy.io import LoadData, SaveData
from toupy.utils import display_slice, iterative_show
from toupy.tomo import full_tomo_recons


# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["phaseonly"] = False  # True for the phase projections.
params["amponly"] = True  # True for the amplitude projections.
params["slicenum"] = 550  # Choose the slice
params["filtertype"] = "hann"  # Filter to use for FBP
params["filtertomo"] = 1.0  # Frequency cutoff (between 0 and 1)
params["circle"] = True
params["algorithm"] = "FBP"  # FBP or SART
params["derivatives"] = True  # only for FBP
params["calc_derivatives"] = False  # Calculate derivatives if not done
params["opencl"] = True
params["autosave"] = False
params["vmin_plot"] = None  # 0.5e-5
params["vmax_plot"] = -1e-3  # None
params["colormap"] = "bone"
params["showrecons"] = False  # to display the reconstructed slice on the fly.
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#

if __name__ == "__main__":

    if params["phaseonly"] == params["amponly"]:
        raise ValueError("phaseonly and amponly cannot be True at the same time")

    # loading the data
    aligned_projections, theta, shiftstack, params = LoadData.load(
        "aligned_amp_projections.h5", **params
    )

    # from now on, voxelsize is pixelsize.
    voxelsize = params["pixelsize"]

    tomogram_amp = full_tomo_recons(aligned_projections, theta, **params)

    a = input("Do you want to display the tomographic slices? (y/[n]) :").lower()
    if str(a) == "y":
        iterative_show(
            tomogram_amp,
            vmin=params["vmin_plot"],
            vmax=params["vmax_plot"],
            colormap=params["colormap"],
        )  # Show aligned projections derivatives

    # save the tomograms
    SaveData.save("tomogram_amp.h5", tomogram, theta, shiftstack, **params)

    # next step
    print('You should run "tiff_convertion.py" now')
    # =============================================================================#
