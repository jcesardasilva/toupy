#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import time

# third party packages
import matplotlib.pyplot as plt
import numpy as np

# local packages
from toupy.io import LoadData, SaveData
from toupy.restoration import calculate_derivatives, chooseregiontoderivatives
from toupy.utils import iterative_show

# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["phaseonly"] = True
params["deltax"] = 2  # From edge of region to edge of image in x
params["limsy"] = (1,2329)  # (top, bottom)
params["shift_method"] = "fourier"
params["autosave"] = True
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # loading the data
    aligned, theta, shiftstack, params = LoadData.load(
        "vertical_alignment.h5", **params
    )

    # chosse the region to apply the derivatives
    roix, roiy = chooseregiontoderivatives(aligned, **params)

    # calculating the projection derivatives
    aligned_diff = calculate_derivatives(
        aligned, roiy, roix, shift_method=params["shift_method"]
    )

    # display the projections after the unwrapping
    showmovie = input(
        "Do you want to show all the projection derivatives?([y]/n): "
    ).lower()

    if str(showmovie) == "" or str(showmovie) == "y":
        iterative_show(aligned_diff, onlyroi=False, vmin=-0.2, vmax=0.2)

    # Save the projection derivatives
    SaveData.save("aligned_derivatives.h5", aligned_diff, theta, shiftstack, **params)

    # next step
    print('You should run "sinogram_inspection.py" now')
    # =============================================================================#
