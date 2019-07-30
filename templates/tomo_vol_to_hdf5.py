#!/usr/bin/env python
# -*- coding: utf-8 -*-

# local packages
from toupy.io import LoadData,SaveData, load_paramsh5

# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#

if __name__ == "__main__":

    # update params
    paramsh5 = load_paramsh5(**params)
    params.update(paramsh5)
    # loading theta and shiftstack for the saving
    shiftstack = LoadData.loadshiftstack("aligned_projections.h5", **params)
    theta = LoadData.loadtheta("aligned_projections.h5", **params)
    
    # actual saving
    SaveData.save_vol_to_h5("tomogramVOL.h5", theta, **params)
 
    # next step
    print('You can now run "tiff_convertion.py" if needed')
    # =============================================================================#
