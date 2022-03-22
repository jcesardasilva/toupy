#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template for loading the projections from phase retrieved files
"""

# third party packages
import numpy as np

# local packages
from toupy.io import LoadProjections, SaveData

# initializing params
params = dict()

### Edit section ###
# =========================
params["account"] = "ma5295"
params["samplename"] = "ACM_101_Zr_reg"
params["pathfilename"] = "/data/visitor/ma5295/id16a/ACM_101_Zr_reg/analysis/recons/ACM_101_Zr_reg_NFP_035nm_subtomo001_0000/ACM_101_Zr_reg_NFP_035nm_subtomo001_0000_ML_pycuda.ptyr"
params["scanprefix"] = "ACM_101_Zr_reg_NFP_035nm"
params["regime"] = "nearfield"  # 'nearfield' or 'farfield'
params["showrecons"] = False
params["autosave"] = True
params["phaseonly"] = False  # put false if you want to do the amplitude tomogram
params["border_crop_x"] = None
params["border_crop_y"] = None
params["checkextraprojs"] = True
params["missingprojs"] = False
params["missingnum"] = []
params["legacy"]=False
params["pycudaprojs"]=True
params["thetameta"]=False
params["detector"]="PCIe"
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#

if __name__ == "__main__":
    # loading the projections from files
    stack_objs, theta, pixelsize, params = LoadProjections.load(**params)

    # Save reconstructed phase projections
    SaveData.save("reconstructed_projections.h5", stack_objs, theta, **params)

    # next step
    print('You should run "remove_phase_ramp.py" for phase tomogram or')
    print('You should run "remove_amp_air.py" for amplitude tomogram')
# =============================================================================#
