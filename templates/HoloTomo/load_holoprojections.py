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
params["account"] = "ma3495"
params["samplename"] = "H2int_15000h_outlet"
params[
    "pathfilename"
] = "/data/id16a/inhouse5/visitor/ma3495/id16a/H2int_15000h_outlet/H2int_15000h_outlet_25nm_holo_/H2int_15000h_outlet_25nm_holo_rec_0000.edf"
params["regime"] = "holoct"
params["showrecons"] = False
params["autosave"] = True
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#

if __name__ == "__main__":
    # loading the projections from files
    stack_objs, theta, pixelsize, params = LoadProjections.loadedf(**params)

    # Save reconstructed phase projections
    SaveData.save("reconstructed_projections.h5", stack_objs, theta, **params)

    # next step
    print('You should run "vertical_alignment_holo.py" for phase tomogram')
# =============================================================================#
