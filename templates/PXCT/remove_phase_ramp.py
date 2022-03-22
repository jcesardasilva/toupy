#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template for the phase ramp removal from the phase projections

This module will call a GUI with button for each task. Don't forget to
close the GUI once finished and proceed to the saving. 
"""

# local packages
from toupy.io import LoadData, SaveData
from toupy.restoration import gui_plotphase

# initializing params
params = dict()

### Edit section ###
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["crop_reg"] = None  # [20,0,20,0] # left, bottom, right, top
params["autosave"] = False
params["hcen"] = 738
params["vmin"] = -1.6
params["vmax"] = 1.6
params["phaseonly"] = True  # ensuring that we are loading only phase
params["colormap"]="bone"
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # loading the data
    stack_objs, theta, shiftstack, params = LoadData.load(
        "reconstructed_projections.h5", **params
    )

    # correcting phase ramp
    stack_phasecorr = gui_plotphase(stack_objs, **params)

    # Save the corrected phase projections
    SaveData.save("linear_phase_corrected.h5", stack_phasecorr, theta, **params)

    # next step
    print('You should run "phase_unwrapping.py" now')
# =============================================================================#
