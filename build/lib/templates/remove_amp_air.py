#!/usr/bin/env python
# -*- coding: utf-8 -*-

# local packages
from toupy.io import LoadData, SaveData
from toupy.restoration import gui_plotphaseamp

# initializing params
params = dict()

### Edit section ###
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["crop_reg"] = None  # [20,0,20,0] # left, bottom, right, top
params["autosave"] = False
params["vmin"] = -np.pi / 2
params["vmax"] = np.pi / 2
params["phaseonly"] = False  # ensuring that we are loading only phase
params["amponly"] = False  # ensuring that we are loading only amplitude
# =========================
# params[u'correct_bad'] = True
# params[u'bad_projs'] = [117] # starting at zero

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # loading the data
    stack_objs, theta, shiftstack, params = LoadData.load(
        "reconstructed_projections.h5", **params
    )

    # correcting phase ramp
    stack_phasecorr = gui_plotamp(stack_objs, **params)

    # Save the corrected phase projections
    SaveData.save("air_corrected_amplitude.h5", stack_phasecorr, theta, **params)

    # next step
    print('You should run "phase_unwrapping.py" now')
# =============================================================================#