#!/usr/bin/env python
# -*- coding: utf-8 -*-

# local packages
from toupy.io import LoadData, SaveData
from toupy.restoration import phaseresidues, chooseregiontounwrap, unwrapping_phase
from toupy.utils import iterative_show, replace_bad

# initializing dictionaries
params = dict()

### Edit section ###
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["phaseonly"] = True
params["autosave"] = True
params["correct_bad"] = True
params["bad_projs"] = [156, 226, 363, 371, 673, 990]  # starting at zero
params["vmin"] = -8
params["vmax"] = None
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # loading the data
    stack_phasecorr, theta, shiftstack, params = LoadData.load(
        "linear_phase_corrected.h5", **params
    )

    # Temporary replacement of bad projections
    if params["correct_bad"]:
        stack_phasecorr = replace_bad(
            stack_phasecorr, list_bad=params["bad_projs"], temporary=True
        )

    # find the residues and choose region to be unwrapped
    rx, ry, airpix = chooseregiontounwrap(stack_phasecorr)

    ansunw = input("Do you want to continue with the unwrapping?([y]/n)").lower()
    if str(ansunw) == "" or str(ansunw) == "y":
        stack_unwrap = unwrapping_phase(stack_phasecorr, rx, ry, airpix, **params)
    else:
        stack_unwrap = stack_phasecorr
        print("The phases have not been unwrapped").lower()

    # display the projections after the unwrapping
    showmovie = input(
        "Do you want to show all the unwrapped projections?([y]/n): "
    ).lower()

    if str(showmovie) == "" or str(showmovie) == "y":
        iterative_show(stack_unwrap, ry, rx, airpix, onlyroi=False)

    # Save the unwrapped phase projections
    SaveData.save("unwrapped_phases.h5", stack_unwrap, theta, **params)
    # next step
    print('You should run "vertical_alignment.py" now')

# =============================================================================#
