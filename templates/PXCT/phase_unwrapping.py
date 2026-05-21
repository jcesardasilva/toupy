#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template for phase unwrapping using the algorithm implement in skimage:
https://scikit-image.org/docs/dev/auto_examples/filters/plot_phase_unwrap.html?highlight=unwrapping
"""

# local packages
from toupy.io import LoadData, SaveData
from toupy.restoration import phaseresidues, chooseregiontounwrap, unwrapping_phase
from toupy.utils import iterative_show, replace_bad
#from toupy.restoration.unwraptools import _unwrapping_phase_parallel

# initializing dictionaries
params = dict()

### Edit section ###
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["phaseonly"] = True
params["threshold"] = 2000
params["vmin"] = -8
params["vmax"] = None
params["colormap"]="bone"
params["parallel"] = True
params["n_cpus"] = -1
params["autosave"] = True
params["correct_bad"] = False
params["bad_projs"] = []  # starting at zero
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
    # chooseregiontounwrap opens an interactive GUI:
    #   1. click top-left corner of unwrap region
    #   2. click bottom-right corner
    #   3. click a pixel in air (inside the rectangle)
    #   then click  Confirm  — or  Reset  to start over.
    #
    # Script mode: plt.show(block=True) has already returned when this
    #              line executes, so rx, ry, airpix are the finished result.
    # Notebook mode (%matplotlib widget): interact with the GUI in the
    #              cell output, then run the next cell to access results:
    #              rx, ry, airpix = picker.rx, picker.ry, picker.airpix
    rx, ry, airpix = chooseregiontounwrap(
                        stack_phasecorr,
                        threshold = params["threshold"],
                        parallel = params["parallel"],
                        ncores = params["n_cpus"],
                    )

    stack_unwrap = unwrapping_phase(
                        stack_phasecorr,
                        rx,
                        ry,
                        airpix,
                        **params
                )

    del stack_phasecorr

    # display the projections after the unwrapping
    iterative_show(stack_unwrap, ry, rx, airpix, onlyroi=False)

    # Save the unwrapped phase projections
    SaveData.save("unwrapped_phases.h5", stack_unwrap, theta, **params)
    # next step
    print('You should run "vertical_alignment.py" now')

# =============================================================================#
