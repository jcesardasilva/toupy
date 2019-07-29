#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:22:45 2016

@author: jdasilva
"""
# standard libraries imports
import sys

# third party packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as snf

# local packages
from io_utils import checkhostname
from io_utils import create_paramsh5, load_paramsh5
from io_utils import LoadData, SaveData
from registration_utils import alignprojections_vertical

from toupy.utils.plot_utils import animated_image

# initializing dictionaries
params = dict()

# Register (align) projections by vertical mass fluctuations and center of mass
# =========================
params[u"samplename"] = u"gp2_NaCl_dif_pitch_ffp_tomo"
params[u"phaseonly"] = True
params[u"disp"] = 2  # = 0 no display, =1 only final display, >1 every iteration
params[u"pixtol"] = 0.1  # Tolerance of registration in pixels
params[u"bias"] = True  # Remove bias for y registration
params[u"polyorder"] = 2  # Max order of bias to remove
params[u"expshift"] = False  # Shift in phasor space
# 'sinc'#'sinc' # 'sinc' or 'linear' better for noise
params[u"shiftmeth"] = "linear"
params[u"alignx"] = False  # Align horizontally with center of mass
params[u"maxit"] = 10  # max of iterations
params[u"deltaxal"] = 20  # From edge of region to edge of image in x
params[u"limsy_cc"] = (10, 2319)
# 1500) #(100,900) #(40,340)#(75,615)#(160+20,640-20)      # Window inside regstack used for alignment delta = 200; % Window should be centered in x to ensure compliance with iradonfast limsx = [1+delta size(regstack,2)-delta];
params[u"limsy"] = (150, 1500)
params[u"autosave"] = True
params[u"use_cc"] = False  # use cross-correlation for alignement
params[u"2Dgaussian_filter"] = False
params[u"2Dgaussian_sigma"] = 3
params[u"shift_gaussian_filter"] = False
params[u"shift_gaussian_sigma"] = 3  # None
params[u"load_previous_shiftstack"] = False  # True
params[u"correct_bad"] = False
params[u"bad_projs"] = []  # starting at zero
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # load unwrapped phase projections
    host_machine = checkhostname()  # always to check in which machine you are working

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()

    # loading parameters from h5file
    kwargs = load_paramsh5(**params)

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()
    inputparams.update(kwargs)  # add/update with new values

    # load the reconstructed phase projections
    L = LoadData(**inputparams)
    stack_unwrap, theta, shiftstack, outkwargs = L("unwrapped_phases.h5")
    inputparams.update(outkwargs)  # updating the params

    # updating parameter h5 file
    create_paramsh5(**inputparams)

    # correcting bad projections after unwrapping
    if params[u"correct_bad"]:
        for ii in params[u"bad_projs"]:
            print("Temporary replacement of bad projection: {}".format(ii + 1))
            stack_unwrap[ii] = stack_unwrap[ii - 1]

    # initializing shiftstack
    if params["load_previous_shiftstack"]:
        shiftstack = L.load_shiftstack("vertical_alignment.h5", **inputparams)
        print("Using previous estimate of shiftstack")
    else:
        shiftstack = np.zeros((2, stack_unwrap.shape[0]))

    if params[u"use_cc"]:
        print("Using cross-correlation alignment in horizontal direction")
        # defining the boundaries of the area to be used for the alignment
        deltaxal = params[u"deltaxal"]
        limsx = (deltaxal, stack_unwrap.shape[2] - deltaxal)  # horizontal

        # convinient conversion to ndarray
        limsy = params[u"limsy_cc"]
        limrow = np.asarray(limsy)
        limcol = np.asarray(limsx)
        shiftstack, aligned = cc_align(stack_unwrap, limrow, limcol, params)

    # defining the boundaries of the area to be used for the alignment
    deltaxal = params[u"deltaxal"]
    limsx = (deltaxal, stack_unwrap.shape[2] - deltaxal)  # horizontal

    # convinient conversion to ndarray
    limsy = params[u"limsy"]
    limrow = np.asarray(limsy)
    limcol = np.asarray(limsx)

    plt.close("all")
    shiftstack, aligned = alignprojections_vertical(
        stack_unwrap, limrow, limcol, shiftstack, params
    )

    a = input("Do you want to refine further the alignment? (y/[n]): ").lower()
    if str(a) == "" or str(a) == "n":
        pass
    elif str(a) == "y":
        a1 = input("Do you want to apply a filter to shiftstack? ([y]/n): ").lower()
        if str(a1) == "" or str(a1) == "y":
            shiftstack[0] = snf.gaussian_filter1d(
                shiftstack[0], params["smooth_shifts"]
            )
        plt.close("all")
        shiftstack, aligned = alignprojections_vertical(
            stack_unwrap, limrow, limcol, shiftstack, params
        )
    else:
        raise SystemExit("Unrecognized answer")

    # correcting bad projections after unwrapping
    if params[u"correct_bad"]:
        a = input("Do you want to correct bad projections?([y]/n)").lower()
        if str(a) == "" or str(a) == "y":
            for ii in params[u"bad_projs"]:
                print("Correcting bad projection: {}".format(ii + 1))
                aligned[ii] = (aligned[ii - 1] + aligned[ii + 1]) / 2  # this is better

    a = input("Do you want to display the aligned projections? (y/[n]) :").lower()
    if str(a) == "" or str(a) == "n":
        pass
    else:
        # Show aligned projections
        animated_image(aligned, limrow, limcol)

    # save vertically aligned_projections
    S = SaveData(**inputparams)
    S("vertical_alignment.h5", aligned, theta, shiftstack)
    # next step
    print("You should run " "projections_derivatives.py" " now")
    # =============================================================================#
