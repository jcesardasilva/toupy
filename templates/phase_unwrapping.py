#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 07 10:21:27 2015

@author: Julio Cesar da Silva (ESRF) - jdasilva@esrf.fr
"""
# Standard library imports
import time

# third party packages
import matplotlib.pyplot as plt
import numpy as np

# local packages
from toupy.io import LoadData, SaveData
from toupy.restoration import phaseresidues, chooseregiontounwrap, \
    unwrapping_phase
from toupy.utils import iterative_show

# initializing dictionaries
params = dict()

### Edit section ###
# =========================
params['samplename'] = 'v97_v_nfptomo2_15nm'
params['phaseonly'] = True
params['autosave'] = True
params['correct_bad'] = False
params['bad_projs'] = [355, 372, 674]  # starting at zero
params['vmin'] = -8
params['vmax'] = None
# =========================

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#
if __name__ == '__main__':
    # loading the data
    stack_phasecorr, theta, shiftstack, params = LoadData.load(
        'linear_phase_corrected.h5', **params)

    # correcting bad projections before unwrapping
    if params['correct_bad']:
        for ii in params['bad_projs']:
            print('Temporary replacement of bad projection: {}'.format(ii))
            stack_phasecorr[ii] = stack_phasecorr[ii-1]

    # find the residues and choose region to be unwrapped
    rx, ry, airpix = chooseregiontounwrap(stack_array)

    showmovie = input(
        'Do you want to show all the projections with the boundaries?(y/[n]): ').lower()

    if str(showmovie) == '' or str(showmovie) == 'y':
        plt.close('all')s
        iterative_show(stack_phasecorr, rx, ry, airpix, onlyroi=False)

    ansunw = input(
        'Do you want to continue with the unwrapping?([y]/n)').lower()
    if str(ansunw) == '' or str(ansunw) == 'y':
        stack_unwrap = unwrapping_phase(
            stack_phasecorr, rx, ry, airpix, **params)
    else:
        stack_unwrap = stack_phasecorr
        anssave = input(
            'The phases have not been unwrapped. Do you want to continue and save the phase anyway?([y]/n)').lower()

    # Save the unwrapped phase projections
    SaveData.save('unwrapped_phases.h5', stack_unwrap, theta, **params)
    # next step
    print('You should run ''vertical_alignment.py'' now')

#=============================================================================#
