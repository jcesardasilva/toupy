#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import sys

# third party packages
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, Button#, RectangleSelector
from matplotlib.widgets import TextBox

# local packages
from io_utils import checkhostname
from io_utils import create_paramsh5, load_paramsh5
from io_utils import LoadData, SaveData
from phase_ramp_utils import gui_plot

#-------------------------------------------------------
# still keep this block, but it should disappear soon
if sys.version_info<(3,0):
    input = raw_input
#-------------------------------------------------------

#initializing dict params
params = dict()

### Edit section ###
#=========================
params['samplename'] = u'gp2_NaCl_dif_pitch_ffp_tomo'
params['crop_reg'] = 20
params['autosave'] = True
params['vmin'] = -np.pi/2
params['vmax'] = np.pi/2
#=========================

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#
if __name__=='__main__':
    #check machine
    host_machine = checkhostname()

    params[u'amponly'] = False
    params[u'phaseonly'] = True

    # loading parameters from h5file
    kwargs = load_paramsh5(**params)

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()
    inputparams.update(kwargs) # add/update with new values

    # load the reconstructed phase projections
    L = LoadData(**inputparams)
    stack_objs, theta, shiftstack, outkwargs = L('reconstructed_projections.h5')
    inputparams.update(outkwargs) # updating the params

    #updating parameter h5 file
    create_paramsh5(**inputparams)

    # cropping the image for the phase ramp removal
    crop_reg = params['crop_reg']
    if crop_reg is not None:
        if crop_reg != 0:
            stack_objs = stack_objs[:,crop_reg:-crop_reg,crop_reg:-crop_reg]
    stack_phasecorr = gui_plot(stack_objs,**inputparams)

    # Save the corrected phase projections
    S = SaveData(**inputparams)
    S('linear_phase_corrected.h5',stack_phasecorr,theta)

    # next step
    print('You should run ''phase_unwrapping.py'' now')
#=============================================================================#
