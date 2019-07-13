#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import os
import sys

# third party packages
import numpy as np

# local packages
from io_utils import checkhostname, LoadProjections, SaveData, create_paramsh5

# initializing params
params=dict()

### Edit section ###
#=========================
params[u'showrecons']=False
params[u'account']=u'ma4352'
params[u'samplename'] = u'gp2_NaCl_dif_pitch_ffp_tomo'
params[u'pathfilename']=u'/data/id16a/inhouse4/visitor/ma4352/id16a/analysis/recons/gp2_NaCl_dif_pitch_ffp_tomo_div1p2_b2um_subtomo001_0000_/gp2_NaCl_dif_pitch_ffp_tomo_div1p2_b2um_subtomo001_0000__ML.ptyr'
params[u'regime'] = 'farfield' #'nearfield' or 'farfield'
params[u'autosave'] = True
params[u'phaseonly'] = False # put false if you want to do the amplitude tomogram
params[u'border_crop_x'] = None
params[u'border_crop_y'] = None
params[u'checkextraprojs'] = False
params[u'missing_proj'] = False
params[u'missing_num'] = []
#params[u'cxientry'] = None
#=========================

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#

if __name__=='__main__':
    #load the projections
    host_machine = checkhostname()

    #-------------------------------------------------------
    # still keep this block, but it should disappear soon
    if sys.version_info<(3,0):
        input = raw_input
    #-------------------------------------------------------

    #stack_objs,obj_shape,theta,pixelsize=load_projection(**params)
    P = LoadProjections(**params)
    stack_objs,theta,pixelsize = P()
    params['pixelsize'] = pixelsize

    # create parameter file
    create_paramsh5(**params)

    # special: insert the information of the missing projections
    if params[u'missing_proj']:
        print('Inserting the missing projections')
        missing = params[u'missing_num']
        print(missing)
        delta_theta = theta[1]-theta[0]
        for ii in missing:
            print('Projection: {}'.format(ii))
            theta = np.insert(theta,ii,theta[ii-1]+delta_theta)
            stack_objs = np.insert(stack_objs,ii,stack_objs[ii-1], axis=0)

    # Save reconstructed phase projections
    S = SaveData(**params)
    S('reconstructed_projections.h5',stack_objs,theta)

    # next step
    print('You should run ''remove_phase_ramp.py'' for phase tomogram or')
    print('You should run ''remove_amp_air.py'' for amplitude tomogram')
#=============================================================================#
