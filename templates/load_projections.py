#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np

# local packages
from toupy.io import LoadProjections, SaveData

# initializing params
params = dict()

### Edit section ###
# =========================
params['account'] = 'ma4351'
params['samplename'] = 'v97_v_nfptomo2_15nm'
params['pathfilename'] = '/data/id16a/inhouse4/visitor/ma4351/id16a/analysis/recons/v97_v_nfptomo2_15nm_subtomo001_0000/v97_v_nfptomo2_15nm_subtomo001_0000_ML.ptyr'
params['regime'] = 'nearfield'  # 'nearfield' or 'farfield'
params['showrecons'] = False
params['autosave'] = True
# put false if you want to do the amplitude tomogram
params['phaseonly'] = False
params['border_crop_x'] = None
params['border_crop_y'] = None
params['checkextraprojs'] = True
params['missingprojs'] = False
params['missingnum'] = []
# =========================

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#

if __name__ == '__main__':
    # loading the projections from files
    stack_objs, theta, pixelsize = LoadProjections.load(**params)

    # add the information of the pixelsize to params
    params['pixelsize'] = pixelsize

    # Save reconstructed phase projections
    SaveData.save('reconstructed_projections.h5', stack_objs, theta, **params)

    # next step
    print('You should run ''remove_phase_ramp.py'' for phase tomogram or')
    print('You should run ''remove_amp_air.py'' for amplitude tomogram')
#=============================================================================#
