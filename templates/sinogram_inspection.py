#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:02:59 2016

@author: jdasilva
"""

# standard libraries imports
import sys
import time

# third packages
import matplotlib.pyplot as plt
import numpy as np

# local packages
from io_utils import checkhostname
from io_utils import create_paramsh5, load_paramsh5
from io_utils import LoadData, SaveData
from iradon import mod_iradon, mod_iradon2

# initializing dictionaries
params = dict()

# Edit section
# =========================
params[u'samplename'] = u'H2int_15000h_inlet'
params[u'phaseonly'] = True
params[u'slice_num'] = 650      # Choose the slice
params[u'filtertype'] = u'hann'   # Filter to use for FBP
params[u'filtertomo'] = 0.9        # Frequency cutoff
# initial guess of the offset of the axis of rotation
params[u'rot_axis_offset'] = 0
params[u'cliplow'] = None  # clip on sample threshold
params[u'cliphigh'] = None  # [-3e-3], # clip on sample threshold
params[u'colormap'] = 'bone'
params[u'sinohigh'] = None  # -0.1
params[u'sinolow'] = None  # 0.1
params[u'sinocmap'] = 'bone'
params[u'derivatives'] = True
params[u'opencl'] = True
# =========================

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#
if __name__ == '__main__':
    # load the aligned derivative projection
    host_machine = checkhostname()

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()

    # loading parameters from h5file
    kwargs = load_paramsh5(**params)

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()
    inputparams.update(kwargs)  # add/update with new values

    # load the reconstructed phase projections
    L = LoadData(**inputparams)
    aligned_diff, theta, shiftstack, outkwargs = L('aligned_derivatives.h5')
    theta -= theta[0]  # to start at zero
    inputparams.update(outkwargs)  # updating the params

    # updating parameter h5 file
    create_paramsh5(**inputparams)

    # Inspection of a sinogram and a tomogram
    slice_num = params['slice_num']
    aligned_diff = np.asarray(aligned_diff).copy()
    while True:
        sinogram_prealign = np.transpose(aligned_diff[:, slice_num, :])
        if np.sign(params['rot_axis_offset']) == -1:
            print('Initial guess of the rotation axis offset : {}'.format(
                params['rot_axis_offset']))
            # np.transpose(aligned_diff[:,slice_num,:])
            sinogram_prealign = np.pad(sinogram_prealign, ((
                0, 2*abs(params['rot_axis_offset'])), (0, 0)), 'constant', constant_values=0)
        elif np.sign(params['rot_axis_offset']) == +1:
            print('Initial guess of the rotation axis offset : {}'.format(
                params['rot_axis_offset']))
            # np.transpose(aligned_diff[:,slice_num,:])
            sinogram_prealign = np.pad(sinogram_prealign, ((
                2*abs(params['rot_axis_offset']), 0), (0, 0)), 'constant', constant_values=0)
        print('Calculating a tomographic slice')
        p0 = time.time()
        if params[u'opencl']:
            print('Using opencl backprojector')
            B = None
            tomogram_prealign = mod_iradon2(sinogram_prealign, theta=theta, output_size=sinogram_prealign.shape[
                                            0], filter_type=params['filtertype'], derivative=params['derivatives'], freqcutoff=params['filtertomo'])
        else:
            tomogram_prealign = mod_iradon(sinogram_prealign, theta=theta, output_size=sinogram_prealign.shape[
                                           0], filter_type=params['filtertype'], derivative=params['derivatives'], freqcutoff=params['filtertomo'])
        print('Time elapsed: {} s'.format(time.time()-p0))
        print('Calculation done')
        # Display slice:
        plt.close('all')
        print("Slice: {}".format(slice_num))
        fig1 = plt.figure(num=5)
        ax1 = fig1.add_subplot(111)
        im1 = ax1.imshow(tomogram_prealign, cmap=params['colormap'],
                         interpolation='none', vmin=params['cliplow'], vmax=params['cliphigh'])
        ax1.set_title(u'Slice'.format(slice_num))
        fig1.colorbar(im1)
        plt.show(block=False)
        fig2 = plt.figure(num=6)
        ax2 = fig2.add_subplot(111)
        im2 = ax2.imshow(sinogram_prealign, cmap=params['sinocmap'],
                         interpolation='none', vmin=params['sinolow'], vmax=params['sinohigh'])
        ax2.axis('tight')
        ax2.set_title(u'Sinogram - Slice'.format(slice_num))
        fig2.colorbar(im2)
        plt.show(block=False)
        a = input('Are you happy with the rotation axis?([y]/n)').lower()
        if a == '' or a == 'y':
            break
        else:
            b = eval(input('Enter new rotation axis estimate: '))
            params['rot_axis_offset'] = b
    # next step
    print('You should run ''horizontal_alignment.py'' now')
