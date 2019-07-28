#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:44:54 2016

@author: jdasilva
"""
# standard libraries imports
import os
import sys
import time

# third packages
import h5py
import numpy as np

# local packages
from io_utils import checkhostname
from io_utils import create_paramsh5, load_paramsh5
from io_utils import LoadData, SaveData
from registration_utils import compute_aligned_stack

# initializing dictionaries
params = dict()

# =========================
params[u'samplename'] = u'H2int_15000h_inlet'
params[u'phaseonly'] = True
params[u'apply_alignement'] = True
params[u'interpmeth'] = 'linear'  # 'sinc' or 'linear'
params[u'sort_theta'] = True
params[u'correct_bad'] = False
params[u'bad_projs'] = []  # starting at zero
params[u'tomo_type'] = u'delta'
params[u'derivatives'] = False
# =========================

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#

if __name__ == '__main__':
    # load the aligned derivative projection
    host_machine = checkhostname()

    # -------------------------------------------------------
    # still keep this block, but it should disappear soon
    if sys.version_info < (3, 0):
        input = raw_input
        range = xrange
    # -------------------------------------------------------

    if params[u'tomo_type'] == u'delta' and params[u'derivatives']:
        load_file1 = 'unwrapped_phases.h5'
        load_file2 = 'aligned_projections.h5'
        foldername = params[u'samplename']  # 'projs'
        pathfile = foldername+'/{}_{:04d}.edf'
        # Forsce shift in phasor space False to avoid phase wrapping
        params[u'expshift'] = False
    elif params[u'tomo_type'] == u'delta' and not params[u'derivatives']:
        load_file1 = 'reconstructed_projections.h5'
        load_file2 = 'aligned_projections.h5'
        foldername = params[u'samplename']  # 'projs'
        pathfile = foldername+'/{}_{:04d}.edf'
        # Forsce shift in phasor space False to avoid phase wrapping
        params[u'expshift'] = False
    elif params[u'tomo_type'] == u'beta':
        load_file1 = 'air_corrected_amplitude.h5'
        load_file2 = 'aligned_projections.h5'
        foldername = params[u'samplename']+'_amp'  # 'projs_amp'
        pathfile = foldername+'/{}_amp_{:04d}.edf'

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

    # Apply alignement if needed
    if params[u'apply_alignement']:
        # load the projections before alignment and derivatives
        output_stack, theta, shiftstack, outkwargs = L(load_file1)
        del shiftstack  # delete it to free memory space
        print('Found {} unwrapped projections'.format(len(output_stack)))
        print('Computing aligned images')
        # load shiftstack from aligned phase projections
        shiftstack = L.load_shiftstack(load_file2)
        print('The shiftstack length is {}'.format(shiftstack.shape[1]))
        output_stack = compute_aligned_stack(output_stack, shiftstack, params)
    else:
        output_stack, theta, shiftstack, outkwargs = L(load_file2)
        print('Found {} aligned projections'.format(len(projections_stack)))

    # updating parameter h5 file
    create_paramsh5(**inputparams)

    # correcting bad projections after unwrapping
    if params[u'correct_bad']:
        a = input('Do you want to correct bad projections?([y]/n)').lower()
        if str(a) == '' or str(a) == 'y':
            for ii in params[u'bad_projs']:
                print('Correcting bad projection: {}'.format(ii+1))
                output_stack[ii] = (output_stack[ii-1] +
                                    output_stack[ii+1])/2  # this is better

    # estimate number of projections
    nprojs = projections_stack.shape[0]
    if nprojs != len(theta):
        raise ValueError(
            'The number of projections is different from the number of angles')
    print('The total number of projections is {}'.format(nprojs))

    if params[u'sort_theta']:
        # sorting dataset by theta
        theta_sorted = np.sort(theta)
        argsort_theta = np.argsort(theta)
        output_stack_sorted = output_stack[argsort_theta]
    else:
        theta_sorted = theta
        output_stack_sorted = output_stack

    # check if directory already exists
    if not os.path.isdir(foldername):
        print('Creating the directory {}'.format(foldername))
        os.makedirs(foldername)
    else:
        print('Directory {} already exists and will be used'.format(foldername))

    # writing the edf files
    for ii in range(nprojs):
        img1 = output_stack_sorted[ii]
        print('Saving projections as edf files.')
        print('Projection: {}'.format(ii))
        p0 = time.time()
        fname = pathfile.format(params[u'samplename'], ii)
        hdedf = inputparams
        write_edf(fname, img1, hd=hdedf)
        pf = time.time()-p0
        print('Time elapsed: {:4.02f} s\n'.format(pf))
