#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:22:45 2016

@author: jdasilva
"""
from __future__ import division, print_function

# standard libraries imports
import sys

# third party packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as snf

# local packages
from io_utils import save_or_load_data, checkhostname, create_paramsh5, load_paramsh5
# ~ from io_utils import save_or_load_projections
from registration_utils import alignprojections_vertical, compute_aligned_stack

# initializing dictionaries
inputkwargs = dict()
params = dict()

# Register (align) projections by vertical mass fluctuations and center of mass
# =========================
inputkwargs[u'samplename'] = u'v97_v_nfptomo2_15nm'
# 'sinc'#'sinc' # 'sinc' or 'linear' better for noise
params[u'interpmeth'] = 'linear'
params[u'correct_bad'] = True
params[u'bad_projs'] = [156, 226, 363, 371, 673, 990]  # starting at zero
# =========================

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#
if __name__ == '__main__':
    # load unwrapped phase projections
    host_machine = checkhostname()  # always to check in which machine you are working

    if sys.version_info > (3, 0):
        raw_input = input
        xrange = range

    inputkwargs[u'amponly'] = True
    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()

    # loading parameters from h5file
    kwargs = load_paramsh5(**inputkwargs)
    inputparams.update(kwargs)
    inputparams.update(params)

    # loading deltastack from alignment with phase projections
    aligned_projections, theta, deltastack, pixelsize, kwargs = save_or_load_data(
        'aligned_projections.h5', **inputparams)
    print(aligned_projections.shape)
    del aligned_projections

    # load the corrected amplitude projections
    stack_amp_corr, theta, _, pixelsize, kwargs = save_or_load_data(
        'air_corrected_amplitude.h5', **inputparams)
    inputparams.update(kwargs)  # updating the inputparams
    inputparams.update(params)  # as second to update to the most recent values

    # updating parameter h5 file
    create_paramsh5(**inputparams)

    # removing extra projections over 180-\Delta\theta degrees
    print(theta[-5:])
    a = str(input('Do you want to remove extra thetas?([y]/n)')).lower()
    if a == '' or a == 'y':
        a1 = eval(input('How many to remove?'))
        # the 3 last angles are 180, 90 and 0 degrees
        stack_amp_corr = stack_amp_corr[:-a1]
        theta = theta[:-a1]  # the 3 last angles are 180, 90 and 0 degrees
    print(theta[-5:])

    # correcting bad projections after unwrapping
    if params[u'correct_bad']:
        for ii in params[u'bad_projs']:
            print('Temporary replacement of bad projection: {}'.format(ii+1))
            stack_amp_corr[ii] = stack_amp_corr[ii-1]

    if stack_amp_corr.shape[0] != deltastack.shape[1]:
        raise ValueError('The size of stack ({}) and deltastack ({}) are different.'.format(
            stack_amp_corr.shape[0], deltastack.shape[1]))

    # aligning the projections
    aligned_amp_projections = compute_aligned_stack(
        stack_amp_corr, deltastack, params)
    previous_shape = aligned_amp_projections.shape

    # cropping projection
    print('Cropping projections to match phase projection size')
    print('Before: {} x {}'.format(previous_shape[1], previous_shape[2]))
    valx = inputparams['valx']
    roix = range(valx, aligned_amp_projections.shape[2]-valx)
    roiy = inputparams['roiy'].astype(np.int)
    aligned_amp_projections = aligned_amp_projections[:,
                                                      roiy[0]:roiy[-1], roix[0]:roix[-1]]
    new_shape = aligned_amp_projections.shape
    print('New: {} x {}'.format(new_shape[1], new_shape[2]))

    # correcting bad projections after alignement
    if params[u'correct_bad']:
        a = raw_input('Do you want to correct bad projections?([y]/n)').lower()
        if str(a) == '' or str(a) == 'y':
            for ii in params[u'bad_projs']:
                print('Correcting bad projection: {}'.format(ii+1))
                aligned_amp_projections[ii] = (
                    aligned_amp_projections[ii-1]+aligned_amp_projections[ii+1])/2  # this is better

    a = raw_input(
        'Do you want to display the aligned projections? (y/[n]) :').lower()
    if str(a) == '' or str(a) == 'n':
        pass
    else:
        # Show aligned projections
        plt.close('all')
        plt.ion()
        fig = plt.figure(4)  # ,figsize=(14,6))
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(aligned_amp_projections[0], cmap='bone')
        for ii in range(aligned_amp_projections.shape[0]):
            print("Projection: {}".format(ii+1))
            projection = aligned_amp_projections[ii]
            im1.set_data(projection)
            ax1.set_title('Projection {}'.format(ii+1))
            fig.canvas.draw()
        plt.ioff()

    # save vertically aligned_projections
    save_or_load_data('aligned_amp_projections.h5', aligned_amp_projections,
                      theta, pixelsize, deltastack, **inputparams)
    # next step
    print('You should run ''tomographic_reconstruction_amp.py'' now')
    #=============================================================================#
