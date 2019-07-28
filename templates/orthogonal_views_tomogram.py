#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:41:50 2016

@author: jdasilva
"""
from __future__ import division, print_function

# standard libraries imports
import sys

# third packages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import filters

# local packages
from io_utils import checkhostname, save_or_load_tomogram, create_paramsh5, load_paramsh5

# initializing dictionaries
params = dict()

# ==================
# photon energy to convert tomogram to delta or beta values
params[u'energy'] = 33.6
params[u'samplename'] = u'H2int_15000h_inlet'
params[u'phaseonly'] = True
# ~ params['roi'] = [600, 1600, 675, 1685]
params[u'tomo_type'] = u'delta'
params[u'slice_num'] = 500
params[u'vmin_plot'] = 1e-7  # None
params[u'vmax_plot'] = 5e-6  # 5e-4
params[u'scale_bar_size'] = 5  # in microns
params[u'scale_bar_height'] = 1
params[u'scale_bar_color'] = u'yellow'
params[u'bar_start'] = [50, 860]
params[u'bar_axial'] = [70, 100]  # [cols,rows]
params[u'save_figures'] = True
params[u'colormap'] = u'bone'  # u'jet'
params[u'interpolation'] = u'nearest'  # u'bilinear'
params[u'gaussian_filter'] = False  # True
params[u'sigma_gaussian'] = 3  # if gaussian filter
# ==================

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#
if __name__ == '__main__':
    # load the tomograms
    host_machine = checkhostname()

    # -------------------------------------------------------
    # still keep this block, but it should disappear soon
    if sys.version_info < (3, 0):
        range = xrange
    # -------------------------------------------------------

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()

    # loading parameters from h5file
    kwargs = load_paramsh5(**params)
    inputparams.update(kwargs)
    inputparams.update(params)

    # loading files
    if params[u'tomo_type'] == 'delta':
        params[u'phaseonly'] = True
        params[u'amponly'] = False
        tomogram, theta, deltastack, voxelsize, kwargs = save_or_load_tomogram(
            'tomogram.h5', **inputparams)  # pathfilename=params['path_filename'],h5name='tomogram.h5')
    elif params[u'tomo_type'] == 'beta':
        params[u'phaseonly'] = False
        params[u'amponly'] = True
        tomogram, theta, deltastack, voxelsize, kwargs = save_or_load_tomogram(
            'tomogram_amp.h5', **inputparams)  # pathfilename=params['path_filename'],h5name='tomogram_amp.h5')
    else:
        raise ValueError('Unrecognized tomography type')

    inputparams.update(kwargs)  # updating the params
    inputparams.update(params)  # updating the params

    # updating parameter h5 file
    create_paramsh5(**inputparams)
    pixelsize = voxelsize[0]

    # conversion from phase-shifts to delta or from amplitude to beta
    energy = params[u'energy']
    wavelen = (12.4/energy)*1e-10  # in meters
    if params[u'tomo_type'] == 'delta':
        # Conversion from phase-shifts tomogram to delta
        print("Converting from phase-shifts values to delta values")
        factor = wavelen/(2*np.pi*voxelsize[0])
        #tomogram_delta = np.zeros_like(tomogram)
        for ii in range(tomogram.shape[0]):
            print('Tomogram {}'.format(ii+1))
            #tomogram_delta[ii] = -tomogram[ii].copy()*factor
            tomogram[ii] *= (-factor)
    elif params[u'tomo_type'] == 'beta':
        # Conversion from amplitude to beta
        print("Converting from amplitude to beta values")
        factor = wavelen/(2*np.pi*voxelsize[0])  # amplitude correction factor
        for ii in range(tomogram.shape[0]):
            print('Tomogram {}'.format(ii+1))
            tomogram[ii] *= (-factor)

    # simple transfer of variables
    pixelsize = voxelsize[0]
    slice_num = params['slice_num']
    vmin_plot = params['vmin_plot']
    vmax_plot = params['vmax_plot']
    scale_bar_size = params['scale_bar_size']
    scale_bar_height = params['scale_bar_height']
    bar_start = params['bar_start']
    bar_axial = params['bar_axial']
    colormap_choice = params['colormap']
    interp_type = params['interpolation']
    scale_bar_color = params['scale_bar_color']

    if params[u'gaussian_filter']:
        print('Applying gaussian filter with sigma = {}'.format(
            params[u'sigma_gaussian']))
        # sagital slice
        sagital_slice = filters.gaussian_filter(tomogram[:, np.round(
            tomogram.shape[1]/2).astype('int'), :], params[u'sigma_gaussian'])
        #sagital_slice = filters.gaussian_filter(tomogram_delta[:,821,:],params[u'sigma_gaussian'])
        # coronal slice
        coronal_slice = filters.gaussian_filter(tomogram[:, :, np.round(
            tomogram.shape[1]/2).astype('int')], params[u'sigma_gaussian'])
        #coronal_slice = filters.gaussian_filter(tomogram_delta[:,:,630],params[u'sigma_gaussian'])
        # axial slice
        axial_slice = filters.gaussian_filter(
            tomogram[slice_num], params[u'sigma_gaussian'])
    else:
        # sagital slice
        sagital_slice = tomogram[:, np.round(
            tomogram.shape[1]/2).astype('int'), :]
        # coronal slice
        coronal_slice = tomogram[:, :, np.round(
            tomogram.shape[1]/2).astype('int')]
        # axial slice
        axial_slice = tomogram[slice_num]

    textstr = r'{} $\mu$m'.format(scale_bar_size)

    plt.close('all')

    # Sagital slice
    figsag = plt.figure(num=1)  # ,figsize=(15,6))
    # plt.subplots(num=6,nrows=1,ncols=1,figsize=(15,6))
    axsag = figsag.add_subplot(111)
    imsag = axsag.imshow(sagital_slice, interpolation=interp_type,
                         cmap=colormap_choice, vmin=vmin_plot, vmax=vmax_plot)
    axsag.set_title(u'Sagital slice - {}'.format(params['tomo_type']))
    axsag.text(bar_start[0]-10, bar_start[1]-5, textstr, fontsize=14,
               verticalalignment='bottom', color=scale_bar_color)
    rectsag = patches.Rectangle(
        (bar_start[0], bar_start[1]),  # (x,y)
        (np.round(scale_bar_size*1e-6/pixelsize)),  # width
        (np.round(scale_bar_height*1e-6/pixelsize)),  # height
        color=scale_bar_color,
    )
    axsag.add_patch(rectsag)
    axsag.set_axis_off()
    if params['save_figures']:
        plt.savefig('sagital_{}.png'.format(
            params['tomo_type']), bbox_inches='tight', dpi=200)

    # fig.colorbar(imsag)

    # Coronal slice
    figcor = plt.figure(num=2)
    axcor = figcor.add_subplot(111)
    imcor = axcor.imshow(coronal_slice, interpolation=interp_type,
                         cmap=colormap_choice, vmin=vmin_plot, vmax=vmax_plot)
    axcor.set_title(u'Coronal slice - {}'.format(params['tomo_type']))
    axcor.text(bar_start[0]-10, bar_start[1]-5, textstr, fontsize=14,
               verticalalignment='bottom', color=scale_bar_color)
    rectcor = patches.Rectangle(
        (bar_start[0], bar_start[1]),  # (x,y)
        (np.round(scale_bar_size*1e-6/pixelsize)),  # width
        (np.round(scale_bar_height*1e-6/pixelsize)),  # height
        color=scale_bar_color,
    )
    axcor.add_patch(rectcor)
    axcor.set_axis_off()
    plt.tight_layout()
    plt.show()
    if params['save_figures']:
        plt.savefig('coronal_{}.png'.format(
            params['tomo_type']), bbox_inches='tight', dpi=200)

    # Axial slice
    figaxial = plt.figure(num=3)  # ,figsize=(15,6))
    # plt.subplots(num=6,nrows=1,ncols=1,figsize=(15,6))
    axaxial = figaxial.add_subplot(111)
    imaxial = axaxial.imshow(axial_slice, interpolation=interp_type,
                             cmap=colormap_choice, vmin=vmin_plot, vmax=vmax_plot)
    axaxial.set_title(
        u'Axial slice {} - {} '.format(slice_num+1, params['tomo_type']))
    axaxial.text(bar_axial[0]-10, bar_axial[1]-5, textstr, fontsize=14,
                 verticalalignment='bottom', color=scale_bar_color)
    rectaxial = patches.Rectangle(
        (bar_axial[0], bar_axial[1]),  # (x,y)
        (np.round(scale_bar_size*1e-6/pixelsize)),  # width
        (np.round(scale_bar_height*1e-6/pixelsize)),  # height
        color=scale_bar_color,
    )
    axaxial.add_patch(rectaxial)
    axaxial.set_axis_off()
    plt.show()
    if params['save_figures']:
        plt.savefig('axial_slice{}_{}.png'.format(
            slice_num+1, params['tomo_type']), bbox_inches='tight', dpi=200)
