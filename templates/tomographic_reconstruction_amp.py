#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:44:54 2016

@author: jdasilva
"""
# standard libraries imports
from registration_utils import radtap, _shift_method
from iradon import mod_iradon, mod_iradon2
from io_utils import LoadData, SaveTomogram
from io_utils import create_paramsh5, load_paramsh5
from io_utils import checkhostname, read_edf
from skimage.transform import iradon_sart
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import socket
import sys
import time

# quick fix to avoid ImportError: dlopen: cannot load any more object with static TLS
# not used when not using GPUs
if re.search('gpu', socket.gethostname()) or re.search('gpid16a', socket.gethostname()):
    import pyfftw  # has to be imported first
###

# third party packages

# local packages

# -------------------------------------------------------
# still keep this block, but it should disappear soon
if sys.version_info < (3, 0):
    input = raw_input
    range = xrange
# -------------------------------------------------------

# initializing dictionaries
params = dict()

# =========================
params[u'samplename'] = u'v97_h_nfptomo_15nm'
params[u'phaseonly'] = True
params[u'slice_num'] = 650      # Choose the slice
params[u'filtertype'] = u'hann'   # Filter to use for FBP
params[u'filtertomo'] = 1.0     # Frequency cutoff (between 0 and 1)
params[u'circle'] = True
params[u'vmin_plot'] = None  # 0.5e-5
params[u'vmax_plot'] = None
params[u'colormap'] = 'bone'
params[u'showrecons'] = False  # to display the reconstructed slice on the fly.
params[u'autosave'] = True
params[u'sinohigh'] = None  # 0.1
params[u'sinolow'] = None  # -0.1
params[u'derivatives'] = True  # only for FBP
params[u'smooth_sino'] = False
params[u'filter_size'] = 2
params[u'unsharp_sino'] = False
# u'FBP' # u'SART' # SART does not work with derivatives
params[u'tomo_algorithm'] = u'FBP'
params[u'opencl'] = True
params[u'pyhst'] = True
# =========================

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#

# monkey patching
if params[u'opencl']:
    print("Using OpenCL projector")
    mod_iradon = mod_iradon2


def smooth_sino(sino, **params):
    return ndimage.median_filter(sino, params['filter_size'])


def sharpening_sino(sino, **params):
    blurred_sino = ndimage.median_filter(sino, params['filter_size'])
    filter_blurred_sino = ndimage.median_filter(blurred_sino, 1)
    alpha = 30
    sharper_sino = blurred_sino + alpha * (blurred_sino - filter_blurred_sino)
    return sharper_sino


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
    aligned_projections, theta, shiftstack, outkwargs = L(
        'aligned_amp_projections.h5')
    theta -= theta[0]  # to start at zero
    inputparams.update(outkwargs)  # updating the params

    # updating parameter h5 file
    create_paramsh5(**inputparams)

    # from now on, voxelsize is pixelsize.
    voxelsize = inputparams['pixelsize']

    if not params[u'pyhst']:
        print('Calculating a slice for display')
        slice_num = params['slice_num']
        vmin_plot = params['vmin_plot']
        vmax_plot = params['vmax_plot']

        sinogram_align = np.transpose(aligned_projections[:, slice_num, :])
        if params['smooth_sino'] and params['unsharp_sino']:
            print('Unsharpening the sinogram')
            sinogram_align = sharpening_sino(sinogram_align, **params)
        elif params['smooth_sino'] and not params['unsharp_sino']:
            print('Smoothing the sinogram')
            sinogram_align = smooth_sino(sinogram_align, **params)

        if params[u'derivatives']:
            print('Calculating the derivatives of the sinogram')
            shiftmeth = _shift_method(inputparams)
            sinogram_align = np.squeeze(
                shiftmeth(sinogram_align, (0.5, 0))-shiftmeth(sinogram_align, (-0.5, 0)))
        if params['smooth_sino'] and params['unsharp_sino']:
            print('Unsharpening the sinogram')
            sinogram_align = sharpening_sino(sinogram_align, **params)
        elif params['smooth_sino'] and not params['unsharp_sino']:
            print('Smoothing the sinogram')
            sinogram_align = smooth_sino(sinogram_align, **params)

        if params['circle'] == True:
            N = aligned_projections.shape[2]
            xt = np.linspace(-np.fix(N/2.), np.ceil(N/2.), N, endpoint=False)
            Xt, Yt = np.meshgrid(xt, xt)
            circle = 1-radtap(Xt, Yt, 10, N/2-10)
            del Xt, Yt
        else:
            circle = 1

        # calculating one slice for estimating sizes
        if params[u'tomo_algorithm'] == u'FBP':
            t0 = time.time()
            tomogram_align = mod_iradon(sinogram_align, theta=theta, output_size=sinogram_align.shape[0], filter_type=params[
                                        'filtertype'], derivative=params[u'derivatives'], freqcutoff=params['filtertomo'])
            print(tomogram_align.shape)
            print('Calculation done. Time elapsed: {} s'.format(time.time()-t0))

        elif params[u'tomo_algorithm'] == u'SART':
            t0 = time.time()
            params[u'derivatives'] = False
            print('Calculating the initial guess for SART using FBP')
            tomogram_align = mod_iradon(sinogram_align, theta=theta, output_size=sinogram_align.shape[0], filter_type=params[
                                        'filtertype'], derivative=params[u'derivatives'], freqcutoff=params['filtertomo'])
            print('Done. Starting SART')
            theta = np.float64(theta)
            sinogram_align = np.float64(sinogram_align)
            tomogram_align = np.float64(tomogram_align)

            relaxation_params = 0.15  # (default 0.15)
            initial_guess = True
            if initial_guess:
                # with initial guess
                reconstruction_sart = iradon_sart(
                    sinogram_align, theta=theta, image=tomogram_align, relaxation=relaxation_params)
            else:
                # without initial guess
                reconstruction_sart = iradon_sart(
                    sinogram_align, theta=theta, relaxation=relaxation_params)
            iteration_num = 2
            for ii in range(iteration_num):
                print('Iteration {}'.format(ii+1))
                tomogram_align = iradon_sart(
                    sinogram_align, theta=theta, image=reconstruction_sart, relaxation=relaxation_params)

            print('Calculation done. Time elapsed: {} s'.format(time.time()-t0))

        # Display slice:
        plt.close('all')
        print("Slice: {}".format(slice_num))
        fig = plt.figure(num=1)
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(tomogram_align*circle,
                         cmap=params['colormap'], interpolation='none', vmin=vmin_plot, vmax=vmax_plot)
        ax1.set_title(u'Slice {}'.format(slice_num))
        fig.colorbar(im1)
        plt.show(block=False)

        nslices = aligned_projections.shape[1]
        print('The total number of slices is {}'.format(nslices))
        a = input(
            'Do you want to start the full reconstruction? ([y]/n): ').lower()
        if str(a) == '' or str(a) == 'y':
            slice_size = tomogram_align.shape
            tomogram = np.zeros(
                (aligned_projections.shape[1], slice_size[0], slice_size[1]))
            plt.close('all')
            plt.ion()
            for ii in range(nslices):  # num_projections):#sorted(frames):
                inputparams[u'slice_num'] = ii
                print("Calculating tomogram slice: {}".format(ii+1))
                p0 = time.time()
                sinogram = np.transpose(aligned_projections[:, ii, :])
                if params[u'derivatives']:
                    print('Calculating the derivatives of the sinogram')
                    shiftmeth = _shift_method(inputparams)
                    sinogram = np.squeeze(
                        shiftmeth(sinogram, (0.5, 0))-shiftmeth(sinogram, (-0.5, 0)))
                tomogram[ii] = mod_iradon(sinogram, theta=theta, output_size=sinogram.shape[0],
                                          filter_type=params['filtertype'], derivative=params[u'derivatives'], freqcutoff=params['filtertomo'])
                if params['circle']:
                    tomogram[ii] = tomogram[ii]*circle
                pf = time.time()-p0
                # time.time()-p0))
                print('Time elapsed: {:4.02f} s'.format(pf))
                if params['showrecons']:
                    fig = plt.figure(num=7)
                    plt.clf()
                    ax1 = fig.add_subplot(111)
                    im1 = ax1.imshow(tomogram[ii], cmap='bone')
                    ax1.set_title(u'Tomographic slice {}'.format(ii+1))
                    plt.tight_layout()
                    fig.colorbar(im1)
                    fig.canvas.draw()
            plt.ioff()

            # save the tomograms
            S = SaveTomogram(**inputparams)
            S('tomogram_amp.h5', tomogram, theta, shiftstack)
        elif str(a) == 'n':
            print('The full tomographic reconstrucion has not been done.')
            print('Nothing was saved')
    else:
        print('The full tomographic reconstrucion has not been done. Saving on the calculated slice.')
        a = input(
            'Save the (e) edf projections from the batch results or (v) vol file in HDF5? Type s to skip (e/[v]/s)').lower()
        if a == '' or a == 'v':
            del aligned_projections
            print('Saving .vol into HDF5')
            filename = 'volfloat/{}_amp.vol'.format(params[u'samplename'])
            # Usually, the file .vol.info contains de size of the volume
            linesff = []
            infofilename = filename+'.info'
            with open(infofilename, 'r') as ff:
                for lines in ff:
                    linesff.append(lines.strip('\n'))
            x_size = int(linesff[1].split('=')[1])
            y_size = int(linesff[2].split('=')[1])
            z_size = int(linesff[3].split('=')[1])
            # Now we read indeed the .vol file
            tomogram = np.fromfile(filename, dtype=np.float32).reshape(
                (z_size, x_size, y_size))
            nslices = tomogram.shape[0]
            print('Found {} slices in the volume.'.format(nslices))
            # save the tomograms
            S = SaveTomogram(**inputparams)
            S('tomogram_amp.h5', tomogram, theta, shiftstack)
        elif a == 'e':
            del aligned_projections
            print('Saving the .edf files into HDF5')
            file_wcard = 'volnfp_amp/{}*.edf'.format(params[u'samplename'])
            filelist = sorted(glob.glob(file_wcard))
            num_files = len(filelist)
            print("I found {} edf files".format(num_files))
            # read one file
            img1, px1 = read_edf(filelist[0])
            tomogram = np.zeros((num_files, img1.shape[0], img1.shape[1]))
            for idx, ii in enumerate(filelist):
                print("Reading file: {}".format(ii))
                tomogram[idx] = read_edf(ii)
            # save the tomograms
            S = SaveTomogram(**inputparams)
            S('tomogram_amp.h5', tomogram, theta, shiftstack)
            # save_or_load_tomogram('tomogram.h5',tomogram,theta,voxelsize,shiftstack,**inputparams)
        else:
            print('Nothing was saved')

    # next step
    print('You should run ''tiff_convertion.py'' now')
    #=============================================================================#
