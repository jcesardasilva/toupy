#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:44:54 2016

@author: jdasilva
"""
# import of standard libraries
import os
import re
import socket
import sys
import time

# import of third party packages
import pyfftw # has to be imported first to avoid ImportError: dlopen: cannot load any more object with static TLS
import h5py
import libtiff
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage.fourier import fourier_shift

# import of local packages
from FSC import FSCPlot, checkhostname, load_data_FSC
from io_utils import save_or_load_data, checkhostname, save_or_load_tomogram, create_paramsh5, load_paramsh5, save_or_load_FSCdata
from iradon import mod_iradon, mod_iradon2
from scipy import ndimage

#initializing params
params = dict()
#=========================
# Edit session
#=========================
params[u'samplename'] = u'v97_h_nfptomo_15nm'
#params[u'pathfilename']='/data/id16a/inhouse2/staff/ap/ihls2664/id16a/analysis/recons/bone2_hoptyfluo_nfptomo_subtomo001_0000/bone2_hoptyfluo_nfptomo_subtomo001_0000_ML.ptyr'
#params[u'pathfilename']='../recons/bone2_hoptyfluo_nfptomo_subtomo001_0000/bone2_hoptyfluo_nfptomo_subtomo001_0000_ML.ptyr'
params[u'slice_num'] = 1000      # Choose the slice
params[u'limsyFSC'] = [1100, 1480] # number of slices for the 3D FSC
params[u'filtertype'] = 'hann' #'ram-lak' #hann'   # Filter to use for FBP
params[u'filtertomo'] = 1     # Frequency cutoff
params[u'derivatives'] = True ### only for FBP
params[u'opencl'] = True # enable reconstruction with Silx and openCL
params[u'apod_width'] = 50 # apodization width in pixels
params[u'thick_ring'] = 4 # number of pixel to average each FRC ring
params[u'crop'] = [1465,1865,935,1335] # [top, bottom, left, right]
params[u'vmin_plot'] = None#0.5e-5
params[u'vmax_plot'] = -0.5e-4#None
params[u'colormap'] = 'bone' # colormap to show images
params[u'oldfileformat'] = False # if one reads projections from old files
### params[u'pyhst'] = True # not implemented yet
#=========================

# monkey patching
if params[u'opencl']:
    print("Using OpenCL projector")
    mod_iradon = mod_iradon2

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#
if __name__=='__main__':
    #load the aligned derivative projection
    host_machine = checkhostname()

    #-------------------------------------------------------
    # still keep this block, but it should disappear soon
    if sys.version_info<(3,0): # backcompatibility
        input = raw_input
        range = xrange
    #-------------------------------------------------------

    # loading parameters from h5file
    inputparams = dict()
    kwargs = load_paramsh5(**params)
    inputparams.update(kwargs)
    inputparams.update(params)

    # load the aligned phase projections
    if params[u'oldfileformat']:
        filename = 'aligned_projections.h5'#'recons/bone2_hoptyfluo_nfptomo_nfpxct/aligned_projections_proj.h5'
        aligned_projections, theta, pixelsize = load_data_FSC(filename,**inputparams) # in principle this one also works for the modern file format
    else:
        aligned_projections,thetaunsort,deltastack,pixelsize,kwargs= save_or_load_data('aligned_projections.h5',**inputparams)
        inputparams.update(kwargs) # updating the inputkwargs
        inputparams.update(params) # updating the inputkwargs

        # sorting theta
        idxsort = np.argsort(thetaunsort)
        theta = thetaunsort[idxsort]
        aligned_projections = aligned_projections[idxsort]

    # convinient change of variables
    voxelsize = pixelsize # from now on, voxelsize is pixelsize.
    slice_num = params[u'slice_num']
    vmin_plot = params[u'vmin_plot']
    vmax_plot = params[u'vmax_plot']

    # calculate the sinogram
    sinogram_align = np.transpose(aligned_projections[:,slice_num,:])

    # split of the data into two datasets
    print('Spliting in 2 datasets')
    sinogram_align1= sinogram_align[:,0::2]
    theta1 = theta[0::2]
    sinogram_align2= sinogram_align[:,1::2]
    theta2 = theta[1::2]

    # tomographic reconstruction
    print('Calculating a slice 1...')
    t0 = time.time()
    tomogram1 = mod_iradon(sinogram_align1,theta=theta1,output_size=sinogram_align1.shape[0],filter_type=params['filtertype'],derivative=params[u'derivatives'],freqcutoff=params['filtertomo'])
    print('Calculation done. Time elapsed: {} s'.format(time.time()-t0))
    print('Calculating a slice 2...')
    t0 = time.time()
    tomogram2 = mod_iradon(sinogram_align2,theta=theta2,output_size=sinogram_align2.shape[0],filter_type=params['filtertype'],derivative=params[u'derivatives'],freqcutoff=params['filtertomo'])
    print('Calculation done. Time elapsed: {} s'.format(time.time()-t0))

    # keep tomogram shape for later use
    nr, nc = tomogram1.shape

    # cropping
    if params['crop'] is not None:
        if params['crop']!=0:
            creg = params['crop']
            tomogram1 = tomogram1[creg[0]:creg[1],creg[2]:creg[3]]
            tomogram2 = tomogram2[creg[0]:creg[1],creg[2]:creg[3]]
        #~ else:
            #~ image1,image2 = tomogram1,tomogram2
    #~ else:
        #~ image1 = tomo1
        #~ image2 = tomo2

    print("Estimating the resolution by FSC...")
    startfsc = time.time()
    FSC2D=FSCPlot(tomogram1,tomogram2,'halfbit',params['thick_ring'],apod_width = params['transv_apod'])
    normfreqs, T, FSC2Dcurve = FSC2D.plot()
    endfsc = time.time()
    print("Time elapsed: {:g} s".format(endfsc-startfsc))

    print("The pixelsize of the data is {:.02f} nm".format(voxelsize[0]*1e9))

    a = input("\nPlease, input the value of the intersection: ")
    inputparams[u'resolution2D'] = voxelsize[0]*1e9/float(a)
    print("------------------------------------------")
    print("| Resolution is estimated to be {:.2f} nm |".format(inputparams[u'resolution2D']))
    print("------------------------------------------")

    input("\n<Hit Return to close images>")
    plt.close('all')

    # save the 2D FSC data
    save_or_load_FSCdata('FSC2D.h5', normfreqs, T, FSC2Dcurve, tomogram1, tomogram2, theta, voxelsize,**inputparams)

    # 3D FSC calculation
    a = str(input('Do you want to calculate the 3D FSC?(y/n)')).lower()
    if a == ''  or a == 'y':
        #del tomo1, tomo2, image1, image2, sinogram_align, sinogram_align1, sinogram_align2#, sagital_slice1, sagital_slice2
        limsyFSC = params['limsyFSC']
        nslices = limsyFSC[-1]-limsyFSC[0]

        # initializing variables
        tomogram1 = np.empty((nslices,nr,nc))
        tomogram2 = np.empty((nslices,nr,nc))
        for idx,ii in enumerate(range(limsyFSC[0],limsyFSC[-1])):
            print('Slice: {}'.format(ii))
            sinogram_align = np.transpose(aligned_projections[:,ii,:])
            # dividing the data into two datasets
            print('Calculating first slice...')
            t0 = time.time()
            sinogram_align1= sinogram_align[:,0::2]
            tomogram1[idx] = mod_iradon(sinogram_align1,theta=theta1,output_size=sinogram_align1.shape[0],filter_type=params['filtertype'],derivative=params[u'derivatives'],freqcutoff=params['filtertomo'])
            print('Calculation done. Time elapsed: {} s'.format(time.time()-t0))
            print('Calculating second slice...')
            t0 = time.time()
            sinogram_align2= sinogram_align[:,1::2]
            tomogram2[idx] = mod_iradon(sinogram_align2,theta=theta2,output_size=sinogram_align2.shape[0],filter_type=params['filtertype'],derivative=params[u'derivatives'],freqcutoff=params['filtertomo'])
            print('Calculation done. Time elapsed: {} s'.format(time.time()-t0))

        # cropping
        if params['crop'] is not None:
            if params['crop']!=0:
                creg = params['crop']
                tomogram1 = tomogram1[:,creg[0]:creg[1],creg[2]:creg[3]]
                tomogram2 = tomogram2[:,creg[0]:creg[1],creg[2]:creg[3]]
            #~ else:
                #~ image1,image2 = tomogram1,tomogram2
        #~ else:
            #~ image1 = tomogram1
            #~ image2 = tomogram2

        # special for this
        #tomogram1 = tomogram1[:,836:1440,866:1470]
        #tomogram2 = tomogram2[:,836:1440,866:1470]

        # 3D FSC
        print("Estimating the resolution by 3D FSC...")
        startfsc = time.time()
        FSC3D = FSCPlot(tomogram1,tomogram2,'halfbit',params['thick_ring'],apod_width = params['transv_apod'])
        normfreqs, T, FSC3Dcurve = FSC3D.plot()
        endfsc = time.time()
        print("Time elapsed: {:g} s".format(endfsc-startfsc))
        print("The voxelsize of the data is {:.02f} nm".format(voxelsize[0]*1e9))

        a = input("\nPlease, input the value of the intersection: ")
        inputparams[u'resolution3D'] = voxelsize[0]*1e9/float(a)
        print("------------------------------------------")
        print("| Resolution is estimated to be {:.2f} nm |".format(inputparams[u'resolution3D']))
        print("------------------------------------------")

        input("\n<Hit Return to close images>")
        plt.close('all')

        # save the 3D FSC data
        save_or_load_FSCdata('FSC3D.h5', normfreqs, T, FSC3Dcurve, tomogram1, tomogram2, theta, voxelsize,**inputparams)

    print("Your program has finished!")
