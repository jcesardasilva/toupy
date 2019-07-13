#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:37:44 2016

@author: jdasilva
"""
# standard libraries imports
import sys
import time

# third party packages
import matplotlib.pyplot as plt
import numpy as np

# local packages
from io_utils import checkhostname
from io_utils import create_paramsh5, load_paramsh5
from io_utils import LoadData, SaveData
from registration_utils import derivatives

#-------------------------------------------------------
# still keep this block, but it should disappear soon
if sys.version_info<(3,0):
    input = raw_input
#-------------------------------------------------------

# initializing dictionaries
params= dict()

# Edit section
#=========================
params[u'samplename'] = u'gp2_NaCl_dif_pitch_ffp_tomo'
params[u'phaseonly'] = True
params[u'valx'] = 1 #140          # From edge of region to edge of image in x
params[u'roiy']  = (1,1913)#(150,650)      # Window inside regstack used for alignment delta = 200; % Window should be centered in x to ensure compliance with iradonfast limsx = [1+delta size(regstack,2)-delta];
params[u'showmovie'] = False
params[u'projections'] = np.arange(0,1) # in the case of showmovie = False, choose the projections you want to plot
params[u'derivatives'] = True
params[u'autosave'] = True
params[u'shift_method'] = 'sinc'
#=========================

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#
if __name__=='__main__':
    # load unwrapped phase projections
    host_machine = checkhostname() # always to check in which machine you are working

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()

    # loading parameters from h5file
    kwargs = load_paramsh5(**params)

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()
    inputparams.update(kwargs) # add/update with new values

    # load the reconstructed phase projections
    L = LoadData(**inputparams)
    aligned, theta, shiftstack, outkwargs = L('vertical_alignment.h5')
    inputparams.update(outkwargs) # updating the params

    #updating parameter h5 file
    create_paramsh5(**inputparams)

    # horizontal ROI
    valx = params['valx']
    roiy = params['roiy']

    # Display the projections
    while True:
        roix=range(valx,aligned.shape[2]-valx)
        plt.close('all')
        #plt.clf()
        fig = plt.figure(5)
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(aligned[0],cmap='bone')
        ax1.plot([roix[0],roix[-1]],[roiy[0],roiy[0]],'r-')
        ax1.plot([roix[0],roix[-1]],[roiy[-1],roiy[-1]],'r-')
        ax1.plot([roix[0],roix[0]],[roiy[0],roiy[-1]],'r-')
        ax1.plot([roix[-1],roix[-1]],[roiy[0],roiy[-1]],'r-')
        ax1.axis('tight')
        plt.show(block=False)

        ans = input('Are you happy with the boundaries? ([y]/n)').lower()
        if str(ans)=='' or str(ans)=='y':
            break
        else:
            print(aligned.shape)
            roiy = eval(input('Enter new range in y (top, bottom): '))
            params['roiy'] = roiy
            valx = eval(input('Enter new value from edge of region to edge of image in x: '))
            params['valx'] = valx

    if params['showmovie']:
        frames = range(0,aligned.shape[0])
    else:
        frames = params['projections']
    plt.ion()
    for ii in frames:#num_projections):#sorted(frames):
        img = aligned[ii]
        print("Projection: {}".format(ii+1))
        im1.set_data(img)
        ax1.set_title(u'projection {}'.format(ii+1))
        fig.canvas.draw()
    plt.ioff()

    if params['derivatives']:
        a = input('Do you want to proceed with the derivative calculation? ([y]/n) :').lower()
        if str(a)=='' or str(a)=='y':
            # compute projection derivatives
            aligned_diff=np.empty_like(aligned[:,roiy[0]:roiy[-1],roix[0]:roix[-1]])
            nprojs = aligned.shape[0]
            for ii in range(nprojs):
                print('Computing derivative of projection {}'.format(ii))
                t0 = time.time()
                img=aligned[ii,roiy[0]:roiy[-1],roix[0]:roix[-1]]
                aligned_diff[ii] = derivatives(img,params[u'shift_method'])
                delta_time = time.time()-t0
                rem_time = nprojs-(ii+1)
                print('Done. Time Elapsed {:.02f}s. Estimated remaining time: {:.02f}s.'.format(delta_time,delta_time*rem_time))
            a = input('Do you want to display the calculated derivatives projections? ([y]/n): ').lower()
        else:
            raise SystemExit('Derivative projections have NOT been calculated. Choose an new ROI.')
    else:
        aligned_diff=aligned[:,roiy[0]:roiy[-1],roix[0]:roix[-1]]
        a = input('Do you want to display the cropped projections? ([y]/n): ').lower()

    if str(a)=='' or str(a)=='y':
        plt.close('all')
        fig = plt.figure(5)
        ax1 = fig.add_subplot(111)
        im1=ax1.imshow(aligned_diff[0],cmap='bone',vmin=-.2,vmax=.2)
        plt.ion()
        for ii in range(aligned_diff.shape[0]):
            img = aligned_diff[ii].copy()
            print("Projection: {}".format(ii+1))
            im1.set_data(img)
            ax1.set_title(u'Projection {}'.format(ii+1))
            #fig.colorbar(im1)
            fig.canvas.draw()
            plt.pause(0.001)
        plt.ioff()

    # to save the derivative projections
    print('Saving the projections')
    # save vertically aligned_projections
    S = SaveData(**inputparams)
    S('aligned_derivatives.h5',aligned_diff,theta,shiftstack)
    #save_or_load_data('aligned_derivatives.h5',aligned_diff,theta,pixelsize,shiftstack,**inputparams)
    # next step
    print('You should run ''sinogram_inspection.py'' now')
    #=============================================================================#