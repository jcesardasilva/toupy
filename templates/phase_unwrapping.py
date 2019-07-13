#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 07 10:21:27 2015

@author: Julio Cesar da Silva (ESRF) - jdasilva@esrf.fr
"""
# Standard library imports
import sys
import time

# third party packages
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unwrap_phase

# local packages
from io_utils import checkhostname
from io_utils import create_paramsh5, load_paramsh5
from io_utils import LoadData, SaveData
from unwrapping_utils import phaseresidues

#-------------------------------------------------------
# still keep this block, but it should disappear soon
if sys.version_info<(3,0):
    input = raw_input
    range = xrange
#-------------------------------------------------------

# initializing dictionaries
params= dict()

### Edit section ###
params[u'samplename'] = u'gp2_NaCl_dif_pitch_ffp_tomo'
params[u'phaseonly'] = True
params[u'autosave'] = True
params[u'correct_bad'] = False
params[u'bad_projs'] = [] # starting at zero
params[u'vmin']=-8
params[u'vmax']=None

#=============================================================================#
# Don't edit below this line, please                                          #
#=============================================================================#
if __name__=='__main__':
    #load the linear phase corrected projections
    host_machine = checkhostname()

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()

    # loading parameters from h5file
    kwargs = load_paramsh5(**params)

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()
    inputparams.update(kwargs) # add/update with new values

    # load the reconstructed phase projections
    L = LoadData(**inputparams)
    stack_phasecorr, theta, shiftstack, outkwargs = L('linear_phase_corrected.h5')
    inputparams.update(outkwargs) # updating the params

    #updating parameter h5 file
    create_paramsh5(**inputparams)

    # correcting bad projections before unwrapping
    if params[u'correct_bad']:
        for ii in params[u'bad_projs']:
            print('Temporary replacement of bad projection: {}'.format(ii))
            stack_phasecorr[ii] = stack_phasecorr[ii-1]

    # Find and plot residues for the phase unwrapping
    resmap = 0
    for ii in range(stack_phasecorr.shape[0]):
        print('Searching for residues in projection {}'.format(ii+1))
        residues,residues_charge = phaseresidues(stack_phasecorr[ii])
        resmap += np.abs(residues)
    yres,xres = np.where(resmap>=1.0)

    # display the residues
    plt.close('all')
    plt.figure(4)
    plt.imshow(resmap,cmap='jet')
    plt.axis('tight')
    plt.plot(xres,yres,'or')
    plt.show(block=False)

    # choosing the are for the unwrapping
    while True:
        plt.ion()
        fig = plt.figure(6)
        plt.clf()
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(stack_phasecorr[0],cmap='bone')
        ax1.plot(xres,yres,'or')
        ax1.axis('tight')
        plt.show(block=False)
        print('The array dimensions are {} x {}'.format(stack_phasecorr[0].shape[0],stack_phasecorr[0].shape[1]))
        print('Please, choose an area for the unwrapping:')
        # while loops in each question to avoid mistapying problems
        while True:
            deltax = eval(input('From edge of region to edge of image in x: '))
            if isinstance(deltax,int):
                rx=range(1+deltax,stack_phasecorr.shape[2]-deltax)
                break
            else:
                print('Wrong typing. Try it again.')
        while True:
            ry = eval(input('Range in y (top, bottom): '))
            if isinstance(ry,tuple):
                #ry = range(ry[0],ry[-1])
                break
            else:
                print('Wrong typing. Try it again.')
        while True:
            airpix = eval(input('Pixel in air (x,y) or (col,row): '))
            if isinstance(airpix,tuple):
                # check if air pixel is inside the image
                if airpix[0]<rx[0] or airpix[0]>rx[-1] or airpix[1]<ry[0] or airpix[1]>ry[-1]:
                    print(u'Pixel of air is outside of region of unwrapping')
                    print('Wrong typing. Try it again.')
                else:
                    break
            else:
                print('Wrong typing. Try it again.')

        # couting residues
        num_residues = int(np.round(resmap[ry[0]:ry[-1],rx[0]:rx[-1]].sum()))
        print('Chosen region contains {} residues in total'.format(num_residues))

        # update images with boudaries
        ax1.plot([rx[0],rx[-1]],[ry[0],ry[0]],'r-')
        ax1.plot([rx[0],rx[-1]],[ry[-1],ry[-1]],'r-')
        ax1.plot([rx[0],rx[0]],[ry[0],ry[-1]],'r-')
        ax1.plot([rx[-1],rx[-1]],[ry[0],ry[-1]],'r-')
        ax1.plot(airpix[0], airpix[1],'ob')
        ax1.set_title('First projection with boundaries')
        plt.show(block=False)

        ans = input('Are you happy with the boundaries?([y]/n)').lower()
        if str(ans)=='' or str(ans)=='y':
            break

    showmovie = input('Do you want to show all the projections with the boundaries?(y/[n]): ').lower()

    if str(showmovie)=='' or str(showmovie)=='y':
        plt.close('all')
        plt.ion()
        fig = plt.figure(6)
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(stack_phasecorr[0],cmap='bone')
        ax1.plot([rx[0],rx[-1]],[ry[0],ry[0]],'r-')
        ax1.plot([rx[0],rx[-1]],[ry[-1],ry[-1]],'r-')
        ax1.plot([rx[0],rx[0]],[ry[0],ry[-1]],'r-')
        ax1.plot([rx[-1],rx[-1]],[ry[0],ry[-1]],'r-')
        ax1.plot(airpix[0], airpix[1],'ob')
        #ax1.plot(xres,yres,'or')
        ax1.axis('tight')
        plt.show(block=False)
        for ii in range(stack_phasecorr.shape[0]):
            img = stack_phasecorr[ii].copy()
            print("Projection: {}".format(ii+1))
            im1.set_data(img)
            ax1.set_title(u'projection {}'.format(ii+1))
            fig.canvas.draw()
        plt.ioff()

    ansunw = input('Do you want to continue with the unwrapping?([y]/n)').lower()
    if str(ansunw)=='' or str(ansunw)=='y':
        stack_unwrap = np.empty_like(stack_phasecorr)
        # test on first projection
        img0_unwrap = stack_phasecorr[0]
        img0_wrap_sel=img0_unwrap[ry[0]:ry[-1],rx[0]:rx[-1]]
        img0_unwrap_sel = unwrap_phase(img0_wrap_sel) # skimage
        img0_unwrap[ry[0]:ry[-1],rx[0]:rx[-1]] = img0_unwrap_sel
        img0_unwrap[ry[0]:ry[-1],rx[0]:rx[-1]] = img0_unwrap_sel- 2*np.pi*np.round(img0_unwrap[airpix[1],airpix[0]]/(2*np.pi))
        # displaying
        plt.close('all')
        plt.ion()
        fig = plt.figure(7)
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(stack_phasecorr[0],cmap='bone',vmin=params[u'vmin'],vmax=params[u'vmax'])
        ax1.plot([rx[0],rx[-1]],[ry[0],ry[0]],'r-')
        ax1.plot([rx[0],rx[-1]],[ry[-1],ry[-1]],'r-')
        ax1.plot([rx[0],rx[0]],[ry[0],ry[-1]],'r-')
        ax1.plot([rx[-1],rx[-1]],[ry[0],ry[-1]],'r-')
        ax1.plot(airpix[0], airpix[1],'ob')
        ax1.axis('tight')
        plt.show(block=False)
        while True:
            a = input('Do you want to edit the color scale?([y]/n)').lower()
            if str(a) == '' or str(a) == 'y':
                while True:
                    color_vmin = eval(input('Minimum color scale value: '))
                    if isinstance(color_vmin,int) or isinstance(color_vmin,float):
                        break
                    else:
                        print('Wrong typing. Try it again.')
                while True:
                    color_vmax = eval(input('Maximum color scale value: '))
                    if isinstance(color_vmax,int) or isinstance(color_vmax,float):
                        break
                    else:
                        print('Wrong typing. Try it again.')
                params[u'vmin'] = color_vmin
                params[u'vmax'] = color_vmax
                print('Using vmin={} and vmax={}'.format(params[u'vmin'],params[u'vmax']))
                # displaying the update images
                plt.close('all') # close previous ones
                plt.ion()
                fig = plt.figure(7)
                ax1 = fig.add_subplot(111)
                im1 = ax1.imshow(stack_phasecorr[0],cmap='bone',vmin=params[u'vmin'],vmax=params[u'vmax'])
                ax1.plot([rx[0],rx[-1]],[ry[0],ry[0]],'r-')
                ax1.plot([rx[0],rx[-1]],[ry[-1],ry[-1]],'r-')
                ax1.plot([rx[0],rx[0]],[ry[0],ry[-1]],'r-')
                ax1.plot([rx[-1],rx[-1]],[ry[0],ry[-1]],'r-')
                ax1.plot(airpix[0], airpix[1],'ob')
                ax1.axis('tight')
                plt.show(block=False)
            else:
                print('Color scale was not changed. Using vmin={} and vmax={}'.format(params[u'vmin'],params[u'vmax']))
                break
        # main loop for the unwrapping
        nprojs = stack_phasecorr.shape[0]
        for ii in range(nprojs):
            t0 = time.time()
            img_unwrap = stack_phasecorr[ii]
            print("Unwrapping projection: {}".format(ii))
            img_wrap_sel=img_unwrap[ry[0]:ry[-1],rx[0]:rx[-1]] # select the region to be unwrapped
            img_unwrap_sel = unwrap_phase(img_wrap_sel) # unwrap the region using the algorithm from skimage
            img_unwrap[ry[0]:ry[-1],rx[0]:rx[-1]] = img_unwrap_sel # update the image in the original array
            img_unwrap[ry[0]:ry[-1],rx[0]:rx[-1]] = img_unwrap_sel-2*np.pi*np.round(img_unwrap[airpix[1],airpix[0]]/(2*np.pi))
            stack_unwrap[ii]=img_unwrap # update the stack
            delta_time = time.time()-t0
            rem_time = nprojs-(ii+1)
            print('Done. Time Elapsed {:.02f}s. Estimated remaining time: {:.02f}s.'.format(delta_time,delta_time*rem_time))
            # displaying
            im1.set_data(img_unwrap)
            ax1.set_title('Unwrapped Projection {}'.format(ii))
            fig.canvas.draw()
        plt.ioff()
        anssave = input('Do you want to continue and save the phase?([y]/n)').lower()
    else:
        stack_unwrap = stack_phasecorr
        anssave = input('The phases have not been unwrapped. Do you want to continue and save the phase anyway?([y]/n)').lower()

    # correcting bad projections after unwrapping
    if params[u'correct_bad']:
        for ii in params[u'bad_projs']:
            print('Correcting bad projection: {}'.format(ii+1))
            stack_phasecorr[ii] = (stack_phasecorr[ii-1]+stack_phasecorr[ii+1])/2

    if str(anssave)=='' or str(anssave)=='y':
        # Save the unwrapped phases
        S = SaveData(**inputparams)
        S('unwrapped_phases.h5',stack_unwrap,theta)
        # next step
        print('You should run ''vertical_alignment.py'' now')
    else:
        print('The projections have not been saved and you cannot proceed to the next step before saving them.')

    #=============================================================================#
