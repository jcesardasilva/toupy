#!/usr/bin/env python
# -*- coding: utf-8 -*-

#third party packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib as mpl

__all__=['show_projections',
         'show_linearphase']

def show_projections(objs,probe,idxproj):
    """
    Show projections and probe
    """
    if objs.shape[0]<objs.shape[1]:
        plotgrid=(3,1)
        plotsize=(6,20)
    else:
        plotgrid=(1,3)
        plotsize=(20,6)
    vabsmean = np.abs(objs).mean()
    perabsmean = 0.2*vabsmean
    #vphasemean = np.angle(objs).mean()
    #perphasemean = 0.3*vphasemean
    plt.clf()
    fig, (ax1,ax2,ax3) = plt.subplots(num=1,nrows=plotgrid[0],ncols=plotgrid[1],figsize=plotsize)
    im1 = ax1.imshow(np.abs(objs),interpolation='none',cmap='gray',vmin=vabsmean-perabsmean,vmax=vabsmean+perabsmean)
    fig.colorbar(im1,ax=ax1)
    ax1.set_title(u'Object magnitude - projection {}'.format(idxproj))
    im2 = ax2.imshow(np.angle(objs),interpolation='none',cmap='bone')#,vmin=vphasemean-perabsmean,vmax=vphasemean+perabsmean)
    fig.colorbar(im2,ax=ax2)
    ax2.set_title(u'Object Phase - projection {}'.format(idxproj))
    # Special tricks for the probe display
    H = np.angle(probe)/(2*np.pi)+0.5
    S = np.ones_like(H).astype(int)
    V = np.abs(probe)/np.max(np.abs(probe))
    probe_hsv = np.dstack((H,S,V))
    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    norm = mpl.colors.Normalize(-np.pi,np.pi)
    cmap = mpl.cm.colors.hsv_to_rgb # TO BE FIXED
    #fig.subplots_adjust(hspace=plotgrid[0]/3.,wspace=plotgrid[1]/3.)
    #fig.tight_layout()
    im3 = ax3.imshow(hsv_to_rgb(probe_hsv),interpolation='none')
    fig.colorbar(im3,ax=ax3,cmap=mpl.cm.get_cmap('hsv'),norm=norm) # TO BE FIXED
    #cb = mpl.colorbar.ColorbarBase(ax3,cmap=mpl.cm.get_cmap('hsv'),norm=norm,orientation = 'horizontal')
    ax3.set_title('Probe - projection {}'.format(idxproj))  
    #plt.tight_layout()
    plt.draw()
    #plt.clf()

def show_linearphase(image,mask,*args):
    """
    Show projections and probe
    """
    try:
        idxproj=args[0]
    except:
        idxproj=''
        
    linecut = np.round(image.shape[0]/2.)
    
    fig, (ax1,ax2) = plt.subplots(num=3,nrows=2,ncols=1,figsize=(14,10))
    im1=ax1.imshow(image+mask,cmap='bone')
    ax1.set_title('Projection {}'.format(idxproj))
    im2=ax2.plot(image[linecut,:])
    ax2.plot([0,image.shape[1]],[0,0])
    ax2.axis('tight')
    plt.draw()
    #ax2.cla()
