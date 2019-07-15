#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import multiprocessing

# third party packages
import numpy as np
from numpy.fft import fftfreq
import pyfftw # has to be imported first to avoid ImportError: dlopen: cannot load any more object with static TLS
import scipy.constants as consts

# enable cache for pyfftw
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(30)

__all__=['crop',
         'mask_borders',
         'convert_to_mu',
         'convert_to_rhoe',
         'convert_to_rhom']

def crop(input_array,delcropx,delcropy):
    """
    Crop images
    Inputs:
        input_array: input image to be cropped
        **params: dict of parameters
    @author: Julio C. da Silva (jdasilva@esrf.fr)
    """
    if delcropx is not None or delcropy is not None:
        print('Cropping ROI of data')
        print('Before: '+input_array.shape)
        print(input_array[delcropy:-delcropy,delcropx:-delcropx].shape)
        if input_array.ndim == 2:
            return input_array[delcropy:-delcropy,delcropx:-delcropx]
        elif input_array.ndim == 3:
            return input_array[:,delcropy:-delcropy,delcropx:-delcropx]
        print('After: '+input_array.shape)
    else:
        print('No cropping of data')
        return input_array

def mask_borders(imgarray,mask_array,threshold=4e-7):
    # mask borders
    gr,gc = np.gradient(imgarray)
    mask_border = np.sqrt(gr**2+gc**2)>threshold
    mask_array *= (~mask_border)
    return mask_array

def convert_to_mu(input_img,wavelen):
    return (4*np.pi/wavelen)*input_img

def convert_to_rhoe(input_img,wavelen):
    # classical electron radius
    r0 = consts.physical_constants['classical electron radius'][0]
    return (2*np.pi/(r0*wavelen**2))*input_img

def convert_to_rhom(input_img,wavelen,A,Z):
    # Avogadro's Constant
    Na = consts.N_A # not used yet
    # classical electron radius
    r0 = consts.physical_constants['classical electron radius'][0]
    # ratio A/Z
    A_Z = A/Z
    #return 1e-6*(2*np.pi*A_Z/(r0*Na*wavelen**2))*input_img
    return 1e-6*(input_img/Na)*(A_Z)
