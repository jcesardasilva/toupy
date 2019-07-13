#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import multiprocessing

# third party packages
import numpy as np
from numpy.fft import fftfreq
import pyfftw # has to be imported first to avoid ImportError: dlopen: cannot load any more object with static TLS

# enable cache for pyfftw
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(30)

__all__=['crop']

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
