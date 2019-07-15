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

__all__=[u'is_power2',
         u'fastfftn',
         u'fastifftn',
         u'nextpoweroftwo',
         u'padwidthbothsides',
         u'padfft'
         ]

def is_power2(num):
    """
    states if a number is a power of two
    """
    return num != 0 and ((num & (num - 1)) == 0)

def nextpoweroftwo(number):
    """
    Returns next power of two following 'number'
    """
    return int(np.ceil(np.log2(number)))

def padwidthbothsides(nbins):
    """
    Returns pad_width for padding both sides
    """
    nextPower = nextpoweroftwo(nbins)
    deficit = int(np.power(2, nextPower) - nbins)
    return int(deficit/2)

def fastfftn(input_array):
    """
    Auxiliary function to use pyFFTW. It does the align, planning and
    apply FFTW transform

    Parameters
    ---------
    input_array: ndarray
        Array to be FFTWed

    Returns
    -------
    fftw_array : ndarray
        Fourier transformed array

    Note: It is fast for array sizes which are power of 2
    """
    # checking number of cores available
    ncores = multiprocessing.cpu_count()
    # stating the precision.
    # np.complex64: single precision; and np.complex128: double precision
    cprecision = np.complex64 # single precision
    planner_type = 'FFTW_MEASURE'
    ## align array
    fftw_array = pyfftw.byte_align(input_array,dtype=cprecision,n=16)
    ## will need to plan once
    fftw_array = pyfftw.interfaces.numpy_fft.fftn(fftw_array, overwrite_input=True, planner_effort=planner_type, threads=ncores)
    return fftw_array

def fastifftn(input_array):
    """
    Auxiliary function to use pyFFTW. It does the align, planning and
    apply inverse FFTW transform

    Parameters
    ---------
    input_array: ndarray
        Array to be FFTWed

    Returns
    -------
    ifftw_array : ndarray
        Inverse Fourier transformed array

    Note: It is fast for array sizes which are power of 2
    """
    # checking number of cores available
    ncores = multiprocessing.cpu_count()
    # stating the precision.
    # np.complex64: single precision; and np.complex128: double precision
    cprecision = np.complex64 # single precision
    planner_type = 'FFTW_MEASURE'
    # align array
    ifftw_array = pyfftw.byte_align(input_array,dtype=cprecision,n=16)
    ifftw_array = pyfftw.interfaces.numpy_fft.ifftn(ifftw_array, overwrite_input=True, planner_effort=planner_type, threads=ncores)
    return ifftw_array

def padfft(input_array,pad_mode='reflect'):
    """
    Auxiliary function to pad arrays for Fourier transforms. It accepts
    1D and 2D arrays.

    Parameters
    ---------
    input_array : ndarray
        Array to be padded
    mode : str (default = 'reflect')
        Padding mode to treat the array borders. See np.pad for modes.

    Returns
    -------
    array_pad : ndarray
        Padded array
    N_pad : ndarray
        padded frequency coordinates
    padw : int or list of ints
        pad width
    """
    #padding to reduce artifacts with FFTs
    if input_array.ndim == 1:
        nr = len(input_array)
        padw = padwidthbothsides(nr) # next power of 2
        array_pad = np.pad(input_array,(padw,padw),mode=pad_mode)
        N_pad = fftfreq(len(array_pad))
    elif input_array.ndim == 2:
        nr, nc = input_array.shape
        padw = [padwidthbothsides(nr), padwidthbothsides(nc)]
        array_pad = np.pad(input_array,((padw[0],padw[0]),(padw[1],padw[1])),mode=pad_mode)
        n_pad = [fftfreq(array_pad.shape[0]),fftfreq(array_pad.shape[1])]
        N_pad = np.meshgrid(n_pad[1],n_pad[0]) # reverted order to be compatible with meshgrid output
    return array_pad, N_pad, padw
