#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import functools
import multiprocessing

# third party packages
import numpy as np
from numpy.fft import fftfreq
import pyfftw  # has to be imported first to avoid ImportError: dlopen: cannot load any more object with static TLS
from scipy import fftpack

# enable cache for pyfftw
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(30)

__all__ = [
    "is_power2",
    "nextpoweroftwo",
    "nextpow2",
    "padwidthbothsides",
    "padrightside",
    "fastfftn",
    "fastifftn",
    "padfft",
]


def is_power2(num):
    """
    States if a number ``num`` is a power of two
    """
    return num != 0 and ((num & (num - 1)) == 0)


def nextpoweroftwo(number):
    """
    Returns next power of two following ``number``
    """
    return int(np.ceil(np.log2(number)))


def _nextpoweroftwo_print(number):
    """
    Return next power of two following ``number``
    """
    nextPower = int(np.ceil(np.log2(number)))
    return np.power(2, nextPower)


def nextpow2(number):
    """
    Find the next power 2 of ``number`` for FFT
    """
    n = 1
    while n < number:
        n *= 2
    return n


def padwidthbothsides(nbins):
    """
    Returns pad_width for padding both sides given a value of ``nbins``
    """
    # ~ nextPower = nextpoweroftwo(nbins)
    deficit = int(nextpow2(nbins) - nbins)
    # ~ deficit = int(np.power(2, nextPower) - nbins)
    return int(deficit / 2)


def padrightside(nbins):
    """
    Returns pad_width for padding at the right side given a value of ``nbins``
    The pad_width is calculated with ``next_fast_len`` function from `PyFFTW` package
    """
    # ~ nextPower = nextpoweroftwo(nbins)
    # ~ nextPower = nextpow2(nbins)
    # ~ nextPower = fftpack.next_fast_len(nbins)
    nextPower = pyfftw.next_fast_len(nbins)
    deficit = int(nextPower - nbins)
    # ~ deficit = int(np.power(2, nextPower) - nbins)
    return deficit


def metafftw(func):
    @functools.wraps(func)
    def wrapper(input_array):
        # checking number of cores available
        kwargs = dict()
        kwargs["ncores"] = multiprocessing.cpu_count()
        # stating the precision.
        # np.complex64: single precision; and np.complex128: double precision
        kwargs["cprecision"] = np.complex64
        kwargs["planner_type"] = "FFTW_MEASURE"
        return func(input_array, **kwargs)

    return wrapper


@metafftw
def fastfftn(input_array, **kwargs):
    """
    Auxiliary function to use pyFFTW. It does the align, planning and
    apply FFTW transform

    Parameters
    ---------
    input_array : array_like
        Array to be FFTWed

    Returns
    -------
    fftw_array : array_like
        Fourier transformed array

    """
    # number of cores available
    ncores = kwargs["ncores"]  # multiprocessing.cpu_count()
    # stating the precision.
    cprecision = kwargs["cprecision"]  # np.complex64 # single precision
    planner_type = kwargs["planner_type"]  # 'FFTW_MEASURE'
    # align array
    fftw_array = pyfftw.byte_align(input_array, dtype=cprecision, n=16)
    # will need to plan once
    fftw_array = pyfftw.interfaces.numpy_fft.fftn(
        fftw_array, overwrite_input=True, planner_effort=planner_type, threads=ncores
    )

    return fftw_array


@metafftw
def fastifftn(input_array, **kwargs):
    """
    Auxiliary function to use pyFFTW. It does the align, planning and
    apply inverse FFTW transform

    Parameters
    ---------
    input_array : array_like
        Array to be FFTWed

    Returns
    -------
    ifftw_array : array_like
        Inverse Fourier transformed array

    """
    # number of cores available
    ncores = kwargs["ncores"]  # multiprocessing.cpu_count()
    # stating the precision.
    cprecision = kwargs["cprecision"]  # np.complex64 # single precision
    planner_type = kwargs["planner_type"]  # 'FFTW_MEASURE'
    # align array
    ifftw_array = pyfftw.byte_align(input_array, dtype=cprecision, n=16)
    ifftw_array = pyfftw.interfaces.numpy_fft.ifftn(
        ifftw_array, overwrite_input=True, planner_effort=planner_type, threads=ncores
    )

    return ifftw_array


def padfft(input_array, pad_mode="reflect"):
    """
    Auxiliary function to pad arrays for Fourier transforms. It accepts
    1D and 2D arrays.

    Parameters
    ---------
    input_array : array_like
        Array to be padded
    mode : str
        Padding mode to treat the array borders. See :py:mod:`numpy.pad` 
        for modes. The default value is `reflect`.

    Returns
    -------
    array_pad : array_like
        Padded array
    N_pad : array_like
        padded frequency coordinates
    padw : int, list of ints
        pad width
    """
    # padding to reduce artifacts with FFTs
    if input_array.ndim == 1:
        nr = len(input_array)
        padw = padrightside(nr)  # next power of 2
        # ~ padw = padwidthbothsides(nr) # next power of 2
        # ~ array_pad = np.pad(input_array,(padw,padw),mode=pad_mode)
        array_pad = np.pad(input_array, (0, padw), mode=pad_mode)
        N_pad = fftfreq(len(array_pad))
    elif input_array.ndim == 2:
        nr, nc = input_array.shape
        padw = [padrightside(nr), padrightside(nc)]
        # ~ padw = [padwidthbothsides(nr), padwidthbothsides(nc)]
        # ~ array_pad = np.pad(input_array,((padw[0],padw[0]),(padw[1],padw[1])),mode=pad_mode)
        array_pad = np.pad(input_array, ((0, padw[0]), (0, padw[1])), mode=pad_mode)
        n_pad = [fftfreq(array_pad.shape[0]), fftfreq(array_pad.shape[1])]
        # reverted order to be compatible with meshgrid output
        N_pad = np.meshgrid(n_pad[1], n_pad[0])
    return array_pad, N_pad, padw
