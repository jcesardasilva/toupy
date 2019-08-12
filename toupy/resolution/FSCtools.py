#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FOURIER SHELL CORRELATION tools
"""

# standar packages
import time

# third party package
import h5py
import matplotlib.pyplot as plt
import numpy as np

# local packages
from ..utils.FFT_utils import fastfftn
from ..utils.funcutils import checkhostname
from ..tomo import tomo_recons

__all__ = ["compute_2tomograms"]

def _split_dataset(sinogram, theta):
    """
    Split the tomographic dataset in 2 datasets
    """
    # split of the data into two datasets
    print("Spliting in 2 datasets")
    sinogram1 = sinogram[:, 0::2]
    theta1 = theta[0::2]
    sinogram2 = sinogram[:, 1::2]
    theta2 = theta[1::2]
    
    return sinogram1, sinogram2, theta1, theta2

def compute_2tomograms(sinogram, theta, **params)
    """
    Compute 2 tomograms from splitted tomographic dataset
    """
    sino1, sino2, theta1, theta2 = _split_dataset(sinogram, theta)

    # tomographic reconstruction
    print("Calculating a slice 1...")
    t0 = time.time()
    recon1 = tomo_recons(sino1, theta1, **params)
    print("Calculation done. Time elapsed: {} s".format(time.time() - t0))
    
    print("Calculating a slice 2...")
    t0 = time.time()s
    recon2 = tomo_recons(sino2, theta2, **params)
    print("Calculation done. Time elapsed: {} s".format(time.time() - t0))

    return recon1, recon2
