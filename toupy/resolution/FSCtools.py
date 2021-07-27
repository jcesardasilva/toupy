#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FOURIER SHELL CORRELATION
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
from ..utils import isnotebook
from ..tomo import tomo_recons

__all__ = ["compute_2tomograms", "compute_2tomograms_splitted", "split_dataset"]


def split_dataset(sinogram, theta):
    """
    Split the tomographic dataset in 2 datasets

    Parameters
    ----------
    sinogram : ndarray
        A 2-dimensional array containing the sinogram
    theta : ndarray
        A 1-dimensional array of thetas

    Returns
    -------
    sinogram1 : ndarray
        A 2-dimensional array containing the 1st sinogram
    sinogram2
        A 2-dimensional array containing the 2nd sinogram
    theta1 : ndarray
        A 1-dimensional array containing the 1st set of thetas
    theta2 : ndarray
        A 1-dimensional array containing the 2nd set of thetas
    """
    # split of the data into two datasets
    # print("Spliting in 2 datasets")
    sinogram1 = sinogram[:, 0::2]
    theta1 = theta[0::2]
    sinogram2 = sinogram[:, 1::2]
    theta2 = theta[1::2]

    return sinogram1, sinogram2, theta1, theta2


def compute_2tomograms(sinogram, theta, **params):
    """
    Split the tomographic dataset in 2 datasets and
    compute 2 tomograms from them.

    Parameters
    ----------
    sinogram : ndarray
        A 2-dimensional array containing the sinogram
    theta : ndarray
        A 1-dimensional array of thetas

    Returns
    -------
    recon1 : ndarray
        A 2-dimensional array containing the 1st reconstruction
    recon2
        A 2-dimensional array containing the 2nd reconstruction
    """
    sino1, sino2, theta1, theta2 = split_dataset(sinogram, theta)

    # tomographic reconstruction
    print("Calculating a slice 1...")
    t0 = time.time()
    recon1 = tomo_recons(sino1, theta1, **params)
    print("Calculation done. Time elapsed: {} s".format(time.time() - t0))

    print("Calculating a slice 2...")
    t0 = time.time()
    recon2 = tomo_recons(sino2, theta2, **params)
    print("Calculation done. Time elapsed: {} s".format(time.time() - t0))

    return recon1, recon2


def compute_2tomograms_splitted(sinogram1, sinogram2, theta1, theta2, **params):
    """
    Compute 2 tomograms from already splitted tomographic dataset

    Parameters
    ----------
    sinogram1 : ndarray
        A 2-dimensional array containing the sinogram 1
    sinogram2 : ndarray
        A 2-dimensional array containing the sinogram 2
    theta1 : ndarray
        A 1-dimensional array of thetas for sinogram1
    theta2 : ndarray
        A 1-dimensional array of thetas for sinogram2

    Returns
    -------
    recon1 : ndarray
        A 2-dimensional array containing the 1st reconstruction
    recon2
        A 2-dimensional array containing the 2nd reconstruction
    """

    # tomographic reconstruction
    if not isnotebook():
        print("Calculating a slice 1...")
    t0 = time.time()
    recon1 = tomo_recons(sinogram1, theta1, **params)
    if not isnotebook():
        print("Calculation done. Time elapsed: {} s".format(time.time() - t0))

    if not isnotebook():
        print("Calculating a slice 2...")
    t0 = time.time()
    recon2 = tomo_recons(sinogram2, theta2, **params)
    if not isnotebook():
        print("Calculation done. Time elapsed: {} s".format(time.time() - t0))

    return recon1, recon2
