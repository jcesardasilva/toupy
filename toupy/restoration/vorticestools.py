#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard packages
import time

# third party packages
import numpy as np
import matplotlib.pyplot as plt
import h5py

# local packages
from ..restoration.ramptools import rmphaseramp
from ..restoration.unwraptools import phaseresidues

__all__ = ['cart2pol',
           'pol2cart',
           'rmvortices']


def cart2pol(x, y):
    """
    Change from cartesian to polar coordinates

    Parameters
    ----------
    x, y : ndarrays
        Values in cartesian coordinates
    Returns
    -------
    rho, phi: ndarrays
        Values in polar coordinates
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    """
    Change from polar to cartesian coordinates

    Parameters
    ----------
    rho, phi: ndarrays
        Values in polar coordinates

    Returns
    -------
    x, y : ndarrays
        Values in cartesian coordinates
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def rmvortices(img_in, to_ignore=100):
    # remove phase ramp
    img_pr = rmphaseramp(img_in)  # [101:-100,93:-93] # to get square image
    # find residues
    residues, residues_charge = phaseresidues(np.angle(img_pr))

    # ignore borders
    if to_ignore != 0:
        print('Ignoring border')
        residues[:to_ignore, :] = 0
        residues[:, :to_ignore] = 0
        residues[-to_ignore:, :] = 0
        residues[:, -to_ignore:] = 0

    img_res = np.zeros_like(np.angle(img_pr))
    img_res[1:-1, 1:-1] = residues.copy()

    # get array of vortice positions
    yres, xres = np.where(np.abs(residues) > 0.1)
    print('Getting vortice positions...')
    print('Found {}'.format(len(xres)))
    # remove the vortices
    img_phase_novort = img_pr.copy().astype(np.complex64)
    n, m = img_res.shape
    x = np.arange(m)
    y = np.arange(n)
    for idx, ii in enumerate(xrange(len(xres))):
        print('{} residues out of {}'.format(idx, len(xres)))
        s0 = time.time()
        X, Y = np.meshgrid(x-xres[ii], y-yres[ii])
        R, T = cart2pol(X, Y)
        #img_phase_novort *= np.exp(-1j*T*residues[yres[ii],xres[ii]])
        img_phase_novort *= np.exp(1j*T*residues[yres[ii], xres[ii]])
        print('Time elapsed = {} s'.format(time.time()-s0))
    return img_phase_novort, xres, yres
