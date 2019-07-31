#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np

import scipy.constants as consts


__all__ = [
    "convert_to_mu",
    "convert_to_rhoe",
    "convert_to_rhom",
    "convert_to_delta",
    "convert_to_beta"
    ]


def convert_to_mu(input_img, wavelen):
    """
    Converts the image gray-levels from absoption index Beta to linear
    attenuation coefficient mu
    """
    return (4 * np.pi / wavelen) * input_img


def convert_to_rhoe(input_img, wavelen):
    """
    Converts the image gray-levels from phase shifts to electron density
    """
    # classical electron radius
    r0 = consts.physical_constants["classical electron radius"][0]
    return (2 * np.pi / (r0 * wavelen ** 2)) * input_img


def convert_to_rhom(input_img, wavelen, A, Z):
    """
    Converts the image gray-levels from electron density to mass density
    """
    # Avogadro's Constant
    Na = consts.N_A  # not used yet
    # classical electron radius
    r0 = consts.physical_constants["classical electron radius"][0]
    # ratio A/Z
    A_Z = A / Z
    # return 1e-6*(2*np.pi*A_Z/(r0*Na*wavelen**2))*input_img
    return 1e-6 * (input_img / Na) * (A_Z)

       
def _converter_factor(input_img, energy, voxelsize):
    """
    Yields the factor to convert image gray-levels to quantitative valuess
    """
    if isinstance(voxelsize,list) or isinstance(voxelsize,np.ndarray):
        if len(voxelsize) >= 1:
            voxelsize = voxelsize[0]

    wavelen = (12.4 / energy) * 1e-10  # in meters
    factor = wavelen / (2 * np.pi * voxelsize)

    # ~ if inputkwargs['pyhst']: #TODO: check conversion value in PyHSTs
    # ~ factor_pyhst = (2*np.pi/wavelen)*voxelsize[0]

    return factor


def convert_to_delta(input_img, energy, voxelsize):
    """
    Converts the image gray-levels from phase-shifts to delta
    """
    factor = _converter_factor(input_img, energy, voxelsize)
    return input_img*(-factor), factor
    

def convert_to_beta(input_img, energy, voxelsize, apply_log=False):
    """
    Converts the image gray-levels from amplitude to beta
    """
    factor = _converter_factor(input_img, energy, voxelsize)

    # In case the log has not yet been applied to the image
    if apply_log:
        input_img = np.log(input_img)
    return input_img*(-factor), factor
