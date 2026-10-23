#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np

import scipy.constants as consts


__all__ = [
    "convert_to_beta",
    "convert_to_delta",
    "convert_to_mu",
    "convert_to_rhoe",
    "convert_to_rhom",
]


def convert_to_mu(input_img, wavelen):
    r"""
    Convert image gray-levels from absorption index Beta to linear attenuation coefficient mu.

    Parameters
    ----------
    input_img : ndarray
        Image with gray-levels proportional to the absorption index Beta.
    wavelen : float
        X-ray wavelength in metres.

    Returns
    -------
    ndarray
        Image with gray-levels in units of linear attenuation coefficient
        mu (m\ :sup:`-1`).
    """
    return (4 * np.pi / wavelen) * input_img


def convert_to_rhoe(input_img, wavelen):
    r"""
    Convert image gray-levels from refractive index decrement delta to electron density.

    Parameters
    ----------
    input_img : ndarray
        Image with gray-levels proportional to delta.
    wavelen : float
        X-ray wavelength in metres.

    Returns
    -------
    ndarray
        Image with gray-levels in units of electron density
        (electrons m\ :sup:`-3`).
    """
    # classical electron radius
    r0 = consts.physical_constants["classical electron radius"][0]
    return (2 * np.pi / (r0 * wavelen ** 2)) * input_img


def convert_to_rhom(input_img, wavelen, A, Z):
    r"""
    Convert image gray-levels from electron density to mass density.

    Parameters
    ----------
    input_img : ndarray
        Image with gray-levels proportional to electron density.
    wavelen : float
        X-ray wavelength in metres.
    A : float
        Atomic mass number of the material.
    Z : float
        Atomic number of the material.

    Returns
    -------
    ndarray
        Image with gray-levels in units of mass density
        (g cm\ :sup:`-3`).
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
    Compute the multiplicative factor to convert image gray-levels to quantitative values.

    Parameters
    ----------
    input_img : ndarray
        Input image (used only to resolve ``voxelsize`` type; not modified).
    energy : float
        X-ray photon energy in keV.
    voxelsize : float, list of float, or ndarray
        Voxel size in metres.  If a list or array, only the first element
        is used.

    Returns
    -------
    factor : float
        Multiplicative conversion factor ``wavelen / (2 * pi * voxelsize)``.
    """
    if isinstance(voxelsize, list):
        if len(voxelsize) >= 1:
            voxelsize = voxelsize[0]
    elif isinstance(voxelsize, np.ndarray):
        if voxelsize.ndim !=0:
            voxelsize = voxelsize[0]

    wavelen = (12.4 / energy) * 1e-10  # in meters
    factor = wavelen / (2 * np.pi * voxelsize)

    # ~ if inputkwargs['pyhst']: #TODO: check conversion value in PyHSTs
    # ~ factor_pyhst = (2*np.pi/wavelen)*voxelsize[0]

    return factor


def convert_to_delta(input_img, energy, voxelsize):
    """
    Convert image gray-levels from phase-shifts to refractive index decrement delta.

    Parameters
    ----------
    input_img : ndarray
        Image with gray-levels proportional to accumulated phase shift.
    energy : float
        X-ray photon energy in keV.
    voxelsize : float, list of float, or ndarray
        Voxel size in metres.

    Returns
    -------
    delta_img : ndarray
        Image with gray-levels in units of delta.
    factor : float
        Conversion factor applied.
    """
    factor = _converter_factor(input_img, energy, voxelsize)
    return input_img * (-factor), factor


def convert_to_beta(input_img, energy, voxelsize, apply_log=False):
    """
    Convert image gray-levels from amplitude (or log-amplitude) to absorption index beta.

    Parameters
    ----------
    input_img : ndarray
        Image with gray-levels proportional to transmitted amplitude.
    energy : float
        X-ray photon energy in keV.
    voxelsize : float, list of float, or ndarray
        Voxel size in metres.
    apply_log : bool, optional
        If ``True``, apply the natural logarithm to ``input_img`` before
        the conversion (for images that have not yet been log-transformed).
        Default ``False``.

    Returns
    -------
    beta_img : ndarray
        Image with gray-levels in units of absorption index beta.
    factor : float
        Conversion factor applied.
    """
    factor = _converter_factor(input_img, energy, voxelsize)

    # In case the log has not yet been applied to the image
    if apply_log:
        input_img = np.log(input_img)
    return input_img * (-factor), factor
