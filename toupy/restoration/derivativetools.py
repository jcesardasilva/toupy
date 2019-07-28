#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np

# local packages
from ..registration.shift import ShiftFunc

__all__ = ['derivatives',
           'derivatives_sino']


def derivatives(input_array, shift_method='fourier'):
    """
    Calculate the derivative of an image

    Parameters
    ----------
    input_array: ndarray
        Input image to calculate the derivatives
    shift_method: str
        Name of the shift method to use. Available options:
        'sinc', 'linear'

    Returns
    -------
    diffimg : ndarray
        Derivatives of the images along the row direction
    """
    S = ShiftFunc(shiftmeth=shift_method)
    rshift = [0, 0.5]
    lshift = [0, -0.5]
    diffimg = S(input_array, rshift) - S(input_array, lshift)
    # ~ if shift_method == 'phasor':
    # ~ diffimg = np.angle(S(np.exp(1j*input_array),rshift,'reflect',True)*S(np.exp(-1j*input_array),lshift,'reflect',True))
    return diffimg


def derivatives_sino(input_sino, shift_method='fourier'):
    """
    Calculate the derivative of the sinogram

    Parameters
    ----------
    input_array : ndarray
        Input sinogram to calculate the derivatives
    shift_method : str
        Name of the shift method to use. Available options:
        'fourier', 'linear', 'spline'

    Returns
    -------
    diffsino : ndarray
        Derivatives of the sinogram along the radial direction
    """
    rollsino = np.rollaxis(input_sino, 1)  # same as np.transpose(input_sino,1)
    rolldiff = derivatives(rollsino, shift_method)
    diffsino = np.rollaxis(rolldiff, 1)  # same as np.transpose(rolldiff,1)
    return diffsino
