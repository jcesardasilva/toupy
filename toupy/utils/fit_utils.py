#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np
from scipy.optimize import leastsq
from scipy.special import erf, erfc

__all__ = [
    "model_erf",
    "model_tanh",
    "residuals_erf",
    "residuals_tanh",
]


def model_erf(t, *coeffs):
    """
    Model for the erf fitting

    P0 + P1*t + (P2/2)*(1-erf(sqrt(2)*(x-P3)/(P4)))
    
    Parameters
    ----------
    t : ndarray
        Input coordinates
    coeffs[0] : float
        P0 (noise)
    coeffs[1] : float
        P1 (linear term)
    coeffs[2] : float
        P2 (Maximum amplitude)
    coeffs[3] : float
        P3 (center)
    coeffs[4] : float
        P4 (width)

    Returns
    -------
    ndarray
        Array containing the model
    """
    return (
        coeffs[0]
        + coeffs[1] * t
        + (coeffs[2] / 2.0) * (1 + erf(np.sqrt(2) * (t - coeffs[3]) / (coeffs[4])))
    )


def model_tanh(t, *coeffs):
    """
    Model for the erf fitting

    P0 + P1*t + (P2/2)*(1-tanh(sqrt(2)*(x-P3)/P4))
    
    Parameters
    ----------
    t : ndarray
        Input coordinates
    coeffs[0] : float
        P0 (noise)
    coeffs[1] : float
        P1 (linear term)
    coeffs[2] : float
        P2 (Maximum amplitude)
    coeffs[3] : float
        P3 (center)
    coeffs[4] : float
        P4 (width)

    Returns
    -------
    ndarray
        Array containing the model

    """
    return (
        coeffs[0]
        + coeffs[1] * t
        + (coeffs[2] / 2.0) * (1 - np.tanh((t - coeffs[3]) / coeffs[4]))
    )


def residuals_erf(coeffs, y, t):
    """
    Residuals for the least-squares optimization
    coeffs as the ones of the model erf function

    Parameters
    ----------
    y : ndarray
        The data
    t : ndarray
        Input coordinates

    Returns
    -------
    ndarray
        Residuals
    """
    return y - model_erf(t, *coeffs)


def residuals_tanh(coeffs, y, t):
    """
    Residuals for the least-squares optimization
    coeffs as the ones of the model tanh function
    
    Parameters
    ----------
    y : ndarray
        The data
    t : ndarray
        Input coordinates

    Returns
    -------
    ndarray
        Residuals
    """
    return y - model_tanh(t, *coeffs)


