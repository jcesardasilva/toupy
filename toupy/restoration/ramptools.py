#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard packages
import time

# third party packages
import numpy as np

# local packages
from ..utils.funcutils import deprecated

__all__ = ["rmphaseramp", "rmlinearphase", "rmair"]


def rmphaseramp(a, weight=None, return_phaseramp=False):
    """
    Auxiliary functions to attempt to remove the phase ramp in a
    two-dimensional complex array ``a``.

    Parameters
    ----------
    a : array_like
        Input image as complex 2D-array.

    weight : array_like, str, optional
        Pass weighting array or use ``'abs'`` for a modulus-weighted
        phaseramp and ``Non`` for no weights.

    return_phaseramp : bool, optional
        Use True to get also the phaseramp array ``p``.

    Returns
    -------
    out : array_like
        Modified 2D-array, ``out=a*p``
    p : array_like, optional
        Phaseramp if ``return_phaseramp = True``, otherwise omitted

    Note
    ----
    Function forked from Ptypy.plot_utils (https://github.com/ptycho/ptypy)
    and ported to Python 3.

    Examples
    --------
    >>> b = rmphaseramp(image)
    >>> b, p = rmphaseramp(image , return_phaseramp=True)
    """
    useweight = True
    if weight is None:
        useweight = False
    elif weight == "abs":
        weight = np.abs(a)

    ph = np.exp(1j * np.angle(a))
    [gx, gy] = np.gradient(ph)
    gx = -np.real(1j * gx / ph)
    gy = -np.real(1j * gy / ph)

    if useweight:
        nrm = weight.sum()
        agx = (gx * weight).sum() / nrm
        agy = (gy * weight).sum() / nrm
    else:
        agx = gx.mean()
        agy = gy.mean()

    (xx, yy) = np.indices(a.shape)
    p = np.exp(-1j * (agx * xx + agy * yy))

    if return_phaseramp:
        return a * p, p
    else:
        return a * p


def rmlinearphase(image, mask):
    """
    Removes linear phase from object

    Parameters
    ----------
    image : array_like
        Input image
    mask : bool
        Boolean array with ones where the linear phase should be
        computed from

    Returns
    -------
    im_output : array_like
        Linear ramp corrected image
    """

    ph = np.exp(1j * np.angle(image))
    [gx, gy] = np.gradient(ph)
    gx = -np.real(1j * gx / ph)
    gy = -np.real(1j * gy / ph)

    nrm = mask.sum()
    agx = (gx * mask).sum() / nrm
    agy = (gy * mask).sum() / nrm

    (xx, yy) = np.indices(image.shape)
    p = np.exp(-1j * (agx * xx + agy * yy))  # ramp
    ph_corr = ph * p  # correcting ramp
    # taking the mask into account
    ph_corr *= np.conj((ph_corr * mask).sum() / nrm)

    # applying to the image
    im_output = np.abs(image) * ph_corr
    # ph_err = (mask * np.angle(ph_corr) ** 2).sum() / nrm

    return im_output  # , ph_err


def rmair(image, mask):
    """
    Correcting amplitude factor using the mask from the phase ramp
    removal considering only pixels where mask is  unity, arrays have
    center on center of array

    Parameters
    ---------
    image : array_like
        Amplitude-contrast image
    mask  : bool
        Boolean array with indicating the locations from where the air
        value should be obtained

    Returns
    -------
    normalizedimage : array_like
        Image normalized by the air values
    """
    norm_val = np.sum(mask * image) / mask.sum()
    print("Normalization value: {}".format(norm_val))
    return image / norm_val
