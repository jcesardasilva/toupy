#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:22:54 2015

@author: jdasilva
"""
# standard packages
import time

# third party packages
import numpy as np

# local packages
from ..registration.register_translation_fast import register_translation
from .funcutils import deprecated

__all__=[
        'rmphaseramp',
        'rmlinearphase',
        'rmair']

def rmphaseramp(a, weight=None, return_phaseramp=False):
    """
    Auxiliary functions to attempt to remove the phase ramp in a
    two-dimensional complex array ``a``.

    Parameters
    ----------
    a : ndarray
        Input image as complex 2D-array.

    weight : ndarray, str, optional
        Pass weighting array or use ``'abs'`` for a modulus-weighted
        phaseramp and ``Non`` for no weights.

    return_phaseramp : bool, optional
        Use True to get also the phaseramp array ``p``.

    Returns
    -------
    out : ndarray
        Modified 2D-array, ``out=a*p``
    p : ndarray, optional
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
    elif weight == 'abs':
        weight = np.abs(a)

    ph = np.exp(1j*np.angle(a))
    [gx, gy] = np.gradient(ph)
    gx = -np.real(1j*gx/ph)
    gy = -np.real(1j*gy/ph)

    if useweight:
        nrm = weight.sum()
        agx = (gx*weight).sum() / nrm
        agy = (gy*weight).sum() / nrm
    else:
        agx = gx.mean()
        agy = gy.mean()

    (xx, yy) = np.indices(a.shape)
    p = np.exp(-1j*(agx*xx + agy*yy))

    if return_phaseramp:
        return a*p, p
    else:
        return a*p

def rmlinearphase(image,mask):
    """
    Removes linear phase from object

    Parameters
    ----------
    image : ndarray
        Input image
    mask : bool
        Boolean array with ones where the linear phase should be
                  computed from

    Returns
    -------
    im_output : ndarray
        Linear ramp corrected image
    """

    ph = np.exp(1j*np.angle(image))
    [gx, gy] = np.gradient(ph)
    gx = -np.real(1j*gx/ph)
    gy = -np.real(1j*gy/ph)

    nrm = mask.sum()
    agx = (gx*mask).sum() / nrm
    agy = (gy*mask).sum() / nrm

    (xx, yy) = np.indices(image.shape)
    p = np.exp(-1j*(agx*xx + agy*yy)) # ramp
    ph_corr = ph*p # correcting ramp
    ph_corr *= np.conj((ph_corr*mask).sum() / nrm) # taking the mask into account

    # applying to the image
    im_output = np.abs(image)*ph_corr
    ph_err = (mask*np.angle(ph_corr)**2).sum() / nrm

    return im_output #, ph_err

@deprecated
def remove_linearphase_old(image,mask,upsamp):
    """
    Removes linear phase from object considering only pixels where mask is
    unity, arrays have center on center of array
    Inputs:
        image  = Image
        mask   = Binary array with ones where the linear phase should be
                  computed from
        upsamp = Linear phase will be removed within 2*pi/upsamp peak to valley
                  in radians
    @author: Julio Cesar da Silva (e-mail:jdasilva@esrf.fr)

    Inspired by remove_linearphase.m created by Manuel Guizar-Sicairos in Aug 19th, 2010.
    Please, cite: Manuel Guizar-Sicairos, Ana Diaz, Mirko Holler, Miriam S. Lucas,
    Andreas Menzel, Roger A. Wepf, and Oliver Bunk, "Phase tomography from x-ray
    coherent diffractive imaging projections," Opt. Express 19, 21345-21357 (2011)
    """
    p0 = time.time()
    shift, error, diffphase = register_translation(np.fft.ifftshift(mask*np.abs(image)),np.fft.ifftshift(mask*image),upsamp)
    #shift, error, diffphase = register_translation(mask*np.abs(image),mask*image,upsamp)
    if shift[0]!=0 or shift[1]!=0:
        nr,nc = image.shape
        ar = np.arange(-np.floor(nr/2),np.ceil(nr/2))
        ac = np.arange(-np.floor(nc/2),np.ceil(nc/2))
        #~ Nr,Nc = fftfreq(nr),fftfreq(nc)
        #Nr,Nc = np.fft.ifftshift(fftfreq(nr)),np.fft.ifftshift(fftfreq(nc))
        #~ Nc,Nr = np.meshgrid(Nc,Nr)
        Nc,Nr = np.meshgrid(ac,ar) # FFT frequencies
        image*=np.exp(1j*2*np.pi*(-shift[0]*Nr/nr-shift[1]*Nc/nc))

    image*=np.exp(1j*diffphase)
    print("shifts: [{} , {}]".format(shift[0],shift[1]))
    print("Phase difference: {}".format(diffphase))
    print('Time elapsed: {} s'.format(time.time()-p0))
    return image#*np.exp(1j*diffphase)

def rmair(image,mask):
    """
    Correcting amplitude factor using the mask from the phase ramp removal
    considering only pixels where mask is
    unity, arrays have center on center of array
    Inputs:
        image  = Amplitude Image
        mask   = Binary array with ones where the linear phase should be
                  computed from
    @author: Julio Cesar da Silva (e-mail:jdasilva@esrf.fr)

    """
    norm_val = np.sum(mask*image)/mask.sum()
    print("Normalization value: {}".format(norm_val))
    return image/norm_val
