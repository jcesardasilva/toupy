#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard packages
import time

# third party packages
from ..utils.plot_utils import plt
import numpy as np


# local packages
from ..restoration.ramptools import rmphaseramp
from ..restoration.unwraptools import phaseresidues

__all__ = [
    "cart2pol",
    "get_object_novort",
    "get_probe_novort",
    "pol2cart",
    "rmvortices_object",
    "rmvortices_probe",
]


def cart2pol(x, y):
    """
    Change from cartesian to polar coordinates

    Parameters
    ----------
    x, y : array_like
        Values in cartesian coordinates
    Returns
    -------
    rho, phi : array_like
        Values in polar coordinates
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """
    Change from polar to cartesian coordinates

    Parameters
    ----------
    rho, phi : array_like
        Values in polar coordinates

    Returns
    -------
    x, y : array_like
        Values in cartesian coordinates
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def get_probe_novort(img_phase, residues):
    """
    Remove the vortices from the probe

    Parameters
    ----------
    img_phase : array_like
        Probe image with vortices to be removed without linear phase ramp
    residues : array_like
        Residues map

    Returns
    -------
    img_phase_novort : array_like
        Probe image without vortices
    xres, yres : array_like
        Coordinates `x` and `y` of the vortices
    """
    n, m = residues.shape
    print("Getting vortice positions...")
    yres, xres = np.nonzero(np.abs(residues) > 0.1)
    # Precompute coordinate grids once (outside the loop)
    X, Y = np.mgrid[: (m + 2), : (n + 2)]
    # X and Y need to be transposed
    X = X.T
    Y = Y.T
    # Initialise output and accumulate corrections in-place
    img_phase_novort = img_phase.copy()
    for ii in range(len(xres)):
        print("{} residues out of {}".format(ii + 1, len(xres)))
        s0 = time.time()
        yr = yres[ii]
        xr = xres[ii]
        T = np.arctan2(Y - yr, X - xr)
        rr = residues[yr, xr]
        img_phase_novort *= np.exp(-1j * T * rr)
        print("Time elapsed = {} s".format(time.time() - s0))
    return img_phase_novort, xres, yres


def rmvortices_probe(img_in, to_ignore=100):
    """
    Remove phase vortices on the probe image ignoring border pixels.

    Parameters
    ----------
    img_in : array_like
        Complex probe image with vortices to be removed.
    to_ignore : int, optional
        Number of pixels to ignore from each border.  Default ``100``.

    Returns
    -------
    img_phase_novort : array_like
        Probe image without vortices.
    xres : ndarray
        X-coordinates (columns) of the detected vortices.
    yres : ndarray
        Y-coordinates (rows) of the detected vortices.

    Notes
    -----
    Any linear phase ramp present in the input image is removed before
    and after the vortex correction.
    """
    # remove phase ramp
    img_pr = rmphaseramp(img_in)  # [101:-100,93:-93] # to get square image
    # find residues
    residues, residues_charge = phaseresidues(np.angle(img_pr))

    # ignore borders
    if to_ignore != 0:
        print("Ignoring border")
        residues[:to_ignore, :] = 0
        residues[:, :to_ignore] = 0
        residues[-to_ignore:, :] = 0
        residues[:, -to_ignore:] = 0

    # initialize array
    img_phase_vort = img_pr.copy().astype(np.complex64)

    # remove vortices
    img_phase_novort, xres, yres = get_probe_novort(img_phase_vort, residues)

    img_phase_novort = rmphaseramp(img_phase_novort)

    return img_phase_novort, xres, yres


def get_object_novort(img_phase, residues):
    """
    Remove the vortices from the phase projections

    Parameters
    ----------
    img_phase : array_like
        Phase image with vortices to be removed without linear phase ramp
    residues : array_like
        Residues map

    Returns
    -------
    img_phase_novort : array_like
        Phase image without vortices
    xres, yres : array_like
        Coordinates `x` and `y` of the vortices
    """
    n, m = residues.shape
    print("Getting vortice positions...")
    yres, xres = np.nonzero(np.abs(residues) > 0.1)
    # Precompute coordinate grids once (outside the loop)
    X, Y = np.mgrid[: (m + 2), : (n + 2)]
    # X and Y need to be transposed
    X = X.T
    Y = Y.T
    # Initialise output and accumulate corrections in-place
    img_phase_novort = img_phase.copy()
    for ii in range(len(xres)):
        print("{} residues out of {}".format(ii + 1, len(xres)))
        s0 = time.time()
        yr = yres[ii]
        xr = xres[ii]
        T = np.arctan2(Y - yr, X - xr)
        rr = residues[yr, xr]
        img_phase_novort *= np.exp(1j * T * rr)
        print("Time elapsed = {} s".format(time.time() - s0))

    return img_phase_novort, xres, yres


def rmvortices_object(img_in, to_ignore=100):
    """
    Remove phase vortices on the object image ignoring border pixels.

    Parameters
    ----------
    img_in : array_like
        Complex phase image with vortices to be removed.
    to_ignore : int, optional
        Number of pixels to ignore from each border.  Default ``100``.

    Returns
    -------
    img_phase_novort : array_like
        Phase image without vortices.
    xres : ndarray
        X-coordinates (columns) of the detected vortices.
    yres : ndarray
        Y-coordinates (rows) of the detected vortices.

    Notes
    -----
    Any linear phase ramp present in the input image is removed before
    and after the vortex correction.
    """
    # remove phase ramp
    img_pr = rmphaseramp(img_in)  # [101:-100,93:-93] # to get square image
    # find residues
    residues, residues_charge = phaseresidues(np.angle(img_pr))

    # ignore borders
    if to_ignore != 0:
        print("Ignoring border")
        residues[:to_ignore, :] = 0
        residues[:, :to_ignore] = 0
        residues[-to_ignore:, :] = 0
        residues[:, -to_ignore:] = 0

    # initialize array
    img_phase_vort = img_pr.copy().astype(np.complex64)

    # remove vortices
    img_phase_novort, xres, yres = get_object_novort(img_phase_vort, residues)

    img_phase_novort = rmphaseramp(img_phase_novort)

    return img_phase_novort, xres, yres


def rmvortices_slow(img_in, to_ignore=100):
    """
    Remove phase vortices on the object image (slow reference implementation).

    Parameters
    ----------
    img_in : array_like
        Complex phase image with vortices to be removed.
    to_ignore : int, optional
        Number of pixels to ignore from each border.  Default ``100``.

    Returns
    -------
    img_phase_novort : array_like
        Phase image without vortices.
    xres : ndarray
        X-coordinates (columns) of the detected vortices.
    yres : ndarray
        Y-coordinates (rows) of the detected vortices.

    Notes
    -----
    Possibly deprecated and will be removed in a future version.
    Prefer :func:`rmvortices_object` for production use.
    """
    # remove phase ramp
    img_pr = rmphaseramp(img_in)  # to get square image
    # find residues
    residues, residues_charge = phaseresidues(np.angle(img_pr))

    # ignore borders
    if to_ignore != 0:
        print("Ignoring border")
        residues[:to_ignore, :] = 0
        residues[:, :to_ignore] = 0
        residues[-to_ignore:, :] = 0
        residues[:, -to_ignore:] = 0

    img_res = np.zeros_like(np.angle(img_pr))
    img_res[1:-1, 1:-1] = residues.copy()

    # get array of vortice positions
    print("Getting vortice positions...")
    # yres, xres = np.nonzero(np.abs(residues) > 0.1)
    yres, xres = np.where(np.abs(residues) > 0.1)
    print("Found {}".format(len(xres)))
    # remove the vortices
    img_phase_novort = img_pr.copy().astype(np.complex64)
    n, m = img_res.shape
    x = np.arange(m)
    y = np.arange(n)
    for idx, ii in enumerate(range(len(xres))):
        print("{} residues out of {}".format(idx, len(xres)))
        s0 = time.time()
        X, Y = np.meshgrid(x - xres[ii], y - yres[ii])
        R, T = cart2pol(X, Y)
        # img_phase_novort *= np.exp(-1j*T*residues[yres[ii],xres[ii]])
        img_phase_novort *= np.exp(1j * T * residues[yres[ii], xres[ii]])
        print("Time elapsed = {} s".format(time.time() - s0))
    return img_phase_novort, xres, yres
