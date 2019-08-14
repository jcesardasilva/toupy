#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np
from silx.opencl.projection import Projection
from skimage.transform import radon

__all__ = ["radonSilx", "projector"]

P = None


def radonSilx(recons, theta):
    """
    Forward Radon transform using Silx and OpenCL

    Parameters
    ----------
    recons : array_like
        Array containing the tomographic slice
    theta : array_like
        Array of thetas

    Return
    ------
    sinogramcomp : array_like
        Reprojected sinogram
    """
    global P
    # using Silx Projector
    # , axis_position=my_axis_pos)
    P = None
    P = Projection(recons.shape, angles=np.pi * (theta) / 180.0)
    sinogramcomp = P(recons.astype(np.float32)).T
    return sinogramcomp


def projector(recons, theta, **params):
    """
    Wrapper to choose between Forward Radon transform using Silx and
    OpenCL or standard reconstruction. 

    Parameters
    ----------
    recons : array_like
        Array containing the tomographic slice
    theta : array_like
        Array of thetas
    params : dict
        Dictionary of parameters to be used
    params["opencl"] : bool
        If True, it will perform the tomographic reconstruction using
        the opencl implementation of Silx.

    Return
    ------
    sinogramcomp : array_like
        Reprojected sinogram
    """
    # array shape
    N = recons.shape[0]
    center = int(N / 2)
    if params["opencl"]:
        # using Silx Projector
        # print("Using OpenCL")
        sinogramcomp = radonSilx(recons, theta)
    else:
        # Not using Silx Projector (very slow)
        # print("Not using OpenCL")
        sinogramcomp = radon(recons, theta, circle=True)
    sinogramcomp = np.squeeze(sinogramcomp)
    return sinogramcomp
