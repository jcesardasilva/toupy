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
    """
    global P
    # using Silx Projector
    # , axis_position=my_axis_pos)
    P = Projection(recons.shape, angles=np.pi * (theta) / 180.0)
    sinogramcomp = P(recons.astype(np.float32)).T
    return sinogramcomp


def projector(recons, theta, **params):
    """
    Wrapper to choose between Forward Radon transform using Silx and
    OpenCL or standard reconstruction
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
