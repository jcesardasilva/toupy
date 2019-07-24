#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np
from silx.opencl.projection import Projection
from skimage.transform import radon

__all__=['radonSilx', 'FBP_projector']

P = None
def radonSilx(recons,theta):
    """
    Forward Radon transform using Silx and OpenCL
    @author: jdasilva
    """
    # using Silx Projector
    P = Projection(recons.shape, angles=np.pi*(theta)/180.)#, axis_position=my_axis_pos)
    sinogramcomp = P(recons.astype(np.float32)).T
    return sinogramcomp

def projector(recons,theta,**params):
    """
    Wrapper to choose between Forward Radon transform using Silx and
    OpenCL or standard reconstruction
    """
    # array shape
    N = recons.shape[0]
    center = int(N/2)
    if params['opencl']:
        # using Silx Projector
        #print("Using OpenCL")
        sinogramcomp = radonSilx(recons,theta)
    else:
        # Not using Silx Projector (very slow)
        #print("Not using OpenCL")
        sinogramcomp = radon(recons,theta,circle=True)
    sinogramcomp = np.squeeze(sinogramcomp)
    return sinogramcomp
