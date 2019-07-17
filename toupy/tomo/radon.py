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
        print("Using OpenCL")
        sinogramcomp = radonSilx(recons,theta)
    else:
        # Not using Silx Projector (very slow)
        print("Not using OpenCL")
        sinogramcomp = radon(recons,theta,circle=True)
    # calculate the derivative or not of the sinogram
    Nbig = np.asarray(sinogramcomp).shape[0]
    centerbig = int(Nbig/2) #np.ceil(Nbig/2.)#np.floor((Nbig+1)/2.)
    if params[u'derivatives']: # if derivatives is used
        sinogramcomp = derivatives_sino(sinogramcomp,shift_method='sinc')
    else:
        sinogramcomp = np.squeeze(sinogramcomp)
    delta_center = centerbig-center
    sinogramcomp = sinogramcomp[delta_center:N+delta_center,:]
    return sinogramcomp
