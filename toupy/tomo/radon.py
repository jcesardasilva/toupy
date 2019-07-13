#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np
from silx.opencl.projection import Projection

__all__=['radonSilx']

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
