#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np

import scipy.constants as consts


__all__=['convert_to_mu',
         'convert_to_rhoe',
         'convert_to_rhom']

def convert_to_mu(input_img,wavelen):
    return (4*np.pi/wavelen)*input_img

def convert_to_rhoe(input_img,wavelen):
    # classical electron radius
    r0 = consts.physical_constants['classical electron radius'][0]
    return (2*np.pi/(r0*wavelen**2))*input_img

def convert_to_rhom(input_img,wavelen,A,Z):
    # Avogadro's Constant
    Na = consts.N_A # not used yet
    # classical electron radius
    r0 = consts.physical_constants['classical electron radius'][0]
    # ratio A/Z
    A_Z = A/Z
    #return 1e-6*(2*np.pi*A_Z/(r0*Na*wavelen**2))*input_img
    return 1e-6*(input_img/Na)*(A_Z)
