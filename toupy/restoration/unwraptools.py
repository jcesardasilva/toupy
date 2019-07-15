#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np

__all__=[
        'wraptopi',
        'wrap',
        'distance',
        'get_charge',
        'phaseresidues'
        #~ u'goldstein_unwrap2D'
        ]

def wraptopi(phase,endpoint=True):
    """
    Wrap a scalar value or an entire array to:
    [-pi, pi) if endpoint=False
    (-pi, pi] if endpoint=True (default)
    Example:
    >>> import numpy as np
    >>> wraptopi(np.linspace(-np.pi,np.pi,7),endpoint=True)
    array([ 3.14159265, -2.0943951 , -1.04719755, -0.        ,  1.04719755,
        2.0943951 ,  3.14159265])
    >>> wraptopi(np.linspace(-np.pi,np.pi,7),endpoint=False)
    array([-3.14159265, -2.0943951 , -1.04719755,  0.        ,  1.04719755,
        2.0943951 , -3.14159265])
    Created 07/10/2015
    """
    if not endpoint: # case [-pi, pi)
        return ( phase + np.pi) % (2 * np.pi ) - np.pi
    else: # case (-pi, pi]
        return (( -phase + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0

def wrap(phase):
    """
    Wrap a scalar value or an entire array to -0.5 <= a < 0.5.
    Created by Sebastian Theilenberg, PyMRR
    Github repository: https://github.com/theilen/PyMRR.git
    """
    if hasattr(phase, '__len__'):
        phase = phase.copy()
        phase[phase > 0.5] -= 1.
        phase[phase <= -0.5] += 1.
    else:
        if phase > 0.5:
            phase -= 1.
        elif phase <= -0.5:
            phase += 1.
    return phase

def distance(pixel1, pixel2):
    """
    Return the Euclidean distance of two pixels.
    Example:
    >>> distance(np.arange(1,10),np.arange(2,11))
    3.0
    Created 26/11/2015
    """
    if (not isinstance(pixel1,np.ndarray)) and (not isinstance(pixel2,np.ndarray)):
        pixel1 = np.asarray(pixel1)
        pixel2 = np.asarray(pixel2)
    return np.sqrt(np.sum((pixel1-pixel2)**2))

def get_charge(residues):
    """
    Get the residues charges
    Parameters
    ----------
    residues : ndarray
        2D arrays with residues
    Returns
    -------
    posres : ndarray
        Positions of the residues with positive charge
    negres : ndarray
        Positions of the residues with negative charge
    """
    posres = np.where(np.round(residues)==1)
    respos = len(posres[0])
    negres = np.where(np.round(residues)==-1)
    resneg = len(negres[0])

    nres = respos+resneg
    print('Found {} residues'.format(nres))

    return posres, negres

def phaseresidues(phimage,disp=1):
    """
    Calculates the phase residues for a given wrapped phase image. 

    Parameters
    ----------
    phimage : ndarray
        Array containing the phase-contrast images with gray-level 
        in radians
    disp : bool
        False -> No feedback
        True ->  Text feedback (additional computation)
    
    Returns
    -------
    residues : ndarray
        Map of residues (valued +1 or -1)
    
    Notes
    -----
    Note that by convention the positions of the phase residues are
    marked on the top left corner of the 2 by 2 regions.

      active---res4---right
         |              |
        res1           res3
         |              |
      below---res2---belowright
    Inspired by PhaseResidues.m created by B.S. Spottiswoode on 07/10/2004
    and by find_residues.m created by Manuel Guizar - Sept 27, 2011
    Relevant literature: R. M. Goldstein, H. A. Zebker and C. L. Werner,
    Radio Science 23, 713-720(1988).
    """
    residues =  wraptopi(phimage[2:,1:-1]   - phimage[1:-1,1:-1])
    residues += wraptopi(phimage[2:,2:]     - phimage[2:,1:-1])
    residues += wraptopi(phimage[1:-1,2:]   - phimage[2:,2:])
    residues += wraptopi(phimage[1:-1,1:-1] - phimage[1:-1,2:])
    residues /= (2*np.pi)

    respos,resneg = get_charge(residues)
    residues_charge = dict(
        pos = respos,
        neg = resneg
        )

    return residues,residues_charge

#~ def goldstein_unwrap2D(phimage,disp=0):
    #~ """
    #~ Implementation of Goldstein unwrap algorithm based on location of
    #~ residues and introduction of branchcuts.
    #~ Inputs:
        #~ phimage = Wrapped phase image in radians, wrapped between (-pi,pi)
        #~ disp (optional) = 1 to show progress (will slow down code)
                 #~ will also display the branch cuts
    #~ Outputs:
        #~ unwrap_phase =    Unwrapped phase ( = fase where phase could not be unwrapped)
        #~ shadow    = 1 where phase could not be unwrapped
    #~
    #~ Inpired in the goldstein_unwrap2D.m by Manuel Guizar 31 August, 2010 - Acknowledge if used
    #~ Please, cite: R. M. Goldstein, H. A. Zebker and C. L. Werner, Radio Science 23, 713-720 (1988).
    #~ """
#~
    #~ nr,nc = phimage.shape
    #~ #position to start unwrapping. Typically faster at the center of the array
    #~ #nrstart = np.round(nr/2.)
    #~ #ncstart = np.round(nc/2.)
    #~
    #~ residues,_ = phaseresidues(phimage,disp=1)
    #~
    #~ ## Find residues
    #~ pposr,pposc = np.where(np.round(residues)==1)
    #~ respos = [pposr,pposc,np.ones_like(pposr)]
    #~ ###respos= len(pposr)
    #~ nposr,nposc = np.where(np.round(residues)==-1)
    #~ resneg = [nposr,nposc,-np.ones_like(pposr)]
    #~ ###resneg = len(nposr)
#~
    #~ nres = len(respos[:][0])+len(resneg[:][0])
    #~ ###nres = respos+resneg
    #~ print('Found {} residues'.format(nres))
#~
    #~ if nres == 0:
        #~ print('No residues found. Unwrapping with standard unwrapping algorithm')
        #~ unwrap_phase = np.unwrap(np.unwrap(phimage))
        #~ shadow = np.zeros_like(unwrap_phase)
    #~ else:
        #~ print('Unwrapping with Goldstein algorithm')
    #~ return unwrap_phase,shadow
