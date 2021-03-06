#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unwrap_phase

# local packages
from ..utils.plot_utils import _plotdelimiters
from ..utils import progbar

__all__ = [
    "wraptopi",
    "wrap",
    "distance",
    "get_charge",
    "phaseresidues",
    "chooseregiontounwrap",
    "unwrapping_phase"
    # ~ u'goldstein_unwrap2D'
]


def wraptopi(phase, endpoint=True):
    """
    Wrap a scalar value or an entire array
    
    Parameters
    ----------
    phase : float or array_like
        The value or signal to wrapped.
    endpoint : bool, optional 
        If ``endpoint=False``, the scalar value or array is wrapped 
        to [-pi, pi), whereas if ``endpoint=True``, it is wrapped to (-pi, pi].
        The default value is ``endpoint=True``
    
    Returns
    -------
    float or array
        Wrapped value or array        

    Example
    -------
    >>> import numpy as np
    >>> wraptopi(np.linspace(-np.pi,np.pi,7),endpoint=True)
    array([ 3.14159265, -2.0943951 , -1.04719755, -0.        ,  1.04719755,
        2.0943951 ,  3.14159265])
    >>> wraptopi(np.linspace(-np.pi,np.pi,7),endpoint=False)
    array([-3.14159265, -2.0943951 , -1.04719755,  0.        ,  1.04719755,
        2.0943951 , -3.14159265])
    """
    if not endpoint:  # case [-pi, pi)
        return (phase + np.pi) % (2 * np.pi) - np.pi
    else:  # case (-pi, pi]
        return ((-phase + np.pi) % (2.0 * np.pi) - np.pi) * -1.0


def wrap(phase):
    """
    Wrap a scalar value or an entire array to [-0.5, 0.5).
    
    Parameters
    ----------
    phase : float or array_like
        The value or signal to wrapped.
    
    Returns
    -------
    float or array
        Wrapped value or array

    Note
    ----
    Created by Sebastian Theilenberg, PyMRR, which is available at
    Github repository: https://github.com/theilen/PyMRR.git
    """
    if hasattr(phase, "__len__"):
        phase = phase.copy()
        phase[phase > 0.5] -= 1.0
        phase[phase <= -0.5] += 1.0
    else:
        if phase > 0.5:
            phase -= 1.0
        elif phase <= -0.5:
            phase += 1.0
    return phase


def distance(pixel1, pixel2):
    """
    Return the Euclidean distance of two pixels.

    Example
    -------
    >>> distance(np.arange(1,10),np.arange(2,11))
    3.0
    """
    if (not isinstance(pixel1, np.ndarray)) and (not isinstance(pixel2, np.ndarray)):
        pixel1 = np.asarray(pixel1)
        pixel2 = np.asarray(pixel2)
    return np.sqrt(np.sum((pixel1 - pixel2) ** 2))


def get_charge(residues):
    """
    Get the residues charges

    Parameters
    ----------
    residues : ndarray
        A 2-dimensional array containing the with residues

    Returns
    -------
    posres : array_like
        Positions of the residues with positive charge
    negres : array_like
        Positions of the residues with negative charge
    """
    posres = np.where(np.round(residues) == 1)
    respos = len(posres[0])
    negres = np.where(np.round(residues) == -1)
    resneg = len(negres[0])

    nres = respos + resneg
    print("Found {:>3.0f} residues".format(nres), end="")

    return posres, negres


def phaseresidues(phimage):
    """
    Calculates the phase residues [1]_ for a given wrapped phase image.

    Parameters
    ----------
    phimage : ndarray
        A 2-dimensional array containing the phase-contrast images with gray-level
        in radians

    Returns
    -------
    residues : ndarray
        A 2-dimensional array containing the map of residues (valued +1 or -1)

    Note
    -----
    Note that by convention the positions of the phase residues are
    marked on the top left corner of the 2 by 2 regions as shown below:

    .. graphviz::
    
        graph g {
            node [shape=plaintext];
            active -- right [label="  res4   "];
            right  -- belowright [label=" res3 "];
            below  -- belowright [label=" res2 "];
            below  -- active [label=" res1 "];
            { rank=same; active right }
            { rank=same; belowright below }
        }

    Inspired by PhaseResidues.m created by B.S. Spottiswoode on 07/10/2004
    and by find_residues.m created by Manuel Guizar - Sept 27, 2011

    References
    ----------
    .. [1] R. M. Goldstein, H. A. Zebker and C. L. Werner,
      Radio Science 23, 713-720 (1988).
    """
    residues = wraptopi(phimage[2:, 1:-1] - phimage[1:-1, 1:-1])
    residues += wraptopi(phimage[2:, 2:] - phimage[2:, 1:-1])
    residues += wraptopi(phimage[1:-1, 2:] - phimage[2:, 2:])
    residues += wraptopi(phimage[1:-1, 1:-1] - phimage[1:-1, 2:])
    residues /= 2 * np.pi

    respos, resneg = get_charge(residues)
    residues_charge = dict(pos=respos, neg=resneg)

    return residues, residues_charge


def phaseresiduesStack(stack_array):
    """
    Calculate the map of residues on the stack

    Parameters
    ----------
    stack_array : ndarray
        A 3-dimensional array containing the stack of projections
        from which to calculate the phase residues.

    Returns
    -------
    resmap : array_like
        Phase residue map
    posres : tuple
        Positions of the residues in the format ``posres = (yres,xres)``
    """
    resmap = 0
    for ii in range(stack_array.shape[0]):
        print(
            "\rSearching for residues in projection {:>4.0f} ... ".format(ii + 1),
            end="",
        )
        residues, residues_charge = phaseresidues(stack_array[ii])
        resmap += np.abs(residues)
    print(". Done")
    posres = np.where(resmap >= 1.0)
    return resmap, posres


def chooseregiontounwrap(stack_array):
    """
    Choose the region to be unwrapped

    Parameters
    ----------
    stack_array : ndarray
        A 3-dimensional array containing the stack of projections
        to be unwrapped.

    Returns
    -------
    rx, ry : tuple
        Limits of the are to be unwrapped
    airpix : tuple
        Position of the pixel which should contains only air/vacuum
    """
    resmap, posres = phaseresiduesStack(stack_array)
    yres, xres = posres

    # display the residues
    plt.close("all")
    plt.figure(1)
    plt.imshow(resmap, cmap="jet")
    plt.axis("tight")
    plt.plot(xres, yres, "or")
    plt.show(block=False)

    # choosing the are for the unwrapping
    while True:
        plt.ion()
        fig = plt.figure(2)
        plt.clf()
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(stack_array[0], cmap="bone")
        ax1.plot(xres, yres, "or")
        ax1.axis("tight")
        plt.show(block=False)
        print(
            "The array dimensions are {} x {}".format(
                stack_array[0].shape[0], stack_array[0].shape[1]
            )
        )
        print("Please, choose an area for the unwrapping:")
        # while loops in each question to avoid mistapying problems
        while True:
            deltax = eval(input("From edge of region to edge of image in x: "))
            if isinstance(deltax, int):
                rx = range(deltax, stack_array.shape[2] - deltax)
                break
            else:
                print("Wrong typing. Try it again.")
        while True:
            ry = eval(input("Range in y (top, bottom): "))
            if isinstance(ry, tuple):
                ry = range(ry[0], ry[-1])
                break
            else:
                print("Wrong typing. Try it again.")
        while True:
            airpix = eval(input("Pixel in air (x,y) or (col,row): "))
            if isinstance(airpix, tuple):
                # check if air pixel is inside the image
                if (
                    airpix[0] < rx[0]
                    or airpix[0] > rx[-1]
                    or airpix[1] < ry[0]
                    or airpix[1] > ry[-1]
                ):
                    print("Pixel of air is outside of region of unwrapping")
                    print("Wrong typing. Try it again.")
                else:
                    break
            else:
                print("Wrong typing. Try it again.")

        # couting residues
        num_residues = int(np.round(resmap[ry[0] : ry[-1], rx[0] : rx[-1]].sum()))
        print("Chosen region contains {} residues in total".format(num_residues))

        # update images with boudaries
        ax1 = _plotdelimiters(ax1, ry, rx, airpix)
        ax1.set_title("First projection with boundaries")
        plt.show(block=False)

        ans = input("Are you happy with the boundaries?([y]/n)").lower()
        if str(ans) == "" or str(ans) == "y":
            break

    return rx, ry, airpix


def _unwrapping_phase(img2unwrap, rx=[], ry=[], airpix=[]):
    """
    Unwrap the phases of a projection

    Parameters
    ----------
    img2unwrap : ndarray
        A 2-dimensional array containing the image to be unwrapped
    rx, ry : tuple or list of ints
        Limits of the are to be unwrapped in x and y
    airpix : tuple or list of ints
        Position of pixel in the air/vacuum area

    Returns
    -------
    img2unwrap : array_like
        Unwrapped image
    """
    if rx == [] and ry == []:
        img2unwrap = unwrap_phase(im2unwrap)
        img2unwrap -= -2 * np.pi * np.round(img2unwrap / (2 * np.pi))
    else:
        # select the region to be unwrapped
        img2wrap_sel = img2unwrap[ry[0] : ry[-1], rx[0] : rx[-1]]
        # unwrap the region using the algorithm from skimage
        img2unwrap_sel = unwrap_phase(img2wrap_sel)
        # update the image in the original array
        img2unwrap[ry[0] : ry[-1], rx[0] : rx[-1]] = img2unwrap_sel
        img2unwrap[ry[0] : ry[-1], rx[0] : rx[-1]] = (
            img2unwrap_sel
            - 2 * np.pi * np.round(img2unwrap[airpix[1], airpix[0]] / (2 * np.pi))
        )

    return img2unwrap


def unwrapping_phase(stack_phasecorr, rx, ry, airpix, **params):
    """
    Unwrap the phase of the projections in a stack.

    Parameters
    ----------
    stack_phasecorr : ndarray
        A 3-dimensional array containing the stack of projections to be unwrapped
    rx, ry : tuple or list of ints
        Limits of the are to be unwrapped in x and y
    airpix : tuple or list of ints
        Position of pixel in the air/vacuum area
    params : dict
        Dictionary of additional parameters
    params["vmin"] : float, None
        Minimum value for the gray level at each display
    params["vmin"] : float, None
        Maximum value for the gray level at each display

    Returns
    -------
    stack_unwrap : ndarray
        A 3-dimensional array containing the stack of unwrapped projections

    Note
    ----
    It uses the phase unwrapping algorithm by Herraez et al. [#skimage]_
    implemented in Scikit-Image (https://scikit-image.org).
    
    References
    ----------
    .. [#skimage] Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor,
      and Munther A. Gdeisat, “Fast two-dimensional phase-unwrapping algorithm 
      based on sorting by reliability following a noncontinuous path”, 
      Journal Applied Optics, Vol. 41, No. 35, pp. 7437, 2002
    """
    stack_unwrap = np.empty_like(stack_phasecorr)
    # test on first projection
    print("Testing unwrapping on the first projection")
    img0_unwrap = _unwrapping_phase(stack_phasecorr[0], rx, ry, airpix)
    # displaying
    plt.close("all")
    plt.ion()
    fig = plt.figure(7)
    ax1 = fig.add_subplot(111)
    im1 = ax1.imshow(
        stack_phasecorr[0], cmap="bone", vmin=params["vmin"], vmax=params["vmax"]
    )
    # update images with boudaries
    ax1 = _plotdelimiters(ax1, ry, rx, airpix)
    ax1.axis("tight")
    plt.show(block=False)
    while True:
        a = input("Do you want to edit the color scale?([y]/n)").lower()
        if str(a) == "" or str(a) == "y":
            while True:
                color_vmin = eval(input("Minimum color scale value: "))
                if isinstance(color_vmin, int) or isinstance(color_vmin, float):
                    break
                else:
                    print("Wrong typing. Try it again.")
            while True:
                color_vmax = eval(input("Maximum color scale value: "))
                if isinstance(color_vmax, int) or isinstance(color_vmax, float):
                    break
                else:
                    print("Wrong typing. Try it again.")
            params["vmin"] = color_vmin
            params["vmax"] = color_vmax
            print("Using vmin={} and vmax={}".format(params["vmin"], params["vmax"]))
            # displaying the update images
            im1.set_data(stack_phasecorr[0])
            im1.set_clim(params["vmin"], params["vmax"])
            ax1 = _plotdelimiters(ax1, ry, rx, airpix)
            ax1.axis("tight")
            plt.show(block=False)
        else:
            print(
                "Color scale was not changed. Using vmin={} and vmax={}".format(
                    params["vmin"], params["vmax"]
                )
            )
            break
    # main loop for the unwrapping
    nprojs = stack_phasecorr.shape[0]
    for ii in range(nprojs):
        strbar = "Unwrapping projection: {}".format(ii + 1)
        img_unwrap = _unwrapping_phase(stack_phasecorr[ii], rx, ry, airpix)
        stack_unwrap[ii] = img_unwrap  # update the stack
        progbar(ii + 1, nprojs, strbar)
    print("\r")

    return stack_unwrap


# TODO: fix function below
# ~ def goldstein_unwrap2D(phimage,disp=0):
# ~ """
# ~ Implementation of Goldstein unwrap algorithm based on location of
# ~ residues and introduction of branchcuts.
# ~ Inputs:
# ~ phimage = Wrapped phase image in radians, wrapped between (-pi,pi)
# ~ disp (optional) = 1 to show progress (will slow down code)
# ~ will also display the branch cuts
# ~ Outputs:
# ~ unwrap_phase =    Unwrapped phase ( = fase where phase could not be unwrapped)
# ~ shadow    = 1 where phase could not be unwrapped
# ~
# ~ Inpired in the goldstein_unwrap2D.m by Manuel Guizar 31 August, 2010 - Acknowledge if used
# ~ Please, cite: R. M. Goldstein, H. A. Zebker and C. L. Werner, Radio Science 23, 713-720 (1988).
# ~ """
# ~
# ~ nr,nc = phimage.shape
# ~ #position to start unwrapping. Typically faster at the center of the array
# ~ #nrstart = np.round(nr/2.)
# ~ #ncstart = np.round(nc/2.)
# ~
# ~ residues,_ = phaseresidues(phimage,disp=1)
# ~
# ~ ## Find residues
# ~ pposr,pposc = np.where(np.round(residues)==1)
# ~ respos = [pposr,pposc,np.ones_like(pposr)]
# ~ ###respos= len(pposr)
# ~ nposr,nposc = np.where(np.round(residues)==-1)
# ~ resneg = [nposr,nposc,-np.ones_like(pposr)]
# ~ ###resneg = len(nposr)
# ~
# ~ nres = len(respos[:][0])+len(resneg[:][0])
# ~ ###nres = respos+resneg
# ~ print('Found {} residues'.format(nres))
# ~
# ~ if nres == 0:
# ~ print('No residues found. Unwrapping with standard unwrapping algorithm')
# ~ unwrap_phase = np.unwrap(np.unwrap(phimage))
# ~ shadow = np.zeros_like(unwrap_phase)
# ~ else:
# ~ print('Unwrapping with Goldstein algorithm')
# ~ return unwrap_phase,shadow
