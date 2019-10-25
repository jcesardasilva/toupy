#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import matplotlib.pyplot as plt
import numpy as np

# local packages
from ..registration.shift import ShiftFunc
from ..utils.plot_utils import _plotdelimiters
from ..utils import progbar

__all__ = [
    "calculate_derivatives",
    "chooseregiontoderivatives",
    "derivatives",
    "derivatives_sino",
    "gradient_axis",
]


def gradient_axis(x, axis=-1):
    """
    Compute the gradient (keeping dimensions) along one dimension only.
    By default, the axis is -1 (diff along columns).
    """
    t1 = np.empty_like(x)
    t2 = np.empty_like(x)
    if axis != 0:
        t1[:, :-1] = x[:, 1:]
        t1[:, -1] = 0
        t2[:, :-1] = x[:, :-1]
        t2[:, -1] = 0
    else:
        t1[:-1, :] = x[1:, :]
        t1[-1, :] = 0
        t2[:-1, :] = x[:-1, :]
        t2[-1, :] = 0
    return t1 - t2


def chooseregiontoderivatives(stack_array, **params):
    """
    Choose the region to be unwrapped
    """
    # horizontal ROI
    deltax = params["deltax"]
    roix = range(deltax, stack_array.shape[2] - deltax)  # update roix
    roiy = range(*params["limsy"])  # tuple unpacking

    # Display the projections
    while True:
        plt.close("all")
        fig = plt.figure(5)
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(stack_array[0], cmap="bone")
        ax1 = _plotdelimiters(ax1, roiy, roix)
        ax1.axis("tight")
        plt.show(block=False)

        ans = input("Are you happy with the boundaries? ([y]/n)").lower()
        if str(ans) == "" or str(ans) == "y":
            break
        else:
            print(
                "The array dimensions are {} x {}".format(
                    stack_array[0].shape[0], stack_array[0].shape[1]
                )
            )
            print(stack_array.shape)
            while True:
                roiy = eval(input("Enter new range in y (top, bottom): "))
                if isinstance(roiy, tuple):
                    roiy = range(roiy[0], roiy[-1])
                    break
                else:
                    print("Wrong typing. Try it again.")
            while True:
                deltax = eval(
                    input("Enter new value from edge of region to edge of image in x: ")
                )
                if isinstance(deltax, int):
                    roix = range(deltax, stack_array.shape[2] - deltax)  # update roix
                    # roix = range(valx, aligned.shape[2] - valx)  # update roix
                    break
                else:
                    print("Wrong typing. Try it again.")

    return roix, roiy


def calculate_derivatives(stack_array, roiy, roix, shift_method="fourier"):
    """
    Compute projection derivatives
    """
    nprojs, nr, nc = stack_array.shape
    aligned_diff = np.empty_like(stack_array[:, roiy[0] : roiy[-1], roix[0] : roix[-1]])
    for ii in range(nprojs):
        strbar = "Derivative of projection {}".format(ii + 1)
        img = stack_array[ii, roiy[0] : roiy[-1], roix[0] : roix[-1]]
        aligned_diff[ii] = derivatives(img, shift_method)
        progbar(ii + 1, nprojs, strbar)

    return aligned_diff


def derivatives(input_array, shift_method="fourier"):
    """
    Calculate the derivative of an image

    Parameters
    ----------
    input_array: array_like
        Input image to calculate the derivatives
    shift_method: str
        Name of the shift method to use. For the available options, please
        see :class:`ShiftFunc()` in :mod:`toupy.registration`

    Returns
    -------
    diffimg : array_like
        Derivatives of the images along the row direction
    """
    S = ShiftFunc(shiftmeth=shift_method)
    rshift = [0, 0.5]
    lshift = [0, -0.5]
    diffimg = S(input_array, rshift) - S(input_array, lshift)
    # ~ if shift_method == 'phasor':
    # ~ diffimg = np.angle(S(np.exp(1j*input_array),rshift,'reflect',True)*S(np.exp(-1j*input_array),lshift,'reflect',True))
    return diffimg


def derivatives_sino(input_sino, shift_method="fourier"):
    """
    Calculate the derivative of the sinogram

    Parameters
    ----------
    input_array : array_like
        Input sinogram to calculate the derivatives
    shift_method : str
        Name of the shift method to use. For the available options, please
        see :class:`ShiftFunc()` in :mod:`toupy.registration`

    Returns
    -------
    diffsino : array_like
        Derivatives of the sinogram along the radial direction
    """
    rollsino = np.rollaxis(input_sino, 1)  # same as np.transpose(input_sino,1)
    rolldiff = derivatives(rollsino, shift_method)
    diffsino = np.rollaxis(rolldiff, 1)  # same as np.transpose(rolldiff,1)
    return diffsino
