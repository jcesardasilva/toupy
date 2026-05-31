#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np
from scipy.ndimage import interpolation

# local packages
from ..utils import deprecated, switch
from ..utils import fastfftn, fastifftn, padfft

__all__ = ["ShiftFunc"]


class Variables(object):
    """
    Auxiliary class that stores default parameter values for shift operations.

    Attributes
    ----------
    shift_method : str
        Default shift method.  ``'linear'``.
    padmod : str
        Default padding mode.  ``'reflect'``.
    complexoutput : bool
        Whether to return a complex-valued array.  ``False``.
    splineorder : int
        Spline interpolation order for the spline shift method.  ``3``.
    """

    shift_method = "linear"
    padmod = "reflect"
    complexoutput = False
    splineorder = 3


class ShiftFunc(Variables):
    """
    Collection of shift functions for 1-D and 2-D arrays.

    The desired method is selected at construction time via the
    ``shiftmeth`` parameter and exposed through ``__call__`` so that the
    same interface works regardless of the underlying algorithm.

    Parameters
    ----------
    **params
        Must contain:

        shiftmeth : str
            Shift algorithm to use.  One of ``'linear'`` (bilinear
            interpolation), ``'fourier'`` (FFT-based sub-pixel shift via
            pyFFTW), or ``'spline'`` (spline interpolation via
            :func:`scipy.ndimage.interpolation.shift`).
    """

    def __init__(self, **params):
        super().__init__()
        self.shift_method = params["shiftmeth"]
        if self.shift_method == "linear":
            self.shiftmeth = self.shift_linear
        elif self.shift_method == "fourier":
            self.shiftmeth = self.shift_fft
            self.padmode = "reflect"
            self.complexoutput = False
        elif self.shift_method == "spline":
            self.shiftmeth = self.shift_spline_wrap
        else:
            raise ValueError("Unknown shift method")

    def __call__(self, *args):  # input_array,shift):
        """
        Apply the selected shift method to an array.

        Parameters
        ----------
        *args
            args[0] : array_like
                Input 1-D or 2-D array.
            args[1] : float or tuple of float
                Shift amplitude in pixels.  For 1-D arrays pass a scalar;
                for 2-D arrays pass ``(row_shift, col_shift)``.
            args[2] : str, optional
                Padding mode (overrides the instance default).
            args[3] : bool, optional
                ``True`` to return a complex array; ``False`` (default) to
                return a real array.

        Returns
        -------
        ndarray
            Shifted array with the same shape as ``args[0]``.  Returns the
            input unchanged when the shift is zero.
        """

        self.input_array = args[0]
        self.shift = np.array(args[1])  # to be consistent with directions

        if len(args) == 3:
            padmode = args[2]
        else:
            self.padmode = "reflect"

        if len(args) == 4:
            complexoutput = args[3]
        else:
            self.complexoutput = False

        if np.count_nonzero(self.shift) is 0:
            return self.input_array
        else:
            self.ndim = self.input_array.ndim
            if self.ndim > 2:
                raise ValueError("Only implemented for 1D and 2D arrays")
            self.n = self.input_array.shape
            return self.shiftmeth(self.input_array, self.shift)

    def _shift_pseudo_linear(self, input_array, shift):
        """
        Shifts an image with wrap around. Performs pixel shift using
        numpy.roll. Simpler than shift_linear.

        Parameters
        ----------
        input_array : array_like
            Input image to calculate the shifts.
        shift : int or tuple
            Number of pixels to shift. For 1D, use a integer value.
            For 2D, use a tuple of integers where the first value
            corresponds to shifts in the rows and the second value
            corresponds to shifts in the columns.

        Returns
        -------
        output_array : array_like
            Shifted image
        """
        shiftfloor = np.floor(shift).astype(int)
        if input_array.ndim == 1:
            output_array = np.roll(input_array, -shift)
        if input_array.ndim == 2:
            rows, cols = shift
            output_array = np.roll(np.roll(input_array, -rows, axis=0), -cols, axis=1)
        return output_array

    def shift_linear(self, input_array, shift):
        """
        Shifts an image with wrap around and bilinear interpolation

        Parameters
        ----------
        input_array : array_like
            Input image to calculate the shifts.
        shift : int or tuple
            Number of pixels to shift. For 1D, use a integer value.
            For 2D, use a tuple of integers where the first value
            corresponds to shifts in the rows and the second value
            corresponds to shifts in the columns.

        Returns
        -------
        output_array : array_like
            Shifted image
        """
        # pixel shift
        shiftfloor = np.floor(shift).astype(int)
        output_array = self._shift_pseudo_linear(input_array, shiftfloor)

        # Subpixel (bilinear)
        tau = shift - shiftfloor

        if np.count_nonzero(tau) is not 0:
            # Subpixel (bilinear)
            if input_array.ndim == 1:
                taux = tau
                output_array = (
                    output_array * (1 - taux)
                    + self._shift_pseudo_linear(output_array, 1) * taux
                )
            elif input_array.ndim == 2:
                tauy, taux = tau
                output_array = (
                    output_array * (1 - tauy) * (1 - taux)
                    + self._shift_pseudo_linear(output_array, (1, 0))
                    * tauy
                    * (1 - taux)
                    + self._shift_pseudo_linear(output_array, (0, 1))
                    * (1 - tauy)
                    * taux
                    + self._shift_pseudo_linear(output_array, (1, 1)) * tauy * taux
                )
        return output_array

    def shift_fft(self, input_array, shift):
        """
        Performs pixel and subpixel shift (with wraping) using pyFFTW.

        Since FFTW has efficient functions for array sizes which can be
        decompose in prime factor, the input_array is padded to the next
        fast size given by pyFFTW.next_fast_len.
        The padding is done in mode = 'reflect' by default to reduce
        border artifacts.

        Parameters
        ----------
        input_array : array_like
            Input image to calculate the shifts.
        shift : int or tuple
            Number of pixels to shift. For 1D, use a integer value.
            For 2D, use a tuple of integers where the first value
            corresponds to shifts in the rows and the second value
            corresponds to shifts in the columns.

        Returns
        -------
        output_array : array_like
            Shifted image
        """
        ni = input_array.shape
        pad_array, N, padw = padfft(input_array, self.padmode)  # padding
        fftw_input_array = fastfftn(pad_array)  # forward FFT
        if input_array.ndim == 1:  # 1D array case
            shiftr = shift
            H = np.exp(1j * 2 * np.pi * ((shiftr * N)))
            output_array = fastifftn((fftw_input_array) * H)[: ni[0]]
            # ~ output_array = fastifftn((fftw_input_array)*H)[padw:-padw]
        elif input_array.ndim == 2:  # 2D array case
            Nc, Nr = N  # reverted order to be compatible with meshgrid output
            shift_rows, shift_cols = shift
            H = np.exp(1j * 2 * np.pi * ((shift_rows * Nr) + (shift_cols * Nc)))
            output_array = fastifftn((fftw_input_array) * H)[: ni[0], : ni[1]]
            # ~ output_array = fastifftn((fftw_input_array)*H)[padw[0]:-padw[0],padw[1]:-padw[1]]
        if not self.complexoutput:
            output_array = output_array.real  # TODO: this is weird, to be checked
            # ~ output_array = np.angle(np.exp(1j*output_array))
        return output_array

    def shift_spline_wrap(self, input_array, shift):
        """
        Performs pixel and subpixel shift (with wraping) using splines

        Parameters
        ----------
        input_array : array_like
            Input image to calculate the shifts.
        shift : int or tuple
            Number of pixels to shift. For 1D, use a integer value.
            For 2D, use a tuple of integers where the first value
            corresponds to shifts in the rows and the second value
            corresponds to shifts in the columns.

        Returns
        -------
        output_array : array_like
            Shifted image
        """
        if input_array.ndim == 1:
            # 1D array case
            shift_rows = shift
            output_array = interpolation.shift(
                input_array, -shift_rows, order=self.splineorder, mode=self.padmode
            )
        elif input_array.ndim == 2:
            # 2D array case
            shift_rows, shift_cols = shift
            output_array = interpolation.shift(
                input_array,
                (-shift_rows, -shift_cols),
                order=self.splineorder,
                mode=self.padmode,
            )
        return output_array
