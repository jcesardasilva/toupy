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
    Auxiliary class to initialize some variables
    """

    shift_method = "linear"
    padmod = "reflect"
    complexoutput = False
    splineorder = 3


class ShiftFunc(Variables):
    """
    Collections of shift fuctions
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
        Implement the shifts

        Parameters
        ----------
        *args:
            args[0] : array_like
                Input array
            args[1] : int or tuple
                Shift amplitude
            args[2] : str (optional)
                Padding mode if necessary
            args[3] : bool (optional)
                True for complex output or False for real output
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
        TODO: Check if same functionality than shift_linear for integer shifts

        Parameters
        ----------
        input_array : array_like
            Input image to calculate the shifts.
        shift : int or tuple
            Number of pixels to shift. For 1D, use a integer value. 
            For 2D, use a tuple of integers where the first value 
            corresponds to shifts in the rows and the second value 
            corresponds to shifts in the columns.

        Return
        ------
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

        Return
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

        Return
        ------
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

        Return
        ------
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
