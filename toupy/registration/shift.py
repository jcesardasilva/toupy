#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np
from scipy.ndimage import interpolation

# local packages
from ..utils.funcutils import deprecated,switch
from ..utils.FFT_utils import fastfftn, fastifftn, padfft

__all__=['ShiftFunc']

class Variables(object):
    """
    Auxiliary class to initialize some variables
    """
    shift_method = 'linear'
    padmod = 'reflect'
    complexoutput = False
    splineorder = 3
    
class ShiftFunc(Variables):
    """
    Collections of shift fuctions
    """
    def __init__(self,**params):
        super().__init__()
        shift_method = params['shiftmeth']
        if shift_method =='linear':
            self.shiftmeth = self.shift_linear
        elif shift_method =='fourier':
            self.shiftmeth = self.shift_fft
            self.padmode = 'reflect'
            self.complexoutput = False
        elif shift_method =='spline':
            self.shiftmeth = self.shift_spline_wrap
        else:
            raise ValueError('Unknown interpolation method')

    def __call__(self,*args): #input_array,shift):
        """
        Implement the shifts
        *args:
            args[0] : ndarray
                Input array
            args[1] : int or tuple
                Shift amplitude
            args[2] : str (optional)
                Padding mode if necessary
            args[3] : bool (optional)
                True for complex output or False for real output
        """

        self.input_array = args[0]
        self.shift = np.array(args[1]) # to be consistent with directions

        if len(args)==3: padmode = args[2]
        else: self.padmode = 'reflect'

        if len(args)==4: complexoutput = args[3]
        else: self.complexoutput = False
        
        if np.count_nonzero(self.shift) is 0:
            return self.input_array
        else:
            self.ndim = self.input_array.ndim
            if self.ndim > 2: 
                raise ValueError('Only implemented for 1D and 2D arrays')
            self.n = self.input_array.shape
            return self.shiftmeth()
        
    def _shift_pseudo_linear(self,input_array,shift):
        """
        Shifts an image with wrap around. Performs pixel shift using
        numpy.roll. Simpler than shift_linear.
        TODO: Check if same functionality than shift_linear for integer shifts
        
        Parameters
        ----------
        input_array: ndarray
            Input image to calculate the shifts.
        shift: int or tuple
            Number of pixels to shift. For 1D, use a integer value. 
            For 2D, use a tuple of integers where the first value 
            corresponds to shifts in the rows and the second value 
            corresponds to shifts in the columns.
        
        Returns
        -------
        output_array : ndarray
            Shifted image
        """
        shiftfloor = np.floor(shift).astype(int)
        if self.ndim == 1:
            output_array = np.roll(input_array, shift)
        if self.ndim == 2:
            rows, cols = shift
            output_array = np.roll(np.roll(input_array,-rows,axis=0),-cols,axis=1)
        return output_array
        
    def shift_linear(self):
        """
        Shifts an image with wrap around and bilinear interpolation
        
        Parameters
        ----------
        input_array: ndarray
            Input image to calculate the shifts.
        shift: int or tuple
            Number of pixels to shift. For 1D, use a integer value. 
            For 2D, use a tuple of integers where the first value 
            corresponds to shifts in the rows and the second value 
            corresponds to shifts in the columns.
        
        Returns
        -------
        output_array : ndarray
            Shifted image
        """
        # pixel shift
        shiftfloor = np.floor(self.shift).astype(int)
        output_array = self._shift_pseudo_linear(self.input_array, shiftfloor)

        # Subpixel (bilinear)
        tau = self.shift-shiftfloor

        if np.count_nonzero(tau) is not 0:
            # Subpixel (bilinear)
            if self.ndim == 1:
                taux = tau
                output_array = output_array*(1-taux)+\
                               self._shift_pseudo_linear(output_array,1)*taux
            elif self.ndim == 2:
                tauy,taux = tau
                output_array =   output_array*(1-tauy)*(1-taux) + \
                self._shift_pseudo_linear(output_array,(1,0))*tauy*(1-taux) + \
                self._shift_pseudo_linear(output_array,(0,1))*(1-tauy)*taux + \
                self._shift_pseudo_linear(output_array,(1,1))*tauy*taux
        return output_array

    def shift_fft(self):
        """
        Performs pixel and subpixel shift (with wraping) using pyFFTW.
        The array is padded to the next power of 2 for faster FFTW if needed
        The padding is done in mode = 'reflect' by default and it used
        to make FFTW faster and to reduce border artifacts.
        
        Parameters
        ----------
        input_array: ndarray
            Input image to calculate the shifts.
        shift: int or tuple
            Number of pixels to shift. For 1D, use a integer value. 
            For 2D, use a tuple of integers where the first value 
            corresponds to shifts in the rows and the second value 
            corresponds to shifts in the columns.
        
        Returns
        -------
        output_array : ndarray
            Shifted image
        """
        pad_array,N,padw = padfft(self.input_array,self.padmode) # padding
        fftw_input_array = fastfftn(pad_array) # forward FFT
        if self.ndim ==1: # 1D array case
            shift = self.shift[0]
            H = np.exp(-1j*2*np.pi*((shift*N)))
            output_array = fastifftn((fftw_input_array)*H)[padw:-padw]
        elif self.ndim==2: # 2D array case
            Nc, Nr = N  # reverted order to be compatible with meshgrid output
            shift_rows, shift_cols = self.shift
            H = np.exp(-1j*2*np.pi*((shift_rows*Nr)+(shift_cols*Nc)))
            output_array = fastifftn((fftw_input_array)*H)[padw[0]:-padw[0],padw[1]:-padw[1]]
        if not self.complexoutput:
            output_array = output_array.real # TODO: this is weird, to be checked
            # ~ output_array = np.angle(np.exp(1j*output_array))
        return output_array

    def shift_spline_wrap(self):
        """
        Performs pixel and subpixel shift (with wraping) using splines
        @author: jdasilva
        """
        if self.ndim==1:
             # 1D array case
            shift_rows = self.shift
            output_array = interpolation.shift(self.input_array,
                        -shift_rows,order=self.splineorder,mode=self.padmode)
        elif self.ndim==2:
            # 2D array case
            shift_rows,shift_cols = self.shift
            output_array = interpolation.shift(self.input_array,
            (-shift_rows,-shift_cols),order=self.splineorder,mode=self.padmode)
        return output_array
