#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FOURIER SHELL CORRELATION modules
"""

# standard library
import os
import re
import time

# third party package
import h5py
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftshift, ifftshift

# local packages
from ..utils.FFT_utils import fastfftn
from ..utils.funcutils import checkhostname
from ..utils.plot_utils import isnotebook

__all__ = ["FourierShellCorr", "FSCPlot"]


class FourierShellCorr:
    """
    Computes the Fourier Shell Correlation [1]_ between image1 and image2,
    and estimate the resolution based on the threshold funcion T of 1 or 1/2 bit.

    Parameters
    ----------
    img1 : ndarray
        A 2-dimensional array containing the first image
    img2 : ndarray
        A 2-dimensional array containing the second image
    threshold : str, optional
        The option `onebit` means 1 bit threshold with ``SNRt = 0.5``, which
        should be used for two independent measurements. The option `halfbit`
        means 1/2 bit threshold with ``SNRt = 0.2071``, which should be
        use for split tomogram. The default option is ``half-bit``.
    ring_thick : int, optional
        Thickness of the frequency rings. Normally the pixels get
        assined to the closest integer pixel ring in Fourier Domain.
        With ring_thick, each ring gets more pixels and more statistics.
        The default value is ``1``.
    apod_width : int, optional
        Width in pixel of the edges apodization. It applies a Hanning
        window of the size of the data to the data before the Fourier
        transform calculations to attenuate the border effects. The
        default value is ``20``.

    Returns
    -------
    FSC : ndarray
        Fourier Shell correlation curve
    T : ndarray
        Threshold curve

    Note
    ----
    If 3D images, the first axis is the number of slices, ie., ``[slices, rows, cols]``

    References
    ----------

    .. [1] M. van Heel, M. Schatzb, `Fourier shell correlation threshold criteria,`
      Journal of Structural Biology 151, 250-262 (2005)

    """

    @checkhostname
    def __init__(self, img1, img2, threshold="halfbit", ring_thick=1, apod_width=20):
        print("Calling the class FourierShellCorr")
        self.img1 = np.array(img1)
        self.img2 = np.array(img2)
        if self.img1.shape != self.img2.shape:
            raise ValueError("Images must have the same size")
        # get dimensions and indices of the images
        self.n = self.img1.shape
        self.ndim = self.img1.ndim
        if self.ndim == 2:
            self.nr, self.nc = self.n
        elif self.img1.ndim == 3:
            self.ns, self.nr, self.nc = self.n
        else:
            print("Number of dimensions is different from 2 or 3.Exiting...")
            raise SystemExit("Number of dimensions is different from 2 or 3.Exiting...")
        self.Y, self.X = np.indices((self.nr, self.nc))
        self.Y -= np.round(self.nr / 2).astype(int)
        self.X -= np.round(self.nc / 2).astype(int)
        self.ring_thick = ring_thick  # ring thickness
        print("Using ring thickness of {} pixels".format(ring_thick))
        self.apod_width = apod_width
        if threshold == "halfbit" or threshold == "half-bit":
            print("Using half-bit threshold")
            self.snrt = 0.2071
        elif threshold == "onebit" or threshold == "one-bit":
            print("Using 1-bit threshold")
            self.snrt = 0.5
        else:
            raise ValueError(
                "You need to choose a between 'halfbit' or 'onebit' threshold"
            )
        print("Using SNRt = {}".format(self.snrt))
        print("Input images have {} dimensions".format(self.img1.ndim))

    def nyquist(self):
        """
        Evaluate the Nyquist Frequency
        """
        nmax = np.max(self.n)
        fnyquist = np.floor(nmax / 2.0)
        f = np.arange(0, fnyquist + 1).astype(np.int)
        return f, fnyquist

    def ringthickness(self):
        """
        Define indexes for ring_thick
        """
        nmax = np.max(self.n)
        x = (
            np.arange(-np.fix(self.nc / 2.0), np.ceil(self.nc / 2.0))
            * np.floor(nmax / 2.0)
            / np.floor(self.nc / 2.0)
        )
        y = (
            np.arange(-np.fix(self.nr / 2.0), np.ceil(self.nr / 2.0))
            * np.floor(nmax / 2.0)
            / np.floor(self.nr / 2.0)
        )
        # bring the central pixel to the corners  (important for odd array dimensions)
        x = ifftshift(x)
        y = ifftshift(y)
        if self.ndim == 2:
            # meshgriding
            X = np.meshgrid(x, y)
        elif self.ndim == 3:
            z = (
                np.arange(-np.fix(self.ns / 2.0), np.ceil(self.ns / 2.0))
                * np.floor(nmax / 2.0)
                / np.floor(self.ns / 2.0)
            )
            # bring the central pixel to the corners  (important for odd array dimensions)
            z = ifftshift(z)
            # meshgriding
            X = np.meshgrid(y, z, x)
        # sum of the squares independent of ndim
        sumsquares = np.zeros_like(X[0])
        for ii in range(len(X)):
            sumsquares += X[ii] ** 2
        index = np.round(np.sqrt(sumsquares)).astype(np.int)
        return index

    def apodization(self):
        """
        Compute the Hanning window of the size of the data for the apodization
        
        Note
        ----
        This method does not depend on the parameter ``apod_width`` from the class
        """
        if self.ndim == 2:
            window = np.outer(np.hanning(self.nr), np.hanning(self.nc))
        elif self.ndim == 3:
            window1 = np.hanning(self.ns)
            window2 = np.hanning(self.nr)
            window3 = np.hanning(self.nc)
            windowaxial = np.outer(window2, window3)
            windowsag = np.array([window1 for ii in range(self.nr)]).swapaxes(0, 1)
            # win2d = np.rollaxis(np.array([np.tile(windowaxial,(1,1)) for ii in range(n[0])]),1).swapaxes(1,2)
            win2d = np.array([np.tile(windowaxial, (1, 1)) for ii in range(self.ns)])
            window = (
                np.array(
                    [np.squeeze(win2d[:, :, ii]) * windowsag for ii in range(self.nc)]
                )
                .swapaxes(0, 1)
                .swapaxes(1, 2)
            )
        else:
            print("Number of dimensions is different from 2 or 3. Exiting...")
            raise SystemExit(
                "Number of dimensions is different from 2 or 3. Exiting..."
            )
        return window

    def circle(self):
        """
        Create circle with apodized edges
        """
        self.axial_apod = self.apod_width
        R = np.sqrt(self.X ** 2 + self.Y ** 2)
        Rmax = np.round(np.max(R.shape) / 2.0)
        maskout = R < Rmax
        t = (
            maskout
            * (1 - np.cos(np.pi * (R - Rmax - 2 * self.axial_apod) / self.axial_apod))
            / 2.0
        )
        t[np.where(R < (Rmax - self.axial_apod))] = 1
        return t

    def transverse_apodization(self):
        """
        Compute a tapered Hanning-like window of the size of the data
        for the apodization
        """
        print("Calculating the transverse apodization")
        self.transv_apod = self.apod_width
        if self.ndim == 2:
            Nr = fftshift(np.arange(self.nr))
            Nc = fftshift(np.arange(self.nc))
            window1D1 = (
                1.0
                + np.cos(
                    2
                    * np.pi
                    * (Nr - np.floor((self.nr - 2 * self.transv_apod - 1) / 2))
                    / (1 + 2 * self.transv_apod)
                )
            ) / 2.0
            window1D2 = (
                1.0
                + np.cos(
                    2
                    * np.pi
                    * (Nc - np.floor((self.nc - 2 * self.transv_apod - 1) / 2))
                    / (1 + 2 * self.transv_apod)
                )
            ) / 2.0
            window1D1[self.transv_apod : -self.transv_apod] = 1
            window1D2[self.transv_apod : -self.transv_apod] = 1
            window = np.outer(window1D1, window1D2)
        elif self.ndim == 3:
            Ns = fftshift(np.arange(self.ns))
            Nr = fftshift(np.arange(self.nr))
            Nc = fftshift(np.arange(self.nc))
            window1D1 = (
                1.0
                + np.cos(
                    2
                    * np.pi
                    * (Ns - np.floor((self.ns - 2 * self.transv_apod - 1) / 2))
                    / (1 + 2 * self.transv_apod)
                )
            ) / 2.0
            window1D2 = (
                1.0
                + np.cos(
                    2
                    * np.pi
                    * (Nr - np.floor((self.nr - 2 * self.transv_apod - 1) / 2))
                    / (1 + 2 * self.transv_apod)
                )
            ) / 2.0
            window1D3 = (
                1.0
                + np.cos(
                    2
                    * np.pi
                    * (Nc - np.floor((self.nc - 2 * self.transv_apod - 1) / 2))
                    / (1 + 2 * self.transv_apod)
                )
            ) / 2.0
            window1D1[self.transv_apod : -self.transv_apod] = 1
            window1D2[self.transv_apod : -self.transv_apod] = 1
            window1D3[self.transv_apod : -self.transv_apod] = 1
            window = [np.outer(window1D1, window1D2), np.outer(window1D1, window1D3)]
        return window

    def fouriercorr(self):
        """
        Method to compute FSC and threshold
        """
        # Apodization
        print("Performing the apodization")
        circular_region = self.circle()
        if self.ndim == 2:
            print("Apodization in 2D")
            if self.snrt == 0.2071:
                self.window = circular_region
            elif self.snrt == 0.5:
                self.window = self.transverse_apodization()
                # ~ self.window = self.apodization()
            img1_apod = self.img1 * self.window
            img2_apod = self.img2 * self.window
        elif self.ndim == 3:
            if self.apod_width == 0:
                self.window = 1
            else:
                print("Apodization in 3D. This takes time and memory...")
                p0 = time.time()
                # TODO: find a more efficient way to do this. It know this is not optimum
                window3D = self.transverse_apodization()
                circle3D = np.asarray([circular_region for ii in range(self.ns)])
                self.window = (
                    np.array(
                        [
                            np.squeeze(circle3D[:, :, ii]) * window3D[0]
                            for ii in range(self.nc)
                        ]
                    )
                    .swapaxes(0, 1)
                    .swapaxes(1, 2)
                )
                self.window = np.array(
                    [
                        np.squeeze(self.window[:, ii, :]) * window3D[1]
                        for ii in range(self.nr)
                    ]
                ).swapaxes(0, 1)
                print("Done. Time elapsed: {:.02f}s".format(time.time() - p0))

            # sagital slices
            slicenum = np.round(self.nr / 2).astype("int")
            img1_apod = (self.window * self.img1)[:, slicenum, :]
            img2_apod = (self.window * self.img2)[:, slicenum, :]

        # display image
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(121)
        ax2 = fig1.add_subplot(122)
        im1 = ax1.imshow(img1_apod, cmap="bone", interpolation="none")
        ax1.set_title("image1")
        ax1.set_axis_off()
        im2 = ax2.imshow(img2_apod, cmap="bone", interpolation="none")
        ax2.set_title("image2")
        ax2.set_axis_off()
        if isnotebook():
            display.display(fig1)
            display.display(fig1.canvas)
            #display.clear_output(wait=True)
        else:
            plt.show(block=False)
        plt.show(block=False)

        # FSC computation
        print("Calling method fouriercorr from the class FourierShellCorr")
        p1 = time.time()
        F1 = fastfftn(self.img1 * self.window)  # FFT of the first image
        F2 = fastfftn(self.img2 * self.window)  # FFT of the second image
        index = self.ringthickness()  # index for the ring thickness
        f, fnyquist = self.nyquist()  # Frequency and Nyquist Frequency
        # initializing variables
        C = np.empty_like(f).astype(np.float)
        C1 = np.empty_like(f).astype(np.float)
        C2 = np.empty_like(f).astype(np.float)
        npts = np.zeros_like(f)
        for ii in f:
            if self.ring_thick == 0 or self.ring_thick == 1:
                auxF1 = F1[np.where(index == ii)]
                auxF2 = F2[np.where(index == ii)]
            else:
                auxF1 = F1[
                    (
                        np.where(
                            (index >= (ii - self.ring_thick / 2))
                            & (index <= (ii + self.ring_thick / 2))
                        )
                    )
                ]
                auxF2 = F2[
                    (
                        np.where(
                            (index >= (ii - self.ring_thick / 2))
                            & (index <= (ii + self.ring_thick / 2))
                        )
                    )
                ]
            C[ii] = np.abs((auxF1 * np.conj(auxF2)).sum())
            C1[ii] = np.abs((auxF1 * np.conj(auxF1)).sum())
            C2[ii] = np.abs((auxF2 * np.conj(auxF2)).sum())
            npts[ii] = auxF1.shape[0]

        # The correlation
        FSC = C / (np.sqrt(C1 * C2))

        # Threshold computation
        Tnum = (
            self.snrt
            + (2 * np.sqrt(self.snrt) / np.sqrt(npts + np.spacing(1)))
            + 1 / np.sqrt(npts)
        )
        Tden = self.snrt + (2 * np.sqrt(self.snrt) / np.sqrt(npts + np.spacing(1))) + 1
        T = Tnum / Tden

        print("Done. Time elapsed: {:.02f}s".format(time.time() - p1))

        return FSC, T


class FSCPlot(FourierShellCorr):
    """
    Upper level object to plot the FSC and threshold curves

    Parameters
    ----------
    img1 : ndarray
        A 2-dimensional array containing the first image
    img2 : ndarray
        A 2-dimensional array containing the second image
    threshold : str, optional
        The option `onebit` means 1 bit threshold with ``SNRt = 0.5``, which
        should be used for two independent measurements. The option `halfbit`
        means 1/2 bit threshold with ``SNRt = 0.2071``, which should be
        use for split tomogram. The default option is ``half-bit``.
    ring_thick : int, optional
        Thickness of the frequency rings. Normally the pixels get
        assined to the closest integer pixel ring in Fourier Domain.
        With ring_thick, each ring gets more pixels and more statistics.
        The default value is ``1``.
    apod_width : int, optional
        Width in pixel of the edges apodization. It applies a Hanning
        window of the size of the data to the data before the Fourier
        transform calculations to attenuate the border effects. The
        default value is ``20``.

    Returns
    -------
    fn : ndarray
        A 1-dimensional array containing the frequencies normalized by 
        the Nyquist frequency
    FSC : ndarray
        A 1-dimensional array containing the Fourier Shell correlation curve
    T : ndarray
        A 1-dimensional array containing the threshold curve
    """

    def __init__(self, img1, img2, threshold="halfbit", ring_thick=1, apod_width=20):
        print("calling the class FSCplot")
        super().__init__(img1, img2, threshold, ring_thick, apod_width)

        self.FSC, self.T = FourierShellCorr.fouriercorr(self)
        self.f, self.fnyquist = FourierShellCorr.nyquist(self)

    def plot(self):
        print("calling method plot from the class FSCplot")
        plt.figure(2)
        plt.clf()
        plt.plot(self.f / self.fnyquist, self.FSC.real, "-b", label="FSC")
        plt.legend()
        if self.snrt == 0.2071:
            plt.plot(self.f / self.fnyquist, self.T, "--r", label="1/2 bit threshold")
            plt.legend()
        elif self.snrt == 0.5:
            plt.plot(self.f / self.fnyquist, self.T, "--r", label="1 bit threshold")
            plt.legend()
        else:
            plotT = plt.plot(self.f / self.fnyquist, self.T)
            plt.legend(plotT, "Threshold SNR = %g " % self.snrt, loc="center")
        fn = self.f / self.fnyquist
        T = self.T
        FSC = self.FSC.real
        plt.xlim(0, 1)
        plt.ylim(0, 1.1)
        plt.xlabel("Spatial frequency/Nyquist")
        plt.ylabel("Magnitude")
        if isnotebook():
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            plt.show(block=False)
        if self.img1.ndim == 2:
            plt.savefig("FSC_2D.png", bbox_inches="tight")
        elif self.img1.ndim == 3:
            plt.savefig("FSC_3D.png", bbox_inches="tight")
        return fn, T, FSC
