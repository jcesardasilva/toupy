#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FOURIER SHELL CORRELATION modules  (optimized)
"""

# standard library
import os
import re
import time

# third party package
import h5py
import numpy as np
from scipy.fft import fftshift, ifftshift
from ..utils import tqdm

# local packages
from ..utils.FFT_utils import fastfftn
from ..utils.funcutils import checkhostname
from ..utils.plot_utils import show_fsc_images, show_fsc_curve

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

    Notes
    -----
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
            raise SystemExit("Number of dimensions is different from 2 or 3. Exiting...")
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
        Evaluate the Nyquist frequency and the corresponding frequency array.

        Returns
        -------
        f : ndarray of int32
            Integer frequency indices from ``0`` to ``fnyquist`` (inclusive).
        fnyquist : float
            Nyquist frequency in pixels (half the largest image dimension).
        """
        nmax = np.max(self.n)
        fnyquist = np.floor(nmax / 2.0)
        f = np.arange(0, fnyquist + 1).astype(np.int32)
        return f, fnyquist

    def ringthickness(self):
        """
        Compute the shell index for each voxel in Fourier space.

        Uses broadcasting instead of ``np.meshgrid`` and a Python loop to
        compute the sum of squares, avoiding large temporary arrays.

        Returns
        -------
        index : ndarray of int32
            Array of the same shape as the input image containing the
            integer shell index (rounded radius in scaled Fourier pixels)
            for each voxel.
        """
        nmax = np.max(self.n)

        def _axis(n):
            ax = (
                np.arange(-np.fix(n / 2.0), np.ceil(n / 2.0))
                * np.floor(nmax / 2.0)
                / np.floor(n / 2.0)
            )
            return ifftshift(ax)

        x = _axis(self.nc)
        y = _axis(self.nr)

        if self.ndim == 2:
            # Broadcasting: y[:, None] + x[None, :] avoids meshgrid copies
            sumsquares = y[:, None] ** 2 + x[None, :] ** 2
        elif self.ndim == 3:
            z = _axis(self.ns)
            # Shape: (ns, nr, nc) via broadcasting — no meshgrid needed
            sumsquares = (
                z[:, None, None] ** 2
                + y[None, :, None] ** 2
                + x[None, None, :] ** 2
            )

        index = np.round(np.sqrt(sumsquares)).astype(np.int32)
        return index

    def apodization(self):
        """
        Compute a Hanning apodization window matching the image dimensions.

        The 3-D window is built via an ``einsum`` outer product instead of
        nested list-comprehensions, reducing memory allocations and Python
        overhead.

        Returns
        -------
        window : ndarray
            Hanning window array of shape ``(nr, nc)`` for 2-D images or
            ``(ns, nr, nc)`` for 3-D volumes.
        """
        if self.ndim == 2:
            window = np.outer(np.hanning(self.nr), np.hanning(self.nc))
        elif self.ndim == 3:
            w1 = np.hanning(self.ns)   # (ns,)
            w2 = np.hanning(self.nr)   # (nr,)
            w3 = np.hanning(self.nc)   # (nc,)
            # einsum outer product: result shape (ns, nr, nc)
            window = np.einsum("i,j,k->ijk", w1, w2, w3)
        else:
            raise SystemExit(
                "Number of dimensions is different from 2 or 3. Exiting..."
            )
        return window

    def circle(self):
        """
        Create a circular mask with apodized (cosine-tapered) edges.

        Returns
        -------
        t : ndarray, shape (nr, nc)
            2-D mask that is ``1`` inside the central circle, smoothly tapered
            to ``0`` over ``apod_width`` pixels at the edges.
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
        t[R < (Rmax - self.axial_apod)] = 1
        return t

    def _make_1d_tukey(self, n, apod):
        """
        Build a 1-D tapered Hanning (Tukey-like) window in fftshift order.

        Extracted to avoid code duplication between the 2-D and 3-D paths
        of :meth:`transverse_apodization`.

        Parameters
        ----------
        n : int
            Window length (number of samples).
        apod : int
            Number of pixels of cosine taper on each side of the flat-top.

        Returns
        -------
        w : ndarray, shape (n,)
            Window array in fftshift order: flat ``1`` in the centre,
            cosine-tapered to ``0`` over the outer ``apod`` pixels on each side.
        """
        N = fftshift(np.arange(n))
        centre = np.floor((n - 2 * apod - 1) / 2)
        w = (1.0 + np.cos(2 * np.pi * (N - centre) / (1 + 2 * apod))) / 2.0
        w[apod:-apod] = 1.0
        return w

    def transverse_apodization(self):
        """
        Compute a tapered Hanning-like (Tukey) apodization window.

        The 1-D window construction is delegated to :meth:`_make_1d_tukey`
        to avoid duplication.  The 3-D window is assembled with broadcasting
        instead of per-column list-comprehensions with ``swapaxes`` calls.

        Returns
        -------
        window : ndarray or list of ndarray
            For 2-D images: a single 2-D window array of shape ``(nr, nc)``.
            For 3-D volumes: a list ``[outer(w_row, w_col), outer(w_sli, w_col)]``
            matching the original API expected by :meth:`fouriercorr`.
        """
        print("Calculating the transverse apodization")
        self.transv_apod = self.apod_width

        if self.ndim == 2:
            w1 = self._make_1d_tukey(self.nr, self.transv_apod)
            w2 = self._make_1d_tukey(self.nc, self.transv_apod)
            window = np.outer(w1, w2)

        elif self.ndim == 3:
            w1 = self._make_1d_tukey(self.ns, self.transv_apod)  # axial   (ns,)
            w2 = self._make_1d_tukey(self.nr, self.transv_apod)  # rows    (nr,)
            w3 = self._make_1d_tukey(self.nc, self.transv_apod)  # cols    (nc,)
            # Return as list matching original API: [outer(w1,w2), outer(w1,w3)]
            # Used by fouriercorr to multiply circle3D and sagittal slices.
            window = [np.outer(w2, w3), np.outer(w1, w3)]

        return window

    def fouriercorr(self):
        """
        Compute the Fourier Shell Correlation (FSC) and its threshold curve.

        Optimizations applied:

        * 3-D apodization window assembled with broadcasting instead of
          nested list-comprehensions and ``swapaxes`` calls.
        * Ring-shell loop: boolean mask computed once per shell and reused
          for both F1 and F2 extractions, halving the number of ``np.where``
          calls.
        * ``np.where`` replaced by direct boolean indexing.
        * Cross/auto-correlation sums use ``np.dot`` on flat views, which is
          faster than ``.sum()`` on fancy-indexed complex arrays for large rings.

        Returns
        -------
        FSC : ndarray, shape (n_shells,)
            Fourier Shell Correlation values for each frequency shell.
        T : ndarray, shape (n_shells,)
            Threshold curve (half-bit or one-bit) for each frequency shell.
        """
        # ------------------------------------------------------------------
        # Apodization
        # ------------------------------------------------------------------
        print("Performing the apodization")
        circular_region = self.circle()

        if self.ndim == 2:
            print("Apodization in 2D")
            if self.snrt == 0.2071:
                self.window = circular_region
            else:
                self.window = self.transverse_apodization()
            img1_apod = self.img1 * self.window
            img2_apod = self.img2 * self.window

        elif self.ndim == 3:
            if self.apod_width == 0:
                self.window = 1
            else:
                print("Apodization in 3D. This takes time and memory...")
                p0 = time.time()

                # --- Optimized 3-D window construction ---
                # transverse_apodization now returns [outer(w_row,w_col),
                #                                     outer(w_sli,w_col)]
                window3D = self.transverse_apodization()
                w_axial  = window3D[0]   # (nr, nc)  — axial plane taper
                w_sagit  = window3D[1]   # (ns, nc)  — sagittal plane taper

                # circle3D: (ns, nr, nc) — broadcast circular mask along slices
                circle3D = circular_region[None, :, :]   # (1, nr, nc) → broadcast

                # Combine: circle × axial-taper (both shape (nr,nc)) then
                # multiply sagittal taper broadcast over the row axis.
                # All operations are fully vectorized / in-place where possible.
                self.window = (
                    circle3D * w_axial[None, :, :]        # (ns, nr, nc)
                    * w_sagit[:, None, :]                  # (ns,  1, nc) broadcast
                )
                print("Done. Time elapsed: {:.02f}s".format(time.time() - p0))

            # sagittal slice for display
            slicenum = np.round(self.nr / 2).astype("int")
            img1_apod = (self.window * self.img1)[:, slicenum, :]
            img2_apod = (self.window * self.img2)[:, slicenum, :]

        # ------------------------------------------------------------------
        # Display
        # ------------------------------------------------------------------
        show_fsc_images(img1_apod, img2_apod)

        # ------------------------------------------------------------------
        # FSC computation
        # ------------------------------------------------------------------
        print("Calling method fouriercorr from the class FourierShellCorr")
        p1 = time.time()

        F1 = fastfftn(self.img1 * self.window)   # FFT of image 1
        F2 = fastfftn(self.img2 * self.window)   # FFT of image 2

        index    = self.ringthickness()           # per-voxel shell index
        f, fnyquist = self.nyquist()

        # Flatten index and FFT arrays once — avoids repeated ravel inside loop
        index_flat = index.ravel()
        F1_flat    = F1.ravel()
        F2_flat    = F2.ravel()

        # Pre-sort by shell index so we can use np.searchsorted for fast slicing
        sort_order = np.argsort(index_flat, kind="stable")
        index_sorted = index_flat[sort_order]
        F1_sorted    = F1_flat[sort_order]
        F2_sorted    = F2_flat[sort_order]

        # Initialise output arrays
        C   = np.empty(len(f), dtype=np.float32)
        C1  = np.empty(len(f), dtype=np.float32)
        C2  = np.empty(len(f), dtype=np.float32)
        npts = np.zeros(len(f), dtype=np.float32)

        half_thick = self.ring_thick / 2.0
        use_thick  = self.ring_thick > 1

        print("Calculating the correlation...")
        for ii in tqdm(f, desc="Computing FSC shells"):
            # --- Fast shell extraction via searchsorted on the sorted index ---
            if use_thick:
                lo = np.searchsorted(index_sorted, ii - half_thick, side="left")
                hi = np.searchsorted(index_sorted, ii + half_thick, side="right")
            else:
                lo = np.searchsorted(index_sorted, ii,     side="left")
                hi = np.searchsorted(index_sorted, ii + 1, side="left")

            f1 = F1_sorted[lo:hi]
            f2 = F2_sorted[lo:hi]

            # Cross-correlation and auto-correlations
            # np.vdot operates on flattened arrays and conjugates the first arg
            C[ii]  = abs(np.vdot(f2, f1))          # Σ f1 · conj(f2)
            C1[ii] = abs(np.vdot(f1, f1))           # Σ |f1|²
            C2[ii] = abs(np.vdot(f2, f2))           # Σ |f2|²
            npts[ii] = hi - lo

        # ------------------------------------------------------------------
        # Correlation and threshold
        # ------------------------------------------------------------------
        FSC = C / np.sqrt(C1 * C2)

        eps = np.spacing(1)
        sqrt_npts = np.sqrt(npts + eps)
        Tnum = self.snrt + 2 * np.sqrt(self.snrt) / sqrt_npts + 1 / np.sqrt(npts)
        Tden = self.snrt + 2 * np.sqrt(self.snrt) / sqrt_npts + 1
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
        """
        Plot the FSC and threshold curves and return the underlying data.

        Delegates the actual plotting to
        :func:`~toupy.utils.plot_utils.show_fsc_curve`.

        Returns
        -------
        fn : ndarray
            Spatial frequencies normalised by the Nyquist frequency
            (range ``[0, 1]``).
        T : ndarray
            Threshold curve (half-bit or one-bit).
        FSC : ndarray
            Real part of the Fourier Shell Correlation values.
        """
        print("calling method plot from the class FSCplot")
        fn  = self.f / self.fnyquist
        FSC = self.FSC.real
        T   = self.T
        show_fsc_curve(fn, FSC, T, self.snrt, self.img1.ndim)
        return fn, T, FSC
