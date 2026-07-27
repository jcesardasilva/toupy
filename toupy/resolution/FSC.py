#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FOURIER SHELL CORRELATION modules  (optimized)
"""

# standard library
import itertools
import os
import re
import time
import warnings

# third party package
import h5py
import numpy as np
from scipy.fft import fftshift, ifftshift
from scipy.fft import fftn as _fftn, fftfreq as _fftfreq, ifftn as _ifftn
from scipy.interpolate import RegularGridInterpolator
from ..utils import tqdm

# local packages
from ..utils.FFT_utils import fastfftn
from ..utils.funcutils import checkhostname
from ..utils.plot_utils import (
    show_fsc_images,
    show_fsc_curve,
    show_ssnr_curve,
    show_random_fsc_curve,
    show_resolution_map,
)

__all__ = ["FourierShellCorr", "FSCPlot", "FRCPlot", "SSNRPlot", "LocalFSC", "RandomFSC"]


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

        The Nyquist ring index is ``nmax // 2`` (integer division), where
        ``nmax`` is the largest image dimension.  For an even-length DFT of
        size N the highest unambiguous positive-frequency bin is exactly N/2,
        so integer division gives the correct result for both even and odd N.

        Returns
        -------
        f : ndarray of int32
            Integer ring indices from ``0`` (DC) to ``fnyquist`` (inclusive).
        fnyquist : int
            Ring index of the Nyquist frequency (``nmax // 2``).
        """
        nmax = np.max(self.n)
        fnyquist = nmax // 2
        f = np.arange(fnyquist + 1, dtype=np.int32)
        return f, fnyquist

    def ringthickness(self):
        """
        Compute the shell index for each voxel in Fourier space.

        Each dimension's frequency axis is built with ``scipy.fft.fftfreq``
        (which returns values in cycles/pixel, already in FFT layout) and
        then scaled so that every dimension's Nyquist maps to the global
        Nyquist ring index ``nmax // 2``.  This is equivalent to the
        previous manual construction
        ``ifftshift(np.arange(-fix(n/2), ceil(n/2))) * (nmax//2) / (n//2)``
        but requires no ``ifftshift``, ``np.fix``, or ``np.ceil`` calls.

        Uses broadcasting instead of ``np.meshgrid`` to avoid large
        temporary arrays.

        Returns
        -------
        index : ndarray of int32
            Array of the same shape as the input image containing the
            integer shell index (rounded radius in scaled Fourier pixels)
            for each voxel.
        """
        nmax = np.max(self.n)
        fnyquist = nmax // 2  # global Nyquist ring index

        def _axis(n):
            # fftfreq(n) * n  →  integer bin indices in FFT layout
            # scaling by fnyquist / (n // 2) maps each dimension's
            # Nyquist bin to the global fnyquist ring index.
            return _fftfreq(n) * n * fnyquist / (n // 2)

        x = _axis(self.nc)
        y = _axis(self.nr)

        if self.ndim == 2:
            sumsquares = y[:, None] ** 2 + x[None, :] ** 2
        elif self.ndim == 3:
            z = _axis(self.ns)
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


    def ssnr(self):
        """
        Compute the Spectral Signal-to-Noise Ratio (SSNR) from the FSC curve.

        The SSNR is a direct transformation of the FSC:

        .. math::

            \\mathrm{SSNR}(r) = \\frac{2 \\cdot \\mathrm{FSC}(r)}{1 - \\mathrm{FSC}(r)}

        SSNR > 1 indicates signal-dominated shells; SSNR = 1 corresponds to the
        resolution limit.  This method calls :meth:`fouriercorr` internally if
        the FSC has not already been computed.

        Returns
        -------
        f : ndarray
            Integer shell indices (same as :meth:`nyquist`).
        fnyquist : float
            Nyquist frequency (in pixels).
        FSC : ndarray
            Fourier Shell Correlation curve.
        SSNR : ndarray
            Spectral Signal-to-Noise Ratio as a function of spatial frequency.

        References
        ----------
        .. [1] M. Unser, B. L. Trus, and A. C. Steven, "A new resolution
           criterion based on spectral signal-to-noise ratios", Ultramicroscopy
           23, 39-52 (1987).
        """
        FSC, _ = FourierShellCorr.fouriercorr(self)
        f, fnyquist = FourierShellCorr.nyquist(self)
        eps = np.spacing(1)
        SSNR = 2.0 * FSC / np.maximum(1.0 - FSC, eps)
        return f, fnyquist, FSC, SSNR


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
    pixel_size : float, optional
        Physical size of one pixel/voxel in any consistent unit (e.g. metres,
        nanometres).  Used to compute :attr:`resolution_full` and
        :attr:`resolution_half` in physical units.  Default ``1.0`` (result
        in pixels).

    Attributes
    ----------
    fn_res : float or None
        Resolution crossing as a fraction of the Nyquist frequency
        (dimensionless, range ``[0, 1]``).  ``None`` if FSC never exceeds
        the threshold.
    fn_res_cpx : float or None
        Resolution crossing frequency in cycles/pixel
        (``fn_res * 0.5``).  ``None`` if FSC never exceeds the threshold.
    resolution_full : float or None
        Full-period resolution in physical units (van Heel / FSC convention):
        ``pixel_size / fn_res_cpx``.  This is the spatial period of the
        finest resolvable grating — the standard quantity reported in FSC
        analyses.  ``None`` if FSC never exceeds the threshold.
    resolution_half : float or None
        Half-period resolution in physical units (Rayleigh / feature-size
        convention): ``resolution_full / 2``.  This equals the smallest
        resolvable feature size and is the quantity most often reported in
        optical microscopy.  ``None`` if FSC never exceeds the threshold.

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

    def __init__(self, img1, img2, threshold="halfbit", ring_thick=1, apod_width=20,
                 pixel_size=1.0):
        print("calling the class FSCplot")
        super().__init__(img1, img2, threshold, ring_thick, apod_width)
        self.pixel_size = float(pixel_size)
        self.FSC, self.T = FourierShellCorr.fouriercorr(self)
        self.f, self.fnyquist = FourierShellCorr.nyquist(self)

        # Resolution crossing: last shell where FSC > T
        # fn_res        — normalised frequency [0, 1] (fraction of Nyquist)
        # fn_res_cpx    — in cycles/pixel  (= fn_res * 0.5)
        # resolution_full — full grating period in physical units (van Heel convention)
        # resolution_half — half-period / feature size in physical units (Rayleigh convention)
        above = self.FSC.real > self.T
        if np.any(above):
            idx = int(np.where(above)[0][-1])
            self.fn_res          = float(self.f[idx] / self.fnyquist)
            self.fn_res_cpx      = self.fn_res * 0.5          # cycles/pixel
            self.resolution_full = self.pixel_size / self.fn_res_cpx
            self.resolution_half = self.resolution_full / 2.0
        else:
            self.fn_res          = None
            self.fn_res_cpx      = None
            self.resolution_full = None
            self.resolution_half = None

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
        fn_res_cpx : float or None
            Resolution crossing frequency in cycles/pixel.
            Use :attr:`resolution_full` or :attr:`resolution_half` for
            results already converted to physical units.
            ``None`` if FSC never exceeds the threshold.
        """
        print("calling method plot from the class FSCplot")
        fn  = self.f / self.fnyquist
        FSC = self.FSC.real
        T   = self.T
        show_fsc_curve(fn, FSC, T, self.snrt, self.img1.ndim)
        if self.fn_res_cpx is not None:
            print(f"  Resolution (full period) : {self.resolution_full:.6g} "
                  f"[in units of pixel_size]")
            print(f"  Resolution (half period) : {self.resolution_half:.6g} "
                  f"[in units of pixel_size]")
        return fn, T, FSC, self.fn_res_cpx


class FRCPlot(FSCPlot):
    """
    Fourier Ring Correlation (FRC) — the 2-D analogue of the FSC.

    Identical to :class:`FSCPlot` but enforces two-dimensional input and
    uses "FRC" in all labels and output filenames.  FRC is standard in
    cryo-EM and increasingly used in X-ray ptychography and nano-tomography.

    Parameters
    ----------
    img1 : ndarray
        A 2-dimensional array containing the first image.
    img2 : ndarray
        A 2-dimensional array containing the second image.
    threshold : str, optional
        ``'halfbit'`` (default) or ``'onebit'``.
    ring_thick : int, optional
        Ring thickness in pixels.  Default ``1``.
    apod_width : int, optional
        Apodization width in pixels.  Default ``20``.

    Returns
    -------
    fn : ndarray
        Spatial frequencies normalised by the Nyquist frequency.
    FRC : ndarray
        Fourier Ring Correlation curve.
    T : ndarray
        Threshold curve.

    Raises
    ------
    ValueError
        If either input image is not 2-dimensional.

    Notes
    -----
    The mathematical definition and threshold criteria are identical to the
    FSC; only the geometry changes (rings in 2-D vs shells in 3-D).

    References
    ----------
    .. [1] M. van Heel and M. Schatz, "Fourier shell correlation threshold
       criteria", Journal of Structural Biology 151, 250-262 (2005).
    """

    def __init__(self, img1, img2, threshold="halfbit", ring_thick=1, apod_width=20,
                 pixel_size=1.0):
        if np.asarray(img1).ndim != 2 or np.asarray(img2).ndim != 2:
            raise ValueError("FRCPlot requires 2-D images (use FSCPlot for 3-D).")
        print("Calling the class FRCPlot")
        super().__init__(img1, img2, threshold, ring_thick, apod_width, pixel_size)

    def plot(self):
        """
        Plot the FRC and threshold curves and return the underlying data.

        Returns
        -------
        fn : ndarray
            Spatial frequencies normalised by the Nyquist frequency.
        T : ndarray
            Threshold curve (half-bit or one-bit).
        FRC : ndarray
            Fourier Ring Correlation curve (real part).
        fn_res_cpx : float or None
            Resolution crossing frequency in cycles/pixel.
            Use :attr:`resolution_full` or :attr:`resolution_half` for
            results already converted to physical units.
            ``None`` if FRC never exceeds the threshold.
        """
        print("Calling method plot from the class FRCPlot")
        fn  = self.f / self.fnyquist
        FRC = self.FSC.real
        T   = self.T
        # Reuse show_fsc_curve but relabel via the ndim=2 path
        show_fsc_curve(fn, FRC, T, self.snrt, ndim=2)
        if self.fn_res_cpx is not None:
            print(f"  Resolution (full period) : {self.resolution_full:.6g} "
                  f"[in units of pixel_size]")
            print(f"  Resolution (half period) : {self.resolution_half:.6g} "
                  f"[in units of pixel_size]")
        return fn, T, FRC, self.fn_res_cpx


class SSNRPlot(FourierShellCorr):
    """
    Compute and plot the Spectral Signal-to-Noise Ratio (SSNR).

    The SSNR is derived from the FSC via:

    .. math::

        \\mathrm{SSNR}(r) = \\frac{2 \\cdot \\mathrm{FSC}(r)}{1 - \\mathrm{FSC}(r)}

    The resolution limit is defined by a **frequency-dependent** SSNR
    threshold curve obtained by applying the same transformation to the
    half-bit (or one-bit) FSC threshold :math:`T(r)`:

    .. math::

        \\mathrm{SSNR}_T(r) = \\frac{2 \\cdot T(r)}{1 - T(r)}

    This curve is *not* a horizontal line — it starts very high at low
    spatial frequencies (few Fourier coefficients, large :math:`T`) and
    asymptotes to a fixed value at high frequencies:

    * **half-bit** threshold: asymptote :math:`\\approx 0.414`
    * **one-bit**  threshold: asymptote :math:`= 1.0`

    Using a horizontal line at SSNR = 1 as the resolution criterion is
    therefore only correct for the one-bit case (two fully independent
    measurements).  For a **split dataset** (``threshold='halfbit'``), the
    correct criterion is the frequency-dependent :math:`\\mathrm{SSNR}_T(r)`
    curve, whose asymptote is ≈ 0.414.

    Parameters
    ----------
    img1 : ndarray
        First half-dataset (2-D or 3-D array).
    img2 : ndarray
        Second half-dataset, same shape as ``img1``.
    threshold : str, optional
        ``'halfbit'`` (default) or ``'onebit'``.
    ring_thick : int, optional
        Ring thickness in pixels.  Default ``1``.
    apod_width : int, optional
        Apodization width in pixels.  Default ``20``.

    Attributes
    ----------
    FSC : ndarray
        Fourier Shell/Ring Correlation curve.
    SSNR : ndarray
        Spectral Signal-to-Noise Ratio curve.
    SSNR_T : ndarray
        Frequency-dependent SSNR threshold derived from the FSC threshold
        curve :math:`T(r)`.
    f : ndarray
        Shell indices.
    fnyquist : float
        Nyquist frequency in pixels.
    fn_res : float or None
        Resolution crossing as a fraction of the Nyquist frequency
        (dimensionless, range ``[0, 1]``).  ``None`` if SSNR never exceeds
        the threshold.
    fn_res_cpx : float or None
        Resolution crossing frequency in cycles/pixel
        (``fn_res * 0.5``).  ``None`` if SSNR never exceeds the threshold.
    resolution_full : float or None
        Full-period resolution in physical units (van Heel / FSC convention):
        ``pixel_size / fn_res_cpx``.  This is the spatial period of the
        finest resolvable grating — the standard quantity reported in FSC
        analyses.  ``None`` if SSNR never exceeds the threshold.
    resolution_half : float or None
        Half-period resolution in physical units (Rayleigh / feature-size
        convention): ``resolution_full / 2``.  This equals the smallest
        resolvable feature size and is the quantity most often reported in
        optical microscopy.  ``None`` if SSNR never exceeds the threshold.

    References
    ----------
    .. [1] M. Unser, B. L. Trus, and A. C. Steven, "A new resolution
       criterion based on spectral signal-to-noise ratios",
       Ultramicroscopy 23, 39-52 (1987).
    .. [2] M. van Heel and M. Schatz, "Fourier shell correlation threshold
       criteria", Journal of Structural Biology 151, 250-262 (2005).
    """

    def __init__(self, img1, img2, threshold="halfbit", ring_thick=1, apod_width=20,
                 pixel_size=1.0):
        print("Calling the class SSNRPlot")
        super().__init__(img1, img2, threshold, ring_thick, apod_width)
        self.pixel_size = float(pixel_size)

        # Compute FSC and the FSC threshold curve in a single call
        self.FSC, self.T    = FourierShellCorr.fouriercorr(self)
        self.f, self.fnyquist = FourierShellCorr.nyquist(self)

        eps = np.spacing(1)
        fsc = self.FSC.real

        # SSNR from the FSC curve
        self.SSNR   = 2.0 * fsc    / np.maximum(1.0 - fsc,    eps)

        # Frequency-dependent SSNR threshold derived from T(r)
        self.SSNR_T = 2.0 * self.T / np.maximum(1.0 - self.T, eps)

        # Resolution crossing: last shell where SSNR > SSNR_T
        # fn_res        — normalised frequency [0, 1] (fraction of Nyquist)
        # fn_res_cpx    — in cycles/pixel  (= fn_res * 0.5)
        # resolution_full — full grating period in physical units (van Heel convention)
        # resolution_half — half-period / feature size in physical units (Rayleigh convention)
        fn    = self.f / self.fnyquist
        above = self.SSNR > self.SSNR_T
        if np.any(above):
            idx = int(np.where(above)[0][-1])
            self.fn_res          = float(fn[idx])
            self.fn_res_cpx      = self.fn_res * 0.5          # cycles/pixel
            self.resolution_full = self.pixel_size / self.fn_res_cpx
            self.resolution_half = self.resolution_full / 2.0
        else:
            self.fn_res          = None
            self.fn_res_cpx      = None
            self.resolution_full = None
            self.resolution_half = None

    def plot(self):
        """
        Plot the SSNR curve with its frequency-dependent threshold and return
        the underlying data.

        The plot shows:

        * The SSNR curve (blue).
        * The frequency-dependent SSNR threshold :math:`\\mathrm{SSNR}_T(r)`
          (red solid curve), derived by transforming the FSC threshold
          :math:`T(r)` through :math:`\\mathrm{SSNR}_T = 2T/(1-T)`.
        * A dotted horizontal line at the asymptotic threshold value
          (≈ 0.414 for half-bit, 1.0 for one-bit).
        * A vertical dashed line at the estimated resolution (last frequency
          where ``SSNR > SSNR_T``).

        Returns
        -------
        fn : ndarray
            Spatial frequencies normalised by the Nyquist frequency.
        FSC : ndarray
            Fourier Shell/Ring Correlation curve.
        SSNR : ndarray
            Spectral Signal-to-Noise Ratio curve.
        SSNR_T : ndarray
            Frequency-dependent SSNR threshold curve.
        fn_res_cpx : float or None
            Resolution crossing frequency in cycles/pixel
            (last frequency where ``SSNR > SSNR_T``).
            Divide ``pixel_size`` by this value to obtain the resolution
            in physical units.  ``None`` if SSNR never exceeds the threshold.
        """
        print("Calling method plot from the class SSNRPlot")
        fn     = self.f / self.fnyquist
        FSC    = self.FSC.real
        SSNR   = self.SSNR
        SSNR_T = self.SSNR_T
        show_ssnr_curve(fn, FSC, SSNR, SSNR_T, self.snrt, self.ndim)
        if self.fn_res_cpx is not None:
            print(f"  Resolution (full period) : {self.resolution_full:.6g} "
                  f"[in units of pixel_size]")
            print(f"  Resolution (half period) : {self.resolution_half:.6g} "
                  f"[in units of pixel_size]")
        return fn, FSC, SSNR, SSNR_T, self.fn_res_cpx


# ---------------------------------------------------------------------------
# LocalFSC
# ---------------------------------------------------------------------------

class LocalFSC:
    """
    Local resolution estimation via block-wise FSC (3-D) or FRC (2-D).

    Divides the volume into overlapping boxes, computes an independent
    Fourier Shell/Ring Correlation within each box, and assembles a
    spatial map of local resolution interpolated to the input shape.

    Parameters
    ----------
    vol1 : ndarray
        First independent half-reconstruction (2-D or 3-D).
    vol2 : ndarray
        Second independent half-reconstruction, same shape as *vol1*.
    pixel_size : float, optional
        Physical voxel size (any consistent unit). Default ``1.0``.
    box_size : int, optional
        Side length (in voxels) of the cubic (or square) analysis box.
        Smaller boxes give finer spatial sampling but noisier estimates.
        Default ``32``.
    step : int or None, optional
        Step size in voxels between adjacent box centres.
        Default ``box_size // 2`` (50 % overlap).
    threshold : float or {'1bit', 'halfbit'}, optional
        Criterion used to define the local resolution limit.

        float
            A fixed FSC threshold applied at every shell, e.g. ``0.143``
            (the Rosenthal & Henderson, 2003, "gold-standard" criterion —
            *not* the half-bit criterion, despite the common conflation
            of the two [3]_) or ``0.5`` (the fixed threshold used by
            Cardone, Heymann & Steven, 2013, in ``blocres`` [1]_, the
            reference local-FSC implementation this class follows).
            Default ``0.143``.
        ``'1bit'`` or ``'halfbit'``
            The shell-size-corrected one-bit or half-bit threshold of
            van Heel & Schatz, 2005 [2]_, evaluated per shell from the
            number of independent Fourier voxels actually present in
            *this* box, rather than the fixed asymptotic value. Rohou,
            2020 [4]_, shows that fixed FSC thresholds (0.143 or 0.5
            alike) are biased toward over-estimating resolution in the
            sparsely populated low-frequency shells of small
            sub-volumes — precisely the regime ``LocalFSC`` operates in
            — so this option is recommended whenever ``box_size`` is
            small relative to the reconstruction.

    Attributes
    ----------
    resolution_map : ndarray
        Local full-period resolution in pixels (van Heel convention),
        same shape as *vol1*.
    resolution_map_phys : ndarray
        Local full-period resolution in physical units
        (``resolution_map * pixel_size``).
    resolution_map_half : ndarray
        Local half-period resolution in pixels (Rayleigh convention):
        ``resolution_map / 2``.
    resolution_map_phys_half : ndarray
        Local half-period resolution in physical units:
        ``resolution_map_phys / 2``.
    resolution_median : float
        Median full-period local resolution in pixels.
    resolution_mean : float
        Mean full-period local resolution in pixels.
    resolution_std : float
        Standard deviation of full-period local resolution in pixels.
    resolution_median_half : float
        Median half-period local resolution in pixels.
    resolution_mean_half : float
        Mean half-period local resolution in pixels.
    resolution_std_half : float
        Standard deviation of half-period local resolution in pixels.

    References
    ----------
    .. [1] G. Cardone, J. B. Heymann, and A. C. Steven, "One number does
       not fit all: Mapping local variations in resolution in cryo-EM
       reconstructions", J. Struct. Biol. 184, 226-236 (2013).
       https://doi.org/10.1016/j.jsb.2013.08.002
    .. [2] M. van Heel and M. Schatz, "Fourier shell correlation threshold
       criteria", J. Struct. Biol. 151, 250-262 (2005).
       https://doi.org/10.1016/j.jsb.2005.05.009
    .. [3] P. B. Rosenthal and R. Henderson, "Optimal determination of
       particle orientation, absolute hand, and contrast loss in
       single-particle electron cryomicroscopy", J. Mol. Biol. 333,
       721-745 (2003). https://doi.org/10.1016/j.jmb.2003.07.013
    .. [4] A. Rohou, "Fourier shell correlation criteria for local
       resolution estimation", bioRxiv 2020.03.01.972067 (2020).
       https://doi.org/10.1101/2020.03.01.972067
    """

    _ADAPTIVE_SNR = {
        "1bit": 0.5,
        "one-bit": 0.5,
        "onebit": 0.5,
        "halfbit": 0.2071,
        "half-bit": 0.2071,
        "1/2bit": 0.2071,
    }

    def __init__(self, vol1, vol2, pixel_size=1.0, box_size=32, step=None, threshold=0.143):
        print("Calling the class LocalFSC")
        vol1 = np.asarray(vol1, dtype=np.float64)
        vol2 = np.asarray(vol2, dtype=np.float64)
        if vol1.shape != vol2.shape:
            raise ValueError("LocalFSC: vol1 and vol2 must have the same shape.")
        if vol1.ndim not in (2, 3):
            raise ValueError("LocalFSC: input must be 2-D or 3-D.")
        self.vol1 = vol1
        self.vol2 = vol2
        self.shape = vol1.shape
        self.ndim = vol1.ndim
        self.pixel_size = float(pixel_size)
        self.box_size = int(box_size)
        self.step = int(step) if step is not None else self.box_size // 2

        # threshold: either a fixed FSC value (as before), or a
        # shell-size-corrected criterion ('1bit' / 'halfbit', van Heel &
        # Schatz, 2005) evaluated per shell from the number of
        # independent Fourier voxels n(xi) actually present in that box.
        # See the class docstring for the rationale (Rohou, 2020).
        if isinstance(threshold, str):
            key = threshold.lower()
            if key not in self._ADAPTIVE_SNR:
                raise ValueError(
                    f"LocalFSC: threshold string must be one of "
                    f"{sorted(set(self._ADAPTIVE_SNR))!r}, got {threshold!r}."
                )
            self.threshold = threshold
            self._adaptive_snr = self._ADAPTIVE_SNR[key]
        else:
            self.threshold = float(threshold)
            self._adaptive_snr = None

        self._compute()

        ps = self.pixel_size
        print(f"  LocalFSC median resolution (full period) : "
              f"{self.resolution_median:.2f} px "
              f"({self.resolution_median * ps:.4g} physical units)")
        print(f"  LocalFSC median resolution (half period) : "
              f"{self.resolution_median_half:.2f} px "
              f"({self.resolution_median_half * ps:.4g} physical units)")
        print(f"  LocalFSC mean   resolution (full period) : "
              f"{self.resolution_mean:.2f} ± {self.resolution_std:.2f} px")

    def _make_window(self):
        """
        Build a Hanning window matching the analysis box shape.

        Returns
        -------
        window : ndarray
            Hanning window of shape ``(box_size,) * ndim``.
        """
        w1d = np.hanning(self.box_size)
        window = w1d.copy()
        for _ in range(self.ndim - 1):
            window = np.multiply.outer(window, w1d)
        return window

    def _fsc_box(self, box1, box2, window):
        """
        Compute local resolution (pixels) from a pair of overlapping boxes.

        Subtracts the mean, applies *window*, computes the FFT, builds
        radial shells using ``scipy.fft.fftfreq``, and returns the local
        resolution at the shell where FSC first drops below the chosen
        threshold — either the fixed value ``self.threshold``, or, when
        ``self._adaptive_snr`` is set, the van Heel & Schatz (2005)
        shell-size-corrected threshold evaluated from the actual number
        of independent Fourier voxels in each shell of *this* box.

        Parameters
        ----------
        box1 : ndarray
            First analysis box (shape ``(box_size,) * ndim``).
        box2 : ndarray
            Second analysis box, same shape as *box1*.
        window : ndarray
            Hanning window of shape ``(box_size,) * ndim``.

        Returns
        -------
        resolution_px : float
            Local resolution in pixels.  Returns ``box_size * 2.0`` when
            the FSC never reaches the chosen threshold (bad estimate).
        """
        b = self.box_size
        eps = np.finfo(np.float64).tiny

        b1 = (box1 - box1.mean()) * window
        b2 = (box2 - box2.mean()) * window

        F1 = _fftn(b1)
        F2 = _fftn(b2)

        # Build radial coordinate grid in cycles/pixel (FFT layout)
        freqs = [_fftfreq(b) for _ in range(self.ndim)]
        grids = np.meshgrid(*freqs, indexing='ij')
        R = np.sqrt(sum(g ** 2 for g in grids))

        # Radial shells: integer bins in units of 1/box_size
        shell_idx = np.round(R * b).astype(np.int32)
        max_shell = b // 2

        fsc_vals = []
        n_vals = []
        for s in range(max_shell + 1):
            mask = shell_idx == s
            n_shell = int(np.sum(mask))
            n_vals.append(n_shell)
            if n_shell == 0:
                fsc_vals.append(0.0)
                continue
            f1 = F1[mask]
            f2 = F2[mask]
            cross = np.abs(np.vdot(f2, f1))
            a1 = np.abs(np.vdot(f1, f1))
            a2 = np.abs(np.vdot(f2, f2))
            denom = np.sqrt(a1 * a2)
            fsc_vals.append(float(cross / denom) if denom > eps else 0.0)

        fsc_arr = np.array(fsc_vals)

        if self._adaptive_snr is not None:
            # Shell-size-corrected threshold (van Heel & Schatz, 2005),
            # evaluated per shell from n(xi) actually present in this box:
            #   T(xi) = [SNR + (2*sqrt(SNR)+1)/sqrt(n)] / [SNR + 1 + 2*sqrt(SNR)/sqrt(n)]
            n_arr = np.maximum(np.array(n_vals, dtype=np.float64), 1.0)
            snr = self._adaptive_snr
            sqrt_snr = np.sqrt(snr)
            sqrt_n = np.sqrt(n_arr)
            thr_arr = (snr + (2.0 * sqrt_snr + 1.0) / sqrt_n) / (
                snr + 1.0 + 2.0 * sqrt_snr / sqrt_n
            )
            above = np.where(fsc_arr[1:] >= thr_arr[1:])[0] + 1  # skip DC
        else:
            # Find the highest shell where FSC >= fixed threshold
            above = np.where(fsc_arr[1:] >= self.threshold)[0] + 1  # skip DC

        if len(above) == 0:
            return float(b * 2.0)
        r_res_shell = above[-1]
        # Convert shell index to resolution in pixels: shell s = s/b cycles/pixel
        r_cpx = r_res_shell / float(b)
        if r_cpx < eps:
            return float(b * 2.0)
        return 1.0 / r_cpx

    def _compute(self):
        """
        Iterate over the coarse grid, compute local FSC per box, and
        interpolate the result to the full volume shape.

        Stores
        ------
        resolution_map : ndarray
        resolution_map_phys : ndarray
        resolution_median : float
        resolution_mean : float
        resolution_std : float
        """
        vol1 = self.vol1
        vol2 = self.vol2
        shape = self.shape
        ndim = self.ndim
        b = self.box_size
        step = self.step
        half = b // 2
        window = self._make_window()

        # Coarse grid: centre positions
        axes = [np.arange(half, shape[d] - half, step) for d in range(ndim)]

        for d, ax in enumerate(axes):
            if len(ax) < 2:
                warnings.warn(
                    f"LocalFSC: axis {d} has fewer than 2 grid points "
                    f"(box_size={b} may be too large for dimension {shape[d]}).  "
                    "Resolution map may be unreliable.",
                    UserWarning,
                    stacklevel=2,
                )

        coarse_shape = tuple(len(ax) for ax in axes)
        coarse_map = np.empty(coarse_shape, dtype=np.float64)

        idx_ranges = [range(len(ax)) for ax in axes]
        for grid_idx in itertools.product(*idx_ranges):
            centres = tuple(int(axes[d][grid_idx[d]]) for d in range(ndim))
            slices = tuple(
                slice(max(0, centres[d] - half), min(shape[d], centres[d] + half))
                for d in range(ndim)
            )
            box1 = vol1[slices]
            box2 = vol2[slices]

            # Pad to box_size if the box is smaller near the border
            if box1.shape != (b,) * ndim:
                pad1 = [(0, b - box1.shape[dd]) for dd in range(ndim)]
                box1 = np.pad(box1, pad1)
                box2 = np.pad(box2, pad1)

            coarse_map[grid_idx] = self._fsc_box(box1, box2, window)

        # Interpolate to full volume using RegularGridInterpolator
        interp = RegularGridInterpolator(
            axes,
            coarse_map,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )

        # Build full meshgrid of integer coordinates
        full_grids = np.mgrid[tuple(slice(0, shape[d]) for d in range(ndim))]
        # full_grids shape: (ndim, *shape)
        pts = np.stack([full_grids[d].ravel() for d in range(ndim)], axis=-1)
        res_full = interp(pts).reshape(shape)

        self.resolution_map       = res_full
        self.resolution_map_phys  = res_full * self.pixel_size
        self.resolution_map_half  = res_full / 2.0
        self.resolution_map_phys_half = self.resolution_map_phys / 2.0

        self.resolution_median      = float(np.median(res_full))
        self.resolution_mean        = float(np.mean(res_full))
        self.resolution_std         = float(np.std(res_full))
        self.resolution_median_half = self.resolution_median / 2.0
        self.resolution_mean_half   = self.resolution_mean   / 2.0
        self.resolution_std_half    = self.resolution_std    / 2.0

    def plot(self, slice_idx=None, axis=0, cmap='viridis_r', vmin=None, vmax=None):
        """
        Display the local resolution map and save it to ``LocalFSC_resmap.png``.

        For 3-D volumes the central slice (or the slice specified by
        *slice_idx*) along *axis* is shown.  The colour bar is in pixels.

        Parameters
        ----------
        slice_idx : int or None, optional
            Slice index along *axis* to display.  Defaults to the central
            slice.  Ignored for 2-D input.
        axis : int, optional
            Volume axis along which to slice.  Default ``0``.
        cmap : str, optional
            Matplotlib colourmap.  Default ``'viridis_r'``.
        vmin : float or None, optional
            Lower colour-scale limit.  ``None`` uses the data minimum.
        vmax : float or None, optional
            Upper colour-scale limit.  ``None`` uses the data maximum.

        Returns
        -------
        resolution_map : ndarray
            The full local-resolution map (same shape as the input volume).
        """
        show_resolution_map(
            self.resolution_map,
            self.ndim,
            title="LocalFSC resolution map",
            filename="LocalFSC_resmap.png",
            slice_idx=slice_idx,
            axis=axis,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        return self.resolution_map


# ---------------------------------------------------------------------------
# RandomFSC
# ---------------------------------------------------------------------------

class RandomFSC(FourierShellCorr):
    """
    Phase-randomization test for FSC validation (Chen et al., 2013).

    After computing the standard FSC, the Fourier phases of both
    half-volumes are independently randomized above a chosen spatial
    frequency *f*_cutoff (the shell where FSC_obs first drops below
    ``fsc_cutoff``).  The FSC computed from the two phase-randomized
    volumes (FSC_rand) is the noise floor expected from model bias or
    overfitting.  A corrected FSC is then defined as:

    .. math::

        \\mathrm{FSC_{corr}}(r) =
            \\frac{\\mathrm{FSC_{obs}}(r) - \\mathrm{FSC_{rand}}(r)}
                  {1 - \\mathrm{FSC_{rand}}(r)}

    If no overfitting is present, FSC_rand drops quickly to zero beyond
    *f*_cutoff and FSC_corr ≈ FSC_obs.  An elevated FSC_rand indicates
    that the two half-maps share information beyond *f*_cutoff via a
    common external model used during iterative reconstruction.

    Parameters
    ----------
    img1 : ndarray
        First half-reconstruction (2-D or 3-D).
    img2 : ndarray
        Second half-reconstruction, same shape as *img1*.
    threshold : str, optional
        ``'halfbit'`` (default) or ``'onebit'``.
    ring_thick : int, optional
        Ring thickness in pixels. Default ``1``.
    apod_width : int, optional
        Apodization width in pixels. Default ``20``.
    fsc_cutoff : float, optional
        FSC value that defines the phase-randomization cut-off frequency.
        Phases are randomized at all shells where FSC_obs < ``fsc_cutoff``.
        Default ``0.8``.
    random_seed : int or None, optional
        Seed for the random phase generator (for reproducibility).
        Default ``None`` (non-reproducible).

    Attributes
    ----------
    FSC_obs : ndarray
        Standard (observed) FSC curve.
    FSC_rand : ndarray
        FSC of the phase-randomized half-volumes.
    FSC_corr : ndarray
        Phase-randomization corrected FSC.
    T : ndarray
        Threshold curve (same as standard FSC).
    f : ndarray
        Shell indices.
    fnyquist : float
        Nyquist frequency in pixels.
    cutoff_shell : int
        Shell index used as the phase-randomization boundary.

    References
    ----------
    .. [1] S. Chen, G. McMullan, A. R. Faruqi, G. N. Murshudov, J. M. Short,
       S. H. Scheres, and R. Henderson, "High-resolution noise substitution to
       measure overfitting and validate resolution in 3D structure determination
       by single particle electron cryomicroscopy", Ultramicroscopy 135, 24-35
       (2013). https://doi.org/10.1016/j.ultramic.2013.06.004
    """

    def __init__(
        self,
        img1,
        img2,
        threshold="halfbit",
        ring_thick=1,
        apod_width=20,
        fsc_cutoff=0.8,
        random_seed=None,
        pixel_size=1.0,
    ):
        print("Calling the class RandomFSC")
        super().__init__(img1, img2, threshold, ring_thick, apod_width)
        self.fsc_cutoff  = float(fsc_cutoff)
        self.random_seed = random_seed
        self.pixel_size  = float(pixel_size)

        # Compute observed FSC and threshold
        self.FSC_obs, self.T = FourierShellCorr.fouriercorr(self)
        self.f, self.fnyquist = FourierShellCorr.nyquist(self)

        # Phase-randomization
        self.cutoff_shell = self._find_cutoff_shell()
        self._compute_randomized()

        # Resolution at the FSC_corr × T crossing (last shell where FSC_corr > T)
        fsc_corr = np.asarray(self.FSC_corr).real
        above    = fsc_corr > np.asarray(self.T)
        if np.any(above):
            idx = int(np.where(above)[0][-1])
            self.fn_res          = float(self.f[idx] / self.fnyquist)
            self.fn_res_cpx      = self.fn_res * 0.5
            self.resolution_full = self.pixel_size / self.fn_res_cpx
            self.resolution_half = self.resolution_full / 2.0
        else:
            self.fn_res          = None
            self.fn_res_cpx      = None
            self.resolution_full = None
            self.resolution_half = None

        print(f"  Phase-randomization cutoff shell : {self.cutoff_shell}")
        print(f"  f_cutoff (cycles/pixel)          : "
              f"{self.cutoff_shell / self.fnyquist * 0.5:.4f}")
        if self.fn_res_cpx is not None:
            print(f"  Resolution FSC_corr (full period): {self.resolution_full:.6g} "
                  f"[in units of pixel_size]")
            print(f"  Resolution FSC_corr (half period): {self.resolution_half:.6g} "
                  f"[in units of pixel_size]")

    def _find_cutoff_shell(self):
        """
        Find the last shell index where ``FSC_obs >= fsc_cutoff``.

        Returns
        -------
        cutoff_shell : int
            Shell index used as the phase-randomization boundary.

        Warns
        -----
        UserWarning
            If ``FSC_obs`` never reaches ``fsc_cutoff``.
        """
        fsc = np.asarray(self.FSC_obs).real
        above = np.where(fsc >= self.fsc_cutoff)[0]
        if len(above) == 0:
            shell = int(len(self.f) // 4)
            warnings.warn(
                f"RandomFSC: FSC_obs never reaches fsc_cutoff={self.fsc_cutoff}. "
                f"Using shell index {shell} as fallback.",
                UserWarning,
                stacklevel=2,
            )
            return shell
        return int(above[-1])

    def _build_radial_grid(self, shape):
        """
        Build a radial frequency grid in cycles/pixel for a volume of *shape*.

        Parameters
        ----------
        shape : tuple of int
            Shape of the volume.

        Returns
        -------
        R : ndarray
            Radial frequency array of the same shape as *shape*,
            in cycles/pixel (FFT layout).
        """
        freqs = [_fftfreq(n) for n in shape]
        grids = np.meshgrid(*freqs, indexing='ij')
        return np.sqrt(sum(g ** 2 for g in grids))

    def _randomize_phases_above(self, vol, f_cutoff_cpx, rng):
        """
        Randomize Fourier phases above *f_cutoff_cpx* cycles/pixel.

        The amplitude spectrum is preserved; only the phase is replaced by
        uniform random values in ``[-π, π]``.

        Parameters
        ----------
        vol : ndarray
            Real-space volume (2-D or 3-D).
        f_cutoff_cpx : float
            Radial frequency threshold in cycles/pixel.  Shells with
            R > f_cutoff_cpx are phase-randomized.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        vol_rand : ndarray
            Real-space volume with randomized phases above *f_cutoff_cpx*.
        """
        F = _fftn(vol)
        R = self._build_radial_grid(vol.shape)
        mask = R > f_cutoff_cpx
        amp = np.abs(F)
        phase_orig = np.angle(F)
        rand_phase = rng.uniform(-np.pi, np.pi, size=F.shape)
        new_phase = np.where(mask, rand_phase, phase_orig)
        F_rand = amp * np.exp(1j * new_phase)
        return np.real(_ifftn(F_rand))

    def _fsc_from_arrays(self, vol1, vol2):
        """
        Compute FSC between *vol1* and *vol2* using the pre-computed window.

        Does not call :meth:`fouriercorr` to avoid re-triggering the
        apodization display and window recomputation.

        Parameters
        ----------
        vol1 : ndarray
            First volume (same shape as ``self.img1``).
        vol2 : ndarray
            Second volume (same shape as ``self.img2``).

        Returns
        -------
        FSC : ndarray
            Fourier Shell Correlation curve.
        """
        F1 = fastfftn(vol1 * self.window)
        F2 = fastfftn(vol2 * self.window)

        index = self.ringthickness()
        f, fnyquist = self.nyquist()

        index_flat = index.ravel()
        F1_flat = F1.ravel()
        F2_flat = F2.ravel()

        sort_order = np.argsort(index_flat, kind="stable")
        index_sorted = index_flat[sort_order]
        F1_sorted = F1_flat[sort_order]
        F2_sorted = F2_flat[sort_order]

        C  = np.empty(len(f), dtype=np.float32)
        C1 = np.empty(len(f), dtype=np.float32)
        C2 = np.empty(len(f), dtype=np.float32)

        half_thick = self.ring_thick / 2.0
        use_thick = self.ring_thick > 1

        for ii in f:
            if use_thick:
                lo = np.searchsorted(index_sorted, ii - half_thick, side="left")
                hi = np.searchsorted(index_sorted, ii + half_thick, side="right")
            else:
                lo = np.searchsorted(index_sorted, ii,     side="left")
                hi = np.searchsorted(index_sorted, ii + 1, side="left")

            f1 = F1_sorted[lo:hi]
            f2 = F2_sorted[lo:hi]

            C[ii]  = abs(np.vdot(f2, f1))
            C1[ii] = abs(np.vdot(f1, f1))
            C2[ii] = abs(np.vdot(f2, f2))

        eps = np.spacing(1)
        FSC = C / np.sqrt(C1 * C2 + eps)
        return FSC

    def _compute_randomized(self):
        """
        Randomize phases above ``cutoff_shell`` and compute FSC_rand and FSC_corr.

        Stores
        ------
        FSC_rand : ndarray
        FSC_corr : ndarray
        """
        f_cutoff_cpx = self.cutoff_shell / self.fnyquist * 0.5
        rng = np.random.default_rng(self.random_seed)

        rand1 = self._randomize_phases_above(self.img1, f_cutoff_cpx, rng)
        rand2 = self._randomize_phases_above(self.img2, f_cutoff_cpx, rng)

        self.FSC_rand = self._fsc_from_arrays(rand1, rand2)

        eps = np.finfo(np.float64).eps
        fsc_obs = np.asarray(self.FSC_obs, dtype=np.float64)
        fsc_rand = np.asarray(self.FSC_rand, dtype=np.float64)
        fsc_corr = (fsc_obs - fsc_rand) / np.maximum(1.0 - fsc_rand, eps)
        self.FSC_corr = np.clip(fsc_corr, -1.0, 1.0)

    def plot(self):
        """
        Plot FSC_obs, FSC_rand, FSC_corr, and the bias (FSC_obs − FSC_rand).

        Left panel: all three FSC curves and the threshold *T*.
        Right panel: ``FSC_obs − FSC_rand`` (overfitting bias).

        Saves ``RandomFSC.png``.

        Returns
        -------
        fn : ndarray
            Spatial frequencies normalised by the Nyquist frequency.
        FSC_obs : ndarray
            Standard FSC.
        FSC_rand : ndarray
            Phase-randomized FSC.
        FSC_corr : ndarray
            Corrected FSC.
        T : ndarray
            Threshold curve.
        """
        print("Calling method plot from the class RandomFSC")
        fn       = self.f / self.fnyquist
        fsc_obs  = np.asarray(self.FSC_obs).real
        fsc_rand = np.asarray(self.FSC_rand).real
        fsc_corr = np.asarray(self.FSC_corr).real
        T        = np.asarray(self.T)
        cutoff_fn = self.cutoff_shell / self.fnyquist
        show_random_fsc_curve(fn, fsc_obs, fsc_rand, fsc_corr, T, cutoff_fn, self.ndim)
        return fn, fsc_obs, fsc_rand, fsc_corr, T
