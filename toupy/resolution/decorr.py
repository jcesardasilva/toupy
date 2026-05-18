#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single-image resolution estimation via decorrelation analysis.
"""

# standard library
import time

# third party
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

# local
from ..utils.plot_utils import plt, isnotebook

__all__ = ["ImageDecorr"]


class ImageDecorr:
    """
    Single-image resolution estimation via decorrelation analysis [1]_.

    For each radial spatial frequency *r* the image is correlated with its
    phase-normalised ring-filtered self.  In the signal-dominated band the
    phase is coherent and the correlation is high; beyond the resolution
    limit the Fourier phases are noise-dominated and the correlation drops.
    The highest spatial frequency at which the normalised correlation
    exceeds ``threshold`` is taken as the resolution limit.

    No second image or half-dataset is required — the estimate is obtained
    from a **single** 2-D image.

    Parameters
    ----------
    image : ndarray
        A 2-dimensional array containing the image to analyse.
    pixel_size : float, optional
        Physical size of one pixel (in any consistent unit, e.g. nm).
        Used to convert the resolution from pixels to physical units.
        Default ``1.0`` (result in pixels).
    n_r : int, optional
        Number of radial frequency bins between the lowest non-zero
        frequency and the Nyquist limit (0.5 cycles/pixel).
        Default ``100``.
    threshold : float, optional
        Correlation threshold used to define the resolution limit.
        The default ``0.15`` matches the FSC/FRC half-bit criterion.
    apod_width : int, optional
        Width in pixels of the Hanning apodization applied to the image
        edges before computing the FFT.  Set to ``0`` to disable
        apodization.  Default ``20``.

    Attributes
    ----------
    r_values : ndarray
        Radial spatial frequencies (cycles/pixel) at which the
        correlation was evaluated.
    A : ndarray
        Normalised ring correlation A(r).
    d : ndarray
        Decorrelation function d(r) = 1 − A(r).
    r_res : float
        Estimated resolution spatial frequency (cycles/pixel).
    resolution_px : float
        Estimated resolution in pixels (= 1 / r_res).
    resolution : float
        Estimated resolution in physical units (= pixel_size / r_res).

    References
    ----------
    .. [1] A. Descloux, K. S. Grußmayer, and A. Radenovic, "Parameter-free
       image resolution estimation based on decorrelation analysis",
       Nature Methods 16, 918-924 (2019).
       https://doi.org/10.1038/s41592-019-0515-7
    """

    def __init__(self, image, pixel_size=1.0, n_r=100, threshold=0.15, apod_width=20):
        print("Calling the class ImageDecorr")
        self.image = np.asarray(image, dtype=np.float64)
        if self.image.ndim != 2:
            raise ValueError("ImageDecorr requires a 2-D image.")
        self.nr, self.nc = self.image.shape
        self.pixel_size  = float(pixel_size)
        self.n_r         = int(n_r)
        self.threshold   = float(threshold)
        self.apod_width  = int(apod_width)

        print(f"  Image size : {self.nr} × {self.nc} pixels")
        print(f"  Pixel size : {self.pixel_size}")
        print(f"  Threshold  : {self.threshold}")

        p0 = time.time()
        self.r_values, self.A, self.d = self._compute()
        self.r_res, self.resolution_px, self.resolution = self._find_resolution()
        print(f"Done. Time elapsed: {time.time() - p0:.2f}s")
        print(f"  Resolution : {self.resolution_px:.1f} px  "
              f"({self.resolution:.4g} in physical units)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apodize(self):
        """
        Apply a Hanning apodization window and subtract the mean.

        Returns
        -------
        img : ndarray
            Zero-mean apodized copy of the image.
        """
        img = self.image.copy()
        if self.apod_width > 0:
            # Tukey-like window: cosine taper of width apod_width on each side
            wr = np.ones(self.nr)
            wc = np.ones(self.nc)
            aw = self.apod_width
            half_h = np.hanning(2 * aw)
            wr[:aw] = half_h[:aw]
            wr[-aw:] = half_h[aw:]
            wc[:aw] = half_h[:aw]
            wc[-aw:] = half_h[aw:]
            img *= np.outer(wr, wc)
        img -= img.mean()
        return img

    def _compute(self):
        """
        Compute the phase-normalised ring correlation A(r).

        For each ring at radius *r* (cycles/pixel):

        1. Divide the FFT by its modulus (keep only the phase):
           ``F_n = F / |F|``
        2. Apply a ring mask of width ``dr`` centred on *r*.
        3. Back-transform to real space: ``I_n_r = Re(IFFT(F_n * ring))``.
        4. Compute the Pearson correlation between the apodized image
           and ``I_n_r``.

        Returns
        -------
        r_values : ndarray
            Radial frequency axis (cycles/pixel).
        A : ndarray
            Normalised ring correlation.
        d : ndarray
            Decorrelation function ``d = 1 − A``.
        """
        img = self._apodize()

        # Phase-normalised FFT
        F   = fft2(img)
        eps = np.finfo(np.float64).tiny
        F_n = F / (np.abs(F) + eps)

        # Radial coordinate map (cycles/pixel, FFT-shifted layout)
        fy = fftfreq(self.nr)   # shape (nr,)
        fx = fftfreq(self.nc)   # shape (nc,)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        R = np.sqrt(FX ** 2 + FY ** 2)

        # Frequency axis
        r_min = 1.0 / min(self.nr, self.nc)
        r_max = 0.5   # Nyquist
        r_values = np.linspace(r_min, r_max, self.n_r)
        dr = r_values[1] - r_values[0]

        # Precompute image statistics (zero-mean, already done in _apodize)
        img_flat = img.ravel()
        img_rms  = np.sqrt(np.mean(img_flat ** 2))
        if img_rms < eps:
            return r_values, np.zeros(self.n_r), np.ones(self.n_r)

        A = np.zeros(self.n_r)
        for i, r in enumerate(r_values):
            ring_mask = (R >= r - 0.5 * dr) & (R < r + 0.5 * dr)
            if not np.any(ring_mask):
                continue
            F_n_r = F_n * ring_mask
            I_n_r = np.real(ifft2(F_n_r))
            ring_rms = np.sqrt(np.mean(I_n_r ** 2))
            if ring_rms < eps:
                continue
            A[i] = np.mean(img_flat * I_n_r.ravel()) / (img_rms * ring_rms)

        d = 1.0 - A
        return r_values, A, d

    def _find_resolution(self):
        """
        Find the resolution spatial frequency from the correlation curve.

        The resolution is the **highest** spatial frequency r for which
        A(r) ≥ ``self.threshold``.  If the correlation never exceeds
        the threshold the Nyquist frequency is returned as a conservative
        (worst-case) estimate.

        Returns
        -------
        r_res : float
            Resolution frequency (cycles/pixel).
        resolution_px : float
            Resolution in pixels (= 1 / r_res).
        resolution : float
            Resolution in physical units (= pixel_size * resolution_px).
        """
        above = self.A >= self.threshold
        if not np.any(above):
            # Correlation never reaches threshold — return Nyquist as limit
            r_res = self.r_values[-1]
        else:
            r_res = self.r_values[above][-1]

        resolution_px   = 1.0 / r_res
        resolution      = resolution_px * self.pixel_size
        return r_res, resolution_px, resolution

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(self):
        """
        Plot the decorrelation curve A(r) and mark the resolution estimate.

        Returns
        -------
        r_values : ndarray
            Radial spatial frequencies (cycles/pixel).
        A : ndarray
            Normalised ring correlation A(r).
        d : ndarray
            Decorrelation function d(r) = 1 − A(r).
        resolution_px : float
            Estimated resolution in pixels.
        """
        print("Calling method plot from the class ImageDecorr")
        r  = self.r_values
        A  = self.A
        d  = self.d
        fn = r / 0.5   # normalise to [0, 1] (Nyquist = 1)

        if isnotebook():
            fig = plt.figure(figsize=(8, 6))
        else:
            fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.plot(fn, A, "-b", label="A(r)  (ring correlation)")
        ax.plot(fn, d, "-g", label="d(r) = 1 − A(r)")
        ax.axhline(self.threshold, color="r", linestyle="--",
                   label=f"Threshold = {self.threshold}")
        ax.axvline(self.r_res / 0.5, color="k", linestyle=":",
                   label=f"Resolution ≈ {self.resolution_px:.1f} px")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Spatial frequency / Nyquist")
        ax.set_ylabel("Normalised correlation")
        ax.set_title("Image Decorrelation Analysis")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.savefig("ImageDecorr.png", bbox_inches="tight")
        if isnotebook():
            from IPython import display
            display.display(fig)
            plt.close(fig)
        else:
            plt.show(block=False)
        return r, A, d, self.resolution_px
