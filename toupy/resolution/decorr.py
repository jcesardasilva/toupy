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
    from a **single** 2-D image (or from a set of 2-D slices sampled from a
    3-D volume).

    Parameters
    ----------
    image : ndarray
        A 2-dimensional array (single image) **or** a 3-dimensional array
        (tomographic volume).  When a 3-D volume is supplied the algorithm
        is applied slice-by-slice along ``axis`` and summary statistics are
        reported.
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
    axis : int, optional
        Axis along which slices are taken when ``image`` is 3-D.
        Default ``0`` (first axis, i.e. slices are ``image[i, :, :]``).
    n_slices : int, optional
        Number of evenly-spaced slices to sample from the 3-D volume
        (or from the sub-range defined by ``slice_range``).
        Ignored when ``image`` is 2-D.  Default ``10``.
    slice_range : tuple of int or None, optional
        ``(start, stop)`` slice indices along ``axis`` that define the
        sub-range from which slices are sampled, following standard
        Python / NumPy half-open convention (``start`` inclusive,
        ``stop`` exclusive).  Negative indices are supported and are
        resolved against the axis length before use.  ``None`` (default)
        samples the full extent of the volume.

    Attributes
    ----------
    r_values : ndarray
        Radial spatial frequencies (cycles/pixel) at which the
        correlation was evaluated.  For 3-D input this is taken from the
        last slice processed.
    A : ndarray
        Normalised ring correlation A(r).  For 3-D input: result of the
        last slice processed.
    d : ndarray
        Decorrelation function d(r) = 1 − A(r).  For 3-D input: result
        of the last slice processed.
    r_res : float
        Estimated resolution spatial frequency (cycles/pixel).
        For 3-D input: median over all sampled slices.
    resolution_px : float
        Estimated resolution in pixels (= 1 / r_res).
        For 3-D input: median over all sampled slices.
    resolution : float
        Estimated resolution in physical units (= pixel_size / r_res).
        For 3-D input: median over all sampled slices.
    resolutions_px : ndarray or None
        Per-slice resolution estimates in pixels.  ``None`` for 2-D input.
    resolution_px_mean : float or None
        Mean per-slice resolution in pixels.  ``None`` for 2-D input.
    resolution_px_median : float or None
        Median per-slice resolution in pixels.  ``None`` for 2-D input.
    resolution_px_std : float or None
        Standard deviation of per-slice resolutions in pixels.
        ``None`` for 2-D input.

    References
    ----------
    .. [1] A. Descloux, K. S. Grußmayer, and A. Radenovic, "Parameter-free
       image resolution estimation based on decorrelation analysis",
       Nature Methods 16, 918-924 (2019).
       https://doi.org/10.1038/s41592-019-0515-7
    """

    def __init__(
        self,
        image,
        pixel_size=1.0,
        n_r=100,
        threshold=0.15,
        apod_width=20,
        axis=0,
        n_slices=10,
        slice_range=None,
    ):
        print("Calling the class ImageDecorr")
        self.image = np.asarray(image, dtype=np.float64)
        if self.image.ndim not in (2, 3):
            raise ValueError("ImageDecorr requires a 2-D or 3-D array.")
        self.pixel_size  = float(pixel_size)
        self.n_r         = int(n_r)
        self.threshold   = float(threshold)
        self.apod_width  = int(apod_width)
        self.axis        = int(axis)
        self.n_slices    = int(n_slices)
        self.slice_range = slice_range   # validated in _run_3d

        # Initialise per-slice attributes (populated only for 3-D input)
        self.resolutions_px    = None
        self.resolution_px_mean   = None
        self.resolution_px_median = None
        self.resolution_px_std    = None

        p0 = time.time()

        if self.image.ndim == 2:
            self._run_2d(self.image)
        else:
            self._run_3d()

        print(f"Done. Time elapsed: {time.time() - p0:.2f}s")

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    def _run_2d(self, img2d):
        """Run the analysis on a single 2-D image and store results."""
        self.nr, self.nc = img2d.shape
        print(f"  Image size : {self.nr} × {self.nc} pixels")
        print(f"  Pixel size : {self.pixel_size}")
        print(f"  Threshold  : {self.threshold}")

        # Temporarily point self.image at the 2-D slice so _apodize works
        _saved = self.image
        self.image = img2d
        self.r_values, self.A, self.d = self._compute()
        self.r_res, self.resolution_px, self.resolution = self._find_resolution()
        self.image = _saved

        print(
            f"  Resolution : {self.resolution_px:.1f} px  "
            f"({self.resolution:.4g} in physical units)"
        )

    def _run_3d(self):
        """
        Apply the 2-D algorithm to evenly-spaced slices along ``self.axis``.

        The set of candidate slice indices is taken from ``self.slice_range``
        when provided, otherwise the full axis length is used.
        ``self.n_slices`` slices are sampled evenly from that range.

        Summary statistics (mean, median, std) are stored as instance
        attributes.  ``self.r_res``, ``self.resolution_px``, and
        ``self.resolution`` are set to the median values so that the
        scalar interface remains consistent.
        """
        vol  = self.image
        n_ax = vol.shape[self.axis]

        # --- resolve slice_range ---
        if self.slice_range is None:
            i_start, i_stop = 0, n_ax
        else:
            try:
                i_start, i_stop = self.slice_range
            except (TypeError, ValueError):
                raise ValueError(
                    "slice_range must be a (start, stop) tuple of integers."
                )
            # Resolve negative indices
            if i_start < 0:
                i_start = max(0, n_ax + i_start)
            if i_stop < 0:
                i_stop = max(0, n_ax + i_stop)
            i_start = int(np.clip(i_start, 0, n_ax))
            i_stop  = int(np.clip(i_stop,  0, n_ax))
            if i_start >= i_stop:
                raise ValueError(
                    f"slice_range ({i_start}, {i_stop}) is empty after "
                    f"resolving against axis length {n_ax}."
                )

        # Choose evenly-spaced indices within [i_start, i_stop)
        n_avail = i_stop - i_start
        n       = min(self.n_slices, n_avail)
        indices = np.round(
            np.linspace(i_start, i_stop - 1, n)
        ).astype(int)
        # Remove duplicates that can appear on very thin sub-ranges
        indices = np.unique(indices)

        range_str = (
            f"[{i_start}:{i_stop}]"
            if self.slice_range is not None
            else "full range"
        )
        print(
            f"  Volume shape : {vol.shape}  —  axis={self.axis}, "
            f"{range_str}, sampling {len(indices)} slice(s)"
        )
        print(f"  Pixel size   : {self.pixel_size}")
        print(f"  Threshold    : {self.threshold}")

        res_px_list = []
        for idx in indices:
            sl = np.take(vol, idx, axis=self.axis)  # always 2-D
            self._run_2d(sl)                         # sets self.r_values, self.A, self.d, …
            res_px_list.append(self.resolution_px)
            print(
                f"    Slice {idx:4d} → {self.resolution_px:.1f} px "
                f"({self.resolution:.4g} in physical units)"
            )

        self.resolutions_px       = np.array(res_px_list)
        self.resolution_px_mean   = float(np.mean(self.resolutions_px))
        self.resolution_px_median = float(np.median(self.resolutions_px))
        self.resolution_px_std    = float(np.std(self.resolutions_px))

        # Set scalar summary to median
        self.resolution_px = self.resolution_px_median
        self.resolution    = self.resolution_px_median * self.pixel_size
        self.r_res         = 1.0 / self.resolution_px_median

        print(
            f"  Resolution (median) : {self.resolution_px_median:.1f} ± "
            f"{self.resolution_px_std:.1f} px  "
            f"(mean {self.resolution_px_mean:.1f} px)  "
            f"({self.resolution:.4g} in physical units)"
        )

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
        Plot the decorrelation analysis result.

        For a **2-D** input (or after 3-D analysis) the decorrelation
        curve A(r) is shown together with the threshold line and the
        resolution estimate marker.

        For a **3-D** input a histogram of the per-slice resolution
        estimates is shown in addition to the A(r) curve of the last
        processed slice, so that the spread across slices is visible.

        Returns
        -------
        r_values : ndarray
            Radial spatial frequencies (cycles/pixel).
        A : ndarray
            Normalised ring correlation A(r).
        d : ndarray
            Decorrelation function d(r) = 1 − A(r).
        resolution_px : float
            Estimated resolution in pixels (median for 3-D input).
        """
        print("Calling method plot from the class ImageDecorr")
        r  = self.r_values
        A  = self.A
        d  = self.d
        fn = r / 0.5   # normalise to [0, 1] (Nyquist = 1)

        is3d = self.resolutions_px is not None

        if is3d:
            # Two-panel figure: A(r) curve on left, histogram on right
            if isnotebook():
                fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            else:
                fig, (ax, ax2) = plt.subplots(1, 2)

            ax2.hist(self.resolutions_px, bins="auto", color="steelblue",
                     edgecolor="white")
            ax2.axvline(self.resolution_px_median, color="k", linestyle="-",
                        label=f"Median = {self.resolution_px_median:.1f} px")
            ax2.axvline(self.resolution_px_mean, color="r", linestyle="--",
                        label=f"Mean = {self.resolution_px_mean:.1f} px")
            ax2.set_xlabel("Resolution (pixels)")
            ax2.set_ylabel("Count")
            ax2.set_title(
                f"Per-slice resolution  (n={len(self.resolutions_px)},  "
                f"std={self.resolution_px_std:.1f} px)"
            )
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.5)
            title_suffix = "  [last slice]"
        else:
            if isnotebook():
                fig = plt.figure(figsize=(8, 6))
            else:
                fig = plt.figure()
            plt.clf()
            ax = fig.add_subplot(111)
            title_suffix = ""

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
        ax.set_title(f"Image Decorrelation Analysis{title_suffix}")
        ax.grid(True, linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig("ImageDecorr.png", bbox_inches="tight")
        if isnotebook():
            from IPython import display
            display.display(fig)
            plt.close(fig)
        else:
            plt.show(block=False)
        return r, A, d, self.resolution_px
