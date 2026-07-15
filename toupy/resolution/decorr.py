#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single-image resolution estimation via decorrelation analysis.

This module is a Python re-implementation of the reference MATLAB code
released by the original authors (github.com/Ades91/ImDecorr,
``funcs/getDcorr.m``, ``funcs/getCorrcoef.m``, ``funcs/getDcorrLocalMax.m``),
translated function-for-function rather than reconstructed from a prose
description of the method. See the module-level references below.
"""

# standard library
import time
import warnings

# third party
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

# local
from ..utils.plot_utils import plt, isnotebook

__all__ = ["ImageDecorr"]


class ImageDecorr:
    """
    Single-image resolution estimation via decorrelation analysis [1]_.

    The image's Fourier spectrum is normalised to unit amplitude at every
    frequency (keeping only its phase), and compared -- via a normalised
    Fourier-domain cross-correlation coefficient -- against the original,
    unnormalised spectrum restricted to a series of cumulative low-pass
    disks of increasing radius. The resulting decorrelation curve rises
    while the disk still admits mostly true signal and falls once it
    admits mostly noise. Because low-frequency image structure can create
    spurious local maxima at low radii, the whole analysis is repeated
    after applying a sequence of increasingly aggressive Gaussian
    high-pass pre-filters to the image, and the highest-frequency
    significant peak found across all repetitions defines the resolution.

    No correlation threshold is chosen anywhere in this procedure --
    "parameter-free" (the title of ref. [1]_) refers specifically to the
    absence of such a threshold, unlike FSC/FRC- or MTF-based criteria.

    No second image or half-dataset is required -- the estimate is obtained
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
        Number of radial frequency bins swept between the lowest non-zero
        frequency and the Nyquist limit (0.5 cycles/pixel), at each
        refinement stage.  The reference implementation enforces a
        minimum of 30; values below this are silently raised.
        Default ``50`` (matches the reference implementation's default).
    n_g : int, optional
        Number of high-pass pre-filtering strengths swept per refinement
        stage (``Ng`` in the reference implementation).  Two refinement
        stages are always used, as in the reference.  Default ``10``.
    threshold : float, optional
        Deprecated and unused.  Earlier versions of this class defined
        the resolution as the highest frequency at which the correlation
        curve crossed a fixed threshold; this does not match the
        published, threshold-free algorithm and has been removed.  The
        parameter is still accepted for backward compatibility and
        triggers a ``DeprecationWarning`` if set to a non-default value.
    apod_width : int, optional
        Width in pixels of the Hanning apodization applied to the image
        edges before computing the FFT.  Set to ``0`` to disable
        apodization.  Default ``20`` (matches the reference
        implementation's default).
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
        Radial spatial frequencies (cycles/pixel) at which the base
        (non-high-pass-filtered) decorrelation curve was evaluated.
        For 3-D input this is taken from the last slice processed.
    A : ndarray
        Base decorrelation curve ``A(r)`` (before high-pass refinement).
        For 3-D input: result of the last slice processed.
    d : ndarray
        ``d(r) = 1 - A(r)``.  For 3-D input: result of the last slice
        processed.
    kc_candidates : ndarray
        Peak frequencies (cycles/pixel) found at every refinement stage
        (base curve plus every high-pass pre-filtering strength tried).
        The resolution is the maximum of the entries that pass the
        SNR >= 0.05 quality filter; if none pass, the lowest swept
        frequency is reported instead.  For 3-D input: result of the last
        slice processed.
    snr_candidates : ndarray
        Peak amplitudes corresponding to ``kc_candidates``.
    r_res : float
        Estimated resolution spatial frequency (cycles/pixel), i.e. the
        highest-frequency accepted peak across all refinement stages.
        For 3-D input: median over all sampled slices.
    resolution_px : float
        Estimated resolution in pixels (= 1 / r_res), full-period
        convention (matching ``d_full = 1 / xi_c``).
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
    .. [2] Reference MATLAB implementation:
       https://github.com/Ades91/ImDecorr
    """

    def __init__(
        self,
        image,
        pixel_size=1.0,
        n_r=50,
        n_g=10,
        threshold=None,
        apod_width=20,
        axis=0,
        n_slices=10,
        slice_range=None,
    ):
        print("Calling the class ImageDecorr")
        self.image = np.asarray(image, dtype=np.float64)
        if self.image.ndim not in (2, 3):
            raise ValueError("ImageDecorr requires a 2-D or 3-D array.")

        if threshold is not None:
            warnings.warn(
                "ImageDecorr: the 'threshold' parameter is deprecated and "
                "unused. The published decorrelation-analysis algorithm "
                "(Descloux et al. 2019) is parameter-free: resolution is "
                "defined by the highest-frequency peak of the "
                "decorrelation curve found via repeated high-pass "
                "pre-filtering, not by a fixed correlation threshold.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.pixel_size  = float(pixel_size)
        self.n_r         = max(int(n_r), 30)   # reference enforces >= 30
        self.n_g         = max(int(n_g), 5)    # reference enforces >= 5
        self.apod_width  = int(apod_width)
        self.axis        = int(axis)
        self.n_slices    = int(n_slices)
        self.slice_range = slice_range   # validated in _run_3d

        # Initialise per-slice attributes (populated only for 3-D input)
        self.resolutions_px    = None
        self.resolution_px_mean   = None
        self.resolution_px_median = None
        self.resolution_px_std    = None

        # Warning flag: fire at most once per instance
        self._tomo_warned = False

        p0 = time.time()

        if self.image.ndim == 2:
            self._run_2d(self.image)
        else:
            self._run_3d()

        print(f"Done. Time elapsed: {time.time() - p0:.2f}s")

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    def _warn_if_tomo(self, img):
        """
        Emit a ``UserWarning`` if *img* looks like a tomographic slice.

        The check tests whether the fraction of negative-valued pixels
        exceeds 5 %.  Tomographic reconstructions (especially filtered
        back-projection) typically contain many negative values due to
        correlated noise, whereas raw 2-D microscopy images (for which
        the decorrelation algorithm was designed) rarely do.

        The warning is emitted at most **once per instance** — subsequent
        calls after the first warning are silently ignored.

        Parameters
        ----------
        img : ndarray
            Two-dimensional array to test.

        Warns
        -----
        UserWarning
            If the negative-pixel fraction exceeds 0.05 and the warning
            has not already been issued for this instance.
        """
        if self._tomo_warned:
            return
        neg_frac = float(np.sum(img < 0)) / img.size
        if neg_frac > 0.05:
            warnings.warn(
                "ImageDecorr: the supplied image appears to be a tomographic "
                "reconstruction slice (negative-pixel fraction = "
                f"{neg_frac:.1%} > 5 %).  "
                "The decorrelation algorithm was designed for raw 2-D "
                "microscopy images; applied to reconstructed slices it gives "
                "unreliable results because correlated noise biases the "
                "correlation curve toward low spatial frequencies.  "
                "For tomographic resolution estimation use FSC/SSNR or "
                "LocalFSC instead.",
                UserWarning,
                stacklevel=3,
            )
            self._tomo_warned = True

    def _run_2d(self, img2d):
        """Run the analysis on a single 2-D image and store results."""
        self._warn_if_tomo(img2d)
        self.nr, self.nc = img2d.shape
        print(f"  Image size : {self.nr} x {self.nc} pixels")
        print(f"  Pixel size : {self.pixel_size}")

        # Temporarily point self.image at the 2-D slice so _apodize works
        _saved = self.image
        self.image = img2d
        (
            self.r_values,
            self.A,
            self.d,
            self.kc_candidates,
            self.snr_candidates,
            self.r_res,
            self.resolution_px,
            self.resolution,
        ) = self._compute()
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

        # Check the full volume for tomographic reconstruction characteristics
        # before the per-slice loop; call _warn_if_tomo on the first slice so
        # the per-slice loop skips subsequent warnings via the flag.
        vol_neg_frac = float(np.sum(vol < 0)) / vol.size
        if vol_neg_frac > 0.05 and not self._tomo_warned:
            first_sl = np.take(vol, indices[0], axis=self.axis)
            self._warn_if_tomo(first_sl)

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
        Subtract the mean, then apply a Hanning apodization window.

        The order matters.  Windowing an image that still carries a DC
        pedestal imprints the window's own taper onto the spectrum; once the
        pedestal dominates the contrast that taper *is* the strongest
        structure present, so the analysis measures the window rather than
        the image and returns a "resolution" that merely tracks
        ``apod_width``.  Removing the mean first leaves the window
        modulating only genuine structure, and lets the blank-image guard in
        ``_compute`` fire as documented.

        Returns
        -------
        img : ndarray
            Zero-mean apodized copy of the image.
        """
        img = self.image.copy()
        img -= img.mean()
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
        return img

    @staticmethod
    def _corrcoef(I1, I2, norm1=None, norm2=None, eps=None):
        """
        Normalised cross-correlation coefficient between two Fourier
        spectra, following ``getCorrcoef.m`` in the reference
        implementation::

            cc = sum(Re(I1 * conj(I2))) / (norm(I1) * norm(I2))

        ``norm1``/``norm2`` may be supplied pre-computed (e.g. the norm of
        a spectrum that does not change within an inner loop) to avoid
        recomputation; otherwise they default to the L2 norm of ``I1``
        and ``I2`` respectively.
        """
        if eps is None:
            eps = np.finfo(np.float64).tiny
        if norm1 is None:
            norm1 = np.sqrt(np.sum(np.abs(I1) ** 2))
        if norm2 is None:
            norm2 = np.sqrt(np.sum(np.abs(I2) ** 2))
        if norm1 < eps or norm2 < eps:
            return 0.0
        cc = float(np.sum(np.real(I1 * np.conj(I2))) / (norm1 * norm2))
        return cc

    @staticmethod
    def _local_max(curve):
        """
        Locate the first genuine local maximum of ``curve``, following
        ``getDcorrLocalMax.m`` in the reference implementation.

        Starting from the global maximum, the trailing tail of the curve
        is progressively discarded: if the maximum sits at the very end
        of the (remaining) array the curve is still rising there and is
        not yet a real interior peak, so the last point is dropped and
        the maximum re-evaluated; this repeats until either the maximum
        falls at the very first point, or its prominence relative to the
        remaining tail (``amplitude - min(tail)``) reaches at least
        ``5e-4``, at which point it is accepted as the peak.

        Parameters
        ----------
        curve : array_like
            1-D decorrelation curve.

        Returns
        -------
        idx : int
            Index of the accepted local maximum within the *original*
            ``curve``.
        amp : float
            Value of ``curve`` at ``idx``.
        """
        c = list(np.asarray(curve, dtype=np.float64))
        n0 = len(c)
        if n0 < 2:
            return 0, (c[0] if c else 0.0)

        while len(c) > 1:
            amp = max(c)
            idx = c.index(amp)
            if idx == len(c) - 1:
                c.pop()
                continue
            if idx == 0:
                break
            if amp - min(c[idx:]) >= 5e-4:
                break
            c.pop()

        amp = max(c)
        idx = c.index(amp)
        return idx, amp

    def _compute(self):
        """
        Run the full decorrelation-analysis algorithm (Descloux et al.
        2019) on ``self.image`` and return all result arrays.

        This follows ``getDcorr.m`` step by step:

        1. Fourier-transform the apodized image and build the
           phase-normalised spectrum ``F_n = F / |F|``, restricted to the
           unit (Nyquist) disk.
        2. Sweep a cumulative low-pass radius ``rn`` from just above 0 to
           1 (Nyquist); at each ``rn``, mask ``F_n`` with the disk
           ``R < rn`` and compute its normalised cross-correlation
           coefficient against the full, unmasked spectrum ``F``. This
           is the base decorrelation curve, ``A(rn)``.
        3. Locate the first genuine local maximum of ``A`` (not a
           threshold crossing).
        4. Repeat steps 2-3 after pre-filtering ``F`` (not ``F_n``) with
           a sequence of increasingly aggressive Gaussian high-pass
           filters, to reject spurious low-frequency maxima. Two
           refinement passes are performed, narrowing the search range
           around the best peak found after the first pass, exactly as
           in the reference implementation.
        5. Discard any candidate peak whose amplitude (a proxy for its
           SNR) is below 0.05, and take the highest surviving frequency
           as the resolution-defining peak.  If no candidate survives,
           report the lowest swept frequency (worst resolution).

        The image is mean-subtracted *before* apodization (see
        ``_apodize``): windowing a DC pedestal would otherwise imprint the
        window taper on the spectrum and make the result track
        ``apod_width`` rather than the image content.

        No fixed correlation threshold is used anywhere in this
        procedure. Frequencies are expressed internally on the
        reference implementation's own normalised scale (0 to 1,
        Nyquist = 1) and converted back to cycles/pixel (Nyquist = 0.5,
        matching the rest of this module) only for the returned values.

        Returns
        -------
        r_values : ndarray
            Radial frequency axis of the base curve (cycles/pixel).
        A : ndarray
            Base decorrelation curve (before high-pass refinement).
        d : ndarray
            ``1 - A``.
        kc_candidates : ndarray
            Peak frequencies found at every stage (cycles/pixel).
        snr_candidates : ndarray
            Peak amplitudes corresponding to ``kc_candidates``.
        r_res : float
            Resolution frequency (cycles/pixel): the highest-frequency
            accepted peak.
        resolution_px : float
            Resolution in pixels (= 1 / r_res), full-period convention.
        resolution : float
            Resolution in physical units.
        """
        eps = np.finfo(np.float64).tiny
        img = self._apodize()

        F   = fft2(img)
        F_n = F / (np.abs(F) + eps)

        # Radial frequency: cycles/pixel (Nyquist = 0.5), then normalised
        # to the reference implementation's own convention (Nyquist = 1).
        fy = fftfreq(self.nr)
        fx = fftfreq(self.nc)
        FY, FX = np.meshgrid(fy, fx, indexing="ij")
        R_cyc = np.sqrt(FX ** 2 + FY ** 2)
        Rn    = R_cyc / 0.5

        disk0 = Rn < 1.0
        F_n   = F_n * disk0
        F0    = F * disk0

        n_r, n_g = self.n_r, self.n_g
        rn_values = np.linspace(1.0 / max(self.nr, self.nc), 1.0, n_r)

        c0 = np.sqrt(np.sum(np.abs(F0) ** 2))
        if c0 < eps:
            # Blank/constant image: nothing to resolve.  Report the LOWEST
            # frequency (worst resolution), as getDcorr.m does when no valid
            # peak exists -- never the optimistic Nyquist end, which would
            # claim the best possible resolution for an empty image.
            r_values = rn_values * 0.5
            A = np.zeros(n_r)
            d = np.ones(n_r)
            return (
                r_values, A, d,
                np.zeros(1), np.zeros(1),
                r_values[0], 1.0 / r_values[0],
                self.pixel_size / r_values[0],
            )

        # --- base curve d0(rn): no high-pass pre-filtering ---
        A0 = np.zeros(n_r)
        for i, rn in enumerate(rn_values):
            mask = Rn < rn
            A0[i] = self._corrcoef(F0, F_n * mask, norm1=c0)
        idx0, snr0 = self._local_max(A0)
        kc0 = rn_values[idx0]

        kc_candidates  = [kc0]
        snr_candidates = [snr0]

        # --- refinement: repeated Gaussian high-pass pre-filtering ---
        g_max = 2.0 / kc0 if kc0 > 0 else max(self.nr, self.nc) / 2.0
        g_values = np.concatenate((
            [max(self.nr, self.nc) / 4.0],
            np.exp(np.linspace(np.log(g_max), np.log(0.15), n_g)),
        ))
        r_lo, r_hi = rn_values[0], rn_values[-1]

        for stage in range(2):  # two-step refinement, as in getDcorr.m
            r_sweep = np.linspace(r_lo, r_hi, n_r)
            stage_kc = []
            for g in g_values:
                Ir = F0 * (1.0 - np.exp(-2.0 * g ** 2 * Rn ** 2))
                c_r = np.sqrt(np.sum(np.abs(Ir) ** 2))
                d_g = np.zeros(n_r)
                for i, rn in enumerate(r_sweep):
                    mask = Rn < rn
                    d_g[i] = self._corrcoef(Ir, F_n * mask, norm1=c_r)
                idx, snr = self._local_max(d_g)
                kc_candidates.append(r_sweep[idx])
                snr_candidates.append(snr)
                stage_kc.append(r_sweep[idx])

            if stage == 0:
                # Narrow the high-pass and radius search ranges around
                # the best (highest-frequency) peak found in this stage,
                # matching the "if refin == 1" block of getDcorr.m.
                best = int(np.argmax(stage_kc))
                if best == 0:
                    g1, g2 = max(self.nr, self.nc), g_values[0]
                else:
                    g1 = g_values[max(best - 1, 0)]
                    g2 = g_values[min(best, len(g_values) - 1)]
                g_values = np.exp(np.linspace(np.log(g1), np.log(g2), n_g))
                r_lo = max(0.0, stage_kc[best] - (rn_values[1] - rn_values[0]))
                r_hi = min(1.0, stage_kc[best] + 0.4)

        # kc0/snr0 are already the first entry, added before the refinement
        # loop; the reference appends them once, so do not duplicate here.
        kc_candidates  = np.array(kc_candidates)
        snr_candidates = np.array(snr_candidates)

        # Discard peaks too weak to be meaningful, then take the
        # highest-frequency survivor.  If nothing survives there is no
        # resolvable structure: fall back to the lowest frequency (worst
        # resolution) as getDcorr.m does, rather than to max() over the
        # rejected candidates, which would report the most optimistic peak
        # precisely when none of them is trustworthy.
        valid = snr_candidates >= 0.05
        if np.any(valid):
            kc_max = float(np.max(kc_candidates[valid]))
        else:
            kc_max = float(rn_values[0])

        # Convert back to cycles/pixel (Nyquist = 0.5) for all outputs.
        r_values = rn_values * 0.5
        A = A0
        d = 1.0 - A0
        kc_candidates_px = kc_candidates * 0.5
        snr_out = snr_candidates

        r_res = max(kc_max * 0.5, eps)
        resolution_px = 1.0 / r_res
        resolution = resolution_px * self.pixel_size

        return (
            r_values, A, d,
            kc_candidates_px, snr_out,
            r_res, resolution_px, resolution,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(self):
        """
        Plot the decorrelation analysis result.

        For a **2-D** input (or after 3-D analysis) the base decorrelation
        curve A(r) is shown together with the accepted resolution peak
        (the highest-frequency candidate surviving the SNR >= 0.05
        filter across all high-pass refinement stages) and, for context,
        the other candidate peaks found during refinement.

        For a **3-D** input a histogram of the per-slice resolution
        estimates is shown in addition to the A(r) curve of the last
        processed slice, so that the spread across slices is visible.

        Returns
        -------
        r_values : ndarray
            Radial spatial frequencies (cycles/pixel).
        A : ndarray
            Base decorrelation curve A(r).
        d : ndarray
            ``1 - A(r)``.
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

        ax.plot(fn, A, "-b", label="A(r)  (base decorrelation curve)")
        ax.plot(fn, d, "-g", label="d(r) = 1 − A(r)")
        if self.kc_candidates.size:
            ax.plot(self.kc_candidates / 0.5, self.snr_candidates, "x",
                    color="0.5", label="Candidate peaks (refinement)")
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
