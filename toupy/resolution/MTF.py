#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modulation Transfer Function (MTF) estimation for X-ray imaging systems.
"""

# standard library
import warnings

# third party
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.ndimage import sobel, gaussian_filter1d

__all__ = ["MTFEstimator"]


class MTFEstimator:
    """
    Modulation Transfer Function (MTF) estimation.

    Two estimation strategies are supported:

    **Edge method** (``method='edge'``):
        Implements the slanted-edge MTF algorithm (ISO 12233 / Burns 2000
        variant).  A sharp edge in the image is located, its orientation
        measured, and an oversampled Edge Spread Function (ESF) is
        constructed by projecting all pixel values onto the edge-normal
        direction.  The ESF is differentiated to give the Line Spread
        Function (LSF), and the MTF is the modulus of the Fourier transform
        of the LSF normalised to unity at zero frequency.

    **Point-source method** (``method='point'``):
        A small bright feature (bead, wire cross-section) is used as an
        approximation to the system Point Spread Function (PSF).  The MTF
        is the modulus of the 2-D FFT of the PSF, radially averaged and
        normalised.

    Parameters
    ----------
    image : ndarray
        A 2-dimensional image containing the edge or point source.
    pixel_size : float, optional
        Physical size of one pixel. Default ``1.0``.
    method : {'edge', 'point'}, optional
        Estimation strategy. Default ``'edge'``.
    roi : tuple of slice or None, optional
        ``(row_slice, col_slice)`` defining the region of interest.
        For ``'edge'``: crop to this region before edge fitting.
        For ``'point'``: crop to this region before bead finding.
        ``None`` uses the full image.
    center : tuple of int or None, optional
        ``(row, col)`` coordinates of the point source centre.  Only used
        when ``method='point'``.  ``None`` finds the brightest pixel.
    roi_size : int, optional
        Half-size (in pixels) of the extraction box around the point
        source.  Only used when ``method='point'``.  Default ``16``.
    oversample : int, optional
        Oversampling factor for the ESF construction (edge method only).
        Typically ``4``.  Default ``4``.

    Attributes
    ----------
    freq : ndarray
        Spatial frequency axis in cycles/pixel.
    MTF : ndarray
        MTF values at each frequency (normalised to 1.0 at zero frequency).
    freq_phys : ndarray
        Spatial frequency in cycles per physical unit (``freq / pixel_size``).
    f50 : float
        Spatial frequency (cycles/pixel) where MTF = 0.5.
    f10 : float
        Spatial frequency (cycles/pixel) where MTF = 0.1.
    resolution_50 : float
        Resolution at MTF = 0.5 in physical units (``pixel_size / f50``).
    resolution_10 : float
        Resolution at MTF = 0.1 in physical units (``pixel_size / f10``).

    References
    ----------
    .. [1] P. B. Burns, "Slanted-edge MTF for digital camera and scanner
       analysis", in *Proc. IS&T 2000 PICS Conference*, pp. 135-138 (2000).
    .. [2] ISO 12233:2023, Photography — Electronic still picture imaging —
       Resolution and spatial frequency responses.
    """

    def __init__(
        self,
        image,
        pixel_size=1.0,
        method='edge',
        roi=None,
        center=None,
        roi_size=16,
        oversample=4,
    ):
        print("Calling the class MTFEstimator")
        image = np.asarray(image, dtype=np.float64)
        if image.ndim != 2:
            raise ValueError("MTFEstimator requires a 2-D image.")
        if method not in ('edge', 'point'):
            raise ValueError("method must be 'edge' or 'point'.")

        self.image = image
        self.pixel_size = float(pixel_size)
        self.method = method
        self.roi = roi
        self.center = center
        self.roi_size = int(roi_size)
        self.oversample = int(oversample)

        if method == 'edge':
            self._from_edge()
        else:
            self._from_point()

        self._find_resolution_frequencies()

        print(f"  MTF method      : {self.method}")
        print(f"  f50 (MTF=0.5)   : {self.f50:.4f} cycles/pixel  "
              f"→ resolution {self.resolution_50:.4g} physical units")
        print(f"  f10 (MTF=0.1)   : {self.f10:.4f} cycles/pixel  "
              f"→ resolution {self.resolution_10:.4g} physical units")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _crop(self):
        """
        Crop the stored image using ``self.roi`` if provided.

        Returns
        -------
        img : ndarray
            Cropped (or full) image.
        """
        if self.roi is not None:
            row_sl, col_sl = self.roi
            return self.image[row_sl, col_sl]
        return self.image

    # ------------------------------------------------------------------
    # Edge method
    # ------------------------------------------------------------------

    def _from_edge(self):
        """
        Estimate the MTF using the slanted-edge (ISO 12233 / Burns 2000) method.

        Steps
        -----
        1. Crop image to ``self.roi``.
        2. Compute gradient magnitude and angle using Sobel filters.
        3. Threshold to retain the top 5 % of gradient-magnitude pixels.
        4. Fit a line through the retained edge pixels.
        5. Compute perpendicular signed distance of every pixel to the line.
        6. Build an oversampled ESF by binning pixel values onto a fine grid.
        7. Smooth the ESF with a Gaussian of sigma=1 oversampled sample.
        8. Differentiate the ESF to get the LSF.
        9. Apply a Hanning window to the LSF.
        10. FFT the LSF; store ``self.freq`` and ``self.MTF``.
        """
        img = self._crop().astype(np.float64)
        nr, nc = img.shape

        # Gradient via Sobel
        gx = sobel(img, axis=1).astype(np.float64)
        gy = sobel(img, axis=0).astype(np.float64)
        gmag = np.sqrt(gx ** 2 + gy ** 2)

        # Top 5 % edge pixels
        thresh = np.percentile(gmag, 95)
        mask = gmag >= thresh

        rows_e, cols_e = np.where(mask)
        if len(rows_e) < 2:
            warnings.warn(
                "MTFEstimator: too few edge pixels found. "
                "Try a cleaner edge or a different ROI.",
                UserWarning,
                stacklevel=2,
            )
            self.freq = np.array([0.0, 0.5])
            self.MTF = np.array([1.0, 0.0])
            return

        # Fit a line: row = slope * col + intercept
        slope, intercept = np.polyfit(cols_e, rows_e, 1)

        # Perpendicular signed distance of every pixel to the fitted line:
        # line: slope*col - row + intercept = 0
        # normal direction: (slope, -1) / sqrt(slope^2 + 1)
        all_rows, all_cols = np.indices(img.shape)
        norm_factor = np.sqrt(slope ** 2 + 1.0)
        dist = (slope * all_cols - all_rows + intercept) / norm_factor
        # dist is in pixels along the edge-normal direction

        # Build oversampled ESF
        d_range = dist.max() - dist.min()
        n_bins = int(np.ceil(d_range * self.oversample)) + 1
        bin_edges = np.linspace(dist.min(), dist.max(), n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        esf = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=np.int64)
        bin_idx = np.searchsorted(bin_edges[1:], dist.ravel(), side='left')
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(esf,    bin_idx, img.ravel())
        np.add.at(counts, bin_idx, 1)
        valid = counts > 0
        esf[valid] /= counts[valid]
        # Fill any empty bins by linear interpolation
        if not np.all(valid):
            esf = np.interp(
                bin_centres,
                bin_centres[valid],
                esf[valid],
            )

        # Smooth ESF
        esf = gaussian_filter1d(esf, sigma=1.0)

        # Differentiate to get LSF
        lsf = np.gradient(esf)

        # Hanning window
        lsf *= np.hanning(len(lsf))

        # FFT → MTF
        MTF_raw = np.abs(fft(lsf))
        freqs_raw = fftfreq(len(lsf)) * self.oversample  # cycles/pixel

        # Keep positive frequencies
        pos = freqs_raw >= 0
        freqs_pos = freqs_raw[pos]
        mtf_pos = MTF_raw[pos]

        # Normalise
        if mtf_pos[0] > 0:
            mtf_pos = mtf_pos / mtf_pos[0]
        else:
            mtf_pos = mtf_pos / (mtf_pos.max() + np.finfo(float).eps)

        # Clip to [0, 0.5] cycles/pixel (Nyquist)
        keep = freqs_pos <= 0.5
        self.freq = freqs_pos[keep]
        self.MTF = mtf_pos[keep]

    # ------------------------------------------------------------------
    # Point-source method
    # ------------------------------------------------------------------

    def _from_point(self):
        """
        Estimate the MTF from a bright point source (bead / PSF).

        Steps
        -----
        1. Crop image to ``self.roi``.
        2. Locate the bead centre (brightest pixel or ``self.center``).
        3. Extract a ``(2*roi_size) × (2*roi_size)`` box.
        4. Subtract background (box minimum).
        5. Refine centre via first-moment centroid.
        6. Build radial distance map; bin into radial PSF profile.
        7. Apply Hanning window; FFT → radially-averaged MTF.
        8. Store ``self.freq``, ``self.MTF``.
        """
        img = self._crop().astype(np.float64)

        # Find bead centre
        if self.center is not None:
            cr, cc = int(self.center[0]), int(self.center[1])
        else:
            flat_idx = np.argmax(img)
            cr, cc = np.unravel_index(flat_idx, img.shape)

        # Extract box (clipped at borders)
        hs = self.roi_size
        r0 = max(0, cr - hs)
        r1 = min(img.shape[0], cr + hs)
        c0 = max(0, cc - hs)
        c1 = min(img.shape[1], cc + hs)
        box = img[r0:r1, c0:c1].copy()

        # Background subtraction
        box -= box.min()

        # Sub-pixel centroid via first moments
        box_norm = box / (box.sum() + np.finfo(float).eps)
        rows_b = np.arange(box.shape[0])
        cols_b = np.arange(box.shape[1])
        sub_r = float(np.sum(rows_b[:, None] * box_norm))
        sub_c = float(np.sum(cols_b[None, :] * box_norm))

        # Radial distances
        R_rows, R_cols = np.indices(box.shape)
        radii = np.sqrt((R_rows - sub_r) ** 2 + (R_cols - sub_c) ** 2)

        max_r = int(np.floor(min(box.shape) / 2))
        bin_edges = np.arange(0, max_r + 1, 1.0)
        n_bins = len(bin_edges) - 1
        psf_profile = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=np.int64)

        bin_idx = np.searchsorted(bin_edges[1:], radii.ravel(), side='left')
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(psf_profile, bin_idx, box.ravel())
        np.add.at(counts,      bin_idx, 1)
        valid = counts > 0
        psf_profile[valid] /= counts[valid]

        # Fill empty bins
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        if not np.all(valid):
            psf_profile = np.interp(
                bin_centres,
                bin_centres[valid],
                psf_profile[valid],
            )

        # Hanning window
        psf_profile *= np.hanning(len(psf_profile))

        # FFT → MTF
        MTF_raw = np.abs(fft(psf_profile))
        freqs_raw = fftfreq(len(psf_profile))  # cycles/pixel

        pos = freqs_raw >= 0
        freqs_pos = freqs_raw[pos]
        mtf_pos = MTF_raw[pos]

        # Normalise
        if mtf_pos[0] > 0:
            mtf_pos = mtf_pos / mtf_pos[0]
        else:
            mtf_pos = mtf_pos / (mtf_pos.max() + np.finfo(float).eps)

        keep = freqs_pos <= 0.5
        self.freq = freqs_pos[keep]
        self.MTF = mtf_pos[keep]

    # ------------------------------------------------------------------
    # Resolution frequencies
    # ------------------------------------------------------------------

    def _find_resolution_frequencies(self):
        """
        Locate the spatial frequencies where MTF crosses 0.5 and 0.1.

        Uses linear interpolation between adjacent samples.  If the MTF
        never drops to a threshold the Nyquist frequency (0.5 cycles/pixel)
        is used.

        Stores
        ------
        f50 : float
        f10 : float
        resolution_50 : float
        resolution_10 : float
        freq_phys : ndarray
        """
        self.freq_phys = self.freq / self.pixel_size

        def _find_crossing(freq, mtf, level):
            """Return frequency where MTF first crosses *level* from above."""
            above = mtf >= level
            # Find first sample below level
            below_idx = np.where(~above)[0]
            if len(below_idx) == 0:
                return freq[-1]
            idx_below = below_idx[0]
            if idx_below == 0:
                return freq[0]
            # Linear interpolation between [idx_below-1, idx_below]
            f0, f1 = freq[idx_below - 1], freq[idx_below]
            m0, m1 = mtf[idx_below - 1], mtf[idx_below]
            if m1 == m0:
                return f0
            frac = (level - m0) / (m1 - m0)
            return float(f0 + frac * (f1 - f0))

        self.f50 = _find_crossing(self.freq, self.MTF, 0.5)
        self.f10 = _find_crossing(self.freq, self.MTF, 0.1)

        eps = np.finfo(float).eps
        self.resolution_50 = self.pixel_size / (self.f50 + eps)
        self.resolution_10 = self.pixel_size / (self.f10 + eps)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self):
        """
        Plot the MTF curve and save to ``MTF_<method>.png``.

        Marks the MTF=0.5 and MTF=0.1 crossings with vertical dashed lines.
        When ``pixel_size != 1.0`` a second x-axis in physical units is added.

        Returns
        -------
        freq : ndarray
            Spatial frequency axis (cycles/pixel).
        MTF : ndarray
            MTF values normalised to 1.0 at zero frequency.
        """
        import matplotlib
        import matplotlib.pyplot as _plt

        try:
            from ..utils.plot_utils import plt, isnotebook
        except Exception:
            plt = _plt
            def isnotebook():
                return False

        fig, ax = _plt.subplots(figsize=(8, 5))
        ax.plot(self.freq, self.MTF, "-b", label="MTF")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(0.1, color="gray", linestyle=":",  linewidth=0.8)

        ax.axvline(
            self.f50,
            color="steelblue",
            linestyle="--",
            label=f"MTF=0.5 → f={self.f50:.3f} cyc/px",
        )
        ax.axvline(
            self.f10,
            color="darkorange",
            linestyle="--",
            label=f"MTF=0.1 → f={self.f10:.3f} cyc/px",
        )

        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Spatial frequency (cycles/pixel)")
        ax.set_ylabel("MTF")
        ax.set_title(f"MTF — {self.method} method")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

        if self.pixel_size != 1.0:
            ax2 = ax.twiny()
            ax2.set_xlim(0, 0.5 / self.pixel_size)
            ax2.set_xlabel(f"Spatial frequency (cycles / physical unit, px={self.pixel_size})")

        fig.tight_layout()
        fname = f"MTF_{self.method}.png"
        fig.savefig(fname, bbox_inches="tight")

        try:
            if isnotebook():
                from IPython import display as _disp
                _disp.display(fig)
                _plt.close(fig)
            else:
                _plt.show(block=False)
        except Exception:
            _plt.show(block=False)

        return self.freq, self.MTF
