#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-map local resolution estimation via a ResMap-inspired statistical test.
"""

# standard library
import time
import warnings

# third party
import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_erosion,
    generate_binary_structure,
)
from scipy.stats import norm as _norm

# local
from ..utils import tqdm
from ..utils.plot_utils import plt, isnotebook

__all__ = ["LocalResolution"]


class LocalResolution:
    """
    Single-map local resolution estimation via a ResMap-inspired statistical test.

    For each voxel and each spatial-frequency band the local signal energy
    is compared with the noise floor via a z-score derived from the
    chi-squared distribution.  The local resolution is the finest spatial
    frequency at which the test passes at significance level ``significance``.

    A single reconstruction is sufficient — no half-datasets are required.
    The noise level is estimated automatically from the empty corners of
    the reconstruction volume (or from a user-supplied mask).

    Parameters
    ----------
    vol : ndarray
        A 2-dimensional image or 3-dimensional reconstruction.
        No half-maps are needed.
    pixel_size : float, optional
        Physical size of one voxel (any consistent unit, e.g. nm).
        Default ``1.0`` (result in pixels).
    significance : float, optional
        Significance level alpha for the one-sided z-test.  Smaller values
        require stronger evidence of signal and yield more conservative
        (coarser) local resolution estimates.  Default ``0.05``.
    n_freq : int, optional
        Number of frequency bands tested between the lowest resolvable
        frequency and the Nyquist limit.  Default ``20``.
    window_sigma : float, optional
        Standard deviation (in voxels) of the Gaussian window used to
        estimate the local signal energy.  Larger values produce a
        smoother but spatially lower-resolution map.  Default ``7.0``.
    mask : ndarray of bool or None, optional
        Binary mask identifying the sample region.  Voxels outside the
        mask are excluded from the noise estimate and set to ``NaN`` in
        the output map.  If ``None`` (default) a mask is estimated
        automatically via Gaussian blurring and thresholding.
    corner_fraction : float, optional
        Fraction of each edge length used to define the corner regions
        from which the noise standard deviation is estimated.  Must be
        in (0, 0.5).  Default ``0.10``.

    Attributes
    ----------
    resolution_map : ndarray
        Local resolution in pixels (``NaN`` outside the sample mask).
        Same shape as *vol*.
    resolution_map_phys : ndarray
        Local resolution in physical units (``resolution_map * pixel_size``).
    sigma_noise : float
        Estimated noise standard deviation from the corner regions.
    freq_bands : ndarray
        Centre frequencies (cycles/pixel) of each tested band.
    mask : ndarray of bool
        Binary sample mask (estimated or user-supplied).
    resolution_median : float
        Median local resolution in pixels (within the mask).
    resolution_mean : float
        Mean local resolution in pixels (within the mask).
    resolution_std : float
        Standard deviation of local resolution in pixels (within the mask).

    Notes
    -----
    The implementation follows the methodology of ResMap [1]_ but uses a
    Gaussian bandpass filter and a z-score threshold in place of the
    original steerable-filter chi-squared test.  For most practical
    purposes the results are equivalent.  The key advantage over the
    block-wise :class:`LocalFSC` approach is that only a single
    reconstruction is required.

    Noise is estimated from the corner regions of the volume, which must
    be free of sample signal.  If the sample fills the entire field of
    view (no empty corners), supply an explicit noise mask via the
    ``mask`` parameter.

    References
    ----------
    .. [1] A. Kucukelbir, F. J. Sigworth, and H. D. Tagare, "Quantifying
       the local resolution of cryo-EM density maps", Nature Methods 11,
       63-65 (2014). https://doi.org/10.1038/nmeth.2727
    """

    def __init__(
        self,
        vol,
        pixel_size=1.0,
        significance=0.05,
        n_freq=20,
        window_sigma=7.0,
        mask=None,
        corner_fraction=0.10,
    ):
        """
        Initialise, validate inputs, estimate noise, build mask, and compute.

        Parameters
        ----------
        vol : ndarray
            2-D or 3-D reconstruction volume.
        pixel_size : float, optional
            Physical voxel size. Default ``1.0``.
        significance : float, optional
            One-sided test significance level. Default ``0.05``.
        n_freq : int, optional
            Number of frequency bands. Default ``20``.
        window_sigma : float, optional
            Gaussian smoothing sigma (voxels) for local energy. Default ``7.0``.
        mask : ndarray of bool or None, optional
            Pre-computed sample mask. ``None`` triggers auto-masking.
        corner_fraction : float, optional
            Fraction of edge used for corner noise regions. Default ``0.10``.

        Raises
        ------
        ValueError
            If *vol* is not 2-D or 3-D, if *corner_fraction* is out of (0, 0.5),
            or if the corner variance is zero (no detectable noise).
        """
        vol = np.asarray(vol, dtype=np.float64)
        if vol.ndim not in (2, 3):
            raise ValueError(
                f"vol must be 2-D or 3-D, got {vol.ndim}-D array."
            )
        if not (0.0 < corner_fraction < 0.5):
            raise ValueError(
                f"corner_fraction must be in (0, 0.5), got {corner_fraction}."
            )
        if pixel_size <= 0:
            raise ValueError(f"pixel_size must be positive, got {pixel_size}.")
        if not (0.0 < significance < 1.0):
            raise ValueError(
                f"significance must be in (0, 1), got {significance}."
            )
        if n_freq < 2:
            raise ValueError(f"n_freq must be >= 2, got {n_freq}.")
        if window_sigma <= 0:
            raise ValueError(
                f"window_sigma must be positive, got {window_sigma}."
            )

        self.vol = vol
        self.pixel_size = float(pixel_size)
        self.significance = float(significance)
        self.n_freq = int(n_freq)
        self.window_sigma = float(window_sigma)
        self.corner_fraction = float(corner_fraction)

        # Validate user-supplied mask shape
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != vol.shape:
                raise ValueError(
                    f"mask shape {mask.shape} does not match vol shape {vol.shape}."
                )
            self.mask = mask
        else:
            self.mask = None  # will be set by _make_mask

        # Step 1: noise estimation from corners
        self._estimate_noise()

        # Step 2: auto-mask if none supplied
        if self.mask is None:
            self._make_mask()

        # Step 3–6: main computation
        self._compute()

        # Summary
        print(
            f"[LocalResolution] Median resolution = {self.resolution_median:.3f} px "
            f"| Mean = {self.resolution_mean:.3f} px "
            f"| Std = {self.resolution_std:.3f} px"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_corners(self):
        """Extract and pool corner voxels; return a 1-D flat array of values."""
        vol = self.vol
        ndim = vol.ndim
        cs = max(1, round(self.corner_fraction * min(vol.shape)))

        if ndim == 2:
            ny, nx = vol.shape
            corners = [
                vol[:cs, :cs],
                vol[:cs, nx - cs:],
                vol[ny - cs:, :cs],
                vol[ny - cs:, nx - cs:],
            ]
        else:  # ndim == 3
            nz, ny, nx = vol.shape
            corners = [
                vol[:cs, :cs, :cs],
                vol[:cs, :cs, nx - cs:],
                vol[:cs, ny - cs:, :cs],
                vol[:cs, ny - cs:, nx - cs:],
                vol[nz - cs:, :cs, :cs],
                vol[nz - cs:, :cs, nx - cs:],
                vol[nz - cs:, ny - cs:, :cs],
                vol[nz - cs:, ny - cs:, nx - cs:],
            ]

        return np.concatenate([c.ravel() for c in corners])

    def _estimate_noise(self):
        """Estimate noise sigma and variance from corner regions; set attributes."""
        corner_vals = self._extract_corners()
        self._corner_vals = corner_vals  # cache for _make_mask
        sigma = float(np.std(corner_vals))
        var = sigma ** 2

        if var == 0.0:
            raise ValueError(
                "The corner regions appear to contain no noise (variance == 0). "
                "The sample may fill the entire field of view, or the corners "
                "are identically zero. "
                "Provide an explicit binary mask via the `mask` parameter or "
                "increase `corner_fraction` to include more background voxels."
            )

        self.sigma_noise = sigma
        self.var_noise = var
        print(f"[LocalResolution] Estimated noise sigma = {self.sigma_noise:.6g}")

    def _make_mask(self):
        """Auto-generate a binary sample mask from the volume; set ``self.mask``."""
        vol = self.vol
        corner_vals = self._corner_vals
        sigma_noise = self.sigma_noise

        # Smooth |vol| then threshold
        smoothed = gaussian_filter(np.abs(vol), sigma=self.window_sigma / 2.0)
        threshold = float(np.mean(corner_vals)) + 2.0 * sigma_noise
        binary = smoothed > threshold

        # Fill holes
        filled = binary_fill_holes(binary)

        # Erode with ball-shaped structure of radius 2
        ndim = vol.ndim
        struct = generate_binary_structure(ndim, ndim)
        # Expand to radius ~2 by iterating erosion twice
        eroded = binary_erosion(filled, structure=struct, iterations=2)

        if not np.any(eroded):
            warnings.warn(
                "Auto-mask is empty after erosion. "
                "Falling back to an all-True mask. "
                "Consider decreasing `window_sigma` or supplying an explicit mask.",
                RuntimeWarning,
                stacklevel=3,
            )
            eroded = np.ones(vol.shape, dtype=bool)

        self.mask = eroded

    def _build_radial_grid(self):
        """Build a radial frequency grid (cycles/pixel); return array same shape as vol.

        Returns
        -------
        R : ndarray
            Radial frequency magnitude at each voxel position, in cycles/pixel.
        """
        vol = self.vol
        grids = [fftfreq(s) for s in vol.shape]
        R2 = np.zeros(vol.shape, dtype=np.float64)
        for i, g in enumerate(grids):
            # Broadcast g along axis i
            shape = [1] * vol.ndim
            shape[i] = len(g)
            R2 += g.reshape(shape) ** 2
        return np.sqrt(R2)

    def _compute(self):
        """
        Run the frequency-band loop; set resolution map and summary statistics.

        Sets
        ----
        resolution_map : ndarray
            Local resolution in pixels (NaN outside mask).
        resolution_map_phys : ndarray
            Local resolution in physical units.
        freq_bands : ndarray
            Tested centre frequencies (cycles/pixel).
        resolution_median, resolution_mean, resolution_std : float
            Summary statistics over masked voxels.
        """
        vol = self.vol
        ndim = vol.ndim
        eps = np.finfo(np.float64).tiny

        # Frequency bands
        f_min = 1.0 / min(vol.shape)
        f_max = 0.50
        freq_bands = np.linspace(f_min, f_max, self.n_freq)
        sigma_f = (f_max - f_min) / max(self.n_freq - 1, 1) * 0.6
        self.freq_bands = freq_bands

        # z threshold (one-sided)
        z_alpha = float(_norm.ppf(1.0 - self.significance))

        # Precompute
        R = self._build_radial_grid()
        VOL_F = fftn(vol)  # FFT of the full volume (computed once)

        # Resolution map initialised to NaN
        resolution_map = np.full(vol.shape, np.nan, dtype=np.float64)

        # n_eff: effective independent samples in the Gaussian window
        n_eff = (2.0 * np.sqrt(np.pi) * self.window_sigma) ** ndim

        # Iterate from low to high frequency so the final assignment is the
        # finest (highest-frequency) band where the test passes.
        for f_i in tqdm(freq_bands, desc="LocalResolution bands"):
            # Gaussian bandpass filter in Fourier space
            H_f = np.exp(-0.5 * ((R - f_i) / sigma_f) ** 2)

            # Bandpass-filtered volume (real part)
            vol_f = np.real(ifftn(VOL_F * H_f))

            # Expected noise power fraction captured by this band
            bp_energy = float(np.mean(H_f ** 2))

            # Local mean-squared signal via Gaussian smoothing
            E_f = gaussian_filter(vol_f ** 2, sigma=self.window_sigma)

            # Null-hypothesis statistics
            mu0 = self.var_noise * bp_energy
            std0 = mu0 * np.sqrt(2.0 / n_eff)

            # Z-score
            z = (E_f - mu0) / (std0 + eps)

            # Decision: passes test AND inside mask
            decision = (z > z_alpha) & self.mask

            # Overwrite resolution map where test passes (resolution = 1/f_i px)
            resolution_map[decision] = 1.0 / f_i

        self.resolution_map = resolution_map
        self.resolution_map_phys = resolution_map * self.pixel_size

        # Summary statistics (masked voxels only, ignoring NaN)
        valid = resolution_map[self.mask]
        valid = valid[~np.isnan(valid)]
        if valid.size > 0:
            self.resolution_median = float(np.median(valid))
            self.resolution_mean = float(np.mean(valid))
            self.resolution_std = float(np.std(valid))
        else:
            self.resolution_median = float("nan")
            self.resolution_mean = float("nan")
            self.resolution_std = float("nan")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(self, slice_idx=None, axis=0, cmap="viridis_r", vmin=None, vmax=None):
        """
        Visualise the local resolution map and save to ``LocalResolution_map.png``.

        For a 3-D volume a single slice is shown; for a 2-D image the full map
        is displayed.

        Parameters
        ----------
        slice_idx : int or None, optional
            Index of the slice to display along *axis* (3-D only).
            Defaults to the central slice when ``None``.
        axis : int, optional
            Axis along which to take the slice (3-D only).  Default ``0``.
        cmap : str, optional
            Matplotlib colormap name.  Default ``'viridis_r'`` so that higher
            resolution (smaller value) appears bright.
        vmin : float or None, optional
            Lower bound for the colormap.  Defaults to the map minimum.
        vmax : float or None, optional
            Upper bound for the colormap.  Defaults to the map maximum.

        Returns
        -------
        resolution_map : ndarray
            The ``resolution_map`` attribute (unchanged).
        """
        rmap = self.resolution_map

        if rmap.ndim == 3:
            if slice_idx is None:
                slice_idx = rmap.shape[axis] // 2
            if axis == 0:
                img = rmap[slice_idx, :, :]
            elif axis == 1:
                img = rmap[:, slice_idx, :]
            else:
                img = rmap[:, :, slice_idx]
            title = f"Local Resolution Map (slice {slice_idx}, axis {axis})"
        else:
            img = rmap
            title = "Local Resolution Map"

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Resolution (px)")
        ax.set_title(title)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        fig.tight_layout()

        fname = "LocalResolution_map.png"
        fig.savefig(fname, dpi=150)
        print(f"[LocalResolution] Saved plot to {fname}")

        if isnotebook():
            plt.show()
        else:
            plt.close(fig)

        return self.resolution_map
