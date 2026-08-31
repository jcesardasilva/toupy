#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local (spatially-resolved) resolution estimation.

Two complementary estimators live here:

* ``LocalResolution`` -- single-map, ResMap-style statistical test; needs
  only one reconstruction (usable for local tomography).
* ``LocalFSC`` -- two-half-map, block-wise FSC/FRC (blocres-style); needs a
  pair of independent half-reconstructions.

Global (single-number) FSC/SSNR live in ``FSC.py``; this module is the home
for the local resolution *maps*.
"""

# standard library
import time
import warnings
import itertools

# third party
import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_erosion,
    generate_binary_structure,
    minimum_filter,
    distance_transform_edt,
)
from scipy.stats import norm as _norm
from scipy.fft import fftn as _fftn, fftfreq as _fftfreq
from scipy.interpolate import RegularGridInterpolator

# local
from ..utils import tqdm
from ..utils.plot_utils import show_resolution_map

__all__ = ["LocalResolution", "LocalFSC"]


class LocalResolution:
    """
    Single-map local resolution estimation via a ResMap-inspired statistical test.

    For each voxel and each spatial-frequency band the local signal energy
    is compared with the noise floor via a z-score derived from the
    chi-squared distribution.  The local resolution is the finest spatial
    frequency at which the test passes at significance level ``significance``.

    A single reconstruction is sufficient — no half-datasets are required.
    Three noise estimation strategies are available (see ``noise_method``),
    making the estimator usable even for **local tomography** datasets where
    the sample fills the entire field of view and the volume corners are not
    empty.

    Parameters
    ----------
    vol : ndarray
        A 2-dimensional image or 3-dimensional reconstruction.
        No half-maps are needed.
    pixel_size : float, optional
        Physical size of one voxel (any consistent unit, e.g. nm).
        Default ``1.0`` (result in pixels).
    significance : float, optional
        Significance level α for the one-sided z-test.  Smaller values
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
        Binary mask identifying the **sample** region.  Voxels outside
        the mask are set to ``NaN`` in the output map.  If ``None``
        (default) a mask is estimated automatically via Gaussian blurring
        and thresholding.
    noise_method : {'corners', 'highfreq', 'mask'}, optional
        Strategy used to estimate the noise standard deviation:

        ``'corners'`` *(default)*
            Noise is estimated from small cubic sub-volumes at the eight
            (four for 2-D) corners of the reconstruction.  Requires the
            corners to be free of sample signal.  **Not suitable for local
            tomography.**
        ``'highfreq'``
            Noise is estimated from the high-frequency tail of the power
            spectrum (frequencies above ``1 - highfreq_fraction`` of the
            Nyquist range).  Works even when the sample fills the entire
            field of view, because signal power typically falls well below
            the noise floor near the Nyquist limit.  Suitable for
            **local tomography** and any dataset where empty corners are
            unavailable.
        ``'mask'``
            Noise is estimated from the voxels indicated by ``noise_mask``.
            The user must supply a boolean array marking known empty
            (noise-only) regions anywhere in the volume.

    noise_mask : ndarray of bool or None, optional
        Boolean array (same shape as *vol*) whose ``True`` entries mark
        voxels that contain **only noise** (i.e. no sample signal).
        Required when ``noise_method='mask'``; ignored otherwise.
    sigma_noise : float or None, optional
        Directly supply the noise standard deviation, bypassing all
        automatic estimation.  When provided, ``noise_method``,
        ``noise_mask``, and ``corner_fraction`` are all ignored.
    corner_fraction : float, optional
        Fraction of each edge length used to define the corner noise
        regions.  Must be in ``(0, 0.5)``.  Only used when
        ``noise_method='corners'``.  Default ``0.10``.
    highfreq_fraction : float, optional
        Fraction of the frequency range (counting down from Nyquist) used
        to estimate noise power.  Only used when
        ``noise_method='highfreq'``.  Default ``0.10`` (top 10 %, i.e.
        frequencies from 0.45 to 0.50 cycles/pixel).

    Attributes
    ----------
    resolution_map : ndarray
        Local full-period resolution in pixels (van Heel convention,
        ``NaN`` outside the sample mask), same shape as *vol*.
    resolution_map_phys : ndarray
        Local full-period resolution in physical units
        (``resolution_map * pixel_size``).
    resolution_map_half : ndarray
        Local half-period resolution in pixels (Rayleigh convention):
        ``resolution_map / 2``.
    resolution_map_phys_half : ndarray
        Local half-period resolution in physical units:
        ``resolution_map_phys / 2``.
    sigma_noise : float
        Noise standard deviation used for the hypothesis test.
    freq_bands : ndarray
        Centre frequencies (cycles/pixel) of each tested band.
    mask : ndarray of bool
        Binary sample mask (estimated or user-supplied).
    resolution_median : float
        Median full-period local resolution in pixels (within the mask).
    resolution_mean : float
        Mean full-period local resolution in pixels (within the mask).
    resolution_std : float
        Standard deviation of full-period local resolution in pixels
        (within the mask).
    resolution_median_half : float
        Median half-period local resolution in pixels (within the mask).
    resolution_mean_half : float
        Mean half-period local resolution in pixels (within the mask).
    resolution_std_half : float
        Standard deviation of half-period local resolution in pixels
        (within the mask).

    Notes
    -----
    The implementation follows the methodology of ResMap [1]_ but uses a
    Gaussian bandpass filter and a z-score threshold in place of the
    original steerable-filter chi-squared test.  For most practical
    purposes the results are equivalent.  The key advantage over the
    block-wise :class:`LocalFSC` approach is that only a single
    reconstruction is required.

    For **local (region-of-interest) tomography**, use
    ``noise_method='highfreq'`` or supply ``sigma_noise`` directly:

    .. code-block:: python

        # Local tomography — corners are not empty
        lr = LocalResolution(vol, pixel_size=28.6e-9,
                             noise_method='highfreq')

        # Or if you already know sigma_noise from the sinogram
        lr = LocalResolution(vol, pixel_size=28.6e-9,
                             sigma_noise=0.003)

    References
    ----------
    .. [1] A. Kucukelbir, F. J. Sigworth, and H. D. Tagare, "Quantifying
       the local resolution of cryo-EM density maps", Nature Methods 11,
       63-65 (2014). https://doi.org/10.1038/nmeth.2727
    """

    _VALID_NOISE_METHODS = ("corners", "highfreq", "mask")

    def __init__(
        self,
        vol,
        pixel_size=1.0,
        significance=0.05,
        n_freq=20,
        window_sigma=7.0,
        mask=None,
        noise_method="corners",
        noise_mask=None,
        sigma_noise=None,
        corner_fraction=0.10,
        highfreq_fraction=0.10,
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
        noise_method : {'corners', 'highfreq', 'mask'}, optional
            Noise estimation strategy.  Default ``'corners'``.
        noise_mask : ndarray of bool or None, optional
            Noise-only voxel mask.  Required when ``noise_method='mask'``.
        sigma_noise : float or None, optional
            Directly supply noise standard deviation.  Overrides
            ``noise_method`` when provided.
        corner_fraction : float, optional
            Fraction of edge used for corner noise regions. Default ``0.10``.
        highfreq_fraction : float, optional
            Fraction of the frequency range (from Nyquist down) used for
            high-frequency noise estimation. Default ``0.10``.

        Raises
        ------
        ValueError
            If inputs are invalid or the noise variance cannot be estimated.
        """
        vol = np.asarray(vol, dtype=np.float64)
        if vol.ndim not in (2, 3):
            raise ValueError(f"vol must be 2-D or 3-D, got {vol.ndim}-D array.")
        if pixel_size <= 0:
            raise ValueError(f"pixel_size must be positive, got {pixel_size}.")
        if not (0.0 < significance < 1.0):
            raise ValueError(f"significance must be in (0, 1), got {significance}.")
        if n_freq < 2:
            raise ValueError(f"n_freq must be >= 2, got {n_freq}.")
        if window_sigma <= 0:
            raise ValueError(f"window_sigma must be positive, got {window_sigma}.")
        if not (0.0 < corner_fraction < 0.5):
            raise ValueError(
                f"corner_fraction must be in (0, 0.5), got {corner_fraction}."
            )
        if not (0.0 < highfreq_fraction < 0.5):
            raise ValueError(
                f"highfreq_fraction must be in (0, 0.5), got {highfreq_fraction}."
            )
        if noise_method not in self._VALID_NOISE_METHODS:
            raise ValueError(
                f"noise_method must be one of {self._VALID_NOISE_METHODS!r}, "
                f"got {noise_method!r}."
            )

        self.vol              = vol
        self.pixel_size       = float(pixel_size)
        self.significance     = float(significance)
        self.n_freq           = int(n_freq)
        self.window_sigma     = float(window_sigma)
        self.corner_fraction  = float(corner_fraction)
        self.highfreq_fraction = float(highfreq_fraction)
        self.noise_method     = noise_method

        # Validate user-supplied sample mask
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != vol.shape:
                raise ValueError(
                    f"mask shape {mask.shape} does not match vol shape {vol.shape}."
                )
            self.mask = mask
        else:
            self.mask = None  # set by _make_mask

        # Validate noise_mask
        if noise_mask is not None:
            noise_mask = np.asarray(noise_mask, dtype=bool)
            if noise_mask.shape != vol.shape:
                raise ValueError(
                    f"noise_mask shape {noise_mask.shape} does not match "
                    f"vol shape {vol.shape}."
                )
        self._noise_mask = noise_mask

        # Step 1: noise estimation
        if sigma_noise is not None:
            # User-supplied directly — skip all estimation
            sigma_noise = float(sigma_noise)
            if sigma_noise <= 0:
                raise ValueError(
                    f"sigma_noise must be positive, got {sigma_noise}."
                )
            self.sigma_noise = sigma_noise
            self.var_noise   = sigma_noise ** 2
            print(f"[LocalResolution] Using supplied sigma_noise = {self.sigma_noise:.6g}")
            self._corner_vals = None
        else:
            self._estimate_noise()

        # Step 2: auto-mask if none supplied
        if self.mask is None:
            self._make_mask()

        # Step 3–6: main computation
        self._compute()

        ps = self.pixel_size
        print(
            f"[LocalResolution] Median resolution (full period) = "
            f"{self.resolution_median:.3f} px "
            f"({self.resolution_median * ps:.4g} physical units)"
        )
        print(
            f"[LocalResolution] Median resolution (half period) = "
            f"{self.resolution_median_half:.3f} px "
            f"({self.resolution_median_half * ps:.4g} physical units)"
        )
        print(
            f"[LocalResolution] Mean = {self.resolution_mean:.3f} px "
            f"| Std = {self.resolution_std:.3f} px  (full period)"
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
        """
        Dispatch noise estimation to the appropriate strategy.

        Calls one of :meth:`_noise_from_corners`, :meth:`_noise_from_highfreq`,
        or :meth:`_noise_from_mask` according to ``self.noise_method``, then
        sets ``self.sigma_noise`` and ``self.var_noise``.
        """
        if self.noise_method == "corners":
            self._noise_from_corners()
        elif self.noise_method == "highfreq":
            self._noise_from_highfreq()
        elif self.noise_method == "mask":
            self._noise_from_mask()

    def _noise_from_corners(self):
        """
        Estimate noise from the corner sub-volumes of the reconstruction.

        Pools voxels from the 8 (3-D) or 4 (2-D) corners — each of side
        ``round(corner_fraction * min(vol.shape))`` voxels — and computes
        their standard deviation.

        Raises
        ------
        ValueError
            If the corner variance is zero, which typically means the
            corners contain sample material (local tomography) rather than
            empty background.  Switch to ``noise_method='highfreq'`` or
            provide ``sigma_noise`` directly in that case.

        Warns
        -----
        UserWarning
            If the absolute mean of the corner values exceeds
            ``2 * sigma_noise``, suggesting the corners may not be
            empty background (soft indicator of local tomography).
        """
        corner_vals = self._extract_corners()
        self._corner_vals = corner_vals  # cache for _make_mask
        sigma = float(np.std(corner_vals))
        var   = sigma ** 2

        if var == 0.0:
            raise ValueError(
                "Corner regions have zero variance — the sample likely fills "
                "the entire field of view (local tomography) or the corners "
                "are identically zero.  Use noise_method='highfreq' or supply "
                "sigma_noise directly."
            )

        # Soft check: if |mean| > 2σ the corners may contain signal
        corner_mean = float(np.abs(np.mean(corner_vals)))
        if corner_mean > 2.0 * sigma:
            warnings.warn(
                f"[LocalResolution] Corner mean ({corner_mean:.4g}) is more than "
                f"2× the corner std ({sigma:.4g}), suggesting the corners may "
                "contain sample signal rather than empty background.  "
                "This is common in local (region-of-interest) tomography.  "
                "Consider using noise_method='highfreq' or supplying sigma_noise "
                "directly to avoid a biased noise estimate.",
                UserWarning,
                stacklevel=4,
            )

        self.sigma_noise = sigma
        self.var_noise   = var
        print(
            f"[LocalResolution] Noise estimated from corners: "
            f"sigma = {self.sigma_noise:.6g}"
        )

    def _noise_from_highfreq(self):
        """
        Estimate noise from the high-frequency tail of the power spectrum.

        For a well-sampled reconstruction the signal power rolls off well
        before the Nyquist limit, so the power spectrum near Nyquist is
        dominated by noise.  The noise variance is:

        .. math::

            \\sigma^2_{\\text{noise}} =
                \\frac{\\langle |\\hat{V}(\\mathbf{k})|^2 \\rangle_{R > f_{\\text{thr}}}}{N}

        where :math:`N` is the total number of voxels and :math:`f_{\\text{thr}}`
        is the lower bound of the high-frequency band
        (``0.5 * (1 - highfreq_fraction)``).

        This method is suitable for **local tomography** because it does
        not require empty corners.
        """
        R    = self._build_radial_grid()
        F    = fftn(self.vol)
        N    = self.vol.size
        f_thr = 0.5 * (1.0 - self.highfreq_fraction)
        hf_mask = R > f_thr

        n_hf = int(np.sum(hf_mask))
        if n_hf == 0:
            raise ValueError(
                f"No frequency bins found above {f_thr:.3f} cycles/pixel.  "
                "Decrease highfreq_fraction or check the volume shape."
            )

        mean_power = float(np.mean(np.abs(F[hf_mask]) ** 2))
        var   = mean_power / N
        sigma = float(np.sqrt(var))

        if var == 0.0:
            raise ValueError(
                "High-frequency power is zero — the volume may be uniformly "
                "zero.  Check the input data."
            )

        self._corner_vals = None  # not available for auto-mask fallback
        self.sigma_noise  = sigma
        self.var_noise    = var
        print(
            f"[LocalResolution] Noise estimated from high frequencies "
            f"(R > {f_thr:.3f} cycles/px, {n_hf} bins): "
            f"sigma = {self.sigma_noise:.6g}"
        )

    def _noise_from_mask(self):
        """
        Estimate noise from a user-supplied noise-only mask.

        Uses the voxels where ``noise_mask`` is ``True`` to compute
        the noise standard deviation.

        Raises
        ------
        ValueError
            If ``noise_mask`` was not provided or marks no voxels.
        """
        if self._noise_mask is None:
            raise ValueError(
                "noise_method='mask' requires a noise_mask array to be "
                "supplied."
            )
        vals = self.vol[self._noise_mask]
        if vals.size == 0:
            raise ValueError(
                "noise_mask contains no True entries — nothing to estimate "
                "noise from."
            )
        sigma = float(np.std(vals))
        var   = sigma ** 2

        if var == 0.0:
            raise ValueError(
                "noise_mask region has zero variance.  Make sure the mask "
                "selects voxels that contain background noise, not a flat "
                "signal region."
            )

        self._corner_vals = vals  # reuse for auto-mask threshold
        self.sigma_noise  = sigma
        self.var_noise    = var
        print(
            f"[LocalResolution] Noise estimated from noise_mask "
            f"({vals.size} voxels): sigma = {self.sigma_noise:.6g}"
        )

    def _make_mask(self):
        """Auto-generate a binary sample mask from the volume; set ``self.mask``."""
        vol         = self.vol
        sigma_noise = self.sigma_noise

        # Smooth |vol| then threshold
        smoothed = gaussian_filter(np.abs(vol), sigma=self.window_sigma / 2.0)

        # Threshold baseline: use corner/noise-mask mean if available,
        # otherwise fall back to the volume mean (works for highfreq method)
        if self._corner_vals is not None and self._corner_vals.size > 0:
            baseline = float(np.mean(self._corner_vals))
        else:
            baseline = float(np.mean(np.abs(vol)))
        threshold = baseline + 2.0 * sigma_noise
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

        self.resolution_map           = resolution_map
        self.resolution_map_phys      = resolution_map * self.pixel_size
        self.resolution_map_half      = resolution_map / 2.0
        self.resolution_map_phys_half = self.resolution_map_phys / 2.0

        # Summary statistics (masked voxels only, ignoring NaN)
        valid = resolution_map[self.mask]
        valid = valid[~np.isnan(valid)]
        if valid.size > 0:
            self.resolution_median      = float(np.median(valid))
            self.resolution_mean        = float(np.mean(valid))
            self.resolution_std         = float(np.std(valid))
        else:
            self.resolution_median      = float("nan")
            self.resolution_mean        = float("nan")
            self.resolution_std         = float("nan")

        self.resolution_median_half = self.resolution_median / 2.0
        self.resolution_mean_half   = self.resolution_mean   / 2.0
        self.resolution_std_half    = self.resolution_std    / 2.0

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
        show_resolution_map(
            self.resolution_map,
            self.vol.ndim,
            title="LocalResolution map",
            filename="LocalResolution_map.png",
            slice_idx=slice_idx,
            axis=axis,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        return self.resolution_map


# ---------------------------------------------------------------------------
# LocalFSC  (block-wise FSC/FRC; moved here from FSC.py so both local
# resolution-map estimators live together)
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

    mask : ndarray of bool or None, optional
        Support of the specimen, same shape as *vol1*.  The mask is
        eroded internally so that a box is evaluated only where it fits
        inside the support; everything outside the eroded support is
        ``NaN`` in the output map.  See *mask_eroded* for the erosion
        convention.  This mirrors the ``-Mask`` option of ``blocres``
        [1]_, the reference implementation this class follows.

        The mask is **never derived automatically**.  For a porous or
        hollow specimen the correct support is the outer envelope of the
        object and not the thresholded material — a judgement that
        depends on what the user means to measure, so it is left to the
        user.  With ``None`` (default) no masking is applied and every
        box on the grid is evaluated, background included.

    mask_eroded : {'auto', 'cube', 'ball'}, float or None, optional
        The structuring element used to erode *mask*.  Whichever is
        chosen is recorded in :attr:`erosion` and
        :attr:`erosion_radius`, so a saved run says which convention
        produced its map.

        ``'auto'`` / ``'cube'`` *(default)*
            Erosion by the ``(box_size,) * ndim`` cube the analysis box
            occupies — the exact reading of "the box fits entirely inside
            the support".  Note that a round support does **not** erode
            to a round one: the cube needs ``box_size/2 * sqrt(ndim)`` of
            clearance towards its corners and ``box_size/2`` along the
            axes, so a disc comes back with a four-fold scalloped
            boundary.  That is correct, not an artefact.
        ``'ball'``
            Erosion by a ball at the box circumradius,
            ``box_size/2 * sqrt(ndim)``.  The ball contains the cube, so
            containment still holds and the eroded boundary is round; the
            price is eroding the axial directions harder than the box
            requires.
        float
            Erosion by a ball of this radius, in voxels.  Use it when the
            geometry constrains fewer than *ndim* axes — for a
            cylindrical specimen in a 3-D volume, with *z* unconstrained,
            the in-plane circumradius ``box_size / sqrt(2)`` is the right
            figure.  Radii below ``box_size/2 * sqrt(ndim)`` are
            permissive: they admit centres whose box corners reach
            outside the support.
        ``None``
            No erosion — use *mask* verbatim as the region to evaluate.
            Pass this when the mask already carries whatever margin is
            wanted, for instance a support you eroded yourself, or a
            region of interest whose exact boundary is the point.  Boxes
            near the edge then reach outside the support and their FSC
            includes whatever lies there; that is the caller's call.

        Ignored, with a warning, when no *mask* is given.

    Attributes
    ----------
    resolution_map : ndarray
        Local full-period resolution in pixels (van Heel convention),
        same shape as *vol1*.  ``NaN`` wherever no resolution could be
        estimated (see Notes).
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
        Median full-period local resolution in pixels, ignoring ``NaN``.
    resolution_mean : float
        Mean full-period local resolution in pixels, ignoring ``NaN``.
    resolution_std : float
        Standard deviation of full-period local resolution in pixels,
        ignoring ``NaN``.
    resolution_median_half : float
        Median half-period local resolution in pixels.
    resolution_mean_half : float
        Mean half-period local resolution in pixels.
    resolution_std_half : float
        Standard deviation of half-period local resolution in pixels.
    mask : ndarray of bool or None
        The support supplied by the user, unmodified (``None`` if none
        was given).
    mask_eroded : ndarray of bool or None
        The region actually evaluated: *mask* eroded by the chosen
        structuring element, or *mask* itself when the constructor was
        given ``mask_eroded=None``.
    erosion : {'cube', 'ball'} or None
        Which structuring element was used, or ``None`` when the mask was
        taken verbatim (or no mask was given).
    erosion_radius : float or None
        For ``'ball'``, the ball radius in voxels; for ``'cube'``, half
        the side of the cube.  ``None`` when no erosion was applied.
    coarse_map : ndarray
        Per-box resolution estimates in pixels on the coarse grid
        (``NaN`` where the box was skipped or the FSC did not cross the
        threshold), before interpolation to the input shape.
    grid_axes : list of ndarray
        Box-centre coordinates along each axis, i.e. the sample points of
        ``coarse_map``.
    n_boxes : int
        Total number of analysis boxes on the coarse grid.
    n_valid : int
        Number of boxes that produced a finite resolution estimate — how
        many local FSC measurements the summary statistics rest on.
        Boxes outside the eroded support and boxes whose FSC never
        crossed the threshold are excluded.

    Notes
    -----
    Voxels are ``NaN`` in ``resolution_map`` wherever the local
    resolution is genuinely unknown, rather than being filled with a
    stand-in value:

    * outside the eroded support, when a *mask* is supplied;
    * where the FSC of the surrounding boxes reached the threshold in no
      shell at all, which is what a region without reproducible signal
      looks like.  The map is interpolated
      bi/tri-linearly from the box centres, so one such box makes the
      grid cells touching it ``NaN`` too — a background box therefore
      blanks a ``2 * step`` square around itself.

    The resolution of a box is read off at the **highest** shell at or
    above the threshold, matching
    :class:`~toupy.resolution.FSC.FSCplot`.  Against a known band limit
    this recovers the true shell in 100 % of trials at SNR 1.0 and 85 %
    at SNR 0.5 for a 64-voxel box (``test/rule_test.py``).  Anchoring the
    run at DC instead — stopping at the first shell to fall below the
    threshold — was tried and reverted: the shell-size-corrected
    threshold is at its highest exactly where such a run begins
    (``T = 0.462`` at shell 1 for ``L = 64``), so the run terminates in
    the first few shells whatever the signal above them, scoring 59 % and
    0 % on the same trials.

    **Where this rule is weak.**  Taking the highest qualifying shell
    means one noise-driven excursion can set the answer, and how often
    that happens depends on how many Fourier voxels a shell holds.  On
    pure noise with ``box_size=32`` and no signal at all:

    ===========  =================  ===========================
    input        ``threshold``      estimates finer than 3 px
    ===========  =================  ===========================
    3-D (FSC)    ``0.143``          0 %
    3-D (FSC)    ``'halfbit'``      0 %
    2-D (FRC)    ``0.143``          86 %
    2-D (FRC)    ``'halfbit'``      20 %
    ===========  =================  ===========================

    A 3-D shell holds :math:`O(4\\pi s^2)` voxels against
    :math:`O(2\\pi s)` for a 2-D ring, so the FRC noise floor of a small
    box sits well above a fixed threshold of 0.143 and spurious
    near-Nyquist estimates follow.  For 2-D maps, and for small boxes
    generally, use ``threshold='halfbit'`` and the largest *box_size* the
    spatial sampling allows.

    Inside the eroded support the map is continuous: boxes skipped
    because they overhang the support are not measurements that failed,
    and they take the nearest measured estimate rather than punching a
    hole through the interpolation.  ``coarse_map`` and ``n_valid``
    expose where the actual measurements are.

    Earlier versions substituted ``2 * box_size`` for a non-crossing FSC
    and let the interpolator extrapolate past the outermost box centres,
    which produced meaningless — and occasionally negative — entries in
    the border band.  A resolution is a length: this implementation never
    returns a value ``<= 0``.

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

    def __init__(self, vol1, vol2, pixel_size=1.0, box_size=32, step=None,
                 threshold=0.143, mask=None, mask_eroded="auto"):
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

        # Erosion convention.  Recorded on the instance so that a saved run
        # says which one produced its map.
        self.erosion = None
        self.erosion_radius = None
        if mask_eroded is None:
            pass
        elif isinstance(mask_eroded, str):
            key = mask_eroded.lower()
            if key in ("auto", "cube"):
                self.erosion = "cube"
                self.erosion_radius = self.box_size / 2.0
            elif key == "ball":
                self.erosion = "ball"
                # Circumradius of the box: the smallest ball that contains it,
                # hence the smallest one that still guarantees containment.
                self.erosion_radius = (
                    self.box_size * np.sqrt(self.ndim) / 2.0
                )
            else:
                raise ValueError(
                    f"LocalFSC: mask_eroded must be 'auto'/'cube', 'ball', a "
                    f"positive ball radius in voxels, or None; "
                    f"got {mask_eroded!r}."
                )
        elif np.isscalar(mask_eroded) and not isinstance(mask_eroded, bool):
            radius = float(mask_eroded)
            if not radius > 0:
                raise ValueError(
                    f"LocalFSC: a numeric mask_eroded is a ball radius in "
                    f"voxels and must be positive, got {radius}."
                )
            self.erosion = "ball"
            self.erosion_radius = radius
        else:
            raise ValueError(
                "LocalFSC: mask_eroded must be 'auto'/'cube', 'ball', a "
                "positive ball radius in voxels, or None (use mask exactly "
                "as given).  To supply your own eroded support, pass it as "
                "mask=... together with mask_eroded=None."
            )
        self._erode = self.erosion is not None

        if mask is None and not self._erode:
            warnings.warn(
                "LocalFSC: mask_eroded=None has no effect without a mask — "
                "there is nothing to erode.  Pass the support as mask=...",
                UserWarning,
                stacklevel=2,
            )

        # Support mask.  Taken as given, never guessed: see the class
        # docstring for why deriving it here would be wrong.
        self.mask = None
        self.mask_eroded = None
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != self.shape:
                raise ValueError(
                    f"LocalFSC: mask shape {mask.shape} does not match the "
                    f"volume shape {self.shape}."
                )
            if not mask.any():
                raise ValueError("LocalFSC: mask has no True entries.")
            self.mask = mask
            if self._erode:
                self.mask_eroded = self._erode_mask(mask)
                if not self.mask_eroded.any():
                    raise ValueError(
                        f"LocalFSC: the support vanishes when eroded by half "
                        f"the analysis box (box_size={self.box_size}); no box "
                        "fits inside it.  Use a smaller box_size, a wider "
                        "mask, or mask_eroded=None to use the mask as given."
                    )
            else:
                # Used verbatim: the caller has taken responsibility for the
                # margin, so boxes centred near the edge of the support may
                # reach outside it.
                self.mask_eroded = mask

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
        print(f"  LocalFSC valid box estimates             : "
              f"{self.n_valid} of {self.n_boxes}")

    def _erode_mask(self, mask):
        """
        Erode *mask* by half the analysis box.

        The eroded support marks the box centres for which the analysis
        box stays clear of the background.  Which structuring element is
        used is set by ``mask_eroded`` and recorded in ``self.erosion``:

        ``'cube'``
            Erosion by the flat ``(box_size,) * ndim`` cube the box
            actually occupies.  This is the exact reading of "the box
            fits entirely inside the support": the set of centres
            :math:`p` with :math:`p + C \\subseteq S`.  A round support
            does **not** erode to a round one — the cube needs
            ``box_size/2 * sqrt(ndim)`` of clearance towards its corners
            against ``box_size/2`` along the axes, so a disc comes back
            with a four-fold scalloped boundary.  That is geometry, not
            an artefact.
        ``'ball'``
            Erosion by a ball of ``self.erosion_radius``.  At the box
            circumradius ``box_size/2 * sqrt(ndim)`` it contains the cube,
            so containment is still guaranteed and the boundary is round,
            at the cost of eroding the axial directions harder than
            necessary.  A smaller radius is permissive: a ball of
            ``box_size/2`` does not contain the cube, and admits centres
            whose box corners reach outside the support.

        Voxels within half a box of any face of the volume are dropped in
        both cases: such a box hangs outside the reconstruction.

        Parameters
        ----------
        mask : ndarray of bool
            User-supplied support, same shape as the input volumes.

        Returns
        -------
        eroded : ndarray of bool
            The eroded support.
        """
        if self.erosion == "cube":
            # Separable minimum filter: an erosion by a flat cube, but
            # without materialising a (box_size,) * ndim element.  Voxels
            # beyond the volume border count as background (cval=0), so
            # boxes hanging over an edge are excluded here as well.
            return minimum_filter(
                mask.astype(np.uint8),
                size=self.box_size,
                mode="constant",
                cval=0,
            ).astype(bool)

        eroded = distance_transform_edt(mask) >= self.erosion_radius

        # The distance transform sees only the array, so the volume faces
        # have to be cleared explicitly.
        h = int(np.ceil(self.box_size / 2.0))
        for d in range(self.ndim):
            idx = [slice(None)] * self.ndim
            idx[d] = slice(0, h)
            eroded[tuple(idx)] = False
            idx[d] = slice(-h, None)
            eroded[tuple(idx)] = False

        return eroded

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
        resolution at the highest shell at or above the chosen threshold
        — either the fixed value ``self.threshold``, or, when
        ``self._adaptive_snr`` is set, the van Heel & Schatz (2005)
        shell-size-corrected threshold evaluated from the actual number
        of independent Fourier voxels in each shell of *this* box.

        Shells are binned over ``0 .. box_size // 2`` only; the sparse
        corner region beyond the Nyquist radius, which is sampled
        anisotropically, takes no part.

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
            Local resolution in pixels, or ``NaN`` when no shell above DC
            reaches the threshold.  That is the signature of a box
            holding no reproducible signal, and it carries no
            resolution: reporting ``NaN`` keeps it out of the map and out
            of the statistics instead of standing in a fabricated coarse
            value.
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
        n_int   = np.array(n_vals, dtype=np.int64)

        if self._adaptive_snr is not None:
            # Shell-size-corrected threshold (van Heel & Schatz, 2005),
            # evaluated per shell from n(xi) actually present in this box:
            #   T(xi) = [SNR + (2*sqrt(SNR)+1)/sqrt(n)] / [SNR + 1 + 2*sqrt(SNR)/sqrt(n)]
            n_arr = np.maximum(n_int.astype(np.float64), 1.0)
            snr = self._adaptive_snr
            sqrt_snr = np.sqrt(snr)
            sqrt_n = np.sqrt(n_arr)
            thr_arr = (snr + (2.0 * sqrt_snr + 1.0) / sqrt_n) / (
                snr + 1.0 + 2.0 * sqrt_snr / sqrt_n
            )
        else:
            thr_arr = np.full(fsc_arr.shape, self.threshold, dtype=np.float64)

        # Highest shell at or above the threshold (skipping DC).
        #
        # The alternative — ending the run at the *first* shell that drops
        # below the threshold — was tried and reverted.  It fails because
        # the shell-size-corrected threshold is at its highest exactly
        # where the run starts: for a 64-voxel box T = 0.462 at shell 1
        # and 0.340 at shell 2, so a run anchored at DC terminates around
        # shells 0-4 no matter how much signal sits above.  Against a
        # known band limit (150 trials, half-bit, band-limited object plus
        # independent noise per half) this rule recovers the true shell in
        # 100 % of trials at SNR 1.0 and 85 % at SNR 0.5 for L = 64, where
        # the run-from-DC rule scores 59 % and 0 %; on a real PXCT dataset
        # it moved the median from 41 nm to 232 nm.  See
        # ``test/rule_test.py``.
        above = np.where(fsc_arr[1:] >= thr_arr[1:])[0] + 1

        if len(above) == 0:
            return float("nan")
        r_res_shell = above[-1]

        # Convert shell index to resolution in pixels: shell s = s/b cycles/pixel
        r_cpx = r_res_shell / float(b)
        if r_cpx < eps:
            return float("nan")
        return 1.0 / r_cpx

    def _compute(self):
        """
        Iterate over the coarse grid, compute local FSC per box, and
        interpolate the result to the full volume shape.

        Boxes whose FSC never crossed the threshold stay ``NaN`` on the
        coarse grid and propagate through the linear interpolation, so
        the map is ``NaN`` wherever the local resolution is unknown.
        Query points are clamped to the extent of the box centres: the
        border band the grid does not cover takes the value of the
        nearest estimate instead of an extrapolation, which is what used
        to turn into negative resolutions there.

        Stores
        ------
        resolution_map : ndarray
        resolution_map_phys : ndarray
        coarse_map : ndarray
        grid_axes : list of ndarray
        n_boxes : int
        n_valid : int
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
        coarse_map = np.full(coarse_shape, np.nan, dtype=np.float64)
        measured   = np.zeros(coarse_shape, dtype=bool)

        idx_ranges = [range(len(ax)) for ax in axes]
        for grid_idx in itertools.product(*idx_ranges):
            centres = tuple(int(axes[d][grid_idx[d]]) for d in range(ndim))

            # Skip boxes that do not fit inside the support; they stay NaN.
            if self.mask_eroded is not None and not self.mask_eroded[centres]:
                continue
            measured[grid_idx] = True

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

        self.coarse_map = coarse_map
        self.grid_axes  = axes
        self.n_boxes    = int(coarse_map.size)
        self.n_valid    = int(np.count_nonzero(np.isfinite(coarse_map)))

        if self.n_valid == 0:
            warnings.warn(
                "LocalFSC: no analysis box produced a resolution estimate — "
                "the FSC never crossed the threshold anywhere inside the "
                "support.  The resolution map is entirely NaN.  Check that "
                "vol1 and vol2 are genuinely correlated half-reconstructions, "
                "and consider a larger box_size or a lower threshold.",
                UserWarning,
                stacklevel=3,
            )

        # Two kinds of missing box carry different meanings, so they are
        # treated differently before interpolation:
        #
        #   * a box skipped because it does not fit inside the support was
        #     never measured, and the map is masked back to the eroded
        #     support at the end anyway.  Leaving it NaN would poison every
        #     grid cell that touches it — linear interpolation needs all
        #     corners — and hollow out the support from the inside.  It
        #     takes the nearest measured estimate instead, the same edge
        #     hold applied at the volume border below;
        #   * a box that *was* measured and whose FSC never crossed the
        #     threshold stays NaN, and that NaN propagates: there the local
        #     resolution is genuinely unknown.
        coarse_interp = coarse_map.copy()
        finite = np.isfinite(coarse_map)
        skipped = ~measured
        if np.any(skipped) and np.any(finite):
            _, nearest = distance_transform_edt(
                ~finite, return_distances=True, return_indices=True
            )
            coarse_interp[skipped] = coarse_map[tuple(nearest)][skipped]

        # Interpolate to full volume using RegularGridInterpolator
        interp = RegularGridInterpolator(
            axes,
            coarse_interp,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )

        # Build full meshgrid of integer coordinates
        full_grids = np.mgrid[tuple(slice(0, shape[d]) for d in range(ndim))]
        # full_grids shape: (ndim, *shape)
        pts = np.stack(
            [full_grids[d].ravel() for d in range(ndim)], axis=-1
        ).astype(np.float64)

        # Clamp to the extent of the box centres.  The coarse grid stops
        # half a box short of every border, and extrapolating a linear fit
        # into that band is meaningless — it used to hand back negative
        # resolutions.  Holding the edge value keeps every entry inside the
        # range actually measured.
        for d in range(ndim):
            np.clip(pts[:, d], axes[d][0], axes[d][-1], out=pts[:, d])

        res_full = interp(pts).reshape(shape)

        # A resolution is a length: nothing <= 0 is an estimate.  Linear
        # interpolation between positive estimates cannot produce one, so
        # this is a guard rather than a correction.
        res_full[res_full <= 0.0] = np.nan

        # Outside the eroded support nothing was measured.
        if self.mask_eroded is not None:
            res_full[~self.mask_eroded] = np.nan

        self.resolution_map       = res_full
        self.resolution_map_phys  = res_full * self.pixel_size
        self.resolution_map_half  = res_full / 2.0
        self.resolution_map_phys_half = self.resolution_map_phys / 2.0

        # NaN-aware statistics; np.nan* would warn on an all-NaN map.
        if np.any(np.isfinite(res_full)):
            self.resolution_median  = float(np.nanmedian(res_full))
            self.resolution_mean    = float(np.nanmean(res_full))
            self.resolution_std     = float(np.nanstd(res_full))
        else:
            self.resolution_median  = float("nan")
            self.resolution_mean    = float("nan")
            self.resolution_std     = float("nan")

        self.resolution_median_half = self.resolution_median / 2.0
        self.resolution_mean_half   = self.resolution_mean   / 2.0
        self.resolution_std_half    = self.resolution_std    / 2.0

    def plot(self, slice_idx=None, axis=0, cmap='viridis_r', vmin=None, vmax=None):
        """
        Display the local resolution map and save it to ``LocalFSC_resmap.png``.

        For 3-D volumes the central slice (or the slice specified by
        *slice_idx*) along *axis* is shown.  The colour bar is in pixels.
        Voxels with no resolution estimate are ``NaN`` and are left blank
        by the colourmap.

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
