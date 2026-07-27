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
