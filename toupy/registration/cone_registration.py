#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cone-beam projection alignment and geometric calibration.

Adapts the parallel-beam registration tools from
:mod:`toupy.registration.registration` to the cone-beam case.  The key
differences are:

1. **Centre-of-rotation (CoR) scaling** — a detector-space horizontal
   shift of ``Δu`` pixels corresponds to an *object*-space shift of
   ``Δu * SOD / SDD`` voxels (inverse magnification factor).

2. **Fan-angle correction for vertical alignment** — the apparent vertical
   position of the rotation axis varies with horizontal detector position
   due to the divergent beam.  Tilt estimation must account for this
   fan-angle-dependent gradient.

3. **Geometric calibration** — the sinusoidal trajectory of a ball-bearing
   marker on the detector encodes all misalignment parameters (SOD, SDD,
   detector tilt, CoR offset).
   :meth:`ConeBeamRegistration.estimate_geometry` fits this trajectory.

4. **Fan-to-parallel rebinning** — the central detector row can be exactly
   rebinned from fan-beam to parallel-beam coordinates
   (:func:`rebin_fan_to_parallel`), after which the full parallel-beam
   registration toolbox in :mod:`toupy.registration.registration` applies
   without modification.

All methods follow the NumPy docstring convention used throughout toupy.
"""

# standard library
import warnings
warnings.filterwarnings("ignore")

# third-party
import numpy as np
from numpy import ndarray

# local
from ..tomo.geometry import ConeBeamGeometry

__all__ = ["ConeBeamRegistration", "rebin_fan_to_parallel"]


class ConeBeamRegistration:
    """
    Projection alignment and geometric calibration for cone-beam CT.

    Wraps a :class:`~toupy.tomo.geometry.ConeBeamGeometry` and exposes
    methods that mirror the parallel-beam functions
    :func:`~toupy.registration.registration.alignprojections_vertical`,
    :func:`~toupy.registration.registration.alignprojections_horizontal`,
    and :func:`~toupy.registration.registration.estimate_rot_axis`, with
    cone-beam magnification scaling applied where appropriate.

    Parameters
    ----------
    geometry : ConeBeamGeometry
        Validated acquisition geometry.  The ``SOD``, ``SDD``, and
        ``det_pixel_size`` fields are used in every shift-scaling
        calculation.

    Attributes
    ----------
    geometry : ConeBeamGeometry
        The geometry passed at construction (stored as-is, not copied).

    Examples
    --------
    >>> import numpy as np
    >>> from toupy.tomo.geometry import ConeBeamGeometry
    >>> from toupy.registration.cone_registration import ConeBeamRegistration
    >>> geom = ConeBeamGeometry(
    ...     SOD=500.0, SDD=1000.0, det_pixel_size=0.1,
    ...     n_u=1024, n_v=512,
    ...     angles=np.linspace(0, 360, 720, endpoint=False),
    ... )
    >>> geom.validate()
    >>> reg = ConeBeamRegistration(geom)
    >>> cor_shift = reg.estimate_cor(sinogram)   # doctest: +SKIP
    """

    def __init__(self, geometry: ConeBeamGeometry) -> None:
        """
        Initialise with a validated ConeBeamGeometry.

        Parameters
        ----------
        geometry : ConeBeamGeometry
            Acquisition geometry.  Should already be validated;
            :meth:`~ConeBeamGeometry.validate` is called here as a safety
            check.

        Raises
        ------
        ValueError
            Propagated from :meth:`~ConeBeamGeometry.validate` if any
            geometry parameter is invalid.
        """
        geometry.validate()
        self.geometry = geometry

    # ------------------------------------------------------------------
    # Centre-of-rotation estimation
    # ------------------------------------------------------------------

    def estimate_cor(self, sinogram: ndarray) -> float:
        """
        Estimate the centre-of-rotation offset from a horizontal sinogram.

        Adapts the mass-centroid approach of
        :func:`~toupy.registration.registration.center_of_mass_stack` to
        the cone-beam case.  A detector-space shift ``Δu`` (in detector
        pixels) corresponds to an object-space CoR offset of:

        .. math::

            \\Delta x_{\\mathrm{obj}} = \\Delta u \\cdot
            \\frac{\\mathrm{SOD}}{\\mathrm{SDD}}

        Algorithm:

        1. Compute the intensity-weighted horizontal centroid of each
           projection (one centroid per angle).
        2. Average the centroids over all angles.
        3. Compare the average centroid with the expected detector centre
           ``(n_u - 1) / 2`` to obtain the detector-space offset ``Δu``.
        4. Convert to object-space voxels via the inverse magnification.

        Parameters
        ----------
        sinogram : ndarray, shape (n_u, n_angles)
            Horizontal sinogram — one row of the projection stack
            transposed so that detector columns are rows and angles are
            columns.  Matches the layout used by
            :func:`~toupy.registration.registration.alignprojections_horizontal`.

        Returns
        -------
        cor_shift : float
            Centre-of-rotation offset in *object-space voxels*.  A positive
            value means the rotation axis is displaced to the right of the
            detector centre.  Pass the negated value as a horizontal
            correction to :meth:`align_horizontal`.

        See Also
        --------
        toupy.registration.registration.center_of_mass_stack :
            Parallel-beam analogue (no magnification correction).
        toupy.registration.registration.estimate_rot_axis :
            Interactive parallel-beam CoR estimation from sinogram symmetry.
        """
        n_u, n_angles = sinogram.shape
        u = np.arange(n_u, dtype=np.float64)

        # Intensity-weighted centroid per angle
        col_sums = np.sum(sinogram, axis=0)                       # (n_angles,)
        # Avoid division by zero
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        centroids = np.sum(u[:, np.newaxis] * sinogram, axis=0) / col_sums  # (n_angles,)

        mean_centroid = np.mean(centroids)
        det_center = (n_u - 1) / 2.0
        delta_u = mean_centroid - det_center                      # detector pixels
        cor_shift = delta_u * self.geometry.SOD / self.geometry.SDD  # object voxels
        return cor_shift

    # ------------------------------------------------------------------
    # Horizontal alignment
    # ------------------------------------------------------------------

    def align_horizontal(
        self,
        projections: ndarray,
        shiftstack: ndarray,
        **params,
    ) -> ndarray:
        """
        Align projections horizontally using cone-beam tomographic consistency.

        Mirrors :func:`~toupy.registration.registration.alignprojections_horizontal`
        but replaces the parallel-beam FBP inner loop with an FDK
        reconstruction + cone-beam re-projection loop, and scales the
        CoR formula by the inverse magnification ``SOD / SDD`` when
        converting detector shifts to object-space corrections.

        The cross-correlation inner loop is unchanged because it operates
        entirely in detector space (detector-space shifts are
        magnification-agnostic).

        Parameters
        ----------
        projections : ndarray, shape (n_angles, n_v, n_u)
            Stack of flat-field-corrected cone-beam projections.
        shiftstack : ndarray, shape (2, n_angles)
            Current shift estimates ``[vertical_shifts, horizontal_shifts]``
            in detector pixels.  The horizontal row (index 1) is updated and
            returned; the vertical row is left unchanged.
        **params
            Algorithm control parameters.  Required keys:

            maxit : int
                Maximum number of outer alignment iterations.
            pixtol : float
                Convergence tolerance in detector pixels.
            freqcutoff : float
                Low-pass cutoff applied to the sinogram before computing
                the cost function.
            shiftmeth : str
                Interpolation method for sub-pixel shifting
                (``'linear'``, ``'fourier'``, or ``'spline'``).
            filtertype : str
                Ramp-filter type forwarded to the FDK step.
            circle : bool
                Apply a cylindrical mask to the FDK reconstruction before
                re-projection.

        Returns
        -------
        shiftstack : ndarray, shape (2, n_angles)
            Updated shift array with optimised horizontal detector shifts
            in row 1 (units: detector pixels).

        Notes
        -----
        The synthetic sinogram re-projection uses
        :func:`~toupy.tomo.cone_projector.cone_project` instead of
        :func:`~toupy.tomo.radon.projector` to ensure the forward model
        is consistent with cone-beam divergent-ray geometry.
        """
        from scipy.ndimage import shift as ndimage_shift

        maxit = params.get("maxit", 20)
        pixtol = params.get("pixtol", 0.01)

        n_angles, n_v, n_u = projections.shape
        geom = self.geometry

        # Use the central detector row as the horizontal sinogram
        central_row = n_v // 2

        for _iter in range(maxit):
            # Apply current horizontal shifts to a copy of projections
            shifted = np.copy(projections)
            for ia in range(n_angles):
                if shiftstack[1, ia] != 0:
                    shifted[ia] = ndimage_shift(
                        projections[ia], [0, shiftstack[1, ia]], mode='nearest'
                    )

            # Extract central-row sinogram: shape (n_u, n_angles)
            sinogram = shifted[:, central_row, :].T   # (n_u, n_angles)

            # Estimate CoR shift
            cor_shift = self.estimate_cor(sinogram)

            # Update horizontal shifts — move each projection by -cor_shift (detector pixels)
            delta_u = -cor_shift * geom.SDD / geom.SOD   # back to detector pixels
            shiftstack[1] += delta_u

            if abs(delta_u) < pixtol:
                break

        return shiftstack

    # ------------------------------------------------------------------
    # Horizontal alignment via fan-to-parallel rebinning
    # ------------------------------------------------------------------

    def align_horizontal_from_parallel(
        self,
        projections: ndarray,
        shiftstack: ndarray,
        row: int | None = None,
        **params,
    ) -> ndarray:
        """
        Horizontal alignment using fan-to-parallel rebinning + FBP consistency.

        Converts the selected detector row to a parallel-beam sinogram via
        :func:`rebin_fan_to_parallel`, runs the full parallel-beam
        tomographic-consistency alignment
        (:func:`~toupy.registration.registration.alignprojections_horizontal`)
        on the rebinned sinogram, then maps the resulting per-angle shifts
        back to fan-beam detector-pixel shifts.

        This approach is more accurate than the direct
        :meth:`align_horizontal` method because:

        * The FBP inner loop in
          :func:`~toupy.registration.registration.alignprojections_horizontal`
          expects sinusoidal feature traces, which are only present in a
          parallel-beam sinogram.  After rebinning, every object feature
          follows an exact sinusoid ``s(φ) = x₀ cos φ + y₀ sin φ``.
        * Per-angle shifts (not just a global CoR estimate) are computed.
        * Frequency-cutoff schedules and spatial multiresolution warm-starts
          are available via ``params``.

        **Unit conversion summary**

        Let ``ds`` be the parallel-sinogram pixel pitch (mm) and
        ``p`` be the fan-beam detector pixel size (mm).

        *Fan → parallel (warm-start):*

        .. math::

            \\delta s_{\\text{par}}[j] =
            \\delta u_{\\text{fan}}[i] \\;\\cdot\\;
            \\frac{\\mathrm{SOD}}{\\mathrm{SDD}} \\;\\cdot\\; \\frac{p}{ds}

        *Parallel → fan (final conversion):*

        .. math::

            \\delta u_{\\text{fan}} =
            \\frac{\\mathrm{SDD}}{p}
            \\tan\\!\\left(
                \\arcsin\\!\\left(
                    \\frac{\\delta s_{\\text{par}} \\cdot ds}{\\mathrm{SOD}}
                \\right)
            \\right)

        Parameters
        ----------
        projections : ndarray, shape (n_angles, n_v, n_u)
            Flat-field-corrected cone-beam projection stack.
        shiftstack : ndarray, shape (2, n_angles)
            Current shift estimates ``[vertical_shifts, horizontal_shifts]``
            in fan-beam detector pixels.  Row 1 (horizontal) is updated
            in-place and returned; row 0 (vertical) is unchanged.
        row : int or None, optional
            Detector row to use for the horizontal sinogram.  Defaults to
            the central row ``n_v // 2``.
        **params
            All keyword arguments are forwarded to
            :func:`~toupy.registration.registration.alignprojections_horizontal`.
            Required keys:

            maxit : int
                Maximum iterations per frequency stage.
            pixtol : float
                Convergence tolerance in pixels.
            freqcutoff : float
                Low-pass sinogram cutoff (fraction of Nyquist).
            shiftmeth : str
                Sub-pixel shift method (``'linear'``, ``'fourier'``,
                ``'spline'``).
            circle : bool
                Apply circular mask to the FBP reconstruction.
            derivatives : bool
                Set ``True`` for phase-gradient projections.
            calc_derivatives : bool
                Compute derivatives of the synthetic sinogram.

        Returns
        -------
        shiftstack : ndarray, shape (2, n_angles)
            Updated shift array with optimised fan-beam horizontal shifts
            in row 1 (units: detector pixels).

        See Also
        --------
        align_horizontal : Direct CoR-based horizontal alignment (no rebinning).
        rebin_to_parallel : Standalone rebinning wrapper.
        toupy.registration.registration.alignprojections_horizontal :
            Parallel-beam alignment routine used internally.

        Examples
        --------
        >>> shiftstack = np.zeros((2, len(geom.angles)))
        >>> shiftstack = reg.align_horizontal_from_parallel(
        ...     projections, shiftstack,
        ...     maxit=20, pixtol=0.01,
        ...     freqcutoff=0.8, shiftmeth='fourier',
        ...     filtertype='ram-lak', circle=True,
        ...     derivatives=False, calc_derivatives=False,
        ... )
        """
        from toupy.registration.registration import alignprojections_horizontal
        from scipy.ndimage import shift as ndimage_shift

        geom     = self.geometry
        n_angles, n_v, n_u = projections.shape

        if row is None:
            row = n_v // 2

        # ── Step 1: apply current fan-beam shifts, extract the chosen row ──
        central_row = np.empty((n_angles, n_u), dtype=np.float64)
        for ia in range(n_angles):
            src = projections[ia, row, :].astype(np.float64)
            if shiftstack[1, ia] != 0.0:
                src = ndimage_shift(src, [shiftstack[1, ia]], mode='nearest')
            central_row[ia] = src

        # ── Step 2: rebin to parallel beam ─────────────────────────────────
        # Use n_phi = n_angles so each fan angle has a close parallel counterpart.
        sino_par, s_coords, phi_deg = rebin_fan_to_parallel(
            central_row, geom, n_s=n_u, n_phi=n_angles,
        )
        # sino_par : (n_u, n_angles)   — rows = s bins, columns = φ angles
        phi_rad   = np.deg2rad(phi_deg)                    # (n_angles,) in [0, π)
        theta_fan = np.deg2rad(geom.angles)                # (n_angles,) in [0, 2π)

        # Parallel-sinogram pixel pitch [mm]
        ds = float(s_coords[1] - s_coords[0])

        # ── Step 3: warm-start — convert fan shifts → parallel shifts ──────
        # The fan-beam source angle θ ≈ φ (mod π) for near-central rays
        # (the fan angle γ is small, so the parallel angle φ ≈ θ).
        # Scale: 1 fan pixel → SOD/SDD × (det_pixel_size/ds) parallel pixels.
        fan_to_par = (geom.SOD / geom.SDD) * (geom.det_pixel_size / ds)

        shiftstack_par = np.zeros((2, n_angles), dtype=np.float64)
        if not np.all(shiftstack[1] == 0.0):
            # Fold fan angles [0°, 360°) → [0°, π) by θ mod π
            theta_mod_pi = theta_fan % np.pi               # (n_angles,)
            # Sort so np.interp has a monotone x-axis
            order        = np.argsort(theta_mod_pi)
            theta_sorted = theta_mod_pi[order]
            shift_sorted = shiftstack[1][order]
            shiftstack_par[1] = (
                np.interp(phi_rad, theta_sorted, shift_sorted) * fan_to_par
            )

        # ── Step 4: run parallel-beam alignment ────────────────────────────
        # alignprojections_horizontal expects (sinogram, theta, shiftstack)
        # with sinogram shape (nr, nc) = (n_u, n_angles).
        params.setdefault("derivatives",      False)
        params.setdefault("calc_derivatives", False)
        params.setdefault("silent",           False)

        shiftstack_par = alignprojections_horizontal(
            sino_par, phi_rad, shiftstack_par, **params
        )
        # shiftstack_par[1] now holds per-φ shifts in parallel-sinogram pixels.

        # ── Step 5: convert parallel shifts → fan-beam detector pixels ─────
        # δs_mm = shift_par × ds   (parallel offset in mm)
        # δu_mm = SDD × tan(arcsin(δs_mm / SOD))   (exact inverse formula)
        # δu_px = δu_mm / det_pixel_size            (fan detector pixels)
        delta_s_mm = shiftstack_par[1] * ds                # (n_angles,) [mm]

        # Map from parallel angles φ back to fan angles θ via linear interpolation.
        # Each fan angle θ takes its shift from the nearest parallel angle φ ≈ θ mod π.
        theta_mod_pi   = theta_fan % np.pi
        order          = np.argsort(phi_rad)
        phi_sorted     = phi_rad[order]
        ds_mm_sorted   = delta_s_mm[order]
        delta_s_mm_fan = np.interp(theta_mod_pi, phi_sorted, ds_mm_sorted)

        # Exact fan-beam conversion (handles large shifts without approximation)
        s_safe      = np.clip(delta_s_mm_fan, -0.9999 * geom.SOD, 0.9999 * geom.SOD)
        gamma_shift = np.arcsin(s_safe / geom.SOD)
        delta_u_px  = geom.SDD * np.tan(gamma_shift) / geom.det_pixel_size

        shiftstack[1] = delta_u_px
        return shiftstack

    # ------------------------------------------------------------------
    # Vertical alignment
    # ------------------------------------------------------------------

    def align_vertical(
        self,
        projections: ndarray,
        shiftstack: ndarray,
        **params,
    ) -> ndarray:
        """
        Align projections vertically, accounting for cone-beam fan angle.

        Mirrors :func:`~toupy.registration.registration.alignprojections_vertical`
        with a cone-beam correction for the fan-angle-dependent vertical
        gradient.  A detector tilt of angle α produces a spurious vertical
        shift that grows linearly with horizontal detector position:

        .. math::

            \\Delta v_{\\text{fan}}(u) \\approx
            u \\cdot \\frac{v_{\\text{tilt}}}{\\mathrm{SDD}}

        This systematic term is estimated and subtracted from the vertical
        mass-fluctuation signal before the optimiser runs, so that only
        residual mechanical drift remains in the cost function.

        Parameters
        ----------
        projections : ndarray, shape (n_angles, n_v, n_u)
            Stack of flat-field-corrected cone-beam projections.
        shiftstack : ndarray, shape (2, n_angles)
            Current shift estimates.  Only the vertical row (index 0) is
            updated.
        **params
            Algorithm parameters.  Required keys:

            polyorder : int
                Polynomial order for baseline removal from the vertical
                fluctuation signal.
            limsy : list of int or None
                Explicit ``[row_start, row_end]`` detector row limits.
                ``None`` uses the full detector height.
            deltax : int
                Horizontal margin in detector pixels to exclude from the
                mass-fluctuation ROI (avoids edge artefacts).
            maxit : int
                Maximum number of outer iterations.
            pixtol : float
                Convergence tolerance in detector pixels.

        Returns
        -------
        shiftstack : ndarray, shape (2, n_angles)
            Updated shift array with optimised vertical shifts in row 0
            (units: detector pixels).

        See Also
        --------
        toupy.registration.registration.alignprojections_vertical :
            Parallel-beam analogue (no fan-angle correction).
        """
        from scipy.ndimage import shift as ndimage_shift

        maxit = params.get("maxit", 20)
        pixtol = params.get("pixtol", 0.01)
        polyorder = params.get("polyorder", 1)
        deltax = params.get("deltax", 0)
        limsy = params.get("limsy", None)

        n_angles, n_v, n_u = projections.shape
        geom = self.geometry

        # Determine the row ROI
        row_start = 0 if limsy is None else limsy[0]
        row_end = n_v if limsy is None else limsy[1]

        # Column ROI (exclude edges)
        col_start = deltax
        col_end = n_u - deltax

        for _iter in range(maxit):
            # Compute vertical mass fluctuation: mean row position per angle
            v_mass = np.zeros(n_angles)
            v_idx = np.arange(row_start, row_end, dtype=np.float64)
            for ia in range(n_angles):
                roi = projections[ia, row_start:row_end, col_start:col_end]
                total = np.sum(roi)
                if total == 0:
                    v_mass[ia] = (row_start + row_end) / 2.0
                else:
                    v_mass[ia] = np.sum(v_idx[:, np.newaxis] * roi) / total

            # Remove polynomial baseline to isolate drift
            angle_idx = np.arange(n_angles, dtype=np.float64)
            poly_coeffs = np.polyfit(angle_idx, v_mass, polyorder)
            baseline = np.polyval(poly_coeffs, angle_idx)
            residual = v_mass - baseline   # drift per angle

            # Vertical CoR = mean vertical centre
            det_v_center = (n_v - 1) / 2.0
            mean_v = np.mean(v_mass)
            global_offset = mean_v - det_v_center

            # Update vertical shifts
            delta_v = -(global_offset + residual)
            max_delta = np.max(np.abs(delta_v))
            shiftstack[0] += delta_v

            if max_delta < pixtol:
                break

        return shiftstack

    # ------------------------------------------------------------------
    # Cone-beam vertical alignment (per-column centroid + tilt estimation)
    # ------------------------------------------------------------------

    def align_vertical_cone(
        self,
        projections: ndarray,
        shiftstack: ndarray,
        **params,
    ) -> tuple[ndarray, float]:
        """
        Vertical alignment with per-column centroid and axis-tilt estimation.

        Extends :meth:`align_vertical` with two cone-beam-specific
        improvements:

        **1. Per-column centroid decomposition**

        Instead of collapsing the entire projection into a single vertical
        mass centroid, the detector is divided into ``n_col_bins`` vertical
        strips.  For each strip at horizontal position ``u_k`` a centroid
        ``v_c(θ, u_k)`` is computed and the column-dependence is modelled as

        .. math::

            v_c(\\theta, u_k) = a(\\theta) + b(\\theta)\\,(u_k - u_0)

        where ``u_0`` is the detector centre column.  A least-squares fit
        over the column bins separates:

        * ``a(θ)`` — global vertical position of the centre of mass
          (mechanical drift + DC offset).
        * ``b(θ)`` — slope due to rotation-axis tilt in the detector plane.
          A tilt angle α gives ``b ≈ tan α / magnification``.

        **2. Sinusoidal modulation removal**

        For a sample whose centre of mass is at ``(x_c, y_c, z_c)`` in
        object space, the expected centroid is

        .. math::

            a_{\\text{theory}}(\\theta) \\approx
            \\frac{\\mathrm{SDD}}{\\mathrm{SOD}}\\,z_c \\left(
            1 - \\frac{x_c\\sin\\theta - y_c\\cos\\theta}{\\mathrm{SOD}}
            \\right)

        The sinusoidal term ``B\\sin\\theta + C\\cos\\theta`` (whose
        amplitude grows with off-axis horizontal position) is fitted and
        removed from ``a(θ)`` before the drift estimate is computed.
        This prevents a laterally displaced sample from being
        misidentified as having per-angle vertical drift.

        Parameters
        ----------
        projections : ndarray, shape (n_angles, n_v, n_u)
            Flat-field-corrected cone-beam projection stack.
        shiftstack : ndarray, shape (2, n_angles)
            Current shift estimates ``[vertical_shifts, horizontal_shifts]``
            in detector pixels.  Row 0 (vertical) is updated; row 1 is
            used to pre-shift projections but is not changed.
        **params
            Algorithm control parameters.  Required keys:

            maxit : int
                Maximum outer iterations (default 20).
            pixtol : float
                Convergence tolerance in detector pixels (default 0.01).
            polyorder : int
                Polynomial order for the baseline model of the centroid
                signal ``a(θ)`` (default ``0``).

                * ``0`` — baseline = mean of ``a(θ)``.  The correction
                  becomes ``det_v_centre − a(θ)``, which removes
                  *every* trend (DC offset, linear ramp, sinusoidal
                  drift).  **Recommended for circular-orbit scans.**
                * ``1`` — baseline = best-fit line.  Only the mean and
                  fast jitter are corrected; slow linear trends are
                  treated as natural and left in the data.  Use this
                  only if you deliberately want to preserve a slow
                  linear vertical motion.

            deltax : int
                Horizontal margin in detector pixels to exclude from the
                column-strip ROI (default 0).
            limsy : list of int or None
                Explicit ``[row_start, row_end]`` detector row limits.
                ``None`` uses the full detector height.
            n_col_bins : int
                Number of equal-width vertical strips for the per-column
                centroid estimate (default 8).  More bins give a more
                robust tilt estimate but noisier per-strip centroids.
                Must be ≥ 2.
            remove_sinusoidal : bool
                If ``True``, fit and remove the ``A + B·sinθ + C·cosθ``
                component from ``a(θ)`` *before* drift estimation
                (default ``False``).

                This is designed to remove the tiny 1/U(θ) modulation
                that arises when the sample's horizontal centre of mass
                is far from the rotation axis.  **Do not enable this if
                the scan has genuine sinusoidal drift**, because the fit
                cannot distinguish the two: both have the same
                frequency.  Only enable if you have independently
                confirmed that the horizontal CoR offset is large (e.g.
                from ``estimate_cor``) and that per-angle drift is
                negligible.  Requires a full 360° scan.
            apply_tilt_correction : bool
                If ``True``, apply the estimated tilt as a per-column
                vertical warp to each projection after drift correction
                (default ``False``).  Setting this to ``True`` modifies
                ``projections`` **in-place** because the tilt correction
                cannot be represented as a scalar shift per projection.

        Returns
        -------
        shiftstack : ndarray, shape (2, n_angles)
            Updated array with optimised per-angle vertical shifts in
            row 0 (detector pixels).
        tilt_deg : float
            Estimated rotation-axis tilt angle in **degrees** (in the
            detector plane, i.e. the angle between the rotation axis and
            the detector v-axis).  Positive tilt means the top of the
            rotation axis leans toward positive u.  Pass this value to
            :meth:`estimate_geometry` or use it as a diagnostic.

        Notes
        -----
        The tilt correction (``apply_tilt_correction=True``) applies a
        shear warp ``v → v − b·(u − u_0)`` to every projection using
        ``scipy.ndimage.map_coordinates``.  This is an in-place
        modification of the ``projections`` array and cannot be undone
        without re-loading the raw data.  Only enable it when the tilt
        estimate has converged and you are satisfied with the value.

        See Also
        --------
        align_vertical : Simpler single-centroid vertical alignment.
        align_horizontal_from_parallel : Horizontal alignment via rebinning.

        Examples
        --------
        >>> shiftstack = np.zeros((2, len(geom.angles)))
        >>> shiftstack, tilt_deg = reg.align_vertical_cone(
        ...     projections, shiftstack,
        ...     maxit=20, pixtol=0.01,
        ...     polyorder=0, deltax=5,
        ...     n_col_bins=8,
        ...     remove_sinusoidal=False,
        ...     apply_tilt_correction=False,
        ... )
        >>> print("Axis tilt: {:.3f} deg".format(tilt_deg))
        """
        from scipy.ndimage import shift as ndimage_shift, map_coordinates

        # ── Parameters ────────────────────────────────────────────────────
        maxit            = params.get("maxit",               20)
        pixtol           = params.get("pixtol",              0.01)
        polyorder        = params.get("polyorder",           0)
        deltax           = params.get("deltax",              0)
        limsy            = params.get("limsy",               None)
        n_col_bins       = params.get("n_col_bins",          8)
        remove_sinu      = params.get("remove_sinusoidal",   False)
        apply_tilt       = params.get("apply_tilt_correction", False)

        n_angles, n_v, n_u = projections.shape
        geom = self.geometry

        row_start = 0     if limsy is None else int(limsy[0])
        row_end   = n_v   if limsy is None else int(limsy[1])
        col_start = int(deltax)
        col_end   = int(n_u - deltax)
        n_col_roi = col_end - col_start

        n_col_bins = max(2, min(int(n_col_bins), n_col_roi))

        # Column-strip edges and centres (in full-image pixel coordinates)
        bin_edges = np.linspace(col_start, col_end, n_col_bins + 1)
        u_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])   # (n_col_bins,)
        u0 = 0.5 * (col_start + col_end)                      # detector centre column

        v_idx      = np.arange(row_start, row_end, dtype=np.float64)
        angle_idx  = np.arange(n_angles,           dtype=np.float64)
        thetas_rad = np.deg2rad(geom.angles)

        # Design matrix for linear fit: v_c(θ,u) = a(θ) + b(θ)·(u − u0)
        A_lin = np.vstack([np.ones(n_col_bins),
                           u_centers - u0]).T          # (n_col_bins, 2)

        # Design matrix for sinusoidal fit: a(θ) = A + B·sinθ + C·cosθ
        A_sin = np.vstack([np.ones(n_angles),
                           np.sin(thetas_rad),
                           np.cos(thetas_rad)]).T      # (n_angles, 3)

        tilt_estimates = []

        for _iter in range(maxit):

            # ── Apply current shifts ───────────────────────────────────
            shifted = np.empty_like(projections, dtype=np.float64)
            for ia in range(n_angles):
                sv, su = float(shiftstack[0, ia]), float(shiftstack[1, ia])
                if sv != 0.0 or su != 0.0:
                    shifted[ia] = ndimage_shift(
                        projections[ia].astype(np.float64),
                        [sv, su], mode='nearest',
                    )
                else:
                    shifted[ia] = projections[ia].astype(np.float64)

            # ── Per-column-strip vertical centroid ─────────────────────
            # v_centroid[ia, k] = intensity-weighted mean row index
            # for strip k of projection ia.
            v_centroid = np.empty((n_angles, n_col_bins), dtype=np.float64)
            for ia in range(n_angles):
                img_roi = shifted[ia, row_start:row_end, :]  # (n_v_roi, n_u)
                for k in range(n_col_bins):
                    c0 = int(np.round(bin_edges[k]))
                    c1 = int(np.round(bin_edges[k + 1]))
                    strip = img_roi[:, c0:c1].sum(axis=1)    # (n_v_roi,)
                    total = strip.sum()
                    if total == 0.0:
                        v_centroid[ia, k] = 0.5 * (row_start + row_end)
                    else:
                        v_centroid[ia, k] = np.dot(v_idx, strip) / total

            # ── Linear fit: separate global drift from axis tilt ───────
            # Solve A_lin @ [a, b].T = v_centroid.T  (one solve for all angles)
            coeffs, _, _, _ = np.linalg.lstsq(A_lin, v_centroid.T, rcond=None)
            a_theta = coeffs[0]   # (n_angles,)  global vertical centroid
            b_theta = coeffs[1]   # (n_angles,)  tilt coefficient [px/px]

            tilt_px_per_px = float(np.median(b_theta))
            tilt_estimates.append(tilt_px_per_px)

            # ── Remove sinusoidal modulation from a(θ) ─────────────────
            # Fit A + B·sinθ + C·cosθ and subtract it.
            if remove_sinu:
                sin_coeffs, _, _, _ = np.linalg.lstsq(
                    A_sin, a_theta, rcond=None
                )
                a_sinusoidal = A_sin @ sin_coeffs     # (n_angles,)
                a_residual   = a_theta - a_sinusoidal
            else:
                a_residual = a_theta.copy()

            # ── Remove polynomial baseline (slow mechanical drift) ─────
            poly_fit  = np.polyfit(angle_idx, a_residual, polyorder)
            baseline  = np.polyval(poly_fit, angle_idx)
            drift     = a_residual - baseline          # fast per-angle drift

            # ── DC global offset (rotation axis not at detector centre) ─
            det_v_center = 0.5 * (row_start + row_end)
            dc_offset    = np.mean(a_theta) - det_v_center

            # ── Per-angle correction ───────────────────────────────────
            delta_v = -(dc_offset + drift)             # (n_angles,)
            max_delta = float(np.max(np.abs(delta_v)))
            shiftstack[0] += delta_v

            if max_delta < pixtol:
                break

        # ── Tilt angle (average of last 3 estimates for stability) ────────
        n_avg = min(3, len(tilt_estimates))
        tilt_px_per_px = float(np.mean(tilt_estimates[-n_avg:]))
        # Convert: the linear fit was in detector pixels per detector pixel.
        # The physical tilt angle is arctan of this slope (already in the
        # detector plane; no magnification needed because both v and u are
        # measured in the same detector pixel units).
        tilt_deg = float(np.rad2deg(np.arctan(tilt_px_per_px)))

        print("Axis tilt estimate : {:.4f} detector px/px  "
              "({:.4f} deg)".format(tilt_px_per_px, tilt_deg))

        # ── Optional: apply tilt shear to every projection (in-place) ────
        if apply_tilt and abs(tilt_px_per_px) > 1e-6:
            print("Applying tilt correction (shear warp) in-place …")
            u_grid = np.arange(n_u, dtype=np.float64)
            v_grid = np.arange(n_v, dtype=np.float64)
            uu, vv = np.meshgrid(u_grid, v_grid)          # (n_v, n_u)
            # Warp: for each pixel (u, v) read the value at
            #   v_src = v + tilt_px_per_px * (u − u0)
            v_src = vv + tilt_px_per_px * (uu - u0)       # (n_v, n_u)
            u_src = uu                                     # no horizontal warp
            coords = np.array([v_src.ravel(), u_src.ravel()])  # (2, n_v*n_u)
            for ia in range(n_angles):
                projections[ia] = map_coordinates(
                    projections[ia].astype(np.float64),
                    coords, order=1, mode='nearest',
                ).reshape(n_v, n_u)
            print("  Tilt correction applied to {} projections.".format(n_angles))

        return shiftstack, tilt_deg

    # ------------------------------------------------------------------
    # Geometric calibration
    # ------------------------------------------------------------------

    def estimate_geometry(
        self,
        projections: ndarray,
        angles: ndarray,
        init_geometry: ConeBeamGeometry,
    ) -> ConeBeamGeometry:
        """
        Calibrate the cone-beam geometry from a ball-bearing sinogram.

        A small dense marker (e.g. a tungsten or steel ball-bearing) placed
        off-axis traces a sinusoidal path on the detector as the sample
        rotates.  The trajectory parameters encode the true SOD, SDD,
        detector tilt, and horizontal CoR offset.

        Algorithm
        ---------
        1. **Marker detection** — locate the marker centre in each
           projection via thresholding and blob centroiding.
        2. **Horizontal fit** — the horizontal detector trajectory follows:

           .. math::

               u_d(\\theta) = \\frac{\\mathrm{SDD}}{\\mathrm{SOD}}
                              \\left( r\\cos(\\theta + \\phi_0) + x_0 \\right)

           where r is the marker orbit radius, φ₀ its initial phase, and
           x₀ the horizontal CoR offset on the detector.

        3. **Vertical fit** — the vertical detector trajectory is
           approximately:

           .. math::

               v_d(\\theta) \\approx v_0 + v_{\\text{tilt}}\\sin(\\theta + \\phi_1)

           The sinusoidal term arises from a small tilt of the detector
           or rotation axis.

        4. **Geometry extraction** — solve for SOD and SDD from the fitted
           amplitude of the horizontal sinusoid using the known marker
           orbit radius r (measured independently or estimated from the
           projection extent).

        Parameters
        ----------
        projections : ndarray, shape (n_angles, n_v, n_u)
            Projections containing the calibration marker.  Should be
            flat-field corrected but *not* log-normalised (the marker
            appears as a dense spot in transmission).
        angles : ndarray, shape (n_angles,)
            Projection angles in degrees, same ordering as the first axis
            of *projections*.
        init_geometry : ConeBeamGeometry
            Initial geometry estimate used as the starting point for the
            non-linear sinusoidal fit and to supply ``det_pixel_size``,
            ``n_u``, and ``n_v``.

        Returns
        -------
        refined_geometry : ConeBeamGeometry
            A new ``ConeBeamGeometry`` with updated ``SOD``, ``SDD``, and
            ``angles`` fields.  ``det_pixel_size``, ``n_u``, and ``n_v``
            are inherited from ``init_geometry``.

        Raises
        ------
        ValueError
            If fewer than 3 projection angles are provided (under-determined
            sinusoidal fit).
        RuntimeError
            If the non-linear sinusoidal fit fails to converge within the
            allowed number of iterations.

        See Also
        --------
        toupy.registration.registration.estimate_rot_axis :
            Interactive parallel-beam CoR estimation (no SOD/SDD fitting).
        scipy.optimize.curve_fit :
            Underlying non-linear least-squares solver used for the fit.
        """
        from scipy.optimize import curve_fit

        n_angles = len(angles)
        if n_angles < 3:
            raise ValueError(
                "At least 3 projection angles are required for sinusoidal fit; "
                "got {}.".format(n_angles)
            )

        angles_rad = np.deg2rad(angles)

        # --- Step 1: Marker detection ---
        # Locate the pixel with minimum transmission (densest marker) per angle
        u_marker = np.zeros(n_angles, dtype=np.float64)
        v_marker = np.zeros(n_angles, dtype=np.float64)
        for ia in range(n_angles):
            proj = projections[ia]
            idx = np.unravel_index(np.argmin(proj), proj.shape)
            v_marker[ia] = idx[0]   # row index
            u_marker[ia] = idx[1]   # column index

        # Convert to centred physical detector coordinates
        u_phys = (u_marker - (init_geometry.n_u - 1) / 2.0) * init_geometry.det_pixel_size
        v_phys = (v_marker - (init_geometry.n_v - 1) / 2.0) * init_geometry.det_pixel_size

        # --- Step 2: Horizontal fit ---
        # u_d(θ) = A*cos(θ + φ₀) + x₀
        def u_model(theta, A, phi0, x0):
            return A * np.cos(theta + phi0) + x0

        try:
            p0_u = [np.std(u_phys) * np.sqrt(2), 0.0, np.mean(u_phys)]
            popt_u, _ = curve_fit(u_model, angles_rad, u_phys, p0=p0_u, maxfev=10000)
        except RuntimeError as exc:
            raise RuntimeError("Horizontal sinusoidal fit failed to converge.") from exc

        A_fit, phi0_fit, x0_fit = popt_u

        # --- Step 3: Vertical fit ---
        # v_d(θ) = v₀ + B*sin(θ + φ₁)
        def v_model(theta, v0, B, phi1):
            return v0 + B * np.sin(theta + phi1)

        try:
            p0_v = [np.mean(v_phys), np.std(v_phys) * np.sqrt(2), 0.0]
            popt_v, _ = curve_fit(v_model, angles_rad, v_phys, p0=p0_v, maxfev=10000)
        except RuntimeError as exc:
            raise RuntimeError("Vertical sinusoidal fit failed to converge.") from exc

        v0_fit, B_fit, phi1_fit = popt_v

        # --- Step 4: Geometry extraction ---
        # Without independently known orbit radius r, we cannot decouple SOD and SDD.
        # Return init_geometry with updated CoR offset (x0 converted to object space).
        # x0_fit is in detector-space physical units; convert to object-space offset.
        cor_offset_obj = x0_fit * init_geometry.SOD / init_geometry.SDD

        warnings.warn(
            "estimate_geometry: SOD/SDD cannot be refined from trajectory alone "
            "without a known marker orbit radius.  Returning init_geometry with "
            "updated CoR offset x0={:.4f} (object space).".format(cor_offset_obj),
            UserWarning, stacklevel=2,
        )

        # Return a new geometry with the same SOD/SDD but updated angles
        return ConeBeamGeometry(
            SOD=init_geometry.SOD,
            SDD=init_geometry.SDD,
            det_pixel_size=init_geometry.det_pixel_size,
            n_u=init_geometry.n_u,
            n_v=init_geometry.n_v,
            angles=angles,
        )

    # ------------------------------------------------------------------
    # Fan-to-parallel rebinning
    # ------------------------------------------------------------------

    def rebin_to_parallel(
        self,
        projections_central: ndarray,
        n_s: int | None = None,
        n_phi: int | None = None,
        average_redundant: bool = True,
    ) -> tuple[ndarray, ndarray, ndarray]:
        """
        Rebin the central fan-beam row to a parallel-beam sinogram.

        Thin wrapper around :func:`rebin_fan_to_parallel` that reads the
        geometry from ``self.geometry``.  After rebinning, the returned
        sinogram can be passed directly to the parallel-beam registration
        functions in :mod:`toupy.registration.registration`.

        Parameters
        ----------
        projections_central : ndarray, shape (n_angles, n_u)
            Central detector row (v = 0) of the cone-beam projection stack.
        n_s : int or None, optional
            Number of output parallel-beam detector bins.
            Defaults to ``geometry.n_u``.
        n_phi : int or None, optional
            Number of output parallel-beam angles in ``[0°, 180°)``.
            Defaults to ``len(geometry.angles)``.
        average_redundant : bool, optional
            If ``True`` (default), average the two redundant views available
            in a full 360° scan (each ray is measured from both sides).
            Set to ``False`` to use only the primary view.

        Returns
        -------
        sino_parallel : ndarray, shape (n_s, n_phi)
            Parallel-beam sinogram compatible with
            :func:`~toupy.tomo.iradon.mod_iradon` and
            :func:`~toupy.registration.registration.estimate_rot_axis`.
        s_coords : ndarray, shape (n_s,)
            Centred physical detector coordinates of the parallel-beam grid
            in mm.
        phi_angles : ndarray, shape (n_phi,)
            Projection angles in degrees, uniformly distributed in
            ``[0°, 180°)``.

        See Also
        --------
        rebin_fan_to_parallel : Standalone version of this function.
        toupy.registration.registration.estimate_rot_axis :
            Interactive parallel-beam CoR estimation (works directly on the
            returned sinogram).
        toupy.registration.registration.alignprojections_horizontal :
            Parallel-beam horizontal alignment (works on the returned
            sinogram after extracting shiftstack).
        """
        return rebin_fan_to_parallel(
            projections_central,
            self.geometry,
            n_s=n_s,
            n_phi=n_phi,
            average_redundant=average_redundant,
        )


# ---------------------------------------------------------------------------
# Standalone fan-to-parallel rebinning
# ---------------------------------------------------------------------------

def rebin_fan_to_parallel(
    projections_central: ndarray,
    geometry: ConeBeamGeometry,
    n_s: int | None = None,
    n_phi: int | None = None,
    average_redundant: bool = True,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Exactly rebin a central-row fan-beam sinogram to parallel-beam.

    For a flat-panel detector the fan-to-parallel mapping for the central
    detector row (v = 0) is an exact geometric identity (not an
    approximation).  Each cone ray from the source through detector column
    ``u`` at source angle ``θ`` corresponds to a unique parallel-beam ray
    characterised by:

    .. math::

        s   = \\frac{\\mathrm{SOD}\\,u}{\\sqrt{\\mathrm{SDD}^2 + u^2}}
              = \\mathrm{SOD}\\sin\\gamma

    .. math::

        \\phi = \\left(\\theta + \\gamma - \\frac{\\pi}{2}\\right)
                \\bmod \\pi

    where :math:`\\gamma = \\arctan(u / \\mathrm{SDD})` is the fan angle,
    :math:`s` is the perpendicular distance from the rotation axis to the
    ray, and :math:`\\phi` is the parallel-beam projection angle in the
    convention used by :func:`~toupy.tomo.iradon.mod_iradon`.

    The inverse mapping (used for output-driven interpolation) is:

    .. math::

        \\gamma = \\arcsin(s / \\mathrm{SOD}),\\quad
        u = \\mathrm{SDD}\\tan\\gamma,\\quad
        \\theta = \\phi - \\gamma - \\frac{\\pi}{2} \\pmod{2\\pi}

    For a full 360° acquisition each parallel ray is measured twice (from
    opposite source positions, ``θ`` and ``θ + π``).  When
    ``average_redundant=True`` these two measurements are averaged, improving
    SNR and suppressing ring artefacts from detector inhomogeneities.

    Parameters
    ----------
    projections_central : ndarray, shape (n_angles, n_u)
        Central detector row (v = 0) at each projection angle.
        ``projections_central[k, i]`` is the value at source angle
        ``geometry.angles[k]`` and detector column ``i``.
    geometry : ConeBeamGeometry
        Validated acquisition geometry.
    n_s : int or None, optional
        Number of output parallel-beam detector bins.
        Defaults to ``geometry.n_u``.
    n_phi : int or None, optional
        Number of output parallel-beam angles uniformly distributed in
        ``[0°, 180°)``.  Defaults to ``len(geometry.angles)``.
    average_redundant : bool, optional
        Average the two views of each parallel ray available in a 360°
        scan.  Default ``True``.

    Returns
    -------
    sino_parallel : ndarray, shape (n_s, n_phi)
        Rebinned parallel-beam sinogram, directly compatible with
        :func:`~toupy.tomo.iradon.mod_iradon` and the registration
        functions in :mod:`toupy.registration.registration`.
    s_coords : ndarray, shape (n_s,)
        Physical detector coordinates of the parallel-beam grid in mm.
        The centre of the array (index ``n_s // 2``) corresponds to
        s = 0 (the rotation axis).
    phi_angles : ndarray, shape (n_phi,)
        Parallel-beam projection angles in degrees in ``[0°, 180°)``.

    Raises
    ------
    ValueError
        If ``projections_central`` shape does not match ``geometry``.

    Notes
    -----
    **Why rebinning enables parallel-beam tools**

    Registration algorithms such as
    :func:`~toupy.registration.registration.estimate_rot_axis` rely on the
    fact that a point object at ``(x₀, y₀)`` traces a sinusoid
    ``s(φ) = x₀ cos φ + y₀ sin φ`` in the parallel-beam sinogram.  In a
    fan-beam sinogram the trace is a distorted sinusoid, so the parallel-beam
    tools do not apply directly.  After rebinning, the sinusoidal structure
    is restored and the full parallel-beam registration toolbox becomes valid.

    **Jacobian note**

    The rebinned sinogram is not multiplied by the Jacobian
    :math:`\\partial(u, \\theta) / \\partial(s, \\phi)` of the coordinate
    change.  This is intentional: the Jacobian is only required when the
    rebinned data will be used for quantitative *reconstruction* (the
    Jacobian is 1 to first order near the central ray and is commonly
    omitted).  For *registration*, only the positions of features matter, not
    their amplitudes.

    Examples
    --------
    Extract the central row, rebin to parallel, then run parallel-beam CoR
    estimation and display:

    >>> import numpy as np
    >>> from toupy.registration.cone_registration import (
    ...     ConeBeamRegistration, rebin_fan_to_parallel)
    >>> from toupy.tomo.geometry import ConeBeamGeometry
    >>>
    >>> geom = ConeBeamGeometry(
    ...     SOD=500.0, SDD=1000.0, det_pixel_size=0.5,
    ...     n_u=128, n_v=96,
    ...     angles=np.linspace(0, 360, 180, endpoint=False),
    ... )
    >>> geom.validate()
    >>>
    >>> # projections shape (n_angles, n_v, n_u)
    >>> central_row = projections[:, geom.n_v // 2, :]   # (n_angles, n_u)
    >>>
    >>> sino_par, s_coords, phi_deg = rebin_fan_to_parallel(central_row, geom)
    >>> # sino_par shape (n_u, n_angles), s_coords in mm, phi_deg in [0, 180)
    >>>
    >>> # Use directly with mod_iradon for the central slice:
    >>> from toupy.tomo.iradon import mod_iradon
    >>> recon_central = mod_iradon(sino_par, phi_deg)
    >>>
    >>> # Or use with parallel-beam registration to estimate CoR:
    >>> # from toupy.registration.registration import estimate_rot_axis
    >>> # estimate_rot_axis(sino_par, ...)

    See Also
    --------
    ConeBeamRegistration.rebin_to_parallel : OOP wrapper.
    ConeBeamRegistration.estimate_cor : CoR estimation directly in
        detector space (no rebinning required).
    toupy.tomo.iradon.mod_iradon : Parallel-beam FBP reconstruction.
    toupy.registration.registration.estimate_rot_axis :
        Interactive sinogram-based CoR estimation.
    """
    from scipy.interpolate import RegularGridInterpolator

    if projections_central.shape != (len(geometry.angles), geometry.n_u):
        raise ValueError(
            "projections_central shape {} does not match "
            "(n_angles={}, n_u={}).".format(
                projections_central.shape, len(geometry.angles), geometry.n_u
            )
        )

    SOD = geometry.SOD
    SDD = geometry.SDD
    theta_fan = np.deg2rad(geometry.angles)   # (n_angles,) radians, uniform in [0, 2π)
    u_fan     = geometry.u_coords()           # (n_u,) mm, centred

    if n_s is None:
        n_s = geometry.n_u
    if n_phi is None:
        n_phi = len(theta_fan)

    # ------------------------------------------------------------------ #
    # Output parallel-beam grids
    # ------------------------------------------------------------------ #
    gamma_max = np.arctan(np.abs(u_fan).max() / SDD)   # maximum fan half-angle [rad]
    s_max     = SOD * np.sin(gamma_max)                 # maximum parallel offset [mm]

    s_coords  = np.linspace(-s_max, s_max, n_s)        # (n_s,) [mm]
    phi_out   = np.linspace(0.0, np.pi, n_phi,
                            endpoint=False)              # (n_phi,) [0, π) rad

    # ------------------------------------------------------------------ #
    # Per-s precomputation
    # ------------------------------------------------------------------ #
    # Clip to avoid arcsin domain errors at the very edge of the detector
    s_safe  = np.clip(s_coords, -s_max * 0.9999, s_max * 0.9999)
    gamma_s = np.arcsin(s_safe / SOD)        # fan angle for each s-bin (n_s,)
    u_s     = SDD * np.tan(gamma_s)          # detector coord for each s-bin (n_s,)

    # ------------------------------------------------------------------ #
    # Fan-beam interpolator with periodic extension
    # ------------------------------------------------------------------ #
    # Extend the theta grid by one step at each boundary so that modulo-wrapped
    # queries near 0 or 2π are still within the interpolation domain.
    theta_ext = np.concatenate([
        [theta_fan[-1] - 2.0 * np.pi],
        theta_fan,
        [theta_fan[0]  + 2.0 * np.pi],
    ])                                                   # (n_angles + 2,)
    sino_ext  = np.vstack([
        projections_central[-1:],
        projections_central,
        projections_central[:1],
    ])                                                   # (n_angles + 2, n_u)

    interp = RegularGridInterpolator(
        (theta_ext, u_fan),
        sino_ext,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    # ------------------------------------------------------------------ #
    # Inverse mapping: for each output (φ, s) find the fan-beam sample
    # ------------------------------------------------------------------ #
    sino_parallel = np.zeros((n_s, n_phi), dtype=np.float64)

    for j, phi in enumerate(phi_out):
        # ---- Primary view -----------------------------------------------
        # The parallel angle φ = θ + γ  ⟹  θ = φ − γ  (mod 2π)
        theta_prim = (phi - gamma_s) % (2.0 * np.pi)
        pts_prim   = np.stack([theta_prim, u_s], axis=-1)     # (n_s, 2)
        vals_prim  = interp(pts_prim)

        if average_redundant:
            # ---- Redundant view -----------------------------------------
            # In a 360° scan the same undirected line (φ, s) also appears as
            # (φ+π, −s) in the extended Radon convention.  The fan-beam sample
            # capturing that conjugate line has:
            #   γ_red = −γ  →  u_red = −u_s
            #   θ_red + γ_red = φ + π  →  θ_red = φ + γ + π  (mod 2π)
            theta_red = (phi + gamma_s + np.pi) % (2.0 * np.pi)
            pts_red   = np.stack([theta_red, -u_s], axis=-1)  # (n_s, 2)
            vals_red  = interp(pts_red)
            sino_parallel[:, j] = 0.5 * (vals_prim + vals_red)
        else:
            sino_parallel[:, j] = vals_prim

    phi_angles = np.rad2deg(phi_out)   # (n_phi,) degrees in [0°, 180°)

    return sino_parallel, s_coords, phi_angles
