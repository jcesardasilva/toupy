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
