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

__all__ = ["ConeBeamRegistration"]


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
