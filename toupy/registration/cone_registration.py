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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
