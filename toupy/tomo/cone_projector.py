#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cone-beam forward projector.

Computes the cone-beam Radon transform (line integrals along divergent
rays) of a 3-D volume for a circular source orbit.

Primary use-cases:

* **Consistency checks** — compare measured projections with the
  re-projection of the FDK reconstruction to quantify residual
  reconstruction artefacts.
* **Iterative reconstruction** — the forward model is the mandatory
  companion to the FDK back-projector in a SART-style cone-beam update
  step.

The function signature mirrors :func:`~toupy.tomo.radon.projector`
for the parallel-beam case.

GPU path  : stub analogous to :func:`~toupy.tomo.radon.radon_cuda`;
            will delegate to a CuPy CUDA ray-casting kernel.
CPU path  : NumPy reference using trilinear interpolation.
"""

# standard library
import warnings
warnings.filterwarnings("ignore")

# third-party
import numpy as np
from numpy import ndarray

# local
from .geometry import ConeBeamGeometry
from .iradon import CUDA_AVAILABLE

if CUDA_AVAILABLE:
    import cupy as cp

__all__ = ["cone_project"]


def cone_project(
    volume: ndarray,
    geometry: ConeBeamGeometry,
    cuda: bool = False,
) -> ndarray:
    """
    Compute the cone-beam forward projection of a 3-D volume.

    For each projection angle θ in ``geometry.angles`` and each detector
    pixel ``(u_i, v_j)`` the function evaluates the line integral along
    the cone ray from the point source through the voxel grid to the
    detector.

    The perspective-projection mapping of a voxel at object-space
    coordinates ``(x, y, z)`` onto the detector at angle θ is:

    .. math::

        U(\\theta) = \\mathrm{SOD} + x\\sin\\theta - y\\cos\\theta

    .. math::

        u_d = \\frac{\\mathrm{SDD}}{U}\\,(x\\cos\\theta + y\\sin\\theta),
        \\qquad
        v_d = \\frac{\\mathrm{SDD}}{U}\\, z

    The voxel grid is assumed isotropic with pitch
    ``geometry.effective_pixel_size``, centred at the origin (the
    rotation axis).

    Parameters
    ----------
    volume : ndarray, shape (n_v, N, N)
        The 3-D volume to project.  Axis order: ``(z, y, x)`` where z is
        the rotation (axial) direction and (y, x) is the transaxial plane.
        ``N`` is the transaxial voxel count and may differ from
        ``geometry.n_u``; the coordinate mapping is scaled accordingly.
    geometry : ConeBeamGeometry
        Validated acquisition geometry.  Provides SOD, SDD,
        ``det_pixel_size``, and the set of projection angles.
    cuda : bool, optional
        If ``True``, attempt to use the CuPy GPU path.  Falls back to
        CPU with a :mod:`warnings` warning when CuPy is unavailable.
        Default ``False``.

    Returns
    -------
    projections : ndarray, shape (n_angles, n_v, n_u)
        Simulated cone-beam projections.  Axis order matches the input
        convention expected by :func:`~toupy.tomo.fdk.fdk_weight`.

    Raises
    ------
    ValueError
        If ``volume.ndim != 3``.
    ValueError
        If ``volume.shape[0] != geometry.n_v`` (axial size mismatch between
        the volume and the detector).

    See Also
    --------
    toupy.tomo.radon.projector : Parallel-beam forward projector (2-D).
    toupy.tomo.fdk.fdk_reconstruct : Paired cone-beam reconstruction.

    Notes
    -----
    This forward projector is the exact adjoint of :func:`fdk_backproject`
    (without the ``(SOD/U)²`` weighting applied in the back-projector).
    For a consistent SART-style iterative solver, the weighting must be
    applied symmetrically in both the forward and back steps, or the
    normal equations must be formed explicitly.
    """
    from scipy.interpolate import RegularGridInterpolator

    if volume.ndim != 3:
        raise ValueError(
            "volume must be 3-D (n_v, N, N), got ndim={}.".format(volume.ndim)
        )
    if volume.shape[0] != geometry.n_v:
        raise ValueError(
            "volume.shape[0]={} does not match geometry.n_v={}.".format(
                volume.shape[0], geometry.n_v
            )
        )

    # GPU path stub
    if cuda:
        if not CUDA_AVAILABLE:
            warnings.warn(
                "CuPy unavailable — falling back to CPU cone_project.",
                UserWarning, stacklevel=2,
            )
        # else: CUDA not yet implemented, fall through to CPU

    n_v_vol, N, _ = volume.shape
    SOD = geometry.SOD
    SDD = geometry.SDD
    p = geometry.effective_pixel_size

    # Centred object-space coordinate grids (N x N)
    coords = (np.arange(N) - (N - 1) / 2.0) * p
    x, y = np.meshgrid(coords, coords, indexing='xy')  # x varies along cols, y along rows

    # Object-space z-coordinates (axial)
    z_coords = (np.arange(n_v_vol) - (n_v_vol - 1) / 2.0) * p  # shape (n_v_vol,)

    # Detector coordinate arrays
    u_det = geometry.u_coords()   # shape (n_u,)
    v_det = geometry.v_coords()   # shape (n_v,)

    n_angles = len(geometry.angles)
    angles_rad = np.deg2rad(geometry.angles)

    # Build interpolator for the volume: points = (z_coords, y_coords, x_coords)
    # volume shape is (n_v_vol, N, N) with axes (z, y, x)
    vol_interp = RegularGridInterpolator(
        (z_coords, coords, coords),
        volume,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    # -----------------------------------------------------------------------
    # Restrict ray sampling to the volume's bounding box
    # -----------------------------------------------------------------------
    # The full source-to-detector path is ~SDD long but the object occupies only
    # a small central fraction.  Sampling the full ray with a fixed n_steps gives
    # only a handful of samples inside the object, causing severe aliasing.
    #
    # Strategy: parameterise t in [0, 1] (source→detector), but restrict
    # t_vals to [t_lo, t_hi] centred at the isocenter crossing (t ≈ SOD/SDD).
    # The half-span covers the bounding-box diagonal of the volume, with a
    # 20 % margin.  n_steps is chosen so that the physical step inside the
    # object is ≈ effective_pixel_size (one sample per voxel).
    # -----------------------------------------------------------------------
    half_xy   = (N      - 1) / 2.0 * p            # object half-extent in x, y  [mm]
    half_z    = (n_v_vol - 1) / 2.0 * p           # object half-extent in z      [mm]
    half_diag = np.sqrt(2.0 * half_xy**2 + half_z**2)  # half of 3-D diagonal [mm]

    t_center = SOD / SDD                           # isocenter crossing in t
    t_span   = 1.2 * half_diag / SDD              # t-half-span with 20 % margin
    t_lo     = max(0.0, t_center - t_span)
    t_hi     = min(1.0, t_center + t_span)

    # Step size ≈ p along the central ray (length ≈ SDD)
    n_steps = max(int(np.ceil(2.0 * t_span * SDD / p)) + 1,
                  max(N, n_v_vol) * 2)

    projections = np.zeros((n_angles, geometry.n_v, geometry.n_u), dtype=np.float64)

    for i_angle, theta in enumerate(angles_rad):
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        # Source position: S = (-SOD*sin_th, SOD*cos_th, 0)
        # Verify: U(S) = SOD + (-SOD*sin_th)*sin_th - (SOD*cos_th)*cos_th = 0.  ✓
        x_src = -SOD * sin_th
        y_src =  SOD * cos_th

        # Build detector pixel grid in 3-D lab frame.
        # Beam axis direction (source → isocenter): d = (sin_th, -cos_th, 0)
        # Detector centre: isocenter + (SDD-SOD)*d
        # Detector u-axis: e_u = (cos_th, sin_th, 0)   [perpendicular to d in xy]
        # Detector v-axis: e_v = (0, 0, 1)
        uu, vv = np.meshgrid(u_det, v_det, indexing='xy')  # (n_v_det, n_u)
        x_det = (SDD - SOD) * sin_th + uu * cos_th    # (n_v_det, n_u)
        y_det = -(SDD - SOD) * cos_th + uu * sin_th   # (n_v_det, n_u)
        z_det = vv                                      # (n_v_det, n_u)

        # Ray direction vector (source → detector pixel)
        dx = x_det - x_src   # (n_v_det, n_u)
        dy = y_det - y_src   # (n_v_det, n_u)
        dz = z_det            # (n_v_det, n_u),  z_src = 0

        # Sample only within the volume's bounding-box region [t_lo, t_hi]
        t_vals = np.linspace(t_lo, t_hi, n_steps)  # (n_steps,)

        x_ray = x_src + t_vals[:, np.newaxis, np.newaxis] * dx[np.newaxis, :, :]
        y_ray = y_src + t_vals[:, np.newaxis, np.newaxis] * dy[np.newaxis, :, :]
        z_ray =         t_vals[:, np.newaxis, np.newaxis] * dz[np.newaxis, :, :]

        # Query volume interpolator: points = (z, y, x)
        pts  = np.stack([z_ray.ravel(), y_ray.ravel(), x_ray.ravel()], axis=-1)
        vals = vol_interp(pts).reshape(n_steps, geometry.n_v, geometry.n_u)

        # Trapezoidal integration: physical step = |ray_dir| * Δt
        ray_length = np.sqrt(dx**2 + dy**2 + dz**2)       # (n_v_det, n_u) [mm]
        dt         = (t_hi - t_lo) / (n_steps - 1)
        step_size  = ray_length * dt                        # (n_v_det, n_u) [mm]
        projections[i_angle] = np.trapezoid(vals, axis=0) * step_size

    return projections
