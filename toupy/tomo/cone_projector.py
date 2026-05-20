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

    projections = np.zeros((n_angles, geometry.n_v, geometry.n_u), dtype=np.float64)

    for i_angle, theta in enumerate(angles_rad):
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        # For each detector pixel (j, i), find the object-space coordinates
        # of the ray intersection at each z-layer.
        # The perspective mapping is: u_d = SDD/U * (x*cos + y*sin),  v_d = SDD/U * z
        # Inverse: for detector pixel (u_det[i], v_det[j]) at z = z_coords[k]:
        #   U = SOD / (1 - (u_det[i]*sin_th - v_rot)/SDD)  -- not needed for forward
        # Instead, for forward projection we sample the volume along each ray.
        # For each detector pixel (u_det[i], v_det[j]), parameterise the ray and integrate.

        # Build detector grid
        uu, vv = np.meshgrid(u_det, v_det, indexing='xy')  # (n_v_det, n_u)

        # Number of integration steps along the ray
        n_steps = max(N, n_v_vol) * 2

        # Source position in object space (rotated frame)
        # Source is at: x_s = -SOD*sin_th,  y_s = SOD*cos_th,  z_s = 0
        x_src = -SOD * sin_th
        y_src =  SOD * cos_th

        # For each detector pixel (uu, vv), the ray goes from source to detector.
        # Detector pixel in 3D: (x_d, y_d, z_d) in lab frame:
        #   x_d = SDD*sin_th + uu*cos_th  (approx flat-panel at SDD along beam axis)
        # Actually for a flat detector perpendicular to the beam at SDD from source:
        # The detector normal is along (sin_th, -cos_th, 0) (source to isocenter dir).
        # Detector pixel at (u, v) in detector frame maps to 3D lab frame:
        #   x_d = (SDD - SOD)*sin_th + uu*cos_th  ... but it's simpler to use:
        # The source is at (-SOD*sin_th, SOD*cos_th, 0).
        # The detector centre is at ((SDD-SOD)*sin_th, -(SDD-SOD)*cos_th, 0).
        # Actually: source is at distance SOD from origin, detector at SDD from source.
        # Source position: (-SOD*sin_th, SOD*cos_th, 0)
        # Detector centre: (SDD-SOD)*sin_th * (-1,...) — let's be precise:
        # The beam axis direction from source toward detector: (sin_th, -cos_th, 0)
        # Detector centre = source + SDD * beam_dir
        #                 = (-SOD*sin_th + SDD*sin_th, SOD*cos_th - SDD*cos_th, 0)
        #                 = ((SDD-SOD)*sin_th, (SOD-SDD)*cos_th, 0)
        # Detector u-axis: (cos_th, sin_th, 0), v-axis: (0, 0, 1)
        # Detector pixel 3D position:
        x_det_3d = (SDD - SOD) * sin_th + uu * cos_th   # (n_v_det, n_u)
        y_det_3d = (SOD - SDD) * cos_th + uu * sin_th   # (n_v_det, n_u)  # wait, let me recalc
        # beam direction: from source (-SOD*sin, SOD*cos, 0) toward object origin.
        # Actually canonical: source is at (0, -SOD, 0) in the rotating frame, but
        # let's use the formulation consistent with fdk_backproject:
        # U = SOD + x*sin_th - y*cos_th (where x,y are object coords)
        # This means source projects at: source pos = (-SOD*sin_th, SOD*cos_th, 0)
        # For a ray through detector point (u_d, v_d):
        #   Direction from source: d = (u_d*cos_th - SOD*sin_th - (-SOD*sin_th), ...) -- need full 3D

        # Simpler: parameterise ray by t in [0, 1] from source to detector edge.
        # Ray: P(t) = source + t*(detector_point - source)
        # At t=SOD/SDD the ray passes through the isocenter plane (U=SOD).

        # Source: (x_s, y_s, z_s) = (-SOD*sin_th, SOD*cos_th, 0)
        # Detector pixel 3D: from the perspective formula:
        #   u_d = SDD/U * (x*cos_th + y*sin_th)  => x*cos_th + y*sin_th = u_d * U/SDD
        #   v_d = SDD/U * z                       => z = v_d * U/SDD
        # But we want the 3D position of the detector pixel, not the object point.
        # The detector is at perpendicular distance SDD from source along beam axis.
        # detector pixel at (u, v):
        #   lab_x = -SOD*sin_th + (SDD)*sin_th + u*cos_th ... no.
        # Correct formulation:
        # The source is at S = (-SOD*sin_th, SOD*cos_th, 0).
        # The beam axis direction (source to det centre): d_beam = (sin_th, -cos_th, 0)
        # Detector centre: D_c = S + SDD * d_beam = (-SOD*sin_th + SDD*sin_th, SOD*cos_th - SDD*cos_th, 0)
        # Detector u-axis: e_u = (cos_th, sin_th, 0)
        # Detector v-axis: e_v = (0, 0, 1)
        # Detector pixel 3D: P_det = D_c + u*e_u + v*e_v
        #   x_det = (SDD-SOD)*sin_th + u*cos_th
        #   y_det = (SOD-SDD)*cos_th + u*sin_th  — note: beam goes in -y dir when th=0
        # Wait, when th=0: beam axis = (0,-1,0)? No: sin(0)=0, cos(0)=1 => d_beam=(0,-1,0).
        # Source at (0, SOD, 0), detector centre at (0, SOD-SDD, 0). That looks wrong.
        # Let me reconsider. In the FDK convention used in fdk_backproject:
        # Source is at distance SOD from origin along the y-axis (when theta=0).
        # The source moves in a circle. At angle theta:
        # Looking at U = SOD + x*sin_th - y*cos_th:
        # When x=0, y=0: U = SOD. So origin is at distance SOD from source along beam.
        # The source position: source is "behind" by SOD along the beam axis.
        # The beam direction from source to isocenter: at theta=0, beam goes in +y direction.
        # Source pos at theta: S = (-SOD*sin_th, -SOD*cos_th... let's verify:
        # At theta=0: U = SOD - y*1 = SOD at y=0. If source is at y=-SOD, beam travels +y.
        # => source at (0, -SOD, 0) when theta=0.
        # At general theta: S = (SOD*sin_th, -SOD*cos_th, 0).
        # Beam direction: toward origin = (-sin_th, cos_th, 0).
        # Hmm, let's re-derive from U = SOD + x*sin - y*cos:
        # U is the distance from source to voxel along the beam axis.
        # If source is at S, then for a voxel at (x,y,z):
        # U = SOD + <(x,y) - S_xy, beam_dir> where beam_dir points from source to detector.
        # Actually U = SOD + x*sin_th - y*cos_th suggests:
        # S_xy = (0,0), beam_dir = (sin_th, -cos_th, 0)? No.
        # The perspective formula u_d = SDD/U * (x*cos_th + y*sin_th) maps (x,y) to u_d.
        # When x = r*cos_th, y = r*sin_th (a point at radius r in direction theta from origin):
        # u_d = SDD/SOD * r. That makes sense for a point on the source-to-detector axis.
        # The source is at angle theta+180 from origin, at distance SOD.
        # Source position: S = (-SOD*sin_th + 0, SOD*cos_th + 0, 0)?
        # Let's check: U for source = SOD + (-SOD*sin_th)*sin_th - (SOD*cos_th)*cos_th
        #            = SOD - SOD*sin²_th - SOD*cos²_th = SOD - SOD = 0.
        # So source is at x_s = -SOD*sin_th, y_s = SOD*cos_th (confirmed).

        # Detector normal direction (from source toward detector):
        # At theta=0: source at (0, SOD). Detector centre at (0, SOD - SDD)? No...
        # Beam goes from source at (-SOD*sin, SOD*cos) toward the isocenter (0,0).
        # Beyond isocenter the detector is at distance (SDD-SOD) from isocenter along beam.
        # Beam direction (from source): (SOD*sin_th, -SOD*cos_th)/SOD = (sin_th, -cos_th).
        # Detector centre: isocenter + (SDD-SOD)*(sin_th, -cos_th, 0)
        #               = ((SDD-SOD)*sin_th, -(SDD-SOD)*cos_th, 0).
        # Detector u-axis: perpendicular to beam in xy-plane = (cos_th, sin_th, 0).
        # Detector v-axis: (0, 0, 1).
        # Detector pixel: D(u,v) = D_c + u*e_u + v*e_v
        x_det = (SDD - SOD) * sin_th + uu * cos_th    # (n_v_det, n_u)
        y_det = -(SDD - SOD) * cos_th + uu * sin_th   # (n_v_det, n_u)
        z_det = vv                                      # (n_v_det, n_u)

        # Ray direction: from source to detector pixel
        dx = x_det - x_src   # (n_v_det, n_u)
        dy = y_det - y_src   # (n_v_det, n_u)
        dz = z_det - 0.0     # (n_v_det, n_u), z_src=0

        # Sample along ray from t=0 (source) to t=1 (detector)
        t_vals = np.linspace(0.0, 1.0, n_steps)  # (n_steps,)

        # Sample points shape (n_steps, n_v_det, n_u, 3)
        x_ray = x_src + t_vals[:, np.newaxis, np.newaxis] * dx[np.newaxis, :, :]
        y_ray = y_src + t_vals[:, np.newaxis, np.newaxis] * dy[np.newaxis, :, :]
        z_ray = 0.0  + t_vals[:, np.newaxis, np.newaxis] * dz[np.newaxis, :, :]

        # Query volume interpolator: points = (z, y, x)
        pts = np.stack([z_ray.ravel(), y_ray.ravel(), x_ray.ravel()], axis=-1)
        vals = vol_interp(pts).reshape(n_steps, geometry.n_v, geometry.n_u)

        # Integrate via trapezoidal rule: step size = ray length / (n_steps - 1)
        ray_length = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # (n_v_det, n_u)
        step_size = ray_length / (n_steps - 1)
        # np.trapz with default dx=1 integrates in index units;
        # multiply by the physical step size to get a physical line integral.
        projections[i_angle] = np.trapz(vals, axis=0) * step_size

    return projections
