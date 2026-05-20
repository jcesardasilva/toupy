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
    # GPU path stub
    # if cuda:
    #     if not CUDA_AVAILABLE:
    #         warnings.warn(
    #             "CuPy unavailable — falling back to CPU cone_project.",
    #             UserWarning, stacklevel=2,
    #         )
    #     else:
    #         raise NotImplementedError("CUDA cone_project not yet implemented.")

    # CPU reference path
    # 1. Build object-space coordinate grids (x, y, z) centred on origin
    # 2. For each angle θ in geometry.angles:
    #    a. Compute U = SOD + x*sinθ - y*cosθ  (broadcast over voxel grid)
    #    b. Compute u_d = SDD/U * (x*cosθ + y*sinθ),  v_d = SDD/U * z
    #    c. Interpolate volume at (u_d, v_d) → accumulate into projection
    # 3. Return projections stack

    raise NotImplementedError
