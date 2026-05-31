#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feldkamp–Davis–Kress (FDK) cone-beam reconstruction.

Implements the three-step FDK pipeline from:

    Feldkamp, L. A., Davis, L. C., & Kress, J. W. (1984).
    Practical cone-beam algorithm.
    *Journal of the Optical Society of America A*, 1(6), 612–619.
    https://doi.org/10.1364/JOSAA.1.000612

Step 1 — :func:`fdk_weight`:
    Multiply each projection pixel (u, v) by the cosine factor
    ``SOD / sqrt(SOD² + u² + v²)``, where u and v are centred physical
    detector coordinates.

Step 2 — :func:`fdk_filter`:
    Apply a 1-D ramp (or windowed-ramp) filter row-by-row along the
    horizontal detector direction u.  Reuses
    :func:`~toupy.tomo.iradon.compute_filter`.

Step 3 — :func:`fdk_backproject`:
    3-D divergent-ray back-projection.  For each output voxel (x, y, z)
    and each projection angle θ, compute the detector coordinates at which
    the cone ray through the voxel intersects the detector, interpolate the
    filtered projection, and accumulate weighted by ``(SOD / U)²`` where U
    is the perspective-projection denominator (see function docstring).

GPU path  : stub for a CuPy/CUDA implementation, falls back to CPU with a
            warning (same pattern as :func:`~toupy.tomo.iradon.mod_iradon_cuda`).
CPU path  : NumPy reference implementation (loops over angles).

Public API
----------
fdk_weight, fdk_filter, fdk_backproject, fdk_reconstruct
"""

# standard library
import warnings
warnings.filterwarnings("ignore")

# third-party
import numpy as np
from numpy import ndarray

# local
from .geometry import ConeBeamGeometry
from .iradon import compute_filter, CUDA_AVAILABLE

# Optional CuPy — same graceful-degradation pattern as iradon.py
if CUDA_AVAILABLE:
    import cupy as cp

__all__ = [
    "fdk_weight",
    "fdk_filter",
    "fdk_backproject",
    "fdk_reconstruct",
]


# ---------------------------------------------------------------------------
# Step 1 — Cosine weighting
# ---------------------------------------------------------------------------

def fdk_weight(projections: ndarray, geometry: ConeBeamGeometry) -> ndarray:
    """
    Apply FDK cosine weights to a stack of cone-beam projections.

    This is Step 1 of the FDK algorithm.  Each pixel at detector position
    (u, v) is multiplied by the angle-independent cosine factor:

    .. math::

        w(u, v) = \\frac{\\mathrm{SOD}}{\\sqrt{\\mathrm{SOD}^2 + u^2 + v^2}}

    The weight map is computed once and broadcast over the angle axis
    because the geometry is fixed across all views.

    Parameters
    ----------
    projections : ndarray, shape (n_angles, n_v, n_u)
        Raw cone-beam projections.  Axis order: angle first, detector row
        (v) second, detector column (u) last.  Values are typically
        Beer-Lambert log-normalised line integrals.
    geometry : ConeBeamGeometry
        Fully validated acquisition geometry.  ``geometry.u_coords()`` and
        ``geometry.v_coords()`` supply the centred physical detector
        coordinates used to evaluate ``w(u, v)``.

    Returns
    -------
    projections_weighted : ndarray, shape (n_angles, n_v, n_u)
        Cosine-weighted projections, same shape and dtype as *projections*.

    Raises
    ------
    ValueError
        If ``projections.shape[1:]`` does not match
        ``(geometry.n_v, geometry.n_u)``.

    Notes
    -----
    The weight approaches 1 on the central beam axis (u = v = 0) and
    decreases toward the detector periphery, compensating for the longer
    effective path length of oblique cone rays.  This is the exact
    extension of the ``cos(γ)`` factor in the fan-beam Parker weighting
    to the 2-D detector case.
    """
    if projections.shape[1:] != (geometry.n_v, geometry.n_u):
        raise ValueError(
            "projections.shape[1:] = {} does not match (n_v={}, n_u={}).".format(
                projections.shape[1:], geometry.n_v, geometry.n_u
            )
        )
    u = geometry.u_coords()  # shape (n_u,)
    v = geometry.v_coords()  # shape (n_v,)
    SOD = geometry.SOD
    # weight map shape (n_v, n_u)
    uu, vv = np.meshgrid(u, v, indexing='xy')
    w = SOD / np.sqrt(SOD ** 2 + uu ** 2 + vv ** 2)
    return projections * w[np.newaxis, :, :]


# ---------------------------------------------------------------------------
# Step 2 — Ramp filtering
# ---------------------------------------------------------------------------

def fdk_filter(
    projections_weighted: ndarray,
    filter_type: str = "ram-lak",
    freqcutoff: float = 1.0,
) -> ndarray:
    """
    Apply a 1-D ramp filter row-by-row to cosine-weighted projections.

    This is Step 2 of the FDK algorithm.  Each detector row (fixed v,
    varying u) is independently filtered along the u direction using an
    FFT-based convolution with the chosen ramp kernel.  The filter is
    computed by :func:`~toupy.tomo.iradon.compute_filter`, so all filter
    types supported by the parallel-beam FBP are available here.

    Parameters
    ----------
    projections_weighted : ndarray, shape (n_angles, n_v, n_u)
        Output of :func:`fdk_weight`.
    filter_type : str, optional
        Frequency-domain filter name.  One of ``'ram-lak'`` (default),
        ``'shepp-logan'``, ``'cosine'``, ``'hamming'``, ``'hann'``, or
        ``None`` (no filtering, useful for debugging).
    freqcutoff : float, optional
        Normalised frequency cutoff in ``(0, 1]``.  Default ``1.0``
        (no cutoff beyond Nyquist).

    Returns
    -------
    projections_filtered : ndarray, shape (n_angles, n_v, n_u)
        Ramp-filtered projections, same shape and dtype as input.

    Notes
    -----
    The filter is applied in the Fourier domain row-by-row:

    .. math::

        p_f(u) = \\mathcal{F}^{-1}\\!\\left[
                    \\mathcal{F}[p_w](\\xi)\\,
                    H(\\xi)
                 \\right]

    where :math:`H(\\xi)` is the chosen window function.  This step is
    mathematically identical to the FBP filtering step; only the
    geometry of the subsequent back-projection differs.
    """
    from scipy.fft import fft, ifft
    n_angles, n_v, n_u = projections_weighted.shape
    # compute_filter returns shape (projection_size_padded, 1)
    fourier_filter = compute_filter(n_u, filter_type=filter_type,
                                   derivatives=False, freqcutoff=freqcutoff)
    pad_size = fourier_filter.shape[0]  # next power of 2 >= 2*n_u

    # Reshape all rows into a 2D array: (n_angles*n_v, n_u)
    rows = projections_weighted.reshape(-1, n_u)

    # Zero-pad each row to pad_size
    pad_width = pad_size - n_u
    rows_padded = np.pad(rows, ((0, 0), (0, pad_width)), mode='constant')

    # Apply filter in frequency domain
    rows_fft = fft(rows_padded, axis=1)                    # (n_angles*n_v, pad_size)
    rows_fft *= fourier_filter.ravel()[np.newaxis, :]      # broadcast (1, pad_size)
    rows_filtered = np.real(ifft(rows_fft, axis=1))        # (n_angles*n_v, pad_size)

    # Trim back to n_u and reshape
    result = rows_filtered[:, :n_u].reshape(n_angles, n_v, n_u)
    return result


# ---------------------------------------------------------------------------
# Step 3 — 3-D divergent-ray back-projection
# ---------------------------------------------------------------------------

def fdk_backproject(
    projections_filtered: ndarray,
    geometry: ConeBeamGeometry,
    output_size: int | None = None,
) -> ndarray:
    """
    3-D cone-beam back-projection (Step 3 of FDK).

    For each output voxel at object-space coordinates (x, y, z) and for
    each projection angle θ, the perspective-projection denominator is:

    .. math::

        U(\\theta) = \\mathrm{SOD} + x\\sin\\theta - y\\cos\\theta

    The detector coordinates of the cone ray through the voxel are:

    .. math::

        u_d = \\frac{\\mathrm{SDD}}{U}\\,
              (x\\cos\\theta + y\\sin\\theta)

    .. math::

        v_d = \\frac{\\mathrm{SDD}}{U}\\, z

    The filtered projection value at ``(u_d, v_d)`` is obtained by
    bilinear interpolation and accumulated into the voxel, weighted by
    the solid-angle correction factor ``(SOD / U)²``.  The final volume
    is scaled by ``π / n_angles``:

    .. math::

        f(x, y, z) = \\frac{\\pi}{N_\\theta}
                     \\sum_{k=1}^{N_\\theta}
                     \\left(\\frac{\\mathrm{SOD}}{U_k}\\right)^2
                     p_f^{\\theta_k}(u_d, v_d)

    Parameters
    ----------
    projections_filtered : ndarray, shape (n_angles, n_v, n_u)
        Output of :func:`fdk_filter`.
    geometry : ConeBeamGeometry
        Acquisition geometry.  ``geometry.angles`` supplies θ values in
        degrees; ``geometry.u_coords()`` and ``geometry.v_coords()``
        supply the centred detector coordinates for interpolation.
    output_size : int, optional
        Side length N (in voxels) of the square transaxial output grid.
        Defaults to ``geometry.n_u`` (one detector column per voxel,
        at ``effective_pixel_size`` sampling).

    Returns
    -------
    volume : ndarray, shape (n_v, N, N)
        Reconstructed 3-D volume.  Axis order: ``(z, y, x)`` where z is
        the rotation axis.  The rotation axis voxel is at index
        ``(n_v//2, N//2, N//2)``.

    Notes
    -----
    The ``(SOD/U)²`` weighting is the cone-beam analogue of the
    ``cos²(γ)`` factor in 2-D fan-beam backprojection.  It corrects for
    the solid-angle element of each voxel as seen from the source.

    The CPU reference uses bilinear interpolation via
    :class:`scipy.interpolate.RegularGridInterpolator`.  A future CUDA
    kernel (stub in :func:`fdk_reconstruct`) will use texture-memory
    interpolation for speed.
    """
    from scipy.interpolate import RegularGridInterpolator

    n_angles, n_v, n_u = projections_filtered.shape
    if output_size is None:
        output_size = geometry.n_u
    N = output_size

    SOD = geometry.SOD
    SDD = geometry.SDD
    p = geometry.effective_pixel_size

    # Centred object-space coordinate grids (N x N transaxial, n_v axial)
    coords = (np.arange(N) - (N - 1) / 2.0) * p
    x, y = np.meshgrid(coords, coords, indexing='xy')  # x varies along cols, y along rows

    # Object-space axial (z) coordinates — pitch is effective_pixel_size, not det_pixel_size
    z_coords = (np.arange(n_v) - (n_v - 1) / 2.0) * p  # shape (n_v,)

    # Detector coordinate arrays (used only as the interpolation grid)
    u_det = geometry.u_coords()   # shape (n_u,)
    v_det = geometry.v_coords()   # shape (n_v,)

    angles_rad = np.deg2rad(geometry.angles)

    volume = np.zeros((n_v, N, N), dtype=np.float64)

    for i_angle, theta in enumerate(angles_rad):
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        # Perspective-projection denominator, shape (N, N)
        U = SOD + x * sin_th - y * cos_th

        # Transaxial detector coordinate of each voxel (N, N)
        u_d_2d = (SDD / U) * (x * cos_th + y * sin_th)

        # Axial detector coordinate for each z-layer:
        #   v_d = (SDD / U) * z_object   [FDK formula]
        # z_coords uses effective_pixel_size (object space), NOT det_pixel_size.
        # Using v_det here would inflate v_d by the magnification factor.
        inv_U = SDD / U  # (N, N)
        v_d = z_coords[:, np.newaxis, np.newaxis] * inv_U[np.newaxis, :, :]  # (n_v, N, N)
        u_d = u_d_2d[np.newaxis, :, :] * np.ones((n_v, 1, 1))                # (n_v, N, N)

        # Build interpolator for this projection
        # RegularGridInterpolator expects points as (rows, cols) = (v_det, u_det)
        interp = RegularGridInterpolator(
            (v_det, u_det),
            projections_filtered[i_angle],   # shape (n_v, n_u)
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )

        # Query points shape (n_v*N*N, 2) — (v_query, u_query)
        pts = np.stack([v_d.ravel(), u_d.ravel()], axis=-1)
        vals = interp(pts).reshape(n_v, N, N)

        # Accumulate with (SOD/U)^2 weight
        weight = (SOD / U) ** 2  # (N, N)
        volume += weight[np.newaxis, :, :] * vals

    volume *= np.pi / n_angles
    return volume


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def fdk_reconstruct(
    projections: ndarray,
    geometry: ConeBeamGeometry,
    filter_type: str = "ram-lak",
    freqcutoff: float = 1.0,
    output_size: int | None = None,
    cuda: bool = False,
) -> ndarray:
    """
    Full FDK cone-beam reconstruction pipeline.

    Chains :func:`fdk_weight` → :func:`fdk_filter` → :func:`fdk_backproject`
    in sequence, mirroring the interface of
    :func:`~toupy.tomo.iradon.mod_iradon` for the parallel-beam case.

    An optional GPU path (``cuda=True``) is stubbed out; when CuPy is
    unavailable it falls back to the CPU path with a :mod:`warnings`
    message, exactly as :func:`~toupy.tomo.iradon.mod_iradon_cuda` does.

    Parameters
    ----------
    projections : ndarray, shape (n_angles, n_v, n_u)
        Raw cone-beam projections (Beer-Lambert log-normalised line
        integrals).
    geometry : ConeBeamGeometry
        Acquisition geometry.  :meth:`~ConeBeamGeometry.validate` is
        called internally; no need to call it again before passing in.
    filter_type : str, optional
        Ramp-filter window forwarded to :func:`fdk_filter`.
        Default ``'ram-lak'``.
    freqcutoff : float, optional
        Normalised frequency cutoff forwarded to :func:`fdk_filter`.
        Default ``1.0``.
    output_size : int, optional
        Transaxial reconstruction grid side length N, forwarded to
        :func:`fdk_backproject`.  Defaults to ``geometry.n_u``.
    cuda : bool, optional
        If ``True``, attempt to use the CuPy GPU back-projector.  Falls
        back to CPU with a :mod:`warnings` warning when CuPy is
        unavailable.  Default ``False``.

    Returns
    -------
    volume : ndarray, shape (n_v, N, N)
        Reconstructed 3-D volume.

    Raises
    ------
    ValueError
        If ``projections.ndim != 3`` or its trailing dimensions are
        inconsistent with ``geometry.n_v`` and ``geometry.n_u``.

    See Also
    --------
    toupy.tomo.iradon.mod_iradon : Parallel-beam FBP (2-D, single slice).
    toupy.tomo.iradon.mod_iradon_cuda : GPU parallel-beam FBP.
    toupy.tomo.tomorecons.fdk_tomo_recons : High-level wrapper with
        interactive display and optional saving.

    Notes
    -----
    The ``π / N_θ`` scale factor is absorbed into :func:`fdk_backproject`.
    It differs from the ``π / (2 N_θ)`` factor in parallel-beam FBP
    because FDK sums over a full 360° orbit while the parallel-beam
    formula assumes 180°.
    """
    # Validate geometry
    geometry.validate()

    if projections.ndim != 3:
        raise ValueError(
            "projections must be 3-D (n_angles, n_v, n_u), got ndim={}.".format(
                projections.ndim
            )
        )
    if projections.shape[1:] != (geometry.n_v, geometry.n_u):
        raise ValueError(
            "projections.shape[1:] = {} does not match geometry (n_v={}, n_u={}).".format(
                projections.shape[1:], geometry.n_v, geometry.n_u
            )
        )

    # GPU path stub — mirrors mod_iradon_cuda fallback pattern
    if cuda:
        if not CUDA_AVAILABLE:
            warnings.warn(
                "CuPy unavailable — falling back to CPU FDK.",
                UserWarning, stacklevel=2,
            )
        # else: CUDA FDK not yet implemented, fall through to CPU

    # CPU path
    projections_weighted = fdk_weight(projections, geometry)
    projections_filtered = fdk_filter(projections_weighted, filter_type, freqcutoff)
    volume = fdk_backproject(projections_filtered, geometry, output_size)
    return volume
