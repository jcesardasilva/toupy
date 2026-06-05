#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cone-beam propagation phase-contrast tomography.

Convenience pipeline that couples the single- (or multi-) distance non-linear
phase retrieval of :func:`toupy.restoration.iterative_phase_retrieval` to the
FDK cone-beam reconstruction of :func:`toupy.tomo.fdk.fdk_reconstruct`.

Physics
-------
For a point source, cone-beam (divergent) propagation phase contrast maps onto
an equivalent **parallel-beam** Fresnel problem via the Fresnel scaling
theorem.  With source-to-object distance :math:`R_1=\\mathrm{SOD}` and
object-to-detector distance :math:`R_2=\\mathrm{SDD}-\\mathrm{SOD}`, the
detector intensity equals (after flat-field normalisation) the parallel-beam
intensity of the object propagated over

.. math::

    z_{\\mathrm{eff}} = \\frac{R_1 R_2}{R_1 + R_2}
                      = \\frac{R_2}{M},
    \\qquad
    \\Delta_{\\mathrm{eff}} = \\frac{\\Delta_{\\mathrm{det}}}{M},
    \\qquad
    M = \\frac{\\mathrm{SDD}}{\\mathrm{SOD}} ,

sampled with the effective (object-space) pixel size
:math:`\\Delta_{\\mathrm{eff}}` already provided by
:attr:`ConeBeamGeometry.effective_pixel_size`.  Phase retrieval is therefore
run per projection in this effective frame --- no change to the solver --- and
the retrieved projected phase (a line integral of the refractive-index
decrement) is fed directly to FDK.

The retrieval and the reconstruction are decoupled: the wave-optics
(propagation) physics lives in the per-projection step, the ray geometry lives
in FDK.  This is the cone-beam analogue of the familiar "Paganin + FDK"
pipeline, with the non-linear solver replacing the Paganin filter.

Limitations
-----------
* FDK is an approximate circular-orbit reconstruction; cone-beam artifacts grow
  with the cone (axial) angle.
* The Fresnel scaling theorem assumes a point source; a finite source size adds
  a magnification-scaled blur.
* A single propagation distance under-determines the lowest spatial frequencies
  (use TV regularisation, or pass several distances --- see below).
* The object is assumed homogeneous (single ``delta_beta``).
"""

import warnings
import numpy as np

from .fdk import fdk_reconstruct
from .geometry import ConeBeamGeometry

__all__ = ["effective_fresnel_distance", "cone_phase_retrieval_fdk"]


def effective_fresnel_distance(geometry):
    r"""
    Effective parallel-beam propagation distance for a cone-beam geometry.

    Returns :math:`z_{\mathrm{eff}} = R_1 R_2 / (R_1 + R_2)` with
    :math:`R_1=\mathrm{SOD}`, :math:`R_2=\mathrm{SDD}-\mathrm{SOD}`, i.e. the
    distance to use in a parallel-beam Fresnel propagator together with
    ``geometry.effective_pixel_size``.

    Parameters
    ----------
    geometry : ConeBeamGeometry
        Acquisition geometry (provides ``SOD`` and ``SDD``).

    Returns
    -------
    z_eff : float
        Effective propagation distance, in the same length units as ``SOD`` /
        ``SDD``.
    """
    R1 = float(geometry.SOD)
    R2 = float(geometry.SDD) - float(geometry.SOD)
    return R1 * R2 / (R1 + R2)


def cone_phase_retrieval_fdk(
    intensity,
    geometry,
    wavelength,
    delta_beta,
    n_iter=200,
    reg_smooth=0.0,
    reg_tv=2e-3,
    tv_eps=1e-3,
    pad=2,
    init=None,
    filter_type="ram-lak",
    freqcutoff=1.0,
    output_size=None,
    cuda=False,
    verbose=False,
    return_phase=False,
):
    r"""
    Cone-beam phase-contrast tomography: per-projection phase retrieval + FDK.

    Each measured (flat-field-normalised) cone-beam intensity projection is
    phase-retrieved with the exact-Fresnel non-linear solver in the *effective*
    parallel-beam frame (Fresnel scaling theorem), and the resulting stack of
    projected phases is reconstructed with FDK.

    Parameters
    ----------
    intensity : ndarray, shape ``(n_angles, n_v, n_u)``
        Measured cone-beam intensity projections, **flat-field normalised so
        the vacuum level is 1**.  The angular and detector dimensions must match
        ``geometry``.
    geometry : ConeBeamGeometry
        Circular-orbit cone-beam geometry.  ``effective_pixel_size`` and the
        effective distance (:func:`effective_fresnel_distance`) are derived
        from it.
    wavelength : float
        X-ray wavelength, in metres.
    delta_beta : float
        Ratio :math:`\delta/\beta` of the (homogeneous) object.
    n_iter, reg_smooth, reg_tv, tv_eps, pad, init : optional
        Forwarded to :func:`toupy.restoration.iterative_phase_retrieval`.
        ``reg_tv`` (total variation) is recommended for the single-distance
        case to suppress residual Fresnel ripples; ``init`` (if given) is used
        as the warm start for *every* projection.
    filter_type, freqcutoff, output_size : optional
        Forwarded to :func:`toupy.tomo.fdk.fdk_reconstruct`.
    cuda : bool, optional
        Use the GPU path for both phase retrieval and (when available) FDK.
    verbose : bool, optional
        Show a progress bar over projections (uses ``tqdm`` if installed).
    return_phase : bool, optional
        If True, also return the retrieved projected-phase stack.

    Returns
    -------
    volume : ndarray, shape ``(n_v, N, N)``
        Reconstructed volume (the refractive-index-decrement contrast).
    phase : ndarray, shape ``(n_angles, n_v, n_u)``, optional
        Retrieved projected phase, returned when ``return_phase=True``.

    Notes
    -----
    *Wave optics* (the Fresnel propagation) is handled per projection in the
    effective frame; *ray geometry* (the divergent cone) is handled by FDK.
    Phase retrieval being band-limited at a single distance, the recovered
    low-frequency / DC content is only approximate --- pass several distances
    to :func:`iterative_phase_retrieval` directly, or accept the TV-regularised
    single-distance result here.

    Examples
    --------
    >>> from toupy.tomo import ConeBeamGeometry, cone_phase_retrieval_fdk
    >>> geom = ConeBeamGeometry(SOD=0.15, SDD=0.30, det_pixel_size=2e-6,
    ...                         n_u=64, n_v=28, angles=angles)
    >>> volume = cone_phase_retrieval_fdk(I, geom, wavelength=7.29e-11,
    ...                                   delta_beta=50.0, reg_tv=2e-3)
    """
    # Lazy import keeps toupy.tomo importable without forcing restoration.
    from ..restoration.phase_retrieval import iterative_phase_retrieval

    geometry.validate()
    intensity = np.asarray(intensity, dtype=float)
    if intensity.ndim != 3 or intensity.shape[1:] != (geometry.n_v, geometry.n_u):
        raise ValueError(
            "intensity must have shape (n_angles, n_v={}, n_u={}), got {}.".format(
                geometry.n_v, geometry.n_u, intensity.shape)
        )

    z_eff = effective_fresnel_distance(geometry)
    px_eff = geometry.effective_pixel_size
    if not (0.05 <= px_eff**2 / (wavelength * z_eff) <= 5.0):
        warnings.warn(
            "Effective Fresnel number {:.3g} is outside the usual "
            "[0.05, 5] phase-contrast range; check SOD/SDD/pixel/wavelength."
            .format(px_eff**2 / (wavelength * z_eff)),
            stacklevel=2,
        )

    n_ang = intensity.shape[0]
    phase = np.empty_like(intensity)

    iterator = range(n_ang)
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="phase retrieval")
        except ImportError:
            pass

    for k in iterator:
        phase[k] = iterative_phase_retrieval(
            intensity[k], z_eff, wavelength, px_eff,
            delta_beta=delta_beta, init=init, n_iter=n_iter,
            reg_smooth=reg_smooth, reg_tv=reg_tv, tv_eps=tv_eps,
            pad=pad, cuda=cuda,
        )

    volume = fdk_reconstruct(
        phase, geometry, filter_type=filter_type, freqcutoff=freqcutoff,
        output_size=output_size, cuda=cuda,
    )
    return (volume, phase) if return_phase else volume
