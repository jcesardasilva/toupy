#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Magnified inline-holography pipeline (ESRF ID16A style) for multi-distance
CTF phase retrieval.

In a focused-beam (cone-beam) inline-holography experiment the sample is placed
at several defocus positions after the focal point.  Each position has a
different geometric magnification, so the recorded holograms must be
**flat-field corrected**, **rescaled to a common (effective) pixel size**, and
**sub-pixel aligned** before the multi-distance contrast-transfer-function
(CTF) inversion of :func:`toupy.restoration.ctf_retrieve` can be applied.

Geometry (Fresnel scaling theorem)
----------------------------------
For a point source at focus, defocus position *i* with focus-to-sample distance
:math:`z_{s,i}` and a fixed focus-to-detector distance :math:`z_D`:

.. math::

   M_i = \\frac{z_D}{z_{s,i}},\\qquad
   z_{\\mathrm{eff},i} = \\frac{z_{s,i}\\,(z_D - z_{s,i})}{z_D},\\qquad
   \\Delta_i = \\frac{\\Delta_{\\mathrm{det}}}{M_i}.

The position **closest to the focus** has the largest magnification and the
smallest effective pixel :math:`\\Delta_i` --- i.e. the highest resolution ---
so it is used as the common reconstruction grid; the others are interpolated
onto it.

Pipeline
--------
``flat-field``  →  ``rescale to common pixel``  →  ``sub-pixel align``  →
``multi-distance CTF`` (:func:`ctf_retrieve` with the effective distances and
the common effective pixel).
"""

import warnings
from collections import namedtuple

import numpy as np
from scipy.ndimage import affine_transform as _affine
from scipy.ndimage import fourier_shift as _fourier_shift
from scipy.ndimage import gaussian_filter as _gaussian_filter

from .phase_retrieval import ctf_retrieve

try:
    from skimage.registration import phase_cross_correlation as _pcc
    _HAVE_SKIMAGE = True
except ImportError:                                    # pragma: no cover
    _HAVE_SKIMAGE = False

__all__ = [
    "flat_field_correct",
    "holo_geometry",
    "rescale_to_common_pixel",
    "align_holograms",
    "holo_ctf_reconstruct",
]

HoloGeometry = namedtuple(
    "HoloGeometry",
    ["magnification", "effective_distance", "effective_pixel_size"],
)


# ---------------------------------------------------------------------------
# 1. Flat-field correction
# ---------------------------------------------------------------------------
def flat_field_correct(sample, reference, dark, eps=1e-6):
    r"""
    Flat-field (white-field) correction of holograms.

    Computes :math:`(S - D) / (R - D)`, removing the (structured) beam profile
    :math:`R` and the detector dark current :math:`D` from the sample frame(s)
    :math:`S`.  The result is normalised so the vacuum (empty-beam) level is 1.

    Parameters
    ----------
    sample : ndarray, shape ``(M, N)`` or ``(D, M, N)``
        Raw sample hologram(s).
    reference : ndarray
        Empty-beam (flat) frame(s).  Accepted shapes: same as *sample*; a
        single ``(M, N)`` frame broadcast to all; or, to average several flats
        per distance, ``(n_ref, M, N)`` for a 2-D *sample* or
        ``(D, n_ref, M, N)`` for a 3-D *sample* (averaged over the ``n_ref``
        axis).
    dark : ndarray or float
        Dark frame ``(M, N)`` or a scalar.
    eps : float, optional
        Floor applied to ``reference - dark`` to avoid division by zero.

    Returns
    -------
    corrected : ndarray, same leading shape as *sample*
        Flat-field-corrected hologram(s), vacuum level ~1.
    """
    sample = np.asarray(sample, dtype=float)
    dark = np.asarray(dark, dtype=float)
    reference = np.asarray(reference, dtype=float)

    # Average multiple flats if an extra leading axis was supplied.
    if reference.ndim == sample.ndim + 1:
        reference = reference.mean(axis=-3)

    denom = reference - dark
    denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps, denom)
    return (sample - dark) / denom


# ---------------------------------------------------------------------------
# 2. Geometry (Fresnel scaling theorem)
# ---------------------------------------------------------------------------
def holo_geometry(sample_focus_distances, focus_detector_distance,
                  detector_pixel_size):
    r"""
    Effective magnification, propagation distance and pixel size per position.

    Parameters
    ----------
    sample_focus_distances : sequence of float
        Focus-to-sample distances :math:`z_{s,i}` (one per defocus position),
        in metres.
    focus_detector_distance : float
        Focus-to-detector distance :math:`z_D`, in metres.
    detector_pixel_size : float
        Physical detector pixel pitch, in metres.

    Returns
    -------
    geom : HoloGeometry
        Named tuple of arrays ``magnification`` :math:`M_i`,
        ``effective_distance`` :math:`z_{\mathrm{eff},i}` and
        ``effective_pixel_size`` :math:`\Delta_i`.
    """
    zs = np.asarray(sample_focus_distances, dtype=float)
    zD = float(focus_detector_distance)
    if np.any(zs <= 0) or np.any(zs >= zD):
        raise ValueError("each sample-focus distance must satisfy 0 < z_s < z_D")
    M = zD / zs
    z_eff = zs * (zD - zs) / zD
    px_eff = detector_pixel_size / M
    return HoloGeometry(M, z_eff, px_eff)


# ---------------------------------------------------------------------------
# 3. Rescaling to a common effective pixel size
# ---------------------------------------------------------------------------
def rescale_to_common_pixel(images, pixel_sizes, target_pixel=None, order=3):
    r"""
    Interpolate magnified holograms onto a common (effective) pixel size.

    Each image is resampled so that a physical feature spans the same number of
    pixels in every image and the same field of view is covered.  The resampling
    is a **centre-preserving** affine map (no spurious sub-pixel shift): an image
    of effective pixel :math:`\Delta_i` is mapped onto the target grid
    :math:`\Delta_t` with coordinate scale :math:`s_i = \Delta_t/\Delta_i` about
    the image centre, and the output covers the target field of view
    ``N · target_pixel`` (the smallest, highest-resolution FOV).

    Parameters
    ----------
    images : ndarray, shape ``(D, M, N)``
        Flat-field-corrected holograms (one per distance).
    pixel_sizes : sequence of float
        Effective pixel size of each image (e.g. ``geom.effective_pixel_size``).
    target_pixel : float or None, optional
        Common pixel size.  Defaults to ``min(pixel_sizes)`` --- the finest
        sampling (highest-magnification / closest-to-focus image), preserving
        the best resolution.
    order : int, optional
        Spline interpolation order for :func:`scipy.ndimage.affine_transform`.
        Default 3.

    Returns
    -------
    common : ndarray, shape ``(D, M, N)``
        Images on the common grid (the finest image's field of view).
    target_pixel : float
        The common pixel size used.
    """
    images = np.asarray(images, dtype=float)
    pixel_sizes = np.asarray(pixel_sizes, dtype=float)
    if target_pixel is None:
        target_pixel = float(pixel_sizes.min())

    ref_idx = int(np.argmin(pixel_sizes))
    out_shape = images[ref_idx].shape
    no_y, no_x = out_shape
    out = np.empty((images.shape[0],) + out_shape, dtype=float)
    for i, img in enumerate(images):
        s = target_pixel / pixel_sizes[i]             # <= 1 (finer steps)
        ni_y, ni_x = img.shape
        off_y = (ni_y - 1) / 2.0 - s * (no_y - 1) / 2.0
        off_x = (ni_x - 1) / 2.0 - s * (no_x - 1) / 2.0
        out[i] = _affine(img, [s, s], offset=[off_y, off_x],
                         output_shape=out_shape, order=order, mode="nearest")
    return out, target_pixel


# ---------------------------------------------------------------------------
# 4. Sub-pixel alignment
# ---------------------------------------------------------------------------
def _subpixel_shift(img, shift):
    """Apply a (sub-pixel) shift to a real 2-D image via the Fourier domain."""
    shifted = _fourier_shift(np.fft.fftn(img), shift)
    return np.real(np.fft.ifftn(shifted))


def align_holograms(images, reference_index=0, upsample=20, blur=2.0,
                    return_shifts=False):
    r"""
    Sub-pixel align a stack of holograms to one reference image.

    Each image is registered to ``images[reference_index]`` by phase
    cross-correlation (upsampled for sub-pixel precision) and shifted with a
    Fourier shift.

    **Low-pass before correlation.**  Holograms recorded at different
    propagation distances carry *different* Fresnel fringe patterns; correlating
    them directly makes the correlator lock onto the (mismatched) fringes and
    return a wrong shift.  The shift is therefore estimated from Gaussian
    low-pass-filtered copies (the object's position lives in the low
    frequencies, common to all distances) while the shift itself is applied to
    the full-resolution image.

    Parameters
    ----------
    images : ndarray, shape ``(D, M, N)``
        Images on a common grid (output of :func:`rescale_to_common_pixel`).
    reference_index : int, optional
        Index of the reference image (default 0 --- the highest-resolution one).
    upsample : int, optional
        Upsampling factor for sub-pixel cross-correlation.  Default 20
        (≈ 1/20-pixel precision).
    blur : float, optional
        Gaussian sigma (pixels) of the low-pass applied to the images **only**
        for the shift estimation.  Set 0 to correlate the raw images.  Default 2.
    return_shifts : bool, optional
        Also return the applied shift vectors.

    Returns
    -------
    aligned : ndarray, shape ``(D, M, N)``
    shifts : ndarray, shape ``(D, 2)``, optional
        Applied ``(dy, dx)`` shift of each image (zero for the reference).
    """
    if not _HAVE_SKIMAGE:
        raise ImportError("align_holograms requires scikit-image "
                          "(skimage.registration.phase_cross_correlation).")
    images = np.asarray(images, dtype=float)

    def _lp(a):
        return _gaussian_filter(a, blur) if blur and blur > 0 else a

    ref_lp = _lp(images[reference_index])
    aligned = np.empty_like(images)
    shifts = np.zeros((images.shape[0], 2))
    for i, img in enumerate(images):
        if i == reference_index:
            aligned[i] = img
            continue
        # Plain cross-correlation (normalization=None): the default 'phase'
        # normalization fails when the two holograms carry different Fresnel
        # fringes (their cross-power-spectrum phase is dominated by the
        # fringe mismatch).  Fall back gracefully on older scikit-image.
        try:
            shift = _pcc(ref_lp, _lp(img), upsample_factor=upsample,
                         normalization=None)[0]
        except TypeError:                              # pragma: no cover
            shift = _pcc(ref_lp, _lp(img), upsample_factor=upsample)[0]
        shifts[i] = shift
        aligned[i] = _subpixel_shift(img, shift)
    if return_shifts:
        return aligned, shifts
    return aligned


# ---------------------------------------------------------------------------
# 5. Full pipeline
# ---------------------------------------------------------------------------
def holo_ctf_reconstruct(
    samples,
    references,
    dark,
    sample_focus_distances,
    focus_detector_distance,
    detector_pixel_size,
    wavelength,
    alpha=1e-2,
    delta_beta=None,
    align=True,
    upsample=20,
    align_blur=2.0,
    interp_order=3,
    cuda=False,
    return_intermediates=False,
):
    r"""
    End-to-end magnified inline-holography CTF reconstruction (ID16A style).

    Chains flat-field correction, rescaling to a common effective pixel size,
    sub-pixel alignment, and the multi-distance CTF inversion.

    Parameters
    ----------
    samples : ndarray, shape ``(D, M, N)``
        Raw sample holograms at the ``D`` defocus positions.
    references : ndarray
        Empty-beam (flat) frames; see :func:`flat_field_correct` for accepted
        shapes (including ``(D, n_ref, M, N)`` to average several flats).
    dark : ndarray or float
        Dark frame.
    sample_focus_distances : sequence of float
        Focus-to-sample distances :math:`z_{s,i}` [m], one per position.
    focus_detector_distance : float
        Focus-to-detector distance :math:`z_D` [m].
    detector_pixel_size : float
        Detector pixel pitch [m].
    wavelength : float
        X-ray wavelength [m].
    alpha : float, optional
        CTF Tikhonov regularisation (see :func:`ctf_retrieve`).  Default 1e-2.
    delta_beta : float or None, optional
        :math:`\delta/\beta` for the homogeneous model (enables DC / low-
        frequency recovery).  ``None`` → pure-phase object.
    align : bool, optional
        Perform sub-pixel alignment (default True).
    upsample, interp_order : optional
        Alignment upsampling factor and rescaling spline order.
    cuda : bool, optional
        Use the GPU path in :func:`ctf_retrieve`.
    return_intermediates : bool, optional
        If True, also return a dict with the geometry and the
        flat-fielded / rescaled / aligned stacks for inspection.

    Returns
    -------
    phase : ndarray, shape ``(M, N)``
        Retrieved phase [rad] on the common (finest) grid.
    info : dict, optional
        Present when ``return_intermediates=True``.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3:
        raise ValueError("samples must be a 3-D stack (D, M, N).")

    geom = holo_geometry(sample_focus_distances, focus_detector_distance,
                         detector_pixel_size)

    # 1. flat-field
    flat = flat_field_correct(samples, references, dark)

    # 2. rescale to the finest effective pixel (closest-to-focus position)
    common, px_common = rescale_to_common_pixel(
        flat, geom.effective_pixel_size, order=interp_order)

    # reference = highest-magnification image (smallest effective pixel)
    ref_idx = int(np.argmin(geom.effective_pixel_size))

    # 3. sub-pixel align
    if align:
        aligned = align_holograms(common, reference_index=ref_idx,
                                  upsample=upsample, blur=align_blur)
    else:
        aligned = common

    # 4. multi-distance CTF on the common grid
    phase = ctf_retrieve(
        aligned, list(geom.effective_distance), wavelength, px_common,
        alpha=alpha, delta_beta=delta_beta, cuda=cuda)

    if return_intermediates:
        info = {
            "geometry": geom,
            "common_pixel_size": px_common,
            "reference_index": ref_idx,
            "flat": flat,
            "rescaled": common,
            "aligned": aligned,
        }
        return phase, info
    return phase
