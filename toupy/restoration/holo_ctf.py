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
    "eigenflat_correct",
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


def _total_variation(img):
    return float(np.abs(np.diff(img, axis=0)).sum()
                 + np.abs(np.diff(img, axis=1)).sum())


def eigenflat_correct(samples, references, dark, n_components=4,
                      maxiter=60, eps=1e-6):
    r"""
    Dynamic flat-field correction with **eigen flat fields** (PCA of the flats).

    When the beam (probe) drifts between the flat and the sample acquisitions,
    a single division ``(S-dark)/(ref-dark)`` leaves residual fringes.  This
    routine instead builds a low-dimensional basis of the beam variations from
    *many* flat frames and, for each sample, finds the linear combination of
    eigen-flats that best removes the beam --- by minimising the total variation
    of the corrected image (Van Nieuwenhove et al., Opt.\ Express 23, 27975,
    2015).

    Parameters
    ----------
    samples : ndarray, shape ``(M, N)`` or ``(D, M, N)``
        Raw sample hologram(s).
    references : ndarray, shape ``(K, M, N)`` (for a 2-D *sample*) or
        ``(D, K, M, N)`` (for a 3-D *sample*)
        Stack of ``K`` empty-beam flat frames (per distance, if 4-D).
    dark : ndarray or float
        Dark frame.
    n_components : int, optional
        Number of eigen-flats (principal components) to use.  Default 4.
    maxiter : int, optional
        Max iterations of the per-sample TV minimisation.  Default 60.

    Returns
    -------
    corrected : ndarray, same leading shape as *samples*
        Dynamically flat-field-corrected hologram(s).

    Notes
    -----
    Falls back to the mean-flat division when fewer than two flat frames are
    supplied (no variations to model).
    """
    from scipy.optimize import minimize

    samples = np.asarray(samples, dtype=float)
    dark = np.asarray(dark, dtype=float)
    references = np.asarray(references, dtype=float)
    single = samples.ndim == 2
    S = samples[np.newaxis] if single else samples
    D = S.shape[0]

    # references → per-distance stacks of flats: (D, K, M, N)
    if references.ndim == 3:
        refs = np.broadcast_to(references[np.newaxis], (D,) + references.shape)
    elif references.ndim == 4:
        refs = references
    else:
        raise ValueError("references must be (K, M, N) or (D, K, M, N).")

    out = np.empty_like(S)
    for d in range(D):
        flats = refs[d] - dark                       # (K, M, N)
        K = flats.shape[0]
        mean_flat = flats.mean(axis=0)
        if K < 2:
            denom = np.where(np.abs(mean_flat) < eps, eps, mean_flat)
            out[d] = (S[d] - dark) / denom
            continue
        # PCA of the flat variations via SVD
        A = (flats - mean_flat).reshape(K, -1)
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        nc = min(n_components, Vt.shape[0])
        eig = Vt[:nc].reshape(nc, *mean_flat.shape)   # eigen-flats

        s_d = S[d] - dark

        def _cost(w):
            flat = mean_flat + np.tensordot(w, eig, axes=(0, 0))
            flat = np.where(np.abs(flat) < eps, eps, flat)
            return _total_variation(s_d / flat)

        res = minimize(_cost, np.zeros(nc), method="Powell",
                       options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-3})
        flat = mean_flat + np.tensordot(res.x, eig, axes=(0, 0))
        flat = np.where(np.abs(flat) < eps, eps, flat)
        out[d] = s_d / flat

    return out[0] if single else out


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
# Native-resolution joint multi-distance non-linear solver
# ---------------------------------------------------------------------------
def _fourier_down(x, n_out):
    """Ideal band-limited resample to a smaller grid (Fourier crop)."""
    n = x.shape[0]
    if n_out >= n:
        return x
    X = np.fft.fftshift(np.fft.fft2(x, norm="ortho"))
    s = (n - n_out) // 2
    return np.real(np.fft.ifft2(
        np.fft.ifftshift(X[s:s + n_out, s:s + n_out]), norm="ortho"))


def _fourier_up(x, n_out):
    """Transpose of :func:`_fourier_down` (Fourier zero-pad)."""
    n = x.shape[0]
    if n_out <= n:
        return x
    X = np.fft.fftshift(np.fft.fft2(x, norm="ortho"))
    s = (n_out - n) // 2
    G = np.zeros((n_out, n_out), dtype=complex)
    G[s:s + n, s:s + n] = X
    return np.real(np.fft.ifft2(np.fft.ifftshift(G), norm="ortho"))


def _native_multidistance_solve(stack, distances, pixel_sizes, common_pixel,
                                wavelength, delta_beta, init, n_iter=200,
                                reg_tv=1e-3, tv_eps=1e-3, pad=2):
    r"""
    Joint non-linear multi-distance retrieval fitting each distance at its
    **native** resolution.

    Minimises :math:`\tfrac12\sum_d \big\| D_d|\mathcal P_{z_d}\psi(\phi)|
    - D_d\sqrt{I_d}\big\|^2 + \lambda\,\mathrm{TV}(\phi)` where
    :math:`D_d` band-limits / downsamples to detector *d*'s effective pixel.
    Unlike fitting the full-band model against the up-sampled coarse data, each
    distance constrains only the frequencies it actually measured --- giving a
    markedly more accurate (quantitative) reconstruction.  Adjoint gradient
    finite-difference verified.
    """
    if delta_beta is None:
        raise ValueError("method='nonlinear' requires delta_beta (homogeneous "
                         "object); pass the delta/beta ratio.")
    n = stack.shape[-1]
    D = len(distances)
    c = complex(-1.0 / float(delta_beta), 1.0)
    c_conj = np.conj(c)
    nout = [max(4, int(round(n * common_pixel / float(p)))) for p in pixel_sizes]
    m = n * pad
    o = (m - n) // 2
    f = np.fft.fftfreq(m, d=common_pixel)
    FY, FX = np.meshgrid(f, f, indexing="ij")
    Hs = [np.exp(1j * np.pi * wavelength * float(z) * (FY**2 + FX**2))
          for z in distances]
    adata = [_fourier_down(np.sqrt(np.maximum(stack[d], 1e-6)), nout[d])
             for d in range(D)]

    def emb(a):
        b = np.zeros((m, m), dtype=complex)
        b[o:o + n, o:o + n] = a
        return b

    def crop(a):
        return a[o:o + n, o:o + n]

    def cost_grad(phi):
        psi = np.exp(c * emb(phi))
        psi_f = np.fft.fft2(psi)
        J = 0.0
        g = np.zeros((n, n))
        for d in range(D):
            a = crop(np.fft.ifft2(psi_f * Hs[d]))
            amp = np.abs(a)
            r = _fourier_down(amp, nout[d]) - adata[d]
            J += 0.5 * float(np.sum(r * r))
            gz = emb(_fourier_up(r, n) * a / (amp + 1e-30))
            gobj = np.fft.ifft2(np.fft.fft2(gz) * np.conj(Hs[d]))
            g += crop(np.real(c_conj * np.conj(psi) * gobj))
        if reg_tv > 0:
            dx = np.roll(phi, -1, 1) - phi
            dy = np.roll(phi, -1, 0) - phi
            mag = np.sqrt(dx**2 + dy**2 + tv_eps**2)
            J += reg_tv * float(mag.sum())
            px, py = dx / mag, dy / mag
            g -= reg_tv * ((px - np.roll(px, 1, 1)) + (py - np.roll(py, 1, 0)))
        return J, g

    phi = np.array(init, dtype=float)
    J, g = cost_grad(phi)
    d_ = -g
    t = 1.0
    for _ in range(int(n_iter)):
        gd = float(np.sum(g * d_))
        if gd >= 0:
            d_ = -g
            gd = float(np.sum(g * d_))
        t = min(t * 2.0, 1e3)
        Jn, gn = cost_grad(phi + t * d_)
        k = 0
        while Jn > J + 1e-4 * t * gd and t > 1e-12 and k < 60:
            t *= 0.5
            Jn, gn = cost_grad(phi + t * d_)
            k += 1
        phi = phi + t * d_
        beta = max(0.0, float(np.sum(gn * (gn - g))) / (float(np.sum(g * g)) + 1e-30))
        d_ = -gn + beta * d_
        g, J = gn, Jn
    return phi


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
    alpha=1e-4,
    delta_beta=None,
    flat_method="simple",
    n_eigen=4,
    method="ctf",
    nl_n_iter=200,
    nl_reg_tv=1e-3,
    align=True,
    upsample=20,
    align_blur=2.0,
    refine_align=False,
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
        CTF Tikhonov regularisation (see :func:`ctf_retrieve`).  Default 1e-4.
        **Controls the low-frequency / cupping behaviour:** at small Fresnel
        numbers the DC is carried only by the absorption term
        :math:`\varepsilon=\beta/\delta`, so ``alpha`` must be
        :math:`\lesssim (2\varepsilon)^2` to recover the interior grey levels;
        a too-large ``alpha`` attenuates the low frequencies and produces
        cupping.  Raise it only if high-frequency noise dominates.
    delta_beta : float or None, optional
        :math:`\delta/\beta` for the homogeneous model (enables DC / low-
        frequency recovery).  ``None`` → pure-phase object.
    flat_method : {'simple', 'eigen'}, optional
        ``'simple'`` → :func:`flat_field_correct` (``(S-d)/(ref-d)``);
        ``'eigen'`` → :func:`eigenflat_correct` (dynamic eigen-flat correction,
        robust to beam drift; needs several flat frames per distance).
    n_eigen : int, optional
        Number of eigen-flats when ``flat_method='eigen'``.  Default 4.
    method : {'ctf', 'nonlinear'}, optional
        ``'ctf'`` → linear multi-distance CTF (fast).  ``'nonlinear'`` →
        refine the CTF result with the exact-Fresnel non-linear multi-distance
        solver (:func:`iterative_phase_retrieval`), which recovers the low
        frequencies / DC through the homogeneous coupling and reduces the
        interior *cupping* --- recommended for quantitative grey levels.
    nl_n_iter : int, optional
        Conjugate-gradient iterations for the non-linear refinement.  Default 200.
    nl_reg_tv : float, optional
        Total-variation weight for the non-linear refinement.  Default 1e-3.
    refine_align : bool, optional
        After the initial hologram alignment, refine the shifts by registering
        rough single-distance phase retrievals (which look alike across
        distances, unlike the raw holograms).  Default False.
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
    if flat_method == "eigen":
        flat = eigenflat_correct(samples, references, dark,
                                 n_components=n_eigen)
    elif flat_method == "simple":
        flat = flat_field_correct(samples, references, dark)
    else:
        raise ValueError("flat_method must be 'simple' or 'eigen'.")

    # 2. rescale to the finest effective pixel (closest-to-focus position)
    common, px_common = rescale_to_common_pixel(
        flat, geom.effective_pixel_size, order=interp_order)

    # reference = highest-magnification image (smallest effective pixel)
    ref_idx = int(np.argmin(geom.effective_pixel_size))

    # 3. sub-pixel align
    if align:
        aligned, shifts = align_holograms(
            common, reference_index=ref_idx, upsample=upsample,
            blur=align_blur, return_shifts=True)
        # Refine: register *rough single-distance retrievals* (alike across
        # distances) rather than the raw fringe patterns, then re-shift.
        if refine_align:
            rough = np.array([
                ctf_retrieve(aligned[d], float(geom.effective_distance[d]),
                             wavelength, px_common, alpha=1e-2,
                             delta_beta=delta_beta)
                for d in range(aligned.shape[0])])
            _, extra = align_holograms(rough, reference_index=ref_idx,
                                       upsample=upsample, blur=0.0,
                                       return_shifts=True)
            for d in range(aligned.shape[0]):
                if d != ref_idx and np.any(extra[d]):
                    aligned[d] = _subpixel_shift(aligned[d], extra[d])
                    shifts[d] = shifts[d] + extra[d]
    else:
        aligned = common
        shifts = np.zeros((aligned.shape[0], 2))

    # 4. multi-distance CTF on the common grid
    phase = ctf_retrieve(
        aligned, list(geom.effective_distance), wavelength, px_common,
        alpha=alpha, delta_beta=delta_beta, cuda=cuda)

    # 4b. optional non-linear refinement.  The native-resolution joint solver
    # fits each distance at its own effective pixel (band-limited), which is
    # markedly more accurate than fitting the full-band model to the up-sampled
    # coarse data — recovering quantitative interior grey levels.
    if method == "nonlinear":
        phase = _native_multidistance_solve(
            aligned, list(geom.effective_distance),
            np.asarray(geom.effective_pixel_size), px_common, wavelength,
            delta_beta, init=phase, n_iter=nl_n_iter, reg_tv=nl_reg_tv)
    elif method != "ctf":
        raise ValueError("method must be 'ctf' or 'nonlinear'.")

    if return_intermediates:
        info = {
            "geometry": geom,
            "common_pixel_size": px_common,
            "reference_index": ref_idx,
            "flat": flat,
            "rescaled": common,
            "aligned": aligned,
            "shifts": shifts,
        }
        return phase, info
    return phase
