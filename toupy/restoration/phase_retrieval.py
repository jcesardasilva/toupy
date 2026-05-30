#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase retrieval for propagation-based X-ray imaging.

Two methods are provided:

TIE-Hom (Paganin et al., 2002)
    Single-distance phase retrieval assuming a homogeneous object
    (constant delta/beta ratio).  Extremely robust and the de-facto
    standard for synchrotron nanotomography.

    Reference:
        Paganin, D., Mayo, S. C., Gureyev, T. E., Miller, P. R., &
        Wilkins, S. W. (2002). Simultaneous phase and amplitude
        extraction from a single defocused image of a homogeneous object.
        Journal of Microscopy, 206(1), 33–40.
        https://doi.org/10.1046/j.1365-2818.2002.01010.x

CTF (Contrast Transfer Function)
    Linear phase retrieval in the weak-phase / weak-absorption regime,
    valid for a single propagation distance.  Handles the zeros of the
    CTF via Tikhonov regularisation.

    Reference:
        Turner, L. D., Dhal, B. B., Hayes, J. P., Mancuso, A. P.,
        Nugent, K. A., Paterson, D., ... & Peele, A. G. (2004).
        X-ray phase imaging: Demonstration of extended conditions
        for homogeneous objects. Optics Express, 12(13), 2960–2965.
        https://doi.org/10.1364/OPEX.12.002960

GPU notes
---------
Both functions accept ``cuda=True`` to run entirely on a CuPy GPU array.
The input may be a CPU numpy array (it is uploaded automatically) and the
output is always returned as a CPU numpy array.  When a stack (3-D array)
is passed with ``cuda=True``, the full stack is batched on the GPU so that
only one upload/download round-trip occurs regardless of the number of
projections.
"""

import warnings
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

# ---------------------------------------------------------------------------
# Optional CuPy import — graceful degradation to CPU if unavailable
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

__all__ = ["tie_hom", "ctf_retrieve", "iterative_phase_retrieval",
           "suggest_holo_distances"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_xp(cuda):
    """Return cupy if cuda is requested and available, else numpy."""
    if cuda:
        if not CUDA_AVAILABLE:
            warnings.warn(
                "CuPy not available — falling back to CPU phase retrieval.",
                stacklevel=3,
            )
            return np
        return cp
    return np


def _freq_grid(ny, nx, pixel_size, xp):
    """Return (fy, fx) frequency grids in units of 1/pixel_size."""
    fy = xp.fft.fftfreq(ny, d=pixel_size)
    fx = xp.fft.fftfreq(nx, d=pixel_size)
    return xp.meshgrid(fy, fx, indexing="ij")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tie_hom(
    intensity,
    distance,
    wavelength,
    pixel_size,
    delta_beta=1e3,
    regularisation=1e-3,
    cuda=False,
):
    """
    Single-distance TIE-Hom (Paganin) phase retrieval.

    Recovers the projected thickness (or phase) from a single defocused
    intensity image assuming a homogeneous object with a fixed
    delta / beta ratio.

    The filter in frequency space is:

    .. math::

        \\hat{T}(\\mathbf{u}) =
        \\frac{\\mathcal{F}\\{I / I_0\\}}
             {1 + \\pi\\,\\lambda\\,z\\,(\\delta/\\beta)\\,|\\mathbf{u}|^2}

    where the projected thickness is
    :math:`T = -1/(\\mu) \\ln \\hat{T}`,
    and the phase is :math:`\\phi = -T \\cdot (\\pi / (\\lambda z))`.

    Parameters
    ----------
    intensity : ndarray, shape (M, N) or (K, M, N)
        Measured intensity image(s).  A 3-D array is treated as a stack
        of projections processed as a single GPU batch when ``cuda=True``.
    distance : float
        Sample-to-detector propagation distance [m].
    wavelength : float
        X-ray wavelength [m].
    pixel_size : float
        Effective detector pixel size at the sample plane [m].
    delta_beta : float, optional
        Ratio delta/beta of the complex refractive index decrement.
        Typical values: 1e2–1e4 for biological soft tissue; 1e1–1e2 for
        metals.  Default 1e3.
    regularisation : float, optional
        Tikhonov regularisation parameter alpha.  Prevents division by
        zero when the denominator approaches unity.  Default 1e-3.
    cuda : bool, optional
        Run on GPU via CuPy.  Falls back to CPU with a warning if CuPy
        is unavailable.  Default False.

    Returns
    -------
    phase : ndarray, same shape as *intensity*
        Retrieved phase image(s) [rad].  Always returned as a CPU array.

    Examples
    --------
    >>> import numpy as np
    >>> from toupy.restoration import tie_hom
    >>> img = np.ones((256, 256))
    >>> phase = tie_hom(img, distance=0.1, wavelength=1e-10,
    ...                 pixel_size=1e-7, delta_beta=500)
    """
    xp = _get_xp(cuda)

    intensity = np.asarray(intensity, dtype=float)
    stack = intensity.ndim == 3
    if not stack:
        intensity = intensity[np.newaxis]

    use_gpu = cuda and CUDA_AVAILABLE

    # Upload entire stack in one round-trip when on GPU
    arr = xp.asarray(intensity) if use_gpu else intensity

    ny, nx = arr.shape[-2:]
    fy, fx = _freq_grid(ny, nx, pixel_size, xp)
    freq2 = fy**2 + fx**2

    denom = 1.0 + np.pi * wavelength * distance * delta_beta * freq2
    denom = xp.maximum(denom, regularisation)

    # Batch FFT: fft2 on a 3-D array operates on the last two axes
    I_mean = arr.mean(axis=(-2, -1), keepdims=True) + 1e-30
    I_norm = arr / I_mean
    filtered = xp.real(xp.fft.ifft2(xp.fft.fft2(I_norm) / denom))
    phase = xp.log(xp.maximum(filtered, 1e-10)) * (-delta_beta / 2.0)

    if use_gpu:
        phase = cp.asnumpy(phase)
    else:
        phase = np.asarray(phase)

    return phase[0] if not stack else phase


def ctf_retrieve(
    intensity,
    distance,
    wavelength,
    pixel_size,
    alpha=1e-3,
    delta_beta=None,
    cuda=False,
):
    """
    CTF (Contrast Transfer Function) phase retrieval — single or multi-distance.

    Linearised phase retrieval valid for a weakly-attenuating, weak-phase
    object (``|φ| << 1 rad``).  Supports both single-distance inversion and
    **multi-distance / holotomographic** inversion, and both pure-phase and
    homogeneous (fixed δ/β) objects.

    For the standard paraxial Fresnel propagator ``H_z = exp(+iπλz|u|²)`` the
    linearised contrast of a homogeneous object is

    .. math::

        \\hat{H}_d(\\mathbf{u}) = -2\\,w_d(\\mathbf{u})\\,\\hat{\\phi}(\\mathbf{u}),
        \\qquad
        w_d = \\sin(\\chi_d) + \\varepsilon \\cos(\\chi_d),

    with :math:`\\chi_d = \\pi\\lambda z_d |\\mathbf{u}|^2` and
    :math:`\\varepsilon = \\beta/\\delta = 1/(\\delta/\\beta)` (``ε = 0`` for a
    pure-phase object).  The least-squares Tikhonov inversion combining all
    distances is

    .. math::

        \\hat{\\phi} = -\\frac{\\sum_d w_d \\hat{H}_d}
                            {2\\left(\\sum_d w_d^2 + \\alpha\\right)} .

    **Why multi-distance + δ/β recovers low frequencies.**  A single distance
    transfers phase only in a band around the first CTF maximum
    (:math:`\\chi \\approx \\pi/2`) and is blind at the CTF zeros and at DC
    (:math:`\\sin\\chi \\to 0`).  Using several distances fills the zeros of
    one with the maxima of another.  The absorption term
    :math:`\\varepsilon\\cos(\\chi_d)` is non-zero at DC
    (:math:`\\cos 0 = 1`), so a homogeneous object additionally recovers the
    low frequencies — this is the basis of quantitative holotomography.

    Parameters
    ----------
    intensity : ndarray
        Measured intensity image(s), normalised so the vacuum level is 1.
        Shapes accepted:

        * ``(M, N)`` — one image, single distance.
        * ``(K, M, N)`` with a **scalar** ``distance`` — a batch of *K*
          independent projections, each retrieved separately.
        * ``(D, M, N)`` with an **array** ``distance`` of length *D* — one
          object imaged at *D* propagation distances, combined into a single
          phase map (multi-distance / holotomography).
    distance : float or sequence of float
        Propagation distance(s) [m].  A sequence triggers the multi-distance
        combination; its length must match ``intensity.shape[0]``.
    wavelength : float
        X-ray wavelength [m].
    pixel_size : float
        Effective detector pixel size [m].
    alpha : float, optional
        Tikhonov regularisation parameter.  Default 1e-3.
    delta_beta : float or None, optional
        Ratio δ/β of the complex refractive index.  When provided the
        homogeneous (single-material) model is used, coupling absorption to
        phase and enabling low-frequency / DC recovery.  ``None`` (default)
        assumes a pure-phase object (``ε = 0``).
    cuda : bool, optional
        Run on GPU via CuPy.  Falls back to CPU with a warning if CuPy is
        unavailable.  Default False.

    Returns
    -------
    phase : ndarray
        Retrieved phase [rad], always a CPU array.  Shape ``(M, N)`` for a
        single image or a multi-distance set, or ``(K, M, N)`` for a batch
        of independent projections.

    Examples
    --------
    Single distance, pure phase:

    >>> import numpy as np
    >>> from toupy.restoration import ctf_retrieve
    >>> img = np.ones((256, 256))
    >>> phase = ctf_retrieve(img, distance=5e-5, wavelength=7.3e-11,
    ...                      pixel_size=5e-8)

    Multi-distance holotomographic retrieval of a homogeneous object:

    >>> stack = np.ones((4, 256, 256))                     # 4 distances
    >>> dists = [3e-5, 1.5e-4, 6e-4, 2.4e-3]
    >>> phase = ctf_retrieve(stack, distance=dists, wavelength=7.3e-11,
    ...                      pixel_size=5e-8, delta_beta=50.0)
    """
    xp = _get_xp(cuda)

    distances = np.atleast_1d(np.asarray(distance, dtype=float))
    multi = distances.size > 1

    intensity = np.asarray(intensity, dtype=float)
    if multi:
        # One object, several distances -> combine into a single phase map.
        if intensity.ndim != 3 or intensity.shape[0] != distances.size:
            raise ValueError(
                "Multi-distance CTF expects intensity of shape "
                "(D, M, N) with D = len(distance) = {}, got {}.".format(
                    distances.size, intensity.shape)
            )
        batch = False
    else:
        # Scalar distance: 2-D -> one image; 3-D -> batch of independent ones.
        batch = intensity.ndim == 3
        if not batch:
            intensity = intensity[np.newaxis]

    use_gpu = cuda and CUDA_AVAILABLE
    arr = xp.asarray(intensity) if use_gpu else intensity

    ny, nx = arr.shape[-2:]
    fy, fx = _freq_grid(ny, nx, pixel_size, xp)
    freq2 = fy**2 + fx**2
    eps = 0.0 if delta_beta is None else 1.0 / float(delta_beta)

    if multi:
        # Combine distances: φ̂ = -Σ w_d Ĥ_d / (2 (Σ w_d² + α))
        num = xp.zeros((ny, nx), dtype=complex)
        den = xp.zeros((ny, nx))
        for d in range(distances.size):
            chi = np.pi * wavelength * float(distances[d]) * freq2
            w = xp.sin(chi) + eps * xp.cos(chi)
            num = num + w * xp.fft.fft2(arr[d] - 1.0)
            den = den + w * w
        phase = -xp.real(xp.fft.ifft2(num / (2.0 * (den + alpha))))
    else:
        # Single distance (optionally a batch of independent projections).
        #
        # Sign convention: for H_z = exp(+iπλz|u|²) the contrast is
        #   I - 1 = -2 [sin(χ) + ε cos(χ)] φ
        # so the inversion carries a leading MINUS sign to return +φ.
        chi = np.pi * wavelength * float(distances[0]) * freq2
        w = xp.sin(chi) + eps * xp.cos(chi)
        H = xp.fft.fft2(arr - 1.0)
        phase = -xp.real(xp.fft.ifft2(w * H / (2.0 * (w**2 + alpha))))

    if use_gpu:
        phase = cp.asnumpy(phase)
    else:
        phase = np.asarray(phase)

    if multi:
        return phase
    return phase if batch else phase[0]


# ---------------------------------------------------------------------------
# Nonlinear iterative phase retrieval (full Fresnel forward model)
# ---------------------------------------------------------------------------

def iterative_phase_retrieval(
    intensity,
    distance,
    wavelength,
    pixel_size,
    delta_beta,
    init=None,
    n_iter=200,
    reg_smooth=1e-4,
    reg_tv=0.0,
    tv_eps=1e-3,
    pad=2,
    tol=1e-7,
    cuda=False,
    verbose=False,
):
    r"""
    Nonlinear iterative phase retrieval using the exact Fresnel forward model.

    Refines a phase map by minimising the mismatch between the measured
    intensity and the intensity predicted by *propagating* a homogeneous-object
    wavefield — i.e. it inverts the **full** near-field diffraction integral
    instead of TIE-Hom's first-order (single-fringe) approximation.  This makes
    it accurate in the **multi-fringe** regime (small Fresnel number) and at
    higher δ/β, where the Paganin/TIE-Hom filter blurs and loses contrast.

    The object is assumed homogeneous (single material), so absorption and
    phase are coupled and the unknown is a single real field φ:

    .. math::

        \psi(\phi) = \exp\!\big[(-1/(\delta/\beta) + i)\,\phi\big].

    For each propagation distance :math:`z_d` the model intensity is
    :math:`I_d = |\mathcal{P}_{z_d}\,\psi(\phi)|^2`, where
    :math:`\mathcal{P}_{z_d}` is the angular-spectrum (Fresnel) propagator.
    The objective is the amplitude (Gaussian) data term plus a smoothness
    (Tikhonov) prior

    .. math::

        J(\phi) = \tfrac12 \sum_d \big\|\,|\mathcal{P}_{z_d}\psi(\phi)|
                  - \sqrt{I_d}\,\big\|^2
                  + \tfrac{\mu}{2}\,\|\nabla\phi\|^2 ,

    minimised by nonlinear conjugate gradient (Polak–Ribière) with an Armijo
    backtracking line search.  The analytic gradient is obtained from the
    adjoint (back-propagation) of the forward model.

    Passing **several distances** turns this into *nonlinear holotomography*,
    which — unlike the linear CTF — remains valid for strong phase
    (:math:`|\phi| \gtrsim 1` rad) and recovers the low frequencies through the
    absorption term.

    Parameters
    ----------
    intensity : ndarray
        Measured intensity, normalised to unit vacuum level.  Shape ``(M, N)``
        for a single distance, or ``(D, M, N)`` with a length-*D* sequence
        ``distance`` for multi-distance retrieval.
    distance : float or sequence of float
        Propagation distance(s) [m].
    wavelength : float
        X-ray wavelength [m].
    pixel_size : float
        Effective detector pixel size at the sample [m].
    delta_beta : float
        δ/β ratio of the (homogeneous) object.
    init : ndarray or None, optional
        Initial phase guess ``(M, N)``.  When ``None`` (default) the TIE-Hom
        result at the first distance is used — the recommended warm start.
    n_iter : int, optional
        Maximum number of conjugate-gradient iterations.  Default 200.
    reg_smooth : float, optional
        Weight μ of the gradient-squared (Tikhonov) smoothness prior.  Set 0
        to disable.  Default 1e-4.  Smooths uniformly (edges included).
    reg_tv : float, optional
        Weight of an (ε-smoothed) **total-variation** prior
        :math:`\sum \sqrt{|\nabla\phi|^2 + \varepsilon^2}`.  Unlike the
        quadratic prior, TV flattens residual ripples / Fresnel artefacts in
        homogeneous regions while preserving sharp edges — recommended for
        piecewise-constant samples and for cleaning single-distance results.
        Set 0 to disable (default).  Typical range 1e-3–1e-2.
    tv_eps : float, optional
        Smoothing parameter ε of the TV norm (keeps it differentiable for the
        conjugate-gradient solver).  Default 1e-3.
    pad : int, optional
        Zero/vacuum padding factor for the propagation FFTs, used to avoid
        circular wrap-around of the Fresnel kernel.  Default 2.
    tol : float, optional
        Stop when the relative cost decrease falls below ``tol``.  Default 1e-7.
    cuda : bool, optional
        Run on GPU via CuPy (falls back to CPU with a warning).  Default False.
    verbose : bool, optional
        Print the cost every 25 iterations.  Default False.

    Returns
    -------
    phase : ndarray, shape (M, N)
        Retrieved phase [rad], always a CPU array.

    Notes
    -----
    * **Phase wrapping.**  For a single distance the absolute phase is only
      well determined up to wrapping; if the true phase greatly exceeds
      :math:`\pi` the iteration may settle in a wrapped local minimum.  Use
      several distances (and/or a good ``init``) for strong-phase objects.
    * The object should sit inside the field of view with a vacuum margin so
      that its propagation fringes are captured by the detector; otherwise the
      boundary is under-constrained.

    Examples
    --------
    >>> from toupy.restoration import iterative_phase_retrieval
    >>> phase = iterative_phase_retrieval(I, distance=8e-4, wavelength=7.3e-11,
    ...                                   pixel_size=5e-8, delta_beta=20.0)
    """
    xp = _get_xp(cuda)
    use_gpu = cuda and CUDA_AVAILABLE

    distances = np.atleast_1d(np.asarray(distance, dtype=float))
    intensity = np.asarray(intensity, dtype=float)
    if distances.size > 1:
        if intensity.ndim != 3 or intensity.shape[0] != distances.size:
            raise ValueError(
                "Multi-distance retrieval expects intensity of shape "
                "(D, M, N) with D = len(distance) = {}, got {}.".format(
                    distances.size, intensity.shape))
        stack = intensity
    else:
        stack = intensity[np.newaxis] if intensity.ndim == 2 else intensity

    ny, nx = stack.shape[-2:]
    pad = max(1, int(pad))
    # Pad only axes with extent > 1 — a size-1 axis (e.g. a single tomographic
    # projection line) then propagates as an exact 1-D Fresnel problem.
    my = ny * pad if ny > 1 else 1
    mx = nx * pad if nx > 1 else 1
    oy, ox = (my - ny) // 2, (mx - nx) // 2

    # Warm start: TIE-Hom at the first distance.
    if init is None:
        init = tie_hom(stack[0], float(distances[0]), wavelength, pixel_size,
                       delta_beta=delta_beta, cuda=cuda)
    phi = xp.asarray(np.asarray(init, dtype=float))

    sqI = [xp.asarray(np.sqrt(stack[d])) for d in range(distances.size)]

    # Padded Fresnel propagators, one per distance.
    fy = xp.fft.fftfreq(my, d=pixel_size)
    fx = xp.fft.fftfreq(mx, d=pixel_size)
    FY, FX = xp.meshgrid(fy, fx, indexing="ij")
    freq2 = FY**2 + FX**2
    Hs = [xp.exp(1j * np.pi * wavelength * float(z) * freq2) for z in distances]
    c = complex(-1.0 / float(delta_beta), 1.0)
    c_conj = np.conj(c)

    def _embed(a):
        b = xp.zeros((my, mx), dtype=a.dtype)
        b[oy:oy + ny, ox:ox + nx] = a
        return b

    def _crop(a):
        return a[oy:oy + ny, ox:ox + nx]

    def _cost_grad(phi):
        psi = xp.exp(c * _embed(phi))
        psi_f = xp.fft.fft2(psi)
        J = 0.0
        grad = xp.zeros((ny, nx))
        for H, s in zip(Hs, sqI):
            psz = xp.fft.ifft2(psi_f * H)            # propagate
            amp = xp.abs(psz)
            res = _crop(amp) - s                     # amplitude residual
            J = J + 0.5 * float(xp.sum(res * res))
            gz = _embed(res * _crop(psz) / (_crop(amp) + 1e-30))
            gobj = xp.fft.ifft2(xp.fft.fft2(gz) * xp.conj(H))   # back-propagate
            grad = grad + _crop(xp.real(c_conj * xp.conj(psi) * gobj))
        if reg_smooth > 0:
            dphi0 = phi - xp.roll(phi, 1, axis=0)
            dphi1 = phi - xp.roll(phi, 1, axis=1)
            J = J + 0.5 * reg_smooth * float(xp.sum(dphi0**2 + dphi1**2))
            lap = (-4.0 * phi
                   + xp.roll(phi, 1, 0) + xp.roll(phi, -1, 0)
                   + xp.roll(phi, 1, 1) + xp.roll(phi, -1, 1))
            grad = grad - reg_smooth * lap
        if reg_tv > 0:
            # Isotropic ε-smoothed TV: Σ sqrt(|∇φ|² + ε²); edge-preserving.
            dx = xp.roll(phi, -1, axis=1) - phi
            dy = xp.roll(phi, -1, axis=0) - phi
            mag = xp.sqrt(dx**2 + dy**2 + tv_eps**2)
            J = J + reg_tv * float(xp.sum(mag))
            px, py = dx / mag, dy / mag
            divp = (px - xp.roll(px, 1, axis=1)) + (py - xp.roll(py, 1, axis=0))
            grad = grad - reg_tv * divp
        return J, grad

    # Nonlinear conjugate gradient (Polak–Ribière+) with Armijo backtracking.
    J, g = _cost_grad(phi)
    d = -g
    t = 1.0
    for it in range(int(n_iter)):
        gd = float(xp.sum(g * d))
        if gd >= 0:                       # not a descent direction → restart
            d = -g
            gd = float(xp.sum(g * d))
        t = min(t * 2.0, 1e3)
        Jn, gn = _cost_grad(phi + t * d)
        n_ls = 0
        while Jn > J + 1e-4 * t * gd and t > 1e-12 and n_ls < 60:
            t *= 0.5
            Jn, gn = _cost_grad(phi + t * d)
            n_ls += 1
        phi = phi + t * d
        denom = float(xp.sum(g * g)) + 1e-30
        beta = max(0.0, float(xp.sum(gn * (gn - g))) / denom)
        d = -gn + beta * d
        rel = (J - Jn) / (abs(J) + 1e-30)
        g, J = gn, Jn
        if verbose and (it % 25 == 0 or it == n_iter - 1):
            print("  iter {:4d}   J = {:.6e}".format(it, J))
        if 0.0 <= rel < tol:
            break

    return cp.asnumpy(phi) if use_gpu else np.asarray(phi)


# ---------------------------------------------------------------------------
# Helper: choose a gap-free multi-distance (holotomography) series
# ---------------------------------------------------------------------------

def suggest_holo_distances(
    pixel_size,
    wavelength,
    n=4,
    delta_beta=None,
    nf_short=0.5,
    nf_long=0.01,
    f_lo=0.03,
    warn_gap=0.02,
    return_info=False,
):
    r"""
    Suggest a geometrically-spaced set of propagation distances for
    multi-distance (holotomographic) phase retrieval, and check it is gap-free.

    The homogeneous phase transfer function at distance :math:`z` is
    :math:`w_z(u) = \sin(\chi) + \varepsilon\cos(\chi)` with
    :math:`\chi = \pi\lambda z u^2` and :math:`\varepsilon = 1/(\delta/\beta)`.
    A single distance loses information at its CTF zeros
    :math:`u_n = \sqrt{n/(\lambda z)}`.  Several distances fill those gaps when
    their zeros **interleave**, which happens for a geometric spacing of *z*.

    The series is built directly from the two Fresnel-number end points:

    * the **shortest** distance has :math:`N_F = N_{F,\mathrm{short}}`
      (:math:`N_F = \mathrm{pixel}^2/(\lambda z)`).  Keep
      ``nf_short ≳ 0.25`` so its first zero sits at/beyond Nyquist — a smooth,
      zero-free backbone that ensures no frequency is entirely lost;
    * the **longest** distance has :math:`N_F = N_{F,\mathrm{long}}`.  Smaller
      ``nf_long`` (longer z) improves the low-frequency / quantitative DC
      transfer; it is bounded in practice by coherence, detector blur, geometry
      and the Fresnel sampling limit :math:`z < N\,\mathrm{pixel}^2/\lambda`;
    * the intermediate distances are spaced **geometrically**
      :math:`z_d = z_0\,R^{\,d}`, :math:`R=(z_{n-1}/z_0)^{1/(n-1)}`, which
      makes the CTF zeros interleave evenly.

    The worst-case combined transfer :math:`\sqrt{\langle w_z^2\rangle}` over
    ``f_lo·u_Nyq ≤ u ≤ u_Nyq`` is evaluated; a warning is issued if it falls
    below ``warn_gap`` (a residual gap).

    Parameters
    ----------
    pixel_size : float
        Effective detector pixel size at the sample [m].
    wavelength : float
        X-ray wavelength [m].
    n : int, optional
        Number of distances.  Default 4.
    delta_beta : float or None, optional
        δ/β of the object.  Includes the absorption term in the transfer
        function (improves low-frequency coverage).  ``None`` → pure phase.
    nf_short : float, optional
        Fresnel number of the **shortest** distance.  Default 0.5 (zero-free,
        with margin above the 0.25 limit).
    nf_long : float, optional
        Fresnel number of the **longest** distance.  Default 0.01.
    f_lo : float, optional
        Lowest frequency (fraction of Nyquist) at which coverage is required;
        below it transfer is intrinsically weak (→ ε at DC).  Default 0.03.
    warn_gap : float, optional
        Warn if the worst-case combined transfer drops below this.  Default
        0.02.
    return_info : bool, optional
        If True, also return a diagnostics dict.  Default False.

    Returns
    -------
    distances : ndarray, shape (n,)
        Suggested propagation distances [m], ascending.
    info : dict, optional
        Present when ``return_info=True``: ``ratio`` *R*, ``min_coverage``
        (worst combined transfer; larger is better, ~0 means a gap), ``NF``
        (per-distance Fresnel numbers).

    Examples
    --------
    >>> from toupy.restoration import suggest_holo_distances, iterative_phase_retrieval
    >>> z = suggest_holo_distances(50e-9, 7.3e-11, n=4, delta_beta=20)
    >>> # ... acquire/simulate the intensity stack at these distances ...
    >>> # phase = iterative_phase_retrieval(stack, z, 7.3e-11, 5e-8, delta_beta=20)
    """
    n = int(n)
    if nf_long >= nf_short:
        raise ValueError("nf_long ({}) must be < nf_short ({}) — the longest "
                         "distance has the smaller Fresnel number."
                         .format(nf_long, nf_short))
    eps = 0.0 if delta_beta is None else 1.0 / float(delta_beta)
    u_nyq = 1.0 / (2.0 * pixel_size)
    z_short = pixel_size**2 / (wavelength * nf_short)
    z_long = pixel_size**2 / (wavelength * nf_long)

    if n <= 1:
        distances = np.array([z_short])
        ratio = float("nan")
    else:
        ratio = (z_long / z_short) ** (1.0 / (n - 1))
        distances = z_short * ratio ** np.arange(n)
    distances = np.asarray(distances, dtype=float)

    # Worst-case combined transfer over the requested band.
    u = np.linspace(f_lo * u_nyq, u_nyq, 1400)
    acc = np.zeros_like(u)
    for z in distances:
        chi = np.pi * wavelength * z * u**2
        acc = acc + (np.sin(chi) + eps * np.cos(chi)) ** 2
    min_cov = float(np.sqrt(acc / n).min())
    if min_cov < warn_gap:
        warnings.warn(
            "Suggested distances leave a residual frequency gap "
            "(min combined transfer {:.3f} < {:.3f}). Consider more distances "
            "or a wider nf_short/nf_long span.".format(min_cov, warn_gap),
            stacklevel=2,
        )

    if return_info:
        info = {
            "ratio": ratio,
            "min_coverage": min_cov,
            "NF": [pixel_size**2 / (wavelength * z) for z in distances],
        }
        return distances, info
    return distances
