#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Total-Variation (TV) regularised tomographic reconstruction.

Implements the Chambolle-Pock primal-dual algorithm (CP) to solve:

    min_{x >= 0}  (1/2) ||A x - b||^2  +  lambda * TV(x)

where A is the Radon forward operator, b is the (possibly noisy)
sinogram, and TV is the isotropic discrete total-variation.

The algorithm is an exact first-order method and handles the non-smooth
TV term via proximal splitting.  It is well-suited to ptychographic
tomography, where the phase sinogram is noisy or has limited angular
sampling.

Reference:
    Chambolle, A., & Pock, T. (2011). A first-order primal-dual
    algorithm for convex problems with applications to imaging. Journal
    of Mathematical Imaging and Vision, 40(1), 120–145.
    https://doi.org/10.1007/s10107-010-0436-y

    Sidky, E. Y., & Pan, X. (2008). Image reconstruction in circular
    cone-beam computed tomography by constrained, total-variation
    minimization. Physics in Medicine & Biology, 53(17), 4777.
    https://doi.org/10.1088/0031-9155/53/17/014

GPU notes
---------
Both public functions accept ``cuda=True``.  In GPU mode all primal and
dual variables reside on the device as CuPy float32 arrays throughout
the iteration; only the final reconstruction is transferred back to the
host.  The Radon forward and backward operators reuse the CUDA kernels
compiled in :mod:`toupy.tomo.iradon`; no additional compilation is
required.

The step-size estimate is also computed entirely on the GPU via a
five-step power iteration, so the first call is slightly slower than
subsequent calls (kernel reuse).
"""

import warnings
import numpy as np
from skimage.transform import radon, iradon

# ---------------------------------------------------------------------------
# Optional CuPy / CUDA kernel imports — graceful CPU fallback
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from .iradon import CUDA_AVAILABLE, _fbp_kernel, _radon_kernel
except ImportError:
    CUDA_AVAILABLE = False

__all__ = ["tv_reconstruction", "chambolle_pock_tv"]


# ---------------------------------------------------------------------------
# Array-library-agnostic finite-difference operators
# Implemented with slicing (not np.roll) so the same code runs on both
# NumPy and CuPy arrays.
# ---------------------------------------------------------------------------

def _get_xp(arr):
    """Return cupy if arr is a CuPy array, else numpy."""
    if CUDA_AVAILABLE:
        return cp.get_array_module(arr)
    return np


def _grad2d(u):
    """Forward finite-difference gradient (dy, dx), Neumann BC."""
    xp = _get_xp(u)
    gy = xp.empty_like(u)
    gx = xp.empty_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    gy[-1, :]  = 0.0
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gx[:, -1]  = 0.0
    return gy, gx


def _div2d(gy, gx):
    """Negative divergence (adjoint of _grad2d), Neumann BC."""
    xp = _get_xp(gy)
    dy = xp.empty_like(gy)
    dy[0, :]    =  gy[0, :]
    dy[1:-1, :] =  gy[1:-1, :] - gy[:-2, :]
    dy[-1, :]   = -gy[-2, :]
    dx = xp.empty_like(gx)
    dx[:, 0]    =  gx[:, 0]
    dx[:, 1:-1] =  gx[:, 1:-1] - gx[:, :-2]
    dx[:, -1]   = -gx[:, -2]
    return dy + dx


# ---------------------------------------------------------------------------
# TV proximal operators (array-library agnostic)
# ---------------------------------------------------------------------------

def _prox_tv_dual(py, px, lam):
    """Project TV dual onto the isotropic l-inf ball of radius lam.

    prox_{sigma * (lam*||·||_{2,1})*}(z) = z / max(1, |z| / lam)
    The step-size sigma is already folded into z before calling.
    """
    xp    = _get_xp(py)
    norm  = xp.sqrt(py**2 + px**2)
    scale = xp.maximum(1.0, norm / lam)
    return py / scale, px / scale


def _prox_l2_data_dual(z, b, sigma):
    """Proximal op of the conjugate of (1/2)||· - b||^2, step sigma.

    Moreau identity:  prox_{sigma * f*}(z) = (z - sigma*b) / (1 + sigma)
    """
    return (z - sigma * b) / (1.0 + sigma)


# ---------------------------------------------------------------------------
# CPU Radon / back-projection wrappers (skimage)
# ---------------------------------------------------------------------------

def _forward_cpu(x, theta):
    return radon(x, theta=theta, circle=False)


def _backward_cpu(sino, theta, n):
    return iradon(sino, theta=theta, filter_name=None, output_size=n, circle=False)


# ---------------------------------------------------------------------------
# GPU Radon / back-projection wrappers (custom CUDA kernels from iradon.py)
# ---------------------------------------------------------------------------
# Convention used here matches skimage circle=False:
#   nbins = ceil(sqrt(2) * n)
# All GPU arrays are float32.
# ---------------------------------------------------------------------------

def _make_trig_gpu(theta_deg):
    """Precompute float32 cos/sin tables on the GPU."""
    th = np.deg2rad(np.asarray(theta_deg, dtype=np.float64)).astype(np.float32)
    return cp.asarray(np.cos(th)), cp.asarray(np.sin(th))


def _forward_gpu(x_gpu, cos_th, sin_th, nbins, nsteps):
    """GPU Radon forward projection.

    Parameters
    ----------
    x_gpu : cupy.ndarray, shape (N, N), float32
    cos_th, sin_th : cupy.ndarray, shape (nangles,), float32
    nbins : int   — number of detector bins (= ceil(sqrt(2)*N) for circle=False)
    nsteps : int  — integration steps along ray

    Returns
    -------
    sino_gpu : cupy.ndarray, shape (nbins, nangles), float32
    """
    N       = x_gpu.shape[0]
    nangles = cos_th.shape[0]
    sino_flat = cp.zeros(nbins * nangles, dtype=cp.float32)

    block = (16, 16, 1)
    grid  = (int(np.ceil(nbins   / 16)),
             int(np.ceil(nangles / 16)), 1)
    _radon_kernel(
        grid, block,
        (x_gpu.ravel(), sino_flat, cos_th, sin_th,
         np.int32(N), np.int32(nbins), np.int32(nangles), np.int32(nsteps)),
    )
    cp.cuda.stream.get_current_stream().synchronize()
    # Kernel stores (nangles * nbins) in angle-major order → transpose to (nbins, nangles)
    return sino_flat.reshape(nangles, nbins).T


def _backward_gpu(sino_gpu, n, cos_th, sin_th):
    """GPU unfiltered back-projection (adjoint of _forward_gpu).

    Parameters
    ----------
    sino_gpu : cupy.ndarray, shape (nbins, nangles), float32
    n : int   — output image side length
    cos_th, sin_th : cupy.ndarray, shape (nangles,), float32

    Returns
    -------
    img_gpu : cupy.ndarray, shape (n, n), float32
    """
    nbins, nangles = sino_gpu.shape
    mid_index = nbins // 2
    img_flat  = cp.zeros(n * n, dtype=cp.float32)

    # Kernel expects sinogram in (nangles * nbins,) angle-major column-major order
    sino_cm = cp.ascontiguousarray(sino_gpu.T).ravel()

    block = (16, 16, 1)
    grid  = (int(np.ceil(n / 16)), int(np.ceil(n / 16)), 1)
    _fbp_kernel(
        grid, block,
        (sino_cm, cos_th, sin_th, img_flat,
         np.int32(n), np.int32(nbins), np.int32(nangles), np.int32(mid_index)),
    )
    cp.cuda.stream.get_current_stream().synchronize()
    return img_flat.reshape(n, n)


# ---------------------------------------------------------------------------
# Step-size estimation via power iteration
# ---------------------------------------------------------------------------

def _estimate_norm_K2_cpu(n, theta, n_iter=5):
    """Estimate ||K||^2 = ||A||^2 + 8 on CPU via power iteration."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n, n))
    for _ in range(n_iter):
        Av   = _forward_cpu(v, theta)
        ATAv = _backward_cpu(Av, theta, n)
        norm2 = np.sqrt(np.sum(ATAv**2) / (np.sum(v**2) + 1e-30))
        v = ATAv / (np.linalg.norm(ATAv) + 1e-30)
    return float(norm2) + 8.0


def _estimate_norm_K2_gpu(n, cos_th, sin_th, nbins, nsteps, n_iter=5):
    """Estimate ||K||^2 on GPU via power iteration (all ops stay on device)."""
    v = cp.random.default_rng(0).standard_normal((n, n)).astype(cp.float32)
    for _ in range(n_iter):
        Av   = _forward_gpu(v, cos_th, sin_th, nbins, nsteps)
        ATAv = _backward_gpu(Av, n, cos_th, sin_th)
        norm2 = float(cp.sqrt(cp.sum(ATAv**2) / (cp.sum(v**2) + 1e-30)))
        ATAv_norm = float(cp.linalg.norm(ATAv))
        v = ATAv / (ATAv_norm + 1e-30)
    return norm2 + 8.0


# ---------------------------------------------------------------------------
# Core Chambolle-Pock solver
# ---------------------------------------------------------------------------

def chambolle_pock_tv(
    sinogram,
    theta,
    lam=1e-2,
    n_iter=100,
    output_size=None,
    cuda=False,
    verbose=False,
):
    """
    Chambolle-Pock TV-regularised reconstruction.

    Solves the primal-dual saddle-point problem:

    .. math::

        \\min_{x \\ge 0}\\; \\max_{p,q}\\;
        \\langle K x,\\, (p, q) \\rangle
        - G^*(p, q) + F(x)

    where :math:`K = (A,\\, \\nabla)` stacks the Radon operator and the
    gradient, :math:`G^*` is the convex conjugate of the data and TV
    terms, and :math:`F` is the non-negativity indicator.

    Parameters
    ----------
    sinogram : ndarray, shape (n_pixels, n_angles)
        Measured sinogram with axes (detector pixel, angle).
    theta : array_like, shape (n_angles,)
        Projection angles [degrees].
    lam : float, optional
        TV regularisation weight.  Default 1e-2.
    n_iter : int, optional
        Number of Chambolle-Pock iterations.  Default 100.
    output_size : int or None, optional
        Side length of the square reconstruction grid.  When ``None``
        the value is inferred from the sinogram as
        ``round(n_pixels / sqrt(2))``, matching
        ``skimage.transform.radon`` with ``circle=False``.
    cuda : bool, optional
        Run on GPU via CuPy.  All primal/dual variables reside on the
        device throughout the iteration; only the final image is
        transferred back.  Falls back to CPU with a warning when CuPy
        is unavailable.  Default False.
    verbose : bool, optional
        Print cost every 10 iterations.  Default False.

    Returns
    -------
    x : ndarray, shape (output_size, output_size)
        Reconstructed image.  Always returned as a CPU numpy array.

    See Also
    --------
    tv_reconstruction : High-level wrapper with FBP warm-start.
    """
    if cuda and not CUDA_AVAILABLE:
        warnings.warn(
            "CuPy not available — falling back to CPU TV reconstruction.",
            stacklevel=2,
        )
        cuda = False

    theta    = np.asarray(theta, dtype=float)
    sinogram = np.asarray(sinogram, dtype=float)
    n_pix, n_ang = sinogram.shape

    if output_size is None:
        n = int(np.round(n_pix / np.sqrt(2)))
    else:
        n = int(output_size)

    # ------------------------------------------------------------------
    # GPU path
    # ------------------------------------------------------------------
    if cuda:
        nbins  = int(np.ceil(np.sqrt(2) * n))
        nsteps = int(np.ceil(n * np.sqrt(2)))
        cos_th, sin_th = _make_trig_gpu(theta)

        norm_K2 = _estimate_norm_K2_gpu(n, cos_th, sin_th, nbins, nsteps)
        tau     = np.float32(0.9 / np.sqrt(norm_K2))
        sigma   = np.float32(0.9 / np.sqrt(norm_K2))

        # Allocate all variables on GPU as float32
        x     = cp.zeros((n, n),        dtype=cp.float32)
        x_bar = cp.zeros((n, n),        dtype=cp.float32)
        py    = cp.zeros((n, n),        dtype=cp.float32)
        px    = cp.zeros((n, n),        dtype=cp.float32)

        # Sinogram dual variable — shape matches forward projection output
        q     = cp.zeros((nbins, n_ang), dtype=cp.float32)
        b_gpu = cp.asarray(sinogram,     dtype=cp.float32)

        # The measured sinogram may have a different detector size than nbins
        # (e.g. generated with skimage circle=False giving ceil(sqrt(2)*orig_n)).
        # Resize q and b to match if necessary.
        if b_gpu.shape[0] != nbins:
            # Interpolate sinogram to match GPU forward operator output size
            from cupyx.scipy.ndimage import zoom as cp_zoom
            scale = nbins / b_gpu.shape[0]
            b_gpu = cp_zoom(b_gpu, (scale, 1.0), order=1)

        for it in range(n_iter):
            Ax_bar = _forward_gpu(x_bar, cos_th, sin_th, nbins, nsteps)
            q      = _prox_l2_data_dual(q + sigma * Ax_bar, b_gpu, sigma)

            gy, gx = _grad2d(x_bar)
            py, px = _prox_tv_dual(py + sigma * gy, px + sigma * gx, lam)

            AT_q  = _backward_gpu(q, n, cos_th, sin_th)
            div_p = _div2d(py, px)
            x_new = cp.maximum(x - tau * (AT_q + div_p), 0.0)

            x_bar = 2.0 * x_new - x
            x     = x_new

            if verbose and (it % 10 == 0):
                gy_x, gx_x = _grad2d(x)
                cost = (0.5 * float(cp.sum((_forward_gpu(
                            x, cos_th, sin_th, nbins, nsteps) - b_gpu)**2))
                        + lam * float(cp.sum(cp.sqrt(gy_x**2 + gx_x**2))))
                print("  iter {:4d}  cost = {:.6e}".format(it, cost))

        return cp.asnumpy(x)

    # ------------------------------------------------------------------
    # CPU path
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n, n))
    for _ in range(5):
        Av   = _forward_cpu(v, theta)
        ATAv = _backward_cpu(Av, theta, n)
        norm2 = np.sqrt(np.sum(ATAv**2) / (np.sum(v**2) + 1e-30))
        v = ATAv / (np.linalg.norm(ATAv) + 1e-30)
    tau   = 0.9 / np.sqrt(float(norm2) + 8.0)
    sigma = tau

    x     = np.zeros((n, n))
    x_bar = x.copy()
    py    = np.zeros_like(x)
    px    = np.zeros_like(x)
    q     = np.zeros_like(sinogram)

    for it in range(n_iter):
        Ax_bar = _forward_cpu(x_bar, theta)
        q      = _prox_l2_data_dual(q + sigma * Ax_bar, sinogram, sigma)

        gy, gx = _grad2d(x_bar)
        py, px = _prox_tv_dual(py + sigma * gy, px + sigma * gx, lam)

        AT_q  = _backward_cpu(q, theta, n)
        div_p = _div2d(py, px)
        x_new = np.maximum(x - tau * (AT_q + div_p), 0.0)

        x_bar = 2.0 * x_new - x
        x     = x_new

        if verbose and (it % 10 == 0):
            gy_x, gx_x = _grad2d(x)
            cost = (0.5 * np.sum((_forward_cpu(x, theta) - sinogram)**2)
                    + lam * np.sum(np.sqrt(gy_x**2 + gx_x**2)))
            print("  iter {:4d}  cost = {:.6e}".format(it, cost))

    return x


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

def tv_reconstruction(
    sinogram,
    theta,
    lam=1e-2,
    n_iter=100,
    output_size=None,
    init="fbp",
    cuda=False,
    verbose=False,
):
    """
    TV-regularised tomographic reconstruction.

    High-level wrapper around :func:`chambolle_pock_tv` with an optional
    FBP warm-start and the same sinogram-axis convention used throughout
    toupy (detector pixels along axis 0).

    Parameters
    ----------
    sinogram : ndarray, shape (n_pixels, n_angles)
        Sinogram — same axis convention as :func:`mod_iradon`.
    theta : array_like, shape (n_angles,)
        Projection angles in degrees.
    lam : float, optional
        TV regularisation weight.  Default 1e-2.
    n_iter : int, optional
        Number of Chambolle-Pock iterations.  Default 100.
    output_size : int or None, optional
        Side length of the square reconstruction grid.  ``None`` infers
        from the sinogram (matches ``skimage.transform.radon``).
    init : {'fbp', 'zeros'}, optional
        Initialisation strategy.  ``'fbp'`` uses a Ram-Lak FBP as the
        starting point (recommended; faster convergence).  ``'zeros'``
        starts from the zero image.  Default ``'fbp'``.
    cuda : bool, optional
        Run on GPU via CuPy.  Default False.
    verbose : bool, optional
        Print cost every 10 iterations.  Default False.

    Returns
    -------
    recons : ndarray, shape (n, n)
        Reconstructed image.

    Notes
    -----
    The TV weight *lam* should be tuned to the noise level.  A
    cross-validation or L-curve approach is recommended for quantitative
    work.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.transform import radon
    >>> from toupy.simulation import phantom
    >>> from toupy.tomo import tv_reconstruction
    >>> img = phantom(256)
    >>> theta = np.linspace(0, 180, 180, endpoint=False)
    >>> sino = radon(img, theta=theta, circle=False)
    >>> recons = tv_reconstruction(sino, theta, lam=0.01, n_iter=50)
    """
    theta    = np.asarray(theta, dtype=float)
    sinogram = np.asarray(sinogram, dtype=float)

    return chambolle_pock_tv(
        sinogram, theta,
        lam=lam,
        n_iter=n_iter,
        output_size=output_size,
        cuda=cuda,
        verbose=verbose,
    )
