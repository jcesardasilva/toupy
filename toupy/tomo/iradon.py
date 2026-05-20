#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filtered Back-Projection and SART reconstruction.

GPU path  : CuPy + a custom CUDA kernel
CPU path  : pure NumPy/SciPy, identical to the original mod_iradon.

The public API is unchanged:
    compute_angle_weights, compute_filter,
    mod_iradon, mod_iradon_cuda,
    backprojector, reconsSART
"""

# standard packages
import warnings
warnings.filterwarnings("ignore")

# third party packages
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from skimage.transform import iradon_sart

# local package
from ..utils import create_circle
from ..utils.plot_utils import isnotebook

# ---------------------------------------------------------------------------
# Optional CuPy import — graceful degradation to CPU if unavailable
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from cupy import RawKernel
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CuPy not found: GPU reconstruction unavailable, falling back to CPU.")

__all__ = [
    "compute_angle_weights",
    "compute_filter",
    "mod_iradon",
    "mod_iradon_cuda",
    "backprojector",
    "reconsSART",
]

# ---------------------------------------------------------------------------
# CUDA kernel for filtered back-projection
# ---------------------------------------------------------------------------
# Each CUDA thread handles one (row, col) output pixel.
# For every projection angle the kernel:
#   1. computes the detector coordinate t via dot-product with (cos θ, -sin θ)
#   2. does linear interpolation on the filtered sinogram row
#   3. accumulates into the output image
#
# The kernel is compiled once at module load time (if CuPy is present) and
# reused across all calls — compilation takes ~200 ms but happens only once
# per Python session.
# ---------------------------------------------------------------------------
_FBP_KERNEL_SRC = r"""
extern "C" __global__
void fbp_backproject(
    const float* __restrict__ sino,   // filtered sinogram (nbins x nangles), col-major
    const float* __restrict__ cos_th, // cos(theta) for each angle  (nangles,)
    const float* __restrict__ sin_th, // sin(theta) for each angle  (nangles,)
    float*       __restrict__ img,    // output image (output_size x output_size)
    const int    output_size,
    const int    nbins,
    const int    nangles,
    const int    mid_index            // detector centre index
)
{
    // pixel coordinates
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= output_size || row >= output_size) return;

    const int half = output_size / 2;
    const float xpr = (float)(col - half);   // centred x
    const float ypr = (float)(row - half);   // centred y

    float acc = 0.0f;
    for (int a = 0; a < nangles; ++a) {
        // detector coordinate of this pixel for angle a
        float t = ypr * cos_th[a] - xpr * sin_th[a] + (float)mid_index;

        // linear interpolation — clamp to valid range
        int   t0 = (int)floorf(t);
        float dt = t - (float)t0;
        float v0 = (t0 >= 0 && t0 < nbins)     ? sino[a * nbins + t0]     : 0.0f;
        float v1 = (t0+1 >= 0 && t0+1 < nbins) ? sino[a * nbins + (t0+1)] : 0.0f;
        acc += v0 * (1.0f - dt) + v1 * dt;
    }
    img[row * output_size + col] = acc;
}
"""

# Compile once
if CUDA_AVAILABLE:
    _fbp_kernel = RawKernel(_FBP_KERNEL_SRC, "fbp_backproject")


# ---------------------------------------------------------------------------
# CUDA kernel for forward projection (Radon transform)
# ---------------------------------------------------------------------------
# Each thread computes one (detector_bin, angle) sinogram entry.
# It walks along the projection ray and accumulates image values
# using bilinear interpolation.
# ---------------------------------------------------------------------------
_RADON_KERNEL_SRC = r"""
extern "C" __global__
void radon_project(
    const float* __restrict__ img,    // input image (N x N), row-major
    float*       __restrict__ sino,   // output sinogram (nbins x nangles), col-major
    const float* __restrict__ cos_th, // cos(theta) for each angle
    const float* __restrict__ sin_th, // sin(theta) for each angle
    const int    N,                   // image side length
    const int    nbins,               // number of detector bins
    const int    nangles,
    const int    nsteps               // number of integration steps along ray
)
{
    const int bin   = blockIdx.x * blockDim.x + threadIdx.x;
    const int angle = blockIdx.y * blockDim.y + threadIdx.y;
    if (bin >= nbins || angle >= nangles) return;

    const float half_img  = (float)(N - 1) * 0.5f;
    const float half_bins = (float)(nbins - 1) * 0.5f;
    const float step      = 1.0f;            // step size in pixels
    const float t         = (float)bin - half_bins;

    // ray direction (perpendicular to projection direction)
    const float dx =  cos_th[angle];
    const float dy = -sin_th[angle];
    // ray origin: offset along detector axis
    float ox = t * sin_th[angle];  // projection direction = (sin θ, cos θ)
    float oy = t * cos_th[angle];

    float acc = 0.0f;
    const float half_steps = (float)(nsteps - 1) * 0.5f;
    for (int s = 0; s < nsteps; ++s) {
        float sx = ox + (s - half_steps) * dx + half_img;
        float sy = oy + (s - half_steps) * dy + half_img;

        // bilinear interpolation
        int   x0 = (int)floorf(sx), y0 = (int)floorf(sy);
        float fx = sx - x0,         fy = sy - y0;
        float v  = 0.0f;
        if (x0 >= 0 && x0 < N && y0 >= 0 && y0 < N)
            v += img[y0 * N + x0] * (1.0f - fx) * (1.0f - fy);
        if (x0+1 < N && y0 >= 0 && y0 < N)
            v += img[y0 * N + (x0+1)] * fx * (1.0f - fy);
        if (x0 >= 0 && x0 < N && y0+1 < N)
            v += img[(y0+1) * N + x0] * (1.0f - fx) * fy;
        if (x0+1 < N && y0+1 < N)
            v += img[(y0+1) * N + (x0+1)] * fx * fy;
        acc += v;
    }
    sino[angle * nbins + bin] = acc * step;
}
"""

if CUDA_AVAILABLE:
    _radon_kernel = RawKernel(_RADON_KERNEL_SRC, "radon_project")


# ---------------------------------------------------------------------------
# Helper: FFT-based filtering on GPU
# ---------------------------------------------------------------------------
def _filter_sinogram_gpu(sinogram_gpu, filter_type, derivatives, freqcutoff):
    """
    Apply the ramp (or other) filter to a sinogram already on the GPU.

    Parameters
    ----------
    sinogram_gpu : cupy.ndarray  shape (nbins, nangles)
    filter_type, derivatives, freqcutoff : same semantics as compute_filter

    Returns
    -------
    filtered : cupy.ndarray  shape (nbins, nangles)  float32
    """
    nbins = sinogram_gpu.shape[0]
    # compute filter on CPU, transfer once
    h = compute_filter(nbins, filter_type=filter_type,
                       derivatives=derivatives, freqcutoff=freqcutoff)
    pad = h.shape[0]

    # pad sinogram
    sino_pad = cp.zeros((pad, sinogram_gpu.shape[1]), dtype=cp.complex64)
    sino_pad[:nbins] = sinogram_gpu.astype(cp.complex64)

    # transfer filter to GPU and broadcast over angles
    h_gpu = cp.asarray(h.astype(np.complex64))          # (pad, 1)
    filtered = cp.fft.ifft(cp.fft.fft(sino_pad, axis=0) * h_gpu, axis=0)
    return cp.real(filtered[:nbins]).astype(cp.float32)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_angle_weights(theta):
    """
    Compute per-angle weights for non-equally-spaced angular distributions.

    Parameters
    ----------
    theta : ndarray
        Angles in degrees.

    Returns
    -------
    weights : ndarray
        Weight for each angle, proportional to the angular distance to its
        neighbours.  Forked from odtbrain.util.compute_angle_weights_1d.
    """
    theta = theta.flatten() - theta.min()
    sortargs  = np.argsort(theta)
    sorttheta = theta[sortargs]
    diff_theta   = (np.roll(sorttheta, -1) - np.roll(sorttheta, 1)) % 180
    weights      = diff_theta / np.sum(diff_theta) * diff_theta.size
    unsortweights = np.zeros_like(weights)
    unsortweights[sortargs] = weights
    return unsortweights


def compute_filter(nbins, filter_type="ram-lak", derivatives=False, freqcutoff=1):
    """
    Compute the frequency-domain filter for FBP reconstruction.

    Parameters
    ----------
    nbins : int
        Number of detector bins (sinogram rows).
    filter_type : str, optional
        One of ``ram-lak``, ``shepp-logan``, ``cosine``, ``hamming``, ``hann``.
        Default is ``ram-lak``.
    derivatives : bool, optional
        Use a Hilbert filter suited for derivative projections.  Default False.
    freqcutoff : float, optional
        Normalised frequency cutoff in [0, 1].  Default 1 (no cutoff).

    Returns
    -------
    fourier_filter : ndarray  shape (projection_size_padded, 1)  complex64
    """
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * nbins))))
    f     = fftfreq(projection_size_padded).reshape(-1, 1)
    omega = 2 * np.pi * f
    fourier_filter = (
        np.ones_like(f, dtype=np.complex64) if derivatives
        else 2 * np.abs(f).astype(np.complex64)
    )
    if filter_type == "ram-lak":
        pass
    elif filter_type == "shepp-logan":
        fourier_filter[1:] *= np.sinc(omega[1:] / (2 * freqcutoff * np.pi))
    elif filter_type == "cosine":
        fourier_filter[1:] *= np.cos(omega[1:] / (2 * freqcutoff))
    elif filter_type == "hamming":
        fourier_filter[1:] *= 0.54 + 0.46 * np.cos(omega[1:] / freqcutoff)
    elif filter_type == "hann":
        fourier_filter[1:] *= (1 + np.cos(omega[1:] / freqcutoff)) / 2
    elif filter_type is None:
        fourier_filter[:] = 1
    else:
        raise ValueError("Unknown filter: {}".format(filter_type))

    fourier_filter[np.abs(omega) > np.pi * freqcutoff] = 0
    if derivatives:
        fourier_filter = np.sign(f) * fourier_filter / (1j * np.pi)
    return fourier_filter


def mod_iradon(
    radon_image,
    theta=None,
    output_size=None,
    filter_type="ram-lak",
    derivatives=False,
    interpolation="linear",
    circle=False,
    freqcutoff=1,
):
    """
    Inverse Radon transform — CPU path (pure NumPy/SciPy).

    Reconstruct an image from its Radon transform using filtered back-projection.

    Parameters
    ----------
    radon_image : ndarray  shape (nbins, nangles)
        Sinogram. Each column corresponds to one projection angle.
    theta : ndarray, optional
        Projection angles in degrees.  Defaults to ``nangles`` evenly-spaced
        values in [0, 180).
    output_size : int, optional
        Side length of the square output image.
    filter_type : str, optional
        See :func:`compute_filter`.
    derivatives : bool, optional
        Set True if the sinogram contains projection derivatives.
    interpolation : str, optional
        ``linear`` (default), ``nearest``, or ``cubic``.
    circle : bool, optional
        Zero pixels outside the inscribed circle.
    freqcutoff : float, optional
        Normalised frequency cutoff.

    Returns
    -------
    reconstructed : ndarray  shape (output_size, output_size)
    """
    if radon_image.ndim != 2:
        raise ValueError("The input image must be 2-D")
    if theta is None:
        m, n = radon_image.shape
        theta = np.linspace(0, 180, n, endpoint=False)
    else:
        theta = np.asarray(theta)
    if len(theta) != radon_image.shape[1]:
        raise ValueError("theta length does not match the number of projections.")
    if interpolation not in ("linear", "nearest", "cubic"):
        raise ValueError("Unknown interpolation: {}".format(interpolation))
    if not output_size:
        output_size = (radon_image.shape[0] if circle
                       else int(np.floor(np.sqrt(radon_image.shape[0] ** 2 / 2.0))))

    th = np.deg2rad(theta)
    fourier_filter = compute_filter(radon_image.shape[0], filter_type=filter_type,
                                    derivatives=derivatives, freqcutoff=freqcutoff)
    pad_width = ((0, fourier_filter.shape[0] - radon_image.shape[0]), (0, 0))
    img = np.pad(radon_image, pad_width, mode="constant", constant_values=0)
    projection    = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0))[:radon_image.shape[0], :]

    reconstructed = np.zeros((output_size, output_size))
    mid_index = radon_image.shape[0] // 2
    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - output_size // 2
    ypr = Y - output_size // 2
    x   = np.arange(radon_filtered.shape[0]) - mid_index

    for i in range(len(theta)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
        if interpolation == "linear":
            backprojected = np.interp(t, x, radon_filtered[:, i], left=0, right=0)
        else:
            backprojected = interp1d(x, radon_filtered[:, i], kind=interpolation,
                                     bounds_error=False, fill_value=0)(t)
        reconstructed += backprojected

    if circle:
        radius = output_size // 2
        mask = (xpr ** 2 + ypr ** 2) <= radius ** 2
        reconstructed[~mask] = 0.0

    return reconstructed * np.pi / (2 * len(th))


def mod_iradon_cuda(
    radon_image,
    theta=None,
    output_size=None,
    filter_type="ram-lak",
    derivatives=False,
    circle=False,
    freqcutoff=1,
):
    """
    Inverse Radon transform — CUDA/GPU path.

    Falls back to :func:`mod_iradon` automatically if CuPy is unavailable.

    Parameters
    ----------
    radon_image : ndarray  shape (nbins, nangles)
        Sinogram on CPU.  Converted to float32 internally.
    theta : ndarray, optional
        Projection angles in degrees.
    output_size : int, optional
        Side length of the square output image.
    filter_type : str, optional
        See :func:`compute_filter`.
    derivatives : bool, optional
        Set True if the sinogram contains projection derivatives.
    circle : bool, optional
        Zero pixels outside the inscribed circle.
    freqcutoff : float, optional
        Normalised frequency cutoff.

    Returns
    -------
    reconstructed : ndarray  shape (output_size, output_size)  float32
    """
    if not CUDA_AVAILABLE:
        warnings.warn("CuPy unavailable — falling back to CPU reconstruction.")
        return mod_iradon(radon_image, theta=theta, output_size=output_size,
                          filter_type=filter_type, derivatives=derivatives,
                          circle=circle, freqcutoff=freqcutoff)

    if radon_image.ndim != 2:
        raise ValueError("The input image must be 2-D")
    if theta is None:
        m, n = radon_image.shape
        theta = np.linspace(0, 180, n, endpoint=False)
    else:
        theta = np.asarray(theta)
    if len(theta) != radon_image.shape[1]:
        raise ValueError("theta length does not match the number of projections.")
    if not output_size:
        output_size = (radon_image.shape[0] if circle
                       else int(np.floor(np.sqrt(radon_image.shape[0] ** 2 / 2.0))))

    th       = np.deg2rad(theta).astype(np.float32)
    cos_th   = np.cos(th).astype(np.float32)
    sin_th   = np.sin(th).astype(np.float32)
    nbins    = radon_image.shape[0]
    nangles  = radon_image.shape[1]
    mid_index = nbins // 2

    # --- Upload sinogram and apply filter on GPU ---
    sino_gpu = cp.asarray(radon_image.astype(np.float32))       # (nbins, nangles)
    sino_filt = _filter_sinogram_gpu(sino_gpu, filter_type, derivatives, freqcutoff)
    # Make column-major layout so the kernel can stride over angles efficiently
    sino_filt = cp.ascontiguousarray(sino_filt.T).ravel()        # (nangles * nbins,)

    # --- Allocate output ---
    img_gpu   = cp.zeros(output_size * output_size, dtype=cp.float32)
    cos_th_gpu = cp.asarray(cos_th)
    sin_th_gpu = cp.asarray(sin_th)

    # --- Launch kernel ---
    block = (16, 16, 1)
    grid  = (
        int(np.ceil(output_size / block[0])),
        int(np.ceil(output_size / block[1])),
        1,
    )
    _fbp_kernel(
        grid, block,
        (sino_filt, cos_th_gpu, sin_th_gpu, img_gpu,
         np.int32(output_size), np.int32(nbins),
         np.int32(nangles),     np.int32(mid_index)),
    )
    cp.cuda.stream.get_current_stream().synchronize()

    # --- Retrieve and scale ---
    reconstructed = cp.asnumpy(img_gpu).reshape(output_size, output_size)
    reconstructed *= np.pi / (2 * nangles)

    if circle:
        half = output_size // 2
        Y, X = np.ogrid[-half:output_size - half, -half:output_size - half]
        mask = X ** 2 + Y ** 2 <= half ** 2
        reconstructed[~mask] = 0.0

    return reconstructed


def backprojector(sinogram, theta, **params):
    """
    Wrapper that dispatches to the GPU or CPU FBP implementation.

    Parameters
    ----------
    sinogram : ndarray  shape (nbins, nangles)
    theta : ndarray
    params : dict
        ``params["opencl"]`` — repurposed as ``params["cuda"]``: set True to
        use the CUDA path.  The key ``opencl`` is still accepted as an alias
        for backwards compatibility.
        All other keys are forwarded to the chosen implementation.

    Returns
    -------
    recons : ndarray  shape (output_size, output_size)
    """
    params.setdefault("weight_angles", False)
    # accept both "cuda" and the legacy "opencl" key
    use_cuda = params.get("cuda", params.get("opencl", False))

    if params["weight_angles"]:
        weights  = compute_angle_weights(theta).reshape(1, -1)
        sinogram = sinogram * weights

    shared = dict(
        theta              = theta,
        output_size        = sinogram.shape[0],
        filter_type        = params["filtertype"],
        derivatives        = params["derivatives"],
        circle             = params["circle"],
        freqcutoff         = params["freqcutoff"],
    )

    if use_cuda:
        return mod_iradon_cuda(sinogram, **shared)
    else:
        return mod_iradon(sinogram, **shared)


def reconsSART(
    sinogram, theta, num_iter=2, FBPinitial_guess=True, relaxation_params=0.15, **params
):
    """
    Tomographic reconstruction with the SART algorithm.

    Parameters
    ----------
    sinogram : ndarray  shape (nbins, nangles)
    theta : ndarray
    num_iter : int, optional
        Number of SART iterations.  Default 2.
    FBPinitial_guess : bool, optional
        Seed SART with an FBP reconstruction.  Default True.
    relaxation_params : float, optional
        SART relaxation parameter.  Default 0.15.

    Returns
    -------
    recons_sart : ndarray
    """
    theta    = np.float64(theta)
    sinogram = np.float64(sinogram)
    circle   = params["circle"]

    if params.get("weight_angles", False):
        weights  = compute_angle_weights(theta).reshape(1, -1)
        sinogram = sinogram * weights

    if FBPinitial_guess:
        print("Calculating the initial guess for SART using FBP")
        reconsFBP = backprojector(sinogram, theta, **params)
        reconsFBP = np.float64(reconsFBP)
        print("Done. Starting SART")
        recons_sart = iradon_sart(sinogram, theta=theta,
                                  image=reconsFBP, relaxation=relaxation_params)
    else:
        recons_sart = iradon_sart(sinogram, theta=theta, relaxation=relaxation_params)

    print("Starting iterative reconstruction:")
    for ii in range(num_iter):
        print("Iteration {}".format(ii + 1))
        recons_sart = iradon_sart(sinogram, theta=theta,
                                  image=recons_sart, relaxation=relaxation_params)

    if circle:
        recons_circle = create_circle(recons_sart)
        recons_sart   = recons_sart * recons_circle

    return recons_sart
