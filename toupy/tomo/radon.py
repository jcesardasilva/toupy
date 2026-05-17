#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Forward Radon transform (projection).

GPU path  : CuPy + the CUDA kernel defined in iradon.py — no Silx / PyOpenCL.
CPU path  : skimage.transform.radon (unchanged fallback).

Public API is unchanged:  projector()
"""

# third party packages
import numpy as np
from skimage.transform import radon

# local libraries
from ..utils.plot_utils import isnotebook
from .iradon import CUDA_AVAILABLE

if CUDA_AVAILABLE:
    import cupy as cp
    from .iradon import _radon_kernel

__all__ = ["radon_cuda", "projector"]


def radon_cuda(recons, theta):
    """
    Forward Radon transform — CUDA/GPU path.

    Replaces :func:`radonSilx` without any Silx/PyOpenCL dependency.
    Falls back to ``skimage.transform.radon`` automatically if CuPy is
    unavailable.

    Parameters
    ----------
    recons : ndarray  shape (N, N)
        The tomographic slice to project.  Converted to float32 internally.
    theta : ndarray
        Projection angles in degrees.

    Returns
    -------
    sinogram : ndarray  shape (N, nangles)
        The computed sinogram, in the same layout as skimage's radon output
        (each *column* = one projection).
    """
    if not CUDA_AVAILABLE:
        return np.squeeze(radon(recons, theta, circle=True))

    N       = recons.shape[0]
    nangles = len(theta)
    nbins   = N                          # detector bins == image side length
    nsteps  = int(np.ceil(N * np.sqrt(2)))  # enough steps to cross the diagonal

    th       = np.deg2rad(np.asarray(theta, dtype=np.float32))
    cos_th   = np.cos(th).astype(np.float32)
    sin_th   = np.sin(th).astype(np.float32)

    # Upload image and trig tables
    img_gpu     = cp.asarray(recons.astype(np.float32))
    cos_th_gpu  = cp.asarray(cos_th)
    sin_th_gpu  = cp.asarray(sin_th)
    sino_gpu    = cp.zeros(nbins * nangles, dtype=cp.float32)

    # Launch kernel — each thread = one (bin, angle) sinogram entry
    block = (16, 16, 1)
    grid  = (
        int(np.ceil(nbins   / block[0])),
        int(np.ceil(nangles / block[1])),
        1,
    )
    _radon_kernel(
        grid, block,
        (img_gpu.ravel(), sino_gpu,
         cos_th_gpu, sin_th_gpu,
         np.int32(N), np.int32(nbins), np.int32(nangles), np.int32(nsteps)),
    )
    cp.cuda.stream.get_current_stream().synchronize()

    # Bring back to CPU; reshape to (nbins, nangles) — same layout as skimage
    sinogram = cp.asnumpy(sino_gpu).reshape(nangles, nbins).T
    return np.squeeze(sinogram)


def projector(recons, theta, **params):
    """
    Wrapper to choose between CUDA and CPU forward Radon transform.

    Parameters
    ----------
    recons : ndarray  shape (N, N)
        Tomographic slice to project.
    theta : ndarray
        Projection angles in degrees.
    params : dict
        ``params["cuda"]``  — True → use GPU path.
        ``params["opencl"]`` — accepted as alias for backwards compatibility.

    Returns
    -------
    sinogram : ndarray  shape (N, nangles)
    """
    use_cuda = params.get("cuda", params.get("opencl", False))

    if use_cuda:
        return radon_cuda(recons, theta)
    else:
        return np.squeeze(radon(recons, theta, circle=True))
