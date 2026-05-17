#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FOURIER SHELL CORRELATION
"""

# standard packages
import time
import concurrent.futures

# third party package
import h5py
from ..utils.plot_utils import plt
import numpy as np

# local packages
from ..utils.FFT_utils import fastfftn
from ..utils.funcutils import checkhostname
from ..utils import isnotebook
from ..tomo import tomo_recons

__all__ = ["compute_2tomograms", "compute_2tomograms_splitted", "split_dataset"]


def split_dataset(sinogram, theta):
    """
    Split the tomographic dataset in 2 datasets

    Parameters
    ----------
    sinogram : ndarray
        A 2-dimensional array containing the sinogram
    theta : ndarray
        A 1-dimensional array of thetas

    Returns
    -------
    sinogram1 : ndarray
        A 2-dimensional array containing the 1st sinogram
    sinogram2
        A 2-dimensional array containing the 2nd sinogram
    theta1 : ndarray
        A 1-dimensional array containing the 1st set of thetas
    theta2 : ndarray
        A 1-dimensional array containing the 2nd set of thetas
    """
    sinogram1 = sinogram[:, 0::2]
    theta1 = theta[0::2]
    sinogram2 = sinogram[:, 1::2]
    theta2 = theta[1::2]
    return sinogram1, sinogram2, theta1, theta2


def compute_2tomograms(sinogram, theta, **params):
    """
    Split the tomographic dataset in 2 datasets and
    compute 2 tomograms from them.

    Parameters
    ----------
    sinogram : ndarray
        A 2-dimensional array containing the sinogram
    theta : ndarray
        A 1-dimensional array of thetas

    Returns
    -------
    recon1 : ndarray
        A 2-dimensional array containing the 1st reconstruction
    recon2
        A 2-dimensional array containing the 2nd reconstruction
    """
    sino1, sino2, theta1, theta2 = split_dataset(sinogram, theta)
    return compute_2tomograms_splitted(sino1, sino2, theta1, theta2, **params)


def compute_2tomograms_splitted(sinogram1, sinogram2, theta1, theta2, **params):
    """
    Compute 2 tomograms from an already-split tomographic dataset.

    The two reconstructions are independent, so they are run concurrently
    using a ThreadPoolExecutor.  Because tomo_recons ultimately calls
    scipy.ndimage / iradon C extensions that release the GIL, two OS threads
    can execute true parallel C code with negligible overhead.

    If threading fails for any reason (e.g. the backend is not thread-safe),
    the function transparently falls back to the original sequential path.

    Parameters
    ----------
    sinogram1 : ndarray
        A 2-dimensional array containing sinogram 1
    sinogram2 : ndarray
        A 2-dimensional array containing sinogram 2
    theta1 : ndarray
        A 1-dimensional array of thetas for sinogram1
    theta2 : ndarray
        A 1-dimensional array of thetas for sinogram2

    Returns
    -------
    recon1 : ndarray
        A 2-dimensional array containing the 1st reconstruction
    recon2 : ndarray
        A 2-dimensional array containing the 2nd reconstruction
    """
    verbose = not isnotebook()

    def _recon(label, sinogram, theta):
        if verbose:
            print(f"Calculating slice {label}...")
        t0 = time.time()
        result = tomo_recons(sinogram, theta, **params)
        if verbose:
            print(f"Slice {label} done. Time elapsed: {time.time() - t0:.3f} s")
        return result

    # --- Parallel path (threads) ---
    # Two workers suffice: one per reconstruction.
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(_recon, 1, sinogram1, theta1)
            f2 = executor.submit(_recon, 2, sinogram2, theta2)
            recon1 = f1.result()
            recon2 = f2.result()
        return recon1, recon2

    except Exception as exc:
        # Fall back to sequential execution if threading caused any problem.
        print(f"Parallel reconstruction failed ({exc}); falling back to sequential.")
        recon1 = _recon(1, sinogram1, theta1)
        recon2 = _recon(2, sinogram2, theta2)
        return recon1, recon2
