#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard packages
import time

# third party package
from ..utils.plot_utils import plt
import numpy as np
from ..utils import tqdm

# local packages
from ..utils import display_slice
from .iradon import backprojector, reconsSART
from ..restoration import derivatives_sino
from .fdk import fdk_reconstruct          # cone-beam FDK
from .geometry import ConeBeamGeometry    # cone-beam geometry descriptor

__all__ = ["full_tomo_recons", "tomo_recons", "fdk_tomo_recons"]


def tomo_recons(sinogram, theta, **params):
    r"""
    Wrapper to select tomographic algorithm

    Parameters
    ----------
    sinogram : ndarray
        A 2-dimensional array containing the sinogram
    theta : ndarray
        A 1-dimensional array of thetas
    params : dict
        Dictionary containing additional parameters
    params["algorithm"] : str
        Choice of algorithm. Two algorithm implemented: "FBP" and "SART"
    params["slicenum"] : int
        Slice number
    params["filtertype"] : str
        Name of the filter to be applied in frequency domain filtering.
        The options are: `ram-lak`, `shepp-logan`, `cosine`, `hamming`,
        `hann`. Assign None to use no filter.
    params["freqcutoff"] : float
        Frequency cutoff (between 0 and 1)
    params["circle"] : bool
        Multiply the reconstructed slice by a circle to remove borders
    params["weight_angles"] : bool
        If `True`, weights each projection with a factor proportional
        to the angular distance between the neighboring
        projections.

        .. math::
            \Delta \phi_0 \longmapsto \Delta \phi_j = \frac{\phi_{j+1} - \phi_{j-1}}{2}

    params["derivatives"] : bool
        If the projections are derivatives. Only for FBP.
    params["calc_derivatives"] : bool
        Calculate derivatives of the sinogram if not done yet.
    params["opencl"] : bool
        Implement the tomographic reconstruction in opencl as implemented
        in Silx
    params["autosave"] : bool
        Save the data at the end without asking
    params["vmin_plot"] : float
        Minimum value for the gray level at each display
    params["vmax_plot"] : float
        Maximum value for the gray level at each display
    params["colormap"] : str
        Colormap
    params["showrecons"] : bool
        If to show the reconstructed slices

    Returns
    -------
    recons : ndarray
        A 2-dimensional array containing the reconstructed slice.
    """
    if params["algorithm"] == "FBP":
        if params["calc_derivatives"]:
            print("Calculating the derivatives of the sinogram")
            sinogram = derivatives_sino(sinogram, shift_method="fourier")
        recons = backprojector(sinogram, theta, **params)
    elif params["algorithm"] == "SART":
        if params["calc_derivatives"]:
            raise ValueError("Reconstruction from derivatives only works with FBP")
        recons = reconsSART(sinogram, theta, **params)
    return recons


def full_tomo_recons(input_stack, theta, **params):
    """
    Full tomographic reconstruction

    Parameters
    ----------
    input_stack : ndarray
        A 3-dimensional array containing the stack of projections.
        The order should be ``[projection_num, row, column]``
    theta : ndarray
        A 1-dimensional array of thetas
    params : dict
        Dictionary containing additional parameters
    params["algorithm"] : str
        Choice of algorithm. Two algorithm implemented: "FBP" and "SART"
    params["slicenum"] : int
        Slice number
    params["filtertype"] : str
        Filter to use for FBP
    params["freqcutoff"] : float
        Frequency cutoff (between 0 and 1)
    params["circle"] : bool
        Multiply the reconstructed slice by a circle to remove borders
    params["derivatives"] : bool
        If the projections are derivatives. Only for FBP.
    params["calc_derivatives"] : bool
        Calculate derivatives of the sinogram if not done yet.
    params["opencl"] : bool
        Implement the tomographic reconstruction in CUDA
    params["autosave"] : bool
        Save the data at the end without asking
    params["vmin_plot"] : float
        Minimum value for the gray level at each display
    params["vmax_plot"] : float
        Maximum value for the gray level at each display
    params["colormap"] : str
        Colormap
    params["showrecons"] : bool
        If to show the reconstructed slices


    Returns
    -------
    tomogram : ndarray
        A 3-dimensional array containing the full reconstructed tomogram.
    """

    print("Calculating a slice for display")
    slicenum = params["slicenum"]

    sinogram0 = np.transpose(input_stack[:, slicenum, :])

    # calculating one slice for estimating sizes
    t0 = time.time()
    tomogram0 = tomo_recons(sinogram0, theta, **params)
    nr, nc = tomogram0.shape  # size of the slices
    print("Calculation done. Time elapsed: {:.02f} s".format(time.time() - t0))

    # display one slice
    plt.close("all")
    display_slice(
        tomogram0,
        colormap=params["colormap"],
        vmin=params["cliplow"],
        vmax=params["cliphigh"],
    )

    # estimate the total number of slices
    nslices = input_stack.shape[1]
    print(
        "The total number of slices is {}.  Starting full reconstruction …".format(
            nslices
        ),
        flush=True,
    )
    print(
        "Tip: interrupt the kernel at any time to abort.  "
        "Adjust params['vmin_plot'] / params['vmax_plot'] if the preview "
        "colour scale needs changing before re-running.",
        flush=True,
    )

    plt.close("all")
    tomogram = np.zeros((nslices, nr, nc), dtype=np.float32)
    for ii in tqdm(range(nslices), desc="Reconstructing slices"):
        sinogram = np.transpose(input_stack[:, ii, :])
        tomogram[ii] = tomo_recons(sinogram, theta, **params)

    return tomogram


# ---------------------------------------------------------------------------
# Cone-beam high-level wrapper
# ---------------------------------------------------------------------------

def fdk_tomo_recons(
    projections,
    geometry: ConeBeamGeometry,
    **params,
):
    """
    High-level FDK cone-beam reconstruction wrapper.

    Mirrors :func:`tomo_recons` for cone-beam geometry.  Validates the
    geometry, applies optional pre-processing, calls the three-step FDK
    pipeline, and optionally displays the central reconstructed slice.

    Parameters
    ----------
    projections : ndarray, shape (n_angles, n_v, n_u)
        Flat-field-corrected, Beer-Lambert log-normalised cone-beam
        projections.
    geometry : ConeBeamGeometry
        Validated acquisition geometry.
    **params
        Algorithm parameters.  Recognised keys:

        filtertype : str
            Ramp-filter window (forwarded to :func:`~toupy.tomo.fdk.fdk_filter`).
            Default ``'ram-lak'``.
        freqcutoff : float
            Normalised frequency cutoff in ``(0, 1]``.  Default ``1``.
        output_size : int or None
            Transaxial reconstruction grid side length.
            ``None`` uses ``geometry.n_u``.
        cuda : bool
            Use CuPy GPU path.  Default ``False``.
        circle : bool
            Multiply the reconstructed volume by a cylindrical mask to
            suppress border artefacts.  Default ``False``.
        showrecons : bool
            Display the central axial slice after reconstruction.
            Default ``False``.
        colormap : str
            Colormap for display.  Default ``'bone'``.
        vmin_plot : float or None
            Display window minimum.
        vmax_plot : float or None
            Display window maximum.
        autosave : bool
            Save the volume to disk without prompting.  Default ``False``.

    Returns
    -------
    volume : ndarray, shape (n_v, N, N)
        Reconstructed 3-D volume.

    Raises
    ------
    ValueError
        If ``projections.ndim != 3`` or shape is inconsistent with
        ``geometry``.

    See Also
    --------
    tomo_recons : Parallel-beam (2-D) slice reconstruction wrapper.
    full_tomo_recons : Slice-by-slice parallel-beam full reconstruction.
    toupy.tomo.fdk.fdk_reconstruct : Bare FDK pipeline (no display/save).
    """
    from ..utils import create_circle

    filtertype  = params.get("filtertype",  "ram-lak")
    freqcutoff  = params.get("freqcutoff",  1.0)
    output_size = params.get("output_size", None)
    cuda        = params.get("cuda",        False)
    circle      = params.get("circle",      False)
    showrecons  = params.get("showrecons",  False)
    colormap    = params.get("colormap",    "bone")
    vmin_plot   = params.get("vmin_plot",   None)
    vmax_plot   = params.get("vmax_plot",   None)

    geometry.validate()

    if projections.ndim != 3:
        raise ValueError(
            "projections must be 3-D (n_angles, n_v, n_u), got ndim={}.".format(
                projections.ndim
            )
        )

    volume = fdk_reconstruct(
        projections, geometry,
        filter_type=filtertype,
        freqcutoff=freqcutoff,
        output_size=output_size,
        cuda=cuda,
    )

    if circle:
        # Apply cylindrical mask to each axial slice
        central = volume[volume.shape[0] // 2]
        mask = create_circle(central)
        volume = volume * mask[np.newaxis, :, :]

    if showrecons:
        central = volume[volume.shape[0] // 2]
        display_slice(central, colormap=colormap, vmin=vmin_plot, vmax=vmax_plot)

    return volume
