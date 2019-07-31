#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from silx.opencl.backprojection import Backprojection
from silx import version

__all__ = [
    "create_circle",
    "mod_iradon",
    "mod_iradonSilx",
    "compute_filter",
    "backprojector",
    "reconsSART",
]


def create_circle(inputimg):
    """
    Create circle with apodized edges
    """
    bordercrop = 10
    nr, nc = inputimg.shape
    Y, X = np.indices((nr, nc))
    Y -= np.round(nr / 2).astype(int)
    X -= np.round(nc / 2).astype(int)
    R = np.sqrt(X ** 2 + Y ** 2)
    Rmax = np.round(np.max(R.shape) / 2.0)
    maskout = R < Rmax
    t = maskout * (1 - np.cos(np.pi * (R - Rmax - 2 * bordercrop) / bordercrop)) / 2.0
    t[np.where(R < (Rmax - bordercrop))] = 1
    return t


def compute_filter(nbins, filter_type="ram-lak", derivative=True, freqcutoff=1):

    # resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * nbins))))

    # Construct the Fourier filter
    f = fftfreq(projection_size_padded).reshape(-1, 1)  # digital frequency
    omega = 2 * np.pi * f  # angular frequency
    if derivative:
        fourier_filter = np.ones_like(f).astype(np.complex)  # differential filter
    else:
        fourier_filter = 2 * np.abs(f)  # ramp filter
    # fourier_filter[0]=3.9579e-4 # value from MATLAB
    if filter_type == "ram-lak":
        pass
    elif filter_type == "shepp-logan":
        # Start from first element to avoid divide by zero
        fourier_filter[1:] = fourier_filter[1:] * (
            np.sin(omega[1:] / (2 * freqcutoff)) / (omega[1:] / (2 * freqcutoff))
        )
        # fourier_filter[1:] = fourier_filter[1:] * (np.sin(omega[1:]/(2*np.pi*freqcutoff)) / (omega[1:]/(2*freqcutoff))) #factor pi
    elif filter_type == "cosine":
        fourier_filter[1:] *= np.cos(omega[1:] / (2 * freqcutoff))
    elif filter_type == "hamming":
        fourier_filter[1:] *= 0.54 + 0.46 * np.cos(omega[1:] / (freqcutoff))
    elif filter_type == "hann":
        fourier_filter[1:] *= (1 + np.cos(omega[1:] / (freqcutoff))) / 2
    elif filter_type is None:
        fourier_filter[:] = 1
    else:
        raise ValueError("Unknown filter: {}".format(fourier_filter))

    # Frequency cutoff
    # ~ fourier_filter[np.where(2*np.abs(f)>freqcutoff)]=0 #equivalent to below code
    # Get rid of unwanted frequencies
    fourier_filter[np.where(np.abs(omega) > np.pi * freqcutoff)] = 0

    # Change the filter to adapte to projection derivative
    if derivative:
        fourier_filter = np.sign(f) * fourier_filter / (1j * np.pi)

    return fourier_filter


def mod_iradon(
    radon_image,
    theta=None,
    output_size=None,
    filter_type="ram-lak",
    derivative=True,
    interpolation="linear",
    circle=False,
    freqcutoff=1,
):
    """
    Inverse radon transform.

    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    Parameters
    ----------
    radon_image (sinogram) : array_like, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different angle. The
        tomography rotation axis should lie at the pixel index
        ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : array_like, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    output_size : int
        Number of rows and columns in the reconstruction.
    filter : str, optional (default ramp)
        Filter used in frequency domain filtering. Ram-Lak filter used by default.
        Filters available: ram-lak, shepp-logan, cosine, hamming, hann.
        Assign None to use no filter.
    derivative : boolean (default True)
        If True, assumes that the radon_image contains the derivates of the
        projections.
    interpolation : str, optional (default 'linear')
        Interpolation method used in reconstruction. Methods available:
        'linear', 'nearest', and 'cubic' ('cubic' is slow).
    circle : boolean, optional
        Assume the reconstructed image is zero outside the inscribed circle.
        Also changes the default output_size to match the behaviour of
        ``radon`` called with ``circle=True``.
    freqcutoff : int (normalized to 1)
        Frequency cutoff of the filter. 1 means no cutoff

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    Notes
    -----
    It applies the Fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.

    """
    if radon_image.ndim != 2:
        raise ValueError("The input image must be 2-D")
    if theta is None:
        m, n = radon_image.shape
        theta = np.linspace(0, 180, n, endpoint=False)
    else:
        theta = np.asarray(theta)
    if len(theta) != radon_image.shape[1]:
        raise ValueError(
            "The given ``theta`` does not match the number of "
            "projections in ``radon_image``."
        )
    interpolation_types = ("linear", "nearest", "cubic")
    if not interpolation in interpolation_types:
        raise ValueError("Unknown interpolation: {}".format(interpolation))
    if not output_size:
        # If output size not specified, estimate from input radon image
        if circle:
            output_size = radon_image.shape[0]
        else:
            output_size = int(np.floor(np.sqrt((radon_image.shape[0]) ** 2 / 2.0)))

    # convertion degrees to radians
    th = (np.pi / 180.0) * theta

    # customized filter
    fourier_filter = compute_filter(
        radon_image.shape[0],
        filter_type=filter_type,
        derivative=derivative,
        freqcutoff=freqcutoff,
    )

    # padding image
    pad_width = ((0, fourier_filter.shape[0] - radon_image.shape[0]), (0, 0))
    img = np.pad(radon_image, pad_width, mode="constant", constant_values=0)

    # Apply filter in Fourier domain
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0))

    # Resize filtered image back to original size
    radon_filtered = radon_filtered[: radon_image.shape[0], :]
    reconstructed = np.zeros((output_size, output_size))
    # Determine the center of the projections (= center of sinogram)
    mid_index = radon_image.shape[0] // 2

    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2

    # Reconstruct image by interpolation
    for i in range(len(theta)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
        x = np.arange(radon_filtered.shape[0]) - mid_index
        if interpolation == "linear":
            backprojected = np.interp(t, x, radon_filtered[:, i], left=0, right=0)
        else:
            interpolant = interp1d(
                x,
                radon_filtered[:, i],
                kind=interpolation,
                bounds_error=False,
                fill_value=0,
            )
            backprojected = interpolant(t)
        reconstructed += backprojected
    if circle:
        radius = output_size // 2
        reconstruction_circle = (xpr ** 2 + ypr ** 2) <= radius ** 2
        reconstructed[~reconstruction_circle] = 0.0

    return reconstructed * np.pi / (2 * len(th))


B = None


def mod_iradonSilx(
    radon_image,
    theta=None,
    output_size=None,
    filter_type="ram-lak",
    derivative=True,
    interpolation="linear",
    circle=False,
    freqcutoff=1,
    use_numpy=False,
):
    """
    Inverse radon transform using Silx and OpenCL.

    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    Parameters
    ----------
    radon_image (sinogram) : array_like, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different angle. The
        tomography rotation axis should lie at the pixel index
        ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : array_like, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    output_size : int
        Number of rows and columns in the reconstruction.
    filter : str, optional (default ramp)
        Filter used in frequency domain filtering. Ram-Lak filter used by default.
        Filters available: ram-lak, shepp-logan, cosine, hamming, hann.
        Assign None to use no filter.
    derivative : boolean (default True)
        If True, assumes that the radon_image contains the derivates of the
        projections.
    interpolation : str, optional (default 'linear')
        Interpolation method used in reconstruction. Methods available:
        'linear', 'nearest', and 'cubic' ('cubic' is slow).
    circle : boolean, optional
        Assume the reconstructed image is zero outside the inscribed circle.
        Also changes the default output_size to match the behaviour of
        ``radon`` called with ``circle=True``.
    freqcutoff : int (normalized to 1)
        Frequency cutoff of the filter. 1 means no cutoff

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    Notes
    -----
    It applies the Fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.
    """
    if radon_image.ndim != 2:
        raise ValueError("The input image must be 2-D")
    if theta is None:
        m, n = radon_image.shape
        theta = np.linspace(0, 180, n, endpoint=False)
    else:
        theta = np.asarray(theta)
    # customized filter
    cust_filter = compute_filter(
        radon_image.shape[0],
        filter_type=filter_type,
        derivative=derivative,
        freqcutoff=freqcutoff,
    )
    # ~ print("Using Silx v{}".format(version))
    silx_version = float(version[2:])
    if silx_version < 10.0:
        B = Backprojection(radon_image.T.shape, angles=np.pi * (theta) / 180.0)
        # ~ print("Initialized OpenCL backprojector on {}".format(B.device))
        B.filter = cust_filter.ravel() / 2.0  # has to be divided by 2.
    else:
        if use_numpy:
            B = Backprojection(
                radon_image.T.shape,
                angles=np.pi * (theta) / 180.0,
                filter_name=filter_type,
                extra_options={"use_numpy_fft": True},
            )
        else:
            B = Backprojection(
                radon_image.T.shape,
                angles=np.pi * (theta) / 180.0,
                filter_name=filter_type,
            )
        # ~ print("Initialized OpenCL backprojector on {}".format(B.device))
        # from version 0.10.0, silx filtering uses R2C Fourier transforms
        cust_filter2 = cust_filter.ravel()[: B.sino_filter.dwidth_padded // 2 + 1]
        cust_filter2 = np.ascontiguousarray(cust_filter2 / 2.0, dtype=np.complex64)
        B.sino_filter.set_filter(cust_filter2)
    recons = B(radon_image.T)

    if circle:
        recons_circle = create_circle(recons)
        recons = recons * recons_circle
    return recons


def backprojector(sinogram, theta, **params):
    """
    Wrapper to choose between Forward Radon transform using Silx and
    OpenCL or standard reconstruction
    """
    if params["opencl"]:
        # using Silx backprojector
        # print("Using OpenCL")
        iradon = mod_iradonSilx
    else:
        # Not using Silx Projector (very slow)
        # print("Not using OpenCL")
        iradon = mod_iradon
    # reconstructing
    recons = iradon(
        sinogram,
        theta=theta,
        output_size=sinogram.shape[0],
        filter_type=params["filtertype"],
        derivative=params["derivatives"],
        circle=params["circle"],
        freqcutoff=params["filtertomo"],
    )
    return recons


def reconsSART(
    sinogram, theta, num_iter=2, FBPinitial_guess=True, relaxation_params=0.15, **params
):
    """
    Reconstruction with SART algorithm
    """
    theta = np.float64(theta)
    sinogram = np.float64(sinogram)
    circle = params["circle"]

    # actual reconstruction
    if FBPinitial_guess:
        print("Calculating the initial guess for SART using FBP")
        reconsFBP = backprojector(sinogram, theta, **params)
        reconsFBP = np.float64(reconsFBP)
        print("Done. Starting SART")

        # with initial guess
        reconsSART = iradon_sart(
            sinogram, theta=theta, image=reconsFBP, relaxation=relaxation_params
        )
    else:
        # without initial guess
        reconsSART = iradon_sart(sinogram, theta=theta, relaxation=relaxation_params)
    print("Starting iterative reconstruction:")
    for ii in range(iteration_num):
        print("Iteration {}".format(ii + 1))
        reconsSART = iradon_sart(
            sinogram, theta=theta, image=reconsSART, relaxation=relaxation_params
        )

    if circle:
        recons_circle = create_circle(inputimg)
        reconsSART = reconsSART * recons_circle

    return reconsSART
