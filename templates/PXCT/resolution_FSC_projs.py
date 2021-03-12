#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Template for resolution estimate of projections using Fourier Ring correlation
"""

# standard packages
import os
import time

# import of third party packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.restoration import unwrap_phase
import tkinter
import tkinter.filedialog as tkFileDialog

# import of local packages
from toupy.utils import normalize_array, cropROI
from toupy.io import read_recon
from toupy.resolution import FSCPlot
from toupy.restoration import rmphaseramp
from toupy.registration import register_2Darrays

# initializing params
params = dict()

# Edit section
# =========================
# Edit session
# =========================
params["apod_width"] = 100  # apodization width in pixels
params["thick_ring"] = 8  # number of pixel to average each FRC ring
params["crop"] = [200, -370, 300, -300]  # cropping [top,bottom,left,right]
params["vmin_plot"] = None
params["vmax_plot"] = -0.5e-4  # None
params["colormap"] = "bone"  # colormap to show images
params["unwrap"] = False  # unwrap the phase
params["flip2ndimage"] = False  # flip the 2nd image
params["normalizeimage"] = False  # normalize the images
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # open GUI to choose file
    root = tkinter.Tk()
    root.withdraw()
    print("You need to load two files for the FSC evalution")
    print("Please, load the first file...")
    pathfilename1 = tkFileDialog.askopenfilename(
        initialdir=".", title="Please, load the first file..."
    )
    print("File 1: {}".format(pathfilename1))
    print("Please, load the second file...")
    pathfilename2 = tkFileDialog.askopenfilename(
        initialdir=".", title="Please, load the second file..."
    )
    print("File 2: {}".format(pathfilename2))

    # Read the files
    fileprefix, fileext = os.path.splitext(pathfilename1)

    if fileext == ".ptyr" or fileext == ".cxi":
        data1, probe1, pixelsize1, energy = read_recon(pathfilename1)  # file1
        data2, probe2, pixelsize2, energy = read_recon(pathfilename2)  # file2
    elif fileext == ".edf":
        data1, pixelsize1, energy, nvue = read_recon(pathfilename1)  # file1
        data2, pixelsize2, energy, nvue = read_recon(pathfilename2)  # file2

    if params["flip2ndimage"]:  # flip one of the images
        print("Flipping 2nd image")
        data2 = np.fliplr(data2)

    print("the pixelsize of data1 is {:0.02f} nm".format(pixelsize1[0] * 1e9))
    print("the pixelsize of data2 is {:0.02f} nm".format(pixelsize2[0] * 1e9))

    # cropping the image to an useful area
    if params["crop"] is not None:
        if params["crop"] != 0:
            img1 = cropROI(data1, roi=params["crop"])
            img2 = cropROI(data2, roi=params["crop"])

    # remove phase ramp
    print("Removing the ramp")
    image1 = rmphaseramp(img1, weight=None, return_phaseramp=False)
    image2 = rmphaseramp(img2, weight=None, return_phaseramp=False)

    if params["unwrap"]:
        # remove also wrapping
        print("Unwrapping the phase image1")
        image1 = unwrap_phase(np.angle(image1))
        print("Unwrapping the phase image2")
        image2 = unwrap_phase(np.angle(image2))
    else:
        image1 = np.angle(image1)
        image2 = np.angle(image2)

    if params["normalizeimage"]:
        image1 = normalize_array(image1)
        image2 = normalize_array(image2)

    # Display the images
    plt.close("all")
    fig, (ax1, ax2, ax3) = plt.subplots(num=1, ncols=3)
    ax1.imshow(image1, interpolation="none", cmap="bone")
    ax1.set_axis_off()
    ax1.set_title("Image 1 (ref.)")
    ax2.imshow(image2, interpolation="none", cmap="bone")
    ax2.set_axis_off()
    ax2.set_title("Image 2")
    # View the output of a cross-correlation
    # TODO: consider to use pyfftw instead: 
    image_product = fft2(image1) * fft2(image2).conj()
    cc_image = fftshift(ifft2(image_product))
    ax3.imshow(cc_image.real)
    # ax3.set_axis_off()
    ax3.set_title("Cross-correlation")
    plt.show(block=False)

    # align the two images
    shift, diffphase, offset_image2 = register_2Darrays(image1, image2)

    # cropping the images beyond the shift amplitude
    regfsc = np.ceil(np.abs(shift)).astype(np.int)
    if regfsc[0] != 0 and regfsc[1] != 0:
        image1FSC = image1[regfsc[0] : -regfsc[0], regfsc[1] : -regfsc[1]]
        offset_image2FSC = offset_image2[regfsc[0] : -regfsc[0], regfsc[1] : -regfsc[1]]

    # display aligned images
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(num=1, ncols=2)
    ax1.imshow(image1FSC, interpolation="none", cmap="bone")
    ax1.set_axis_off()
    ax1.set_title("Image1 (ref.)")
    ax2.imshow(offset_image2FSC.real, interpolation="none", cmap="bone")
    ax2.set_axis_off()
    ax2.set_title("Offset corrected image2")
    plt.show(block=False)

    # estimate the resolution
    print("Estimating the resolution by FSC. Press <Enter> to continue")
    a = input()
    plt.close("all")

    startfsc = time.time()
    # transv_apod=params['transv_apod'],axial_apod=params['axial_apod'])
    FSC2D = FSCPlot(
        image1FSC,
        offset_image2FSC.real,
        "onebit",
        params["thick_ring"],
        apod_width=params["apod_width"],
    )
    normfreqs, T, FSC2Dcurve = FSC2D.plot()
    endfsc = time.time()
    print("Time elapsed: {:g} s".format(endfsc - startfsc))

    print("The pixelsize of the data is {:.02f} nm".format(pixelsize1[0] * 1e9))

    a = input("\nPlease, input the value of the intersection: ")
    print("------------------------------------------")
    print(
        "| Resolution is estimated to be {:.02f} nm |".format(
            pixelsize1[0] * 1e9 / float(a)
        )
    )
    print("------------------------------------------")

    input("\n<Hit Return to close images>")
    plt.close("all")
