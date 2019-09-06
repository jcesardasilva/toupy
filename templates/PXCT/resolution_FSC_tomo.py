#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template for resolution estimate of tomograms using FSC
"""

# import of third party packages
import matplotlib.pyplot as plt
import numpy as np

# import of local packages
from toupy.resolution import FSCPlot, compute_2tomograms
from toupy.io import LoadData, SaveData
from toupy.utils import sort_array, cropROI
from toupy.tomo import tomo_recons

# initializing params
params = dict()

# Edit section
# =========================
# Edit session
# =========================
params["samplename"] = "v97_h_nfptomo_15nm"
params["slicenum"] = 1000  # Choose the slice
params["limsyFSC"] = [1100, 1480]  # number of slices for the 3D FSC
params["filtertype"] = "hann"  # Filter to use for FBP
params["freqcutoff"] = 1.0  # Frequency cutoff (between 0 and 1)
params["circle"] = True
params["algorithm"] = "FBP"  # FBP or SART
params["derivatives"] = True  # only for FBP
params["calc_derivatives"] = False  # Calculate derivatives if not done
params["opencl"] = True  # enable reconstruction with Silx and openCL
params["apod_width"] = 50  # apodization width in pixels
params["thick_ring"] = 4  # number of pixel to average each FRC ring
params["crop"] = [1465, 1865, 935, 1335]  # [top, bottom, left, right]
params["vmin_plot"] = None  # 0.5e-5
params["vmax_plot"] = -0.5e-4  # None
params["colormap"] = "bone"  # colormap to show images
params["oldfileformat"] = False  # if one reads projections from old files
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#

if __name__ == "__main__":

    # loading the data
    filename = "aligned_projections.h5"
    if params["oldfileformat"]:
        aligned_projections, theta, shiftstack, params = LoadData.load_olddata(
            filename, **params
        )
    else:
        aligned_projections, theta, shiftstack, params = LoadData.load(
            filename, **params
        )

    # sorting theta
    print("Sorting theta...")
    aligned_projections, theta = sort_array(aligned_projections, theta)

    # convinient change of variables
    voxelsize = params["pixelsize"]  # from now on, voxelsize is pixelsize.
    slice_num = params["slicenum"]
    vmin_plot = params["vmin_plot"]
    vmax_plot = params["vmax_plot"]

    # =========================
    # 2D FRC calculation
    # =========================

    # calculate the sinogram
    sinogram_align = np.transpose(aligned_projections[:, slice_num, :])

    # tomographic reconstruction
    tomogram1, tomogram2 = compute_2tomograms(sinogram, theta, **params)

    # keep tomogram shape for later use
    nr, nc = tomogram1.shape

    # cropping
    if params["crop"] is not None:
        if params["crop"] != 0:
            tomogram1 = cropROI(tomogram1, roi=params["crop"])
            tomogram2 = cropROI(tomogram2, roi=params["crop"])

    print("Estimating the resolution by FSC...")
    FSC2D = FSCPlot(
        tomogram1,
        tomogram2,
        "halfbit",
        params["thick_ring"],
        apod_width=params["transv_apod"],
    )
    normfreqs, T, FSC2Dcurve = FSC2D.plot()

    print("The pixelsize of the data is {:.02f} nm".format(voxelsize[0] * 1e9))

    a = input("\nPlease, input the value of the intersection: ")
    params["resolution2D"] = voxelsize[0] * 1e9 / float(a)
    print("------------------------------------------")
    print("| Resolution is estimated to be {:.2f} nm |".format(params["resolution2D"]))
    print("------------------------------------------")

    input("\n<Hit Return to close images>")
    plt.close("all")

    # save the 2D FSC data
    SaveData.saveFSC(
        "FSC2D.h5",
        normfreqs,
        T,
        FSC2Dcurve,
        tomogram1,
        tomogram2,
        theta,
        voxelsize,
        **params
    )

    # =========================
    # 3D FSC calculation
    # =========================

    a = str(input("Do you want to calculate the 3D FSC?(y/n)")).lower()
    if a == "" or a == "y":
        # del tomo1, tomo2, image1, image2, sinogram_align, sinogram_align1, sinogram_align2#, sagital_slice1, sagital_slice2
        limsyFSC = params["limsyFSC"]
        nslices = limsyFSC[-1] - limsyFSC[0]

        # initializing variables
        tomogram1 = np.empty((nslices, nr, nc))
        tomogram2 = np.empty((nslices, nr, nc))
        for idx, ii in enumerate(range(limsyFSC[0], limsyFSC[-1])):
            print("Slice: {}".format(ii))
            sinogram_align = np.transpose(aligned_projections[:, ii, :])
            # dividing the data into two datasets
            print("Calculating first slice...")
            tomogram1[idx], tomogram2[idx] = compute_2tomograms(
                sinogram, theta, **params
            )

        # cropping
        if params["crop"] is not None:
            if params["crop"] != 0:
                creg = params["crop"]
                tomogram1 = cropROI(tomogram1, roi=params["crop"])
                tomogram2 = cropROI(tomogram2, roi=params["crop"])

        # special for this
        # tomogram1 = tomogram1[:,836:1440,866:1470]
        # tomogram2 = tomogram2[:,836:1440,866:1470]

        # 3D FSC
        print("Estimating the resolution by 3D FSC...")
        FSC3D = FSCPlot(
            tomogram1,
            tomogram2,
            "halfbit",
            params["thick_ring"],
            apod_width=params["transv_apod"],
        )
        normfreqs, T, FSC3Dcurve = FSC3D.plot()

        print("The voxelsize of the data is {:.02f} nm".format(voxelsize[0] * 1e9))

        a = input("\nPlease, input the value of the intersection: ")
        params["resolution3D"] = voxelsize[0] * 1e9 / float(a)
        print("------------------------------------------")
        print(
            "| Resolution is estimated to be {:.2f} nm |".format(params["resolution3D"])
        )
        print("------------------------------------------")

        input("\n<Hit Return to close images>")
        plt.close("all")

        # save the 3D FSC data
        SaveData.saveFSC(
            "FSC3D.h5",
            normfreqs,
            T,
            FSC3Dcurve,
            tomogram1,
            tomogram2,
            theta,
            voxelsize,
            **params
        )
