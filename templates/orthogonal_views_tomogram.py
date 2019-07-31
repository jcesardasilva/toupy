#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third packages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import filters

# local packages
from toupy.io import LoadTomogram, SaveTomogram

# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["energy"] = 17.05  # photon energy to convert tomogram to delta or beta values
params["phaseonly"] = True
# ~ params['roi'] = [600, 1600, 675, 1685]
params["tomo_type"] = "delta"
params["slice_num"] = 650
params["vmin_plot"] = 1e-7  # None
params["vmax_plot"] = 5e-6  # 5e-4
params["scale_bar_size"] = 5  # in microns
params["scale_bar_height"] = 1
params["scale_bar_color"] = u"yellow"
params["bar_start"] = [50, 860]
params["bar_axial"] = [70, 100]  # [cols,rows]
params["save_figures"] = True
params["colormap"] = "bone"
params["interpolation"] = "nearest"
params["gaussian_filter"] = False  # True
params["sigma_gaussian"] = 3  # if gaussian filter
# ==================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#

if __name__ == "__main__":

    # loading files
    if params["tomo_type"] == "delta":
        tomogram, theta, shiftstack, params = LoadTomogram.load("tomogram.h5", **params)
    elif params["tomo_type"] == "beta":
        tomogram, theta, shiftstack, params = LoadTomogram.load(
            "tomogram_amp.h5", **params
        )
    else:
        raise ValueError("Unrecognized tomography type")

    # conversion from phase-shifts to delta or from amplitude to beta
    pixelsize = params["voxelsize"][0]
    energy = params["energy"]
    wavelen = (12.4 / energy) * 1e-10  # in meters
    if params[u"tomo_type"] == "delta":
        # Conversion from phase-shifts tomogram to delta
        print("Converting from phase-shifts values to delta values")
        factor = wavelen / (2 * np.pi * voxelsize[0])
        for ii in range(tomogram.shape[0]):
            print("Tomogram {}".format(ii + 1))
            tomogram[ii] *= -factor
    elif params[u"tomo_type"] == "beta":
        # Conversion from amplitude to beta
        print("Converting from amplitude to beta values")
        factor = wavelen / (2 * np.pi * voxelsize[0])  # amplitude correction factor
        for ii in range(tomogram.shape[0]):
            print("Tomogram {}".format(ii + 1))
            tomogram[ii] *= -factor

    # simple transfer of variables
    pixelsize = voxelsize[0]
    slice_num = params["slice_num"]
    vmin_plot = params["vmin_plot"]
    vmax_plot = params["vmax_plot"]
    scale_bar_size = params["scale_bar_size"]
    scale_bar_height = params["scale_bar_height"]
    bar_start = params["bar_start"]
    bar_axial = params["bar_axial"]
    colormap_choice = params["colormap"]
    interp_type = params["interpolation"]
    scale_bar_color = params["scale_bar_color"]

    if params["gaussian_filter"]:
        print(
            "Applying gaussian filter with sigma = {}".format(params[u"sigma_gaussian"])
        )
        # sagital slice
        sagital_slice = filters.gaussian_filter(
            tomogram[:, np.round(tomogram.shape[1] / 2).astype("int"), :],
            params[u"sigma_gaussian"],
        )
        # sagital_slice = filters.gaussian_filter(tomogram_delta[:,821,:],params[u'sigma_gaussian'])
        # coronal slice
        coronal_slice = filters.gaussian_filter(
            tomogram[:, :, np.round(tomogram.shape[1] / 2).astype("int")],
            params["sigma_gaussian"],
        )
        # coronal_slice = filters.gaussian_filter(tomogram_delta[:,:,630],params[u'sigma_gaussian'])
        # axial slice
        axial_slice = filters.gaussian_filter(
            tomogram[slice_num], params[u"sigma_gaussian"]
        )
    else:
        # sagital slice
        sagital_slice = tomogram[:, np.round(tomogram.shape[1] / 2).astype("int"), :]
        # coronal slice
        coronal_slice = tomogram[:, :, np.round(tomogram.shape[1] / 2).astype("int")]
        # axial slice
        axial_slice = tomogram[slice_num]

    textstr = r"{} $\mu$m".format(scale_bar_size)

    plt.close("all")

    # Sagital slice
    figsag = plt.figure(num=1)  # ,figsize=(15,6))
    axsag = figsag.add_subplot(
        111
    )  # plt.subplots(num=6,nrows=1,ncols=1,figsize=(15,6))
    imsag = axsag.imshow(
        sagital_slice,
        interpolation=interp_type,
        cmap=colormap_choice,
        vmin=vmin_plot,
        vmax=vmax_plot,
    )
    axsag.set_title(u"Sagital slice - {}".format(params["tomo_type"]))
    axsag.text(
        bar_start[0] - 10,
        bar_start[1] - 5,
        textstr,
        fontsize=14,
        verticalalignment="bottom",
        color=scale_bar_color,
    )
    rectsag = patches.Rectangle(
        (bar_start[0], bar_start[1]),  # (x,y)
        (np.round(scale_bar_size * 1e-6 / pixelsize)),  # width
        (np.round(scale_bar_height * 1e-6 / pixelsize)),  # height
        color=scale_bar_color,
    )
    axsag.add_patch(rectsag)
    axsag.set_axis_off()
    if params["save_figures"]:
        plt.savefig(
            "sagital_{}.png".format(params["tomo_type"]), bbox_inches="tight", dpi=200
        )

    # fig.colorbar(imsag)

    # Coronal slice
    figcor = plt.figure(num=2)
    axcor = figcor.add_subplot(111)
    imcor = axcor.imshow(
        coronal_slice,
        interpolation=interp_type,
        cmap=colormap_choice,
        vmin=vmin_plot,
        vmax=vmax_plot,
    )
    axcor.set_title(u"Coronal slice - {}".format(params["tomo_type"]))
    axcor.text(
        bar_start[0] - 10,
        bar_start[1] - 5,
        textstr,
        fontsize=14,
        verticalalignment="bottom",
        color=scale_bar_color,
    )
    rectcor = patches.Rectangle(
        (bar_start[0], bar_start[1]),  # (x,y)
        (np.round(scale_bar_size * 1e-6 / pixelsize)),  # width
        (np.round(scale_bar_height * 1e-6 / pixelsize)),  # height
        color=scale_bar_color,
    )
    axcor.add_patch(rectcor)
    axcor.set_axis_off()
    plt.tight_layout()
    plt.show()
    if params["save_figures"]:
        plt.savefig(
            "coronal_{}.png".format(params["tomo_type"]), bbox_inches="tight", dpi=200
        )

    # Axial slice
    figaxial = plt.figure(num=3)  # ,figsize=(15,6))
    axaxial = figaxial.add_subplot(
        111
    )  # plt.subplots(num=6,nrows=1,ncols=1,figsize=(15,6))
    imaxial = axaxial.imshow(
        axial_slice,
        interpolation=interp_type,
        cmap=colormap_choice,
        vmin=vmin_plot,
        vmax=vmax_plot,
    )
    axaxial.set_title(
        u"Axial slice {} - {} ".format(slice_num + 1, params["tomo_type"])
    )
    axaxial.text(
        bar_axial[0] - 10,
        bar_axial[1] - 5,
        textstr,
        fontsize=14,
        verticalalignment="bottom",
        color=scale_bar_color,
    )
    rectaxial = patches.Rectangle(
        (bar_axial[0], bar_axial[1]),  # (x,y)
        (np.round(scale_bar_size * 1e-6 / pixelsize)),  # width
        (np.round(scale_bar_height * 1e-6 / pixelsize)),  # height
        color=scale_bar_color,
    )
    axaxial.add_patch(rectaxial)
    axaxial.set_axis_off()
    plt.show()
    if params["save_figures"]:
        plt.savefig(
            "axial_slice{}_{}.png".format(slice_num + 1, params["tomo_type"]),
            bbox_inches="tight",
            dpi=200,
        )
