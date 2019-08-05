#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third packages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import filters

# local packages
from toupy.io import LoadTomogram
from toupy.utils import convert_to_beta, convert_to_delta

# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["tomo_type"] = "delta"
params["slicenum"] = 650
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
    voxelsize = params["pixelsize"][0]
    energy = params["energy"]
    if params["tomo_type"] == "delta":
        # Conversion from phase-shifts tomogram to delta
        print("Converting from phase-shifts values to delta values")
        for ii in range(nslices):
            strbar = "Slice {} out of {}".format(ii + 1, nslices)
            tomogram[ii], factor = convert_to_delta(tomogram[ii], energy, voxelsize)
            progbar(ii + 1, nslices, strbar)
    elif params["tomo_type"] == "beta":
        # Conversion from amplitude to beta
        print("Converting from amplitude to beta values")
        for ii in range(slices):
            strbar = "Slice {} out of {}".format(ii + 1, nslices)
            tomogram[ii], factor = convert_to_beta(tomogram[ii], energy, voxelsize)
            progbar(ii + 1, nslices, strbar)
    print("\r")
        
    
    # simple transfer of variables
    slice_num = params["slicenum"]
    vmin_plot = params["vmin_plot"]
    vmax_plot = params["vmax_plot"]
    scale_bar_size = params["scale_bar_size"]
    scale_bar_height = params["scale_bar_height"]
    bar_start = params["bar_start"]
    bar_axial = params["bar_axial"]
    colormap_choice = params["colormap"]
    interp_type = params["interpolation"]
    scale_bar_color = params["scale_bar_color"]

    # text style for the scale bar text
    textstr = r"{} $\mu$m".format(scale_bar_size)

    # sagital slice
    sagital_slice = tomogram[:, np.round(tomogram.shape[1] / 2).astype("int"), :]
    # coronal slice
    coronal_slice = tomogram[:, :, np.round(tomogram.shape[1] / 2).astype("int")]
    # axial slice
    axial_slice = tomogram[slice_num]

    if params["gaussian_filter"]:
        print(
            "Applying gaussian filter with sigma = {}".format(params["sigma_gaussian"])
        )
        # sagital slice
        sagital_slice = filters.gaussian_filter(sagital_slice)
        # coronal slice
        coronal_slice = filters.gaussian_filter(coronal_slice)
        # axial slice
        axial_slice = filters.gaussian_filter(axial_slice)

    # display the figures
    plt.close("all")
    
    # Sagital slice
    figsag = plt.figure(num=1)
    axsag = figsag.add_subplot(111)
    imsag = axsag.imshow(
        sagital_slice,
        interpolation=interp_type,
        cmap=colormap_choice,
        vmin=vmin_plot,
        vmax=vmax_plot,
    )
    axsag.set_title("Sagital slice - {}".format(params["tomo_type"]))
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
        (np.round(scale_bar_size * 1e-6 / voxelsize)),  # width
        (np.round(scale_bar_height * 1e-6 / voxelsize)),  # height
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
    axcor.set_title("Coronal slice - {}".format(params["tomo_type"]))
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
        (np.round(scale_bar_size * 1e-6 / voxelsize)),  # width
        (np.round(scale_bar_height * 1e-6 / voxelsize)),  # height
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
    figaxial = plt.figure(num=3)
    axaxial = figaxial.add_subplot(111)
    imaxial = axaxial.imshow(
        axial_slice,
        interpolation=interp_type,
        cmap=colormap_choice,
        vmin=vmin_plot,
        vmax=vmax_plot,
    )
    axaxial.set_title(
        "Axial slice {} - {} ".format(slice_num + 1, params["tomo_type"])
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
        (np.round(scale_bar_size * 1e-6 / voxelsize)),  # width
        (np.round(scale_bar_height * 1e-6 / voxelsize)),  # height
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
