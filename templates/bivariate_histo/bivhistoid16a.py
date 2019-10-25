#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries
import fnmatch
import glob
import os
import time

# third party packages
import libtiff
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter, MaxNLocator
import numpy as np
import scipy.constants as consts

# local package
from toupy.io.filesrw import read_tiff, convert16bitstiff, convert8bitstiff
from toupy.utils import (
    create_mask_borders,
    convert_to_mu,
    convert_to_rhoe,
    convert_to_rhom,
    create_circle,
)

# ========================
# Edit session
# ========================
params["slice_numbers"] = [150, 950]  # [50,700]#[50,200]#[0,299]
params["bins"] = 256  # number of bins of the histogram
params["histo_scale"] = "lin"  # 'lin', 'log', lin seems to be better
params[
    "pathfilename"
] = "/data/id16a/inhouse4/visitor/ma4351/id16a/analysis/recons/v97_v_nfptomo2_15nm_nfpxct"
params["histtype"] = "delta_beta"  # 'rhoe_mu' # 'delta_beta'
params["bits"] = 16
params["apply_circle"] = True
params["apply_mask_borders"] = False
params["factor_scale_beta"] = 1e6  # 1e7
params["factor_scale_delta"] = 1e6
params["energy"] = 17.05  # in keV
# ========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":

    if params["bits"] == 16:
        convert_tiff_value = convert16bitstiff
    elif params["bits"] == 8:
        convert_tiff_value = convert8bitstiff

    # wavelength
    params["wavelen"] = 12.4e-10 / params["energy"]

    # path to files
    base_path = params["pathfilename"]

    factor_scale_beta = params["factor_scale_beta"] = 1e6
    factor_scale_delta = params["factor_scale_delta"] = 1e6
    slice_numbers = params["slice_numbers"]

    # Finding the Tiff files in the path
    tiff_folders = glob.glob(os.path.join(base_path, "TIFF*"))
    print(
        "I have found {} folder: \n{} \n{}".format(
            len(tiff_folders),
            os.path.split(tiff_folders[0])[1],
            os.path.split(tiff_folders[1])[1],
        )
    )

    # check if both exist
    if not np.any([fnmatch.fnmatch(ii, "*beta*") for ii in tiff_folders]):
        raise SystemExit("It does not contain the beta tomogram")
    elif not np.any([fnmatch.fnmatch(ii, "*delta*") for ii in tiff_folders]):
        raise SystemExit("It does not contain the delta tomogram")

    # check and read the some infos from the tif files (.txt and path for files)
    for ii in tiff_folders:
        if fnmatch.fnmatch(ii, "*beta*"):
            tiff_info_beta = os.path.join(ii, os.path.split(ii)[1] + "_cutoffs.txt")
            tiff_beta_wcard = os.path.join(ii, "tomo_beta_filter_type_*_freqscl_*.tif")
        elif fnmatch.fnmatch(ii, "*delta*"):
            tiff_info_delta = os.path.join(ii, os.path.split(ii)[1] + "_cutoffs.txt")
            tiff_delta_wcard = os.path.join(
                ii, "tomo_delta_filter_type_*_freqscl_*.tif"
            )

    ####### DELTA #######
    print("Reading the info file for the delta tomogram")
    low_cutoff_delta, high_cutoff_delta, factor_delta, pixelsize_delta = read_info_file(
        tiff_info_delta
    )

    # read the tiff delta files
    t0 = time.time()
    print("Reading the delta tiff files")
    list_tiff_files_delta = sorted(glob.glob(tiff_delta_wcard))
    print(
        "I have found {} tiff files for the delta tomogram".format(
            len(list_tiff_files_delta)
        )
    )
    print("Reading first slice to get size")
    nslices = slice_numbers[-1] - slice_numbers[0] + 1
    tomodelta0 = read_tiff(list_tiff_files_delta[0]).astype(np.float)
    tomogram_delta = np.empty((nslices, tomodelta0.shape[0], tomodelta0.shape[1]))
    for idx, ii in enumerate(
        list_tiff_files_delta[slice_numbers[0] : slice_numbers[-1]]
    ):
        print("Slice {}".format(idx + 1))
        print("File {}".format(ii))
        tiff_image = read_tiff(ii).astype(np.float)
        # Convert to float so we don't have clipping to the range 0-255
        tomogram_delta[idx] = convert_tiff_value(
            tiff_image, low_cutoff_delta, high_cutoff_delta
        )
        if params["histtype"] == "rhoe_mu":
            print("Converting to electron density: {}".format(idx + 1), end="\r")
            tomogram_delta[idx] = convert_to_rhoe(tomogram_delta[idx], wavelen)
    del tomodelta0

    print("Phase tomogram tif slices loaded and converted to Deta")
    print("Time elapsed {} seconds".format(time.time() - t0))

    # we first create the mask based on tomogram_delta
    # mask - find the indices corresponding to the materials phase only (exclude air)
    print("Calculating mask for the sample")
    mask_delta = (tomogram_delta > 1e-6).astype(np.int)

    # mask borders
    if params["apply_mask_borders"]:
        mask_delta = create_mask_border(tomogram_delta, mask_delta)

    # Reshape the images into 1D vectors
    mask_ind = np.where(mask_delta != 0)

    # mask the circular region excluded during tomographic reconstruction
    if params["apply_circle"]:
        circle = create_circle(tomogram_delta)
    else:
        circle = 1

    # create mask of air/vacuum region
    print("Calculating mask for the air/vacuum")
    comp_mask_delta = (1 - mask_delta) * circle  # broadcasting
    del circle
    # subtracting air/vacuum from tomogram_delta
    # indices of air/vacuum (also used for beta)
    air_ind = np.where(comp_mask_delta != 0)
    # np.mean(tomogram_beta*comp_mask_delta)
    air_delta = np.mean(tomogram_delta[air_ind])
    print("Subtracting air/vacuum delta values")
    tomogram_delta -= air_delta
    print("Done")
    print("Air delta = {}".format(air_delta))

    print("Taking the values of the tomogram_delta only where there is sample")
    y = tomogram_delta[mask_ind]  # uses previous mask_ind
    print("Done")

    # delete tomogram_delta to reduce memory imprint
    del tomogram_delta
    del comp_mask_delta
    del mask_delta

    ###### BETA #######
    print("Reading the info file for the beta tomogram")
    low_cutoff_beta, high_cutoff_beta, factor_beta, pixelsize_beta = read_info_file(
        tiff_info_beta
    )

    # read the tiff beta files
    t0 = time.time()
    # load a slice from the beta tomogram to get the slice dimensions
    print("Reading the beta tiff files")
    list_tiff_files_beta = sorted(glob.glob(tiff_beta_wcard))
    print(
        "I have found {} tiff files for the beta tomogram".format(
            len(list_tiff_files_beta)
        )
    )
    print("Reading first slice to get size")
    nslices = slice_numbers[-1] - slice_numbers[0] + 1
    tomobeta0 = read_tiff(list_tiff_files_beta[0]).astype(np.float)
    tomogram_beta = np.empty((nslices, tomobeta0.shape[0], tomobeta0.shape[1]))
    for idx, ii in enumerate(
        list_tiff_files_beta[slice_numbers[0] : slice_numbers[-1]]
    ):
        print("Slice {}".format(idx + 1))
        print("File {}".format(ii))
        tiff_image = read_tiff(ii).astype(np.float)
        # Convert to float so we don't have clipping to the range 0-255
        tomogram_beta[idx] = convert_tiff_value(
            tiff_image, low_cutoff_beta, high_cutoff_beta
        )
        if params["histtype"] == "rhoe_mu":
            print(
                "Converting to linear attenuation coefficient: {}".format(idx + 1),
                end="\r",
            )
            tomogram_beta[idx] = convert_to_mu(tomogram_beta[idx], wavelen)
    del tomodelta0

    print("Absorption tomogram tif slices loaded and converted to beta")
    print("Time elapsed {} seconds".format(time.time() - t0))

    # taking the average in the air areas
    # np.mean(tomogram_beta*comp_mask_delta)
    air_beta = np.mean(tomogram_beta[air_ind])
    print("Subtracting air/vacuum beta values")
    tomogram_beta -= air_beta
    print("Done")
    print("Air beta = {}".format(air_beta))

    print("Taking the values of the tomogram_delta only where there is sample")
    # uses previous mask_ind, which is the same for delta
    x = tomogram_beta[mask_ind]
    print("Done")

    # delete tomogram_beta to reduce memory imprint
    del tomogram_beta
    # more memory clean up
    del mask_ind

    # plt.hexbin(x,y)

    # Create the bivariate histogram of delta and beta

    # Build the 2D histogram
    # set up default x and y limits
    xlims = [x.min(), x.max()]  # beta
    ylims = [y.min(), y.max()]  # delta

    # define the locations for the axes
    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left + width + 0.02

    # set up the geometry of the three plots
    print("Setting the geometry of three plots")
    # dimensions of the 2D histo plot
    rect_2d_histo = [left, bottom, width, height]
    # dimensions of the beta-histogram
    rect_histbeta = [left, bottom_h, width, 0.25]
    # dimensions of the delta-histogram
    rect_histdelta = [left_h, bottom, 0.25, height]
    print("Done")

    # Find the min/max of the data
    xmin = min(xlims)  # beta
    xmax = max(xlims)  # beta
    ymin = min(ylims)  # delta
    ymax = max(ylims)  # delta

    # Make the 'main' 2D histogram plot
    print("Making the main 2D histogram plot")
    # Define the number of bins
    nbetabins = params["bins"]  # 4*256
    ndeltabins = params["bins"]  # 4*256
    n2dbins = params["bins"]  # 2*512

    betabins = np.linspace(start=xmin, stop=xmax, num=nbetabins)
    deltabins = np.linspace(start=ymin, stop=ymax, num=ndeltabins)
    betacenter = (betabins[0:-1] + betabins[1:]) / 2.0
    deltacenter = (deltabins[0:-1] + deltabins[1:]) / 2.0
    aspectratio = (
        1.0 * (xmax * factor_scale_beta - 0) / (1.0 * ymax * factor_scale_delta - 0)
    )

    H, xedges, yedges = np.histogram2d(y, x, bins=(deltabins, betabins))
    X = betacenter
    Y = deltacenter
    Z = H
    print("Done")

    # set up the size of the figure
    print("Starting the diplay of the Bivariate histogram. Please, wait...")
    plt.close("all")
    fig = plt.figure(1, figsize=(9.5, 9))
    plt.pause(0.5)
    # Make the three plots
    ax2dhisto = plt.axes(rect_2d_histo)  # 2D histogram plot
    axhistbeta = plt.axes(rect_histbeta)  # beta histogram
    axhistdelta = plt.axes(rect_histdelta)  # delta histogram

    # Remove the inner axes numbers of the histogram
    nullfmt = NullFormatter()
    axhistbeta.xaxis.set_major_formatter(nullfmt)
    axhistdelta.yaxis.set_major_formatter(nullfmt)

    print("Plotting...")
    # plot the 2D histogram data
    cax2d = ax2dhisto.imshow(
        np.log(H + np.spacing(1)),
        extent=[
            xmin * factor_scale_beta,
            xmax * factor_scale_beta,
            ymin * factor_scale_delta,
            ymax * factor_scale_delta,
        ],
        vmin=0,
        interpolation="nearest",
        origin="lower",
        aspect="auto",
        cmap="jet",
    )  # ,aspect=aspectratio))

    # ~ # plot the 2D histogram data
    # ~ cax2d = (ax2dhisto.matshow(H,
    # ~ #extent=[xmin*factor_scale_beta,xmax*factor_scale_beta,ymin*factor_scale_delta,ymax*factor_scale_delta],
    # ~ norm=LogNorm(),
    # ~ vmin=0,
    # ~ interpolation='nearest', origin='lower',aspect='auto',cmap='jet'))#,aspect=aspectratio))
    # ~ ax2dhisto.xaxis.set_ticks_position('bottom')

    cbaxes = fig.add_axes([0.58, 0.15, 0.03, 0.2])
    cb = plt.colorbar(cax2d, cax=cbaxes)
    cb.ax.tick_params(axis="y", colors="white")

    # Plot the axes labels
    if params["histtype"] == "rhoe_mu":
        pass
    elif params["histtype"] == "delta_beta":
        # set up the x and y labels
        xlabel = r"Absorption index, $ \beta$ [x$10^{0:}$]".format(
            "{" + str(-int(np.log10(factor_scale_beta))) + "}"
        )
        ylabel = r"Refractive index decrement, $\delta$ [x$10^{0:}$]".format(
            "{" + str(-int(np.log10(factor_scale_delta))) + "}"
        )
    # add the axes labels
    ax2dhisto.set_xlabel(xlabel, fontsize=18)
    ax2dhisto.set_ylabel(ylabel, fontsize=18)

    # Make the tickmarks pretty
    ticklabels = ax2dhisto.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family("serif")

    ticklabels = ax2dhisto.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family("serif")

    # Set up the plot limits
    # [xlims[0]*factor_scale_delta,xlims[1]*factor_scale_delta]#[-7,12]
    betaplotlims = [-0.4, 1.2]
    deltaplotlims = [ylims[0] * factor_scale_delta, 6]
    # [xlims[0]*factor_scale_beta,xlims[1]*factor_scale_beta])
    ax2dhisto.set_xlim(betaplotlims)
    # [ylims[0]*factor_scale_delta,ylims[1]*factor_scale_delta])
    ax2dhisto.set_ylim(deltaplotlims)

    # Set up the histogram bins
    xbins = np.arange(xmin, xmax, (xmax - xmin) / n2dbins)  # *factor_scale_beta
    ybins = np.arange(ymin, ymax, (ymax - ymin) / n2dbins)  # *factor_scale_delta

    # Plot the histograms
    axhistbeta.hist(x, bins=xbins, color="blue")
    axhistdelta.hist(y, bins=ybins, orientation="horizontal", color="red")

    # Set up the histogram limits
    axhistbeta.set_xlim(
        [betaplotlims[0] / factor_scale_beta, betaplotlims[1] / factor_scale_beta]
    )  # x.min(), x.max() )
    axhistbeta.xaxis.set_major_locator(plticker.MultipleLocator(0.2e-6))
    axhistdelta.set_ylim(
        [deltaplotlims[0] / factor_scale_delta, deltaplotlims[1] / factor_scale_delta]
    )  # y.min(), y.max()  )

    # Set up the spacing of the tickerlabels
    axhistbeta.yaxis.set_major_locator(plticker.MultipleLocator(2e6))
    axhistdelta.xaxis.set_major_locator(plticker.MultipleLocator(1e6))

    def y_fmt(x, y):
        return "{:2.0e}".format(x).replace("e+", "x10^")

    def exp_fmt(x, y):
        f = plticker.ScalarFormatter(useOffset=False, useMathText=True)
        return "${}$".format(f._formatSciNotation("%1.10e" % x))

    #
    # g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    # plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(g))

    # Make the tickmarks pretty
    ticklabels = axhistbeta.get_yticklabels()
    # ~ axhistbeta.yaxis.set_major_formatter(plticker.FormatStrFormatter('%0.01e'))
    # ~ axhistbeta.yaxis.set_major_formatter(plticker.FuncFormatter(y_fmt))
    axhistbeta.yaxis.set_major_formatter(plticker.FuncFormatter(exp_fmt))
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family("serif")

    # Make the tickmarks pretty
    ticklabels = axhistdelta.get_xticklabels()
    # ~ axhistdelta.xaxis.set_major_formatter(plticker.FormatStrFormatter('%0.01e'))
    # ~ axhistdelta.xaxis.set_major_formatter(plticker.FuncFormatter(y_fmt))
    axhistdelta.xaxis.set_major_formatter(plticker.FuncFormatter(exp_fmt))
    #'{:2.2e}'.format(x).replace('e', 'x10^')
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family("serif")
        label.set_rotation(270)

    # Cool trick that changes the number of tickmarks for the histogram axes
    axhistbeta.xaxis.set_major_locator(
        plticker.MultipleLocator(0.2e-6)
    )  # MaxNLocator(7))
    # ~ axhistbeta.xaxis.set_major_locator(MaxNLocator(7))
    axhistdelta.yaxis.set_major_locator(MaxNLocator(6))

    # Show the plot
    plt.show(block=False)
    print("Done")

    savefigure = True
    if savefigure:
        plt.savefig("bivariate_histo.png", dpi=200, bbox_inches="tight")
