#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:44:54 2016

@author: jdasilva
"""
# standard libraries imports
import sys
import os
import re

# third packages
import libtiff
import numpy as np

# from scipy.misc import imsave

# local packages
from io_utils import (
    checkhostname,
    save_or_load_tomogram,
    create_paramsh5,
    load_paramsh5,
)
from tomographic_reconstruction import params

# =========================
# Edit session
# =========================
energy = 17.05  # In keV
tomo_type = "delta"  # 'delta' or 'beta'
inputkwargs = dict()
inputkwargs[u"samplename"] = "v97_v_nfptomo2_15nm"
inputkwargs[u"pyhst"] = True
inputkwargs[u"bits"] = 16  # 8 or 16 bits
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # load the tomograms
    host_machine = checkhostname()
    if sys.version_info > (3, 0):
        raw_input = input
        xrange = range

    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()

    # loading parameters from h5file
    kwargs = load_paramsh5(**inputkwargs)
    inputparams.update(kwargs)
    inputparams.update(params)

    # loading files
    if tomo_type == "delta":
        inputkwargs[u"phaseonly"] = True
        inputkwargs[u"amponly"] = False
        tomogram, theta, deltastack, voxelsize, kwargs = save_or_load_tomogram(
            "tomogram.h5", **inputparams
        )  # pathfilename=inputkwargs['path_filename'],h5name='tomogram.h5')
    elif tomo_type == "beta":
        inputkwargs[u"phaseonly"] = False
        inputkwargs[u"amponly"] = True
        # pathfilename=inputkwargs['path_filename'],h5name='tomogram_amp.h5')
        tomogram, theta, deltastack, voxelsize, kwargs = save_or_load_tomogram(
            "tomogram_amp.h5", **inputparams
        )
    else:
        raise ValueError("Unrecognized tomography type")

    nslices = tomogram.shape[1]
    print("The total number of slides is {}".format(nslices))
    print("The voxel size is {} nm".format(voxelsize * 1e9))

    # updating parameters after loading files
    inputparams.update(kwargs)  # updating the inputparams
    inputparams.update(params)  # as second to update to the most recent values
    create_paramsh5(**inputparams)  # updating parameter h5 file

    # ~ if inputkwargs[u'pyhst']:
    # ~ factor_pyhst = (2*np.pi/wavelen)*voxelsize[0]

    # Save Tiff files for 3D visualization with external program
    print("Saving Tiff files")
    pathfilename = inputparams["pathfilename"]
    bodypath, filename = os.path.split(pathfilename)
    # get fileprefix and extension
    fileprefix, ext = os.path.splitext(filename)
    if ext == ".ptyr":
        aux_wcard = re.sub(
            u"_subtomo\d{3}_\d{4}_\w*", "", os.path.splitext(filename)[0]
        )
    elif ext == ".cxi":
        aux_wcard = re.sub(u"_subtomo\d{3}_\d{4}", "", os.path.splitext(filename)[0])
    else:
        raise IOError(
            "File {} is not a .ptyr or a .cxi file. Please, load a .ptyr or a .cxi file.".format(
                filename
            )
        )
    aux_path = os.path.join(os.path.dirname(bodypath), aux_wcard + "_nfpxct")
    # bodypath = os.path.split(bodypath)[0]
    # aux_wcard = re.sub(u'_subtomo\d{3}_\d{4}_\w*$','', os.path.splitext(filename)[0])
    # aux_path = os.path.join(os.path.abspath(bodypath),aux_wcard+'_nfpxct')

    # create the TIFF folder
    tiff_subfolder_name = "TIFF_{}_{}_freqscl_{:0.2f}_{:d}bits".format(
        tomo_type, params["filtertype"], params["filtertomo"], inputkwargs[u"bits"]
    )
    tiff_folder_path = os.path.join(aux_path, tiff_subfolder_name)
    # aux_path+'/unwrapped_phases.h5'):
    if not os.path.isdir(tiff_folder_path):
        print(
            "Folder {} does not exists and will be create".format(tiff_subfolder_name)
        )
        os.makedirs(tiff_folder_path)
    else:
        print("Folder exists:{}".format(tiff_folder_path))
        userans = raw_input(
            "Do you want to overwrite TIFFs in this folder ([y]/n)?"
        ).lower()
        if userans == "" or userans == "y":
            print("Overwritting")
        else:
            print("Writting of TIFFs aborted")
            raise SystemExit("Writting of TIFFs aborted")

    # conversion from phase-shifts to delta or from amplitude to beta
    wavelen = (12.4 / energy) * 1e-10  # in meters
    if tomo_type == "delta":
        # Conversion from phase-shifts tomogram to delta
        print("Converting from phase-shifts values to delta values")
        factor = wavelen / (2 * np.pi * voxelsize[0])
        # tomogram_delta = np.zeros_like(tomogram)
        for ii in range(tomogram.shape[0]):
            print("Tomogram {}".format(ii + 1))
            # tomogram_delta[ii] = -tomogram[ii].copy()*factor
            tomogram[ii] *= -factor
    elif tomo_type == "beta":
        # Conversion from amplitude to beta
        print("Converting from amplitude to beta values")
        factor = wavelen / (2 * np.pi * voxelsize[0])  # amplitude correction factor
        for ii in range(tomogram.shape[0]):
            print("Tomogram {}".format(ii + 1))
            tomogram[ii] *= -factor

    # low and high cutoff for Tiff normalization
    low_cutoff = np.min(tomogram)
    high_cutoff = np.max(tomogram)

    # Convertion to tiff
    for ii in range(tomogram.shape[0]):
        # imgtiff = tomogram[ii].copy()-low_cutoff
        # fliplr to have same orientation as holoct
        imgtiff = np.fliplr(tomogram[ii].copy()) - low_cutoff
        imgtiff /= high_cutoff - low_cutoff
        if inputkwargs[u"bits"] == 16:
            # Tiff normalization - 16bits
            imgtiff *= 2 ** 16 - 1  # 16 bits
            imgtiff = np.uint16(imgtiff)
        elif inputkwargs[u"bits"] == 8:
            # Tiff normalization - 8bits
            imgtiff *= 2 ** 8 - 1  # 8 bits
            imgtiff = np.uint8(imgtiff)
        else:
            raise ValueError("Tiffs can only be saved in 8 bits or 16 bits")

        # Writing to file
        tiff = libtiff.TIFF.open(
            tiff_folder_path
            + "/tomo_{}_filter_type_{}_freqscl_{:0.2f}_{:04d}.tif".format(
                tomo_type, params["filtertype"], params["filtertomo"], ii
            ),
            mode="w",
        )
        tiff.write_image(imgtiff)
        tiff.close()
        print(
            "writing slice {} of {:03d}".format(ii, tomogram.shape[0], tiff_folder_path)
        )

    # Creates a txt file with the information about the Tiff normalization
    fid = open(tiff_folder_path + "/" + tiff_subfolder_name + "_cutoffs.txt", "w")
    fid.write("# low_cutoff = {}\n".format(low_cutoff))
    fid.write("# high_cutoff = {}\n".format(high_cutoff))
    fid.write("# factor = {}\n".format(factor))
    fid.write("# voxel size = {} nm\n".format(voxelsize * 1e9))
    fid.write("# to convert back to quantitative values:\n")
    fid.write(
        "# low_cutoff + [(high_cutoff-low_cutoff)*tiff_image/(2^{:d}-1)] \n".format(
            inputkwargs[u"bits"]
        )
    )
    fid.close()
