#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard packages
import os
import shutil
import time

# third party packages
import h5py
import matplotlib.pyplot as plt
import numpy as np

# local package
from toupy.io import read_recon
from toupy.restoration import rmvortices_object

# initializing dictionaries
params = dict()

# remove vortices from the probe
# =========================
params["filename"] = "/data/visitor/ma3495/id16a/analysis/recons/O2_LSCF_CGO_25nm_subtomo001_0000/O2_LSCF_CGO_25nm_subtomo001_0000_ML.ptyr"
params["overwrite_h5"] = True  # True or False
params["to_ignore"] = 100
params["show_figures"] = True
# =========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":

    # Open the file and extract the image data
    # The orientation is not changed because the image is fed back into file
    objimg, probe, pixelsize, energy = read_recon(
        params["filename"], correct_orientation=False
    )

    # get the phase-shifts of the object
    obj_phase = np.angle(objimg[0])

    # Removing vortices
    print("Removing vortices of object...")
    obj_phase_novort, xres, yres = rmvortices_object(
        objimg[0], to_ignore=params["to_ignore"]
    )

    # display the object with the residues and without vortices
    if params["show_figures"]:
        plt.close("all")
        fig = plt.figure(1)
        ax1 = fig.add_subplot(121)
        ax1.imshow(obj_phase, cmap="bone")
        ax1.axis("image")
        ax1.set_title("Object with vortices")
        ax1.plot(xres, yres, "or")
        # display the probes after vortices removal
        ax2 = fig.add_subplot(122)
        ax2.imshow(np.angle(obj_phase_novort), cmap="bone")
        ax2.axis("image")
        ax2.set_title("Object - no vortices".format(1))
        plt.show(block=False)

    if params["overwrite_h5"]:
        # only to keep a copy of the file and prevent overwritting
        if not os.path.isfile(params["filename"] + ".vort"):
            shutil.copy(params["filename"], params["filename"] + ".vort")
        print("Overwritting probe information in the h5 file")
        with h5py.File(params["filename"], "r+") as fid:
            probe_new = fid["content/obj/S00G00/data"]
            probe_new[...] = obj_phase_novort
