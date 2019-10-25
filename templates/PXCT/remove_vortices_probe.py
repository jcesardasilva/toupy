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
from toupy.restoration import rmvortices_probe
from toupy.utils import progbar

# initializing dictionaries
params = dict()

# remove vortices from the probe
# =========================
params[
    "filename"
] = "/data/visitor/ma3495/id16a/analysis/recons/O2_LSCF_CGO_25nm_subtomo001_0000/O2_LSCF_CGO_25nm_subtomo001_0000_ML.ptyr"
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

    n_probes = probe.shape[0]
    print("I found {} probe modes.".format(n_probes))
    # get the phase of the modes
    probe_phase = np.angle(probe)  # don't know how many in advance

    # Removing vortices
    p_phase_novort = np.empty_like(probe)
    p_xres = []
    p_yres = []
    for ii in range(n_probes):
        strbar = "Removing vortices of probe mode {}".format(ii + 1)
        p_phase_novort, xres, yres = rmvortices_probe(
            probe[ii], to_ignore=params["to_ignore"]
        )
        p_xres.append(xres)
        p_yres.append(yres)
        progbar(ii, n_probes, strbar)
    p_xres = np.array(p_xres)
    p_yres = np.array(p_yres)

    # feed the new array
    probe_novort = np.empty_like(probe)
    for ii in range(n_probes):
        probe_novort[ii] = p_phase_novort[ii]

    # display the probes with the residues and without vortices
    if params["show_figures"]:
        plt.close("all")
        if n_probes == 1:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121)
            ax1.imshow(probe_phase[0], cmap="bone")
            ax1.axis("image")
            ax1.set_title("pr. {} - with vortices".format(1))
            ax1.plot(p_xres, p_yres, "or")
            # display the probes after vortices removal
            ax2 = fig.add_subplot(122)
            ax2.imshow(np.angle(probe_novort)[0], cmap="bone")
            ax2.axis("image")
            ax2.set_title("pr. {} - no vortices".format(1))
        else:
            fig, ax = plt.subplots(2, n_probes, num=1)
            for ii in range(n_probes):
                # display the probes with the residues
                ax[0, ii].imshow(probe_phase[ii], cmap="bone")
                ax[0, ii].axis("image")
                ax[0, ii].set_title("pr. {}".format(ii + 1))
                ax[0, ii].plot(p_xres[ii], p_yres[ii], "or")
                # display the probes after vortices removal
                ax[1, ii].imshow(np.angle(probe_novort[ii]), cmap="bone")
                ax[1, ii].axis("image")
                ax[1, ii].set_title("pr. {} - no vortice".format(ii + 1))

    if params["overwrite_h5"]:
        # only to keep a copy of the file and prevent overwritting
        if not os.path.isfile(params["filename"] + ".vort"):
            shutil.copy(params["filename"], params["filename"] + ".vort")
        print("Overwritting object information in the h5 file")
        with h5py.File(params["filename"], "r+") as fid:
            obj_new = fid["content/probe/S00G00/data"]
            obj_new[...] = probe_novort
