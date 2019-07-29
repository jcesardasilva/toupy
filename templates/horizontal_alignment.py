#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import re
import sys
import socket

### quick fix to avoid ImportError: dlopen: cannot load any more object with static TLS
### not used when not using GPUs
if re.search('gpu', socket.gethostname()) or re.search('gpid16a', socket.gethostname()):
    import pyfftw # has to be imported first
    from skimage.transform import radon
###

# local packages
from toupy.io import LoadData, SaveData
from toupy.utils import sort_array, replace_bad
from iradon import mod_iradon, mod_iradon2
from registration_utils import radtap, shift_fft, shift_linear
from registration_utils import alignprojections_horizontal, compute_aligned_stack

# initializing dictionaries
params = dict()

# Edit section
# =========================
params["samplename"] = "v97_v_nfptomo2_15nm"
params["phaseonly"] = True
params["slicenum"] = 550  # Choose the slice
params["filtertype"] = "hann"  # Filter to use for FBP
params["filtertomo"] = 0.2  # Frequency cutoff
params["circle"] = True
# initial guess of the offset of the axis of rotation
params["rot_axis_offset"] = 0
params["pixtol"] = 0.01 # Tolerance of registration in pixels
params["shiftmeth"] = "fourier" # 'sinc' or 'linear' better for noise
params["maxit"] = 20 # max of iterations
params["cliplow"] = None # clip air threshold
params["cliphigh"] = None# clip on sample threshold
params["sinohigh"] = None
params["sinolow"] = None
params["derivatives"] = True
params["opencl"] = True
params["autosave"]= False
params["apply_alignement"] = True
params["load_previous_shiftstack"] = False
params["correct_bad"] = True
params["bad_projs"] = [156, 226, 363, 371, 673, 990]  # starting at zero
#=========================

# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == '__main__':
    # loading the data
    aligned_diff, theta, shiftstack, params = LoadData.load(
        "aligned_derivatives.h5", **params
    )

    # to start at zero
    theta -= theta.min()

    # if you want to sort theta, uncomment line below:
    # aligned_diff, theta = sort_array(aligned_diff, theta)
    
    # initializing shiftstack
    if params["load_previous_shiftstack"]:
        shiftstack = LoadData.loadshiftstack("aligned_projections.h5", **inputparams)
        print("Using previous estimate of shiftstack")
    else:
        shiftstack[1] = np.zeros((1, aligned_diff.shape[0]))+params['rot_axis_offset']

    # calculate the sinogram
    sinogram = np.transpose(aligned_diff[:,params["slicenum"],:]).copy()

    # actual alignement
    shiftstack,aligned_sino = alignprojections_horizontal(sinogram,theta,shiftstack,**params)

    # alignement refinement
    while True:
        a = input("Do you want to refine further the alignment? ([y]/n): ").lower()
        if str(a) == "" or str(a) == "y":
            a1 = input("Do you want to use the same parameters? ([y]/n): ").lower()
            if a1 == "n":
                a1 = input("Slice number (e.g. {}): ".format(params["slicenum"]))
                if a1 != "":
                    params["slice_num"] = eval(a1)
                a2 = input("Pixel tolerance (e.g. {}): ".format(params["pixtol"]))
                if a2 != "":
                    params["pixtol"] = eval(a2)
                a3 = input(
                    "Filter Tomo cutoff (e.g. {}): ".format(params["filtertomo"])
                )
                if a3 != "":
                    params["filtertomo"] = eval(a3)
                a4 = input("Number of iterations (e.g. {}): ".format(params["maxit"]))
                if a4 != "":
                    params["maxit"] = eval(a4)
                a5 = input("Apply a circle (e.g. {}): ".format(params["circle"]))
                if a5 != "":
                    params["circle"] = eval(a5)
                a6 = input("Clipping high (e.g. {}): ".format(params["cliphigh"]))
                if a6 != "":
                    params["cliphigh"] = eval(a6)
            plt.close("all")

            # correcting bad projections
            if params["correct_bad"]:
                aligned_diff = replace_bad(aligned_diff, list_bad=params["bad_projs"], temporary=False)

            # calculate again the sinogram with corrected bad projections
            sinogram = np.transpose(aligned_diff[:, params["slice_num"], :]).copy()

            # actual alignment
            print('Starting the refinement of the alignment')
            shiftstack,aligned_sino = alignprojections_horizontal(sinogram,theta,shiftstack,**params)

        elif str(a) == "n":
            print("No further refinement done")
            break
        else:
            raise SystemExit("Unrecognized answer")


    a = input(
        "Do you want to reconstruct the slice with different parameters? ([y]/n) :"
    ).lower()
    if str(a) == "" or str(a) == "y":
        filtertomo = input("filtertomo (e.g. 0.7) = ")
        if filtertomo == "":
            filtertomo = 0.7
        else:
            filtertomo = eval(filtertomo)
        filtertype = str(input("filtertype (e.g. ram-lak) = ").lower())
        if filtertype == "":
            filtertype = "ram-lak"
        else:
            filtertype = str(filtertype)
        print("Calculating a tomographic slice")

        if params[u"opencl"]:
            print("Using opencl backprojector")
            B = None
            tomogram_align = mod_iradon2(
                aligned_sino,
                theta=theta,
                output_size=sinogram.shape[0],
                filter_type=filtertype,
                derivative=params[u"derivatives"],
                freqcutoff=filtertomo,
            )
        else:
            tomogram_align = mod_iradon(
                aligned_sino,
                theta=theta,
                output_size=sinogram.shape[0],
                filter_type=filtertype,
                derivative=params[u"derivatives"],
                freqcutoff=filtertomo,
            )

        print("Calculation done")

        if params["circle"] == True:
            N = tomogram_align.shape[0]
            xt = np.linspace(-np.fix(N / 2.0), np.ceil(N / 2.0), N, endpoint=False)
            Xt, Yt = np.meshgrid(xt, xt)
            circle = 1 - radtap(Xt, Yt, 10, N / 2 - 10)
        else:
            circle = 1
        # Display slice:
        plt.close("all")
        print("Slice: {}".format(params["slice_num"]))
        fig = plt.figure(num=5)
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(
            tomogram_align * circle,
            cmap="bone",
            interpolation="none",
            vmin=None,
            vmax=None,
        )  # ,vmin=-1.6,vmax=0.8)
        ax1.set_title(u"Slice {}".format(params["slice_num"]))
        fig.colorbar(im1)
        plt.show()
    elif str(a) == "n":
        pass
    else:
        print("Unrecognized answer")
        print("Calculation of tomographic slice not done")

    a = input("Are you happy with the reconstructed slice? ([y]/n) :").lower()
    if str(a) == "" or str(a) == "y":
        a1 = input(
            "Do you want to perform Tomographic consistency on multiples slices? (y/[n]):"
        ).lower()
    else:
        a1 = "n"
        raise SystemExit("Alignment not concluded")

    if str(a1) == "y":
        print("Starting Tomographic consistency on multiples slices")
        # Tomographic consistency on multiple slices
        # =========================
        # Repeat tomomographic consistency for 10 slices
        slices = np.arange(params["slice_num"] - 5, params["slice_num"] + 5)
        # =========================
        plt.close("all")
        deltaslice_prev = deltaslice.copy()
        deltaxrefine = []  # np.zeros((2,obj_shape[0]))
        for ii in slices:
            print("\nAligning slice {}".format(ii + 1))
            # create the sinogram
            sinogram = np.transpose(aligned_diff[:, ii, :])
            params[u"apply_alignement"] = False
            delta_aux, aligned_sino_aux = alignprojections_horizontal(
                sinogram, theta + 0.01, deltaslice, params
            )
            deltaxrefine.append(delta_aux)  # (ii,:) = deltaaux;
            deltaslice = delta_aux.copy()

        # deltaxrefine(slices,:)
        deltaxrefine = np.asarray(np.squeeze(deltaxrefine))
        deltaxrefine_avg = deltaxrefine.mean(axis=0)  # 2,1)

        plt.close("all")
        fig = plt.figure(num=6, figsize=(14, 8))
        ax1 = fig.add_subplot(211)
        ax1.imshow(deltaxrefine.astype(np.float), interpolation="none", cmap="jet")
        ax1.axis("tight")
        ax1.set_xlabel("Projection number")
        ax1.set_ylabel("Slice number")
        ax1.set_title("Displacements in x")
        ax2 = fig.add_subplot(212)
        ax2.plot(deltaxrefine_avg.astype(np.float), "b-", label="average")
        ax2.plot(deltaslice_prev[0], "r--", label="previous")
        ax2.legend()
        ax2.axis("tight")
        ax2.set_xlim([0, len(deltaxrefine_avg)])
        ax2.set_title("Average displacements in x")
        ax2.set_xlabel("Projection number")
        plt.tight_layout()
        plt.show()
        a = input(
            "Are you happy with the tomographic consistency alignment of the multiples slices? ([y]/n) "
        ).lower()
        if a == "" or a == "y":
            shiftstack[1] = deltaxrefine_avg.copy()
            print("Using the average of all shiftstack")
        else:
            shiftstack[1] = deltaslice_prev[0].copy()
            print(
                "Using the shiftstack before tomographic consisteny in multiple slices"
            )
    else:
        print("Tomo consistency on multiples slices not done")
        # special for this one because we used the cross-correlation
        shiftstack[1] = deltaslice[0].copy()

    # Shift projections (Only shift the horizontal)
    print("Shifting the projections")
    aligned_projections = np.zeros_like(aligned_diff)
    print("Shifting projections according to the alignment in x")

    # Compute the shifted images
    print("Computing aligned images")

    # TODO: Consider to use the function or separate the compute alignment to other function
    nprojs = aligned_diff.shape[0]
    for ii in range(nprojs):
        deltashift = (0, shiftstack[1][ii])
        if params["interpmeth"] == "sinc":
            if params["expshift"]:
                aligned_projections[ii] = np.angle(
                    shift_fft(
                        np.exp(1j * aligned_diff[ii]), deltashift, output_complex=True
                    )
                )
            else:
                aligned_projections[ii] = shift_fft(aligned_diff[ii], deltashift)
        elif params["interpmeth"] == "linear":
            aligned_projections[ii] = shift_linear(aligned_diff[ii], deltashift)
        else:
            raise ValueError("Unknown interpolation method")
        print("Image {} of {}".format(ii + 1, nprojs), end="\r")
    print("\r")

    # correcting bad projections after unwrapping
    if params[u"correct_bad"]:
        a = input("Do you want to correct bad projections?([y]/n)").lower()
        if str(a) == "" or str(a) == "y":
            for ii in params[u"bad_projs"]:
                print("Correcting bad projection (starts at 0): {}".format(ii))
                aligned_projections[ii] = (
                    aligned_projections[ii - 1] + aligned_projections[ii + 1]
                ) / 2  # this is better

    a = input("Do you want to display the aligned projections? (y/[n]) :").lower()
    if str(a) == "" or str(a) == "n":
        pass
    else:
        # Show aligned projections
        plt.close("all")
        plt.ion()
        fig = plt.figure(4)  # ,figsize=(14,6))
        ax1 = fig.add_subplot(111)
        im1 = ax1.imshow(aligned_projections[0], cmap="bone", vmin=-0.2, vmax=0.2)
        for ii in range(len(aligned_projections)):
            print("Projection: {}".format(ii + 1))
            projection = aligned_projections[ii]
            im1.set_data(projection)
            ax1.set_title("Projection {}".format(ii + 1))
            fig.canvas.draw()
        plt.ioff()

    # reconstructing the slice after alignement
    print("Reconstructing another slice for display")
    aligned_sino = np.transpose(aligned_projections[:, params["slice_num"], :]).copy()
    if params[u"opencl"]:
        print("Using opencl backprojector")
        B = None
        tomogram_align = mod_iradon2(
            aligned_sino,
            theta=theta,
            output_size=sinogram.shape[0],
            filter_type=filtertype,
            derivative=params[u"derivatives"],
            freqcutoff=filtertomo,
        )
    else:
        tomogram_align = mod_iradon(
            aligned_sino,
            theta=theta,
            output_size=sinogram.shape[0],
            filter_type=filtertype,
            derivative=params[u"derivatives"],
            freqcutoff=filtertomo,
        )
    print("Calculation done")

    # Display slice:
    plt.close("all")
    print("Slice: {}".format(params["slice_num"]))
    fig = plt.figure(num=6)
    ax1 = fig.add_subplot(111)
    im1 = ax1.imshow(
        tomogram_align * circle, cmap="bone", interpolation="none", vmin=None, vmax=None
    )  # ,vmin=-1.6,vmax=0.8)
    ax1.set_title(u"Slice {}".format(params["slice_num"]))
    fig.colorbar(im1)
    plt.show(block=False)

    # save horizontally aligned_projections
    S = SaveData(**inputparams)
    S("aligned_projections.h5", aligned_projections, theta, shiftstack)
    # next step
    print("You should run " "tomographic_reconstruction.py" " now")
    # =============================================================================#
