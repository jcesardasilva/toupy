#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
import functools
import fnmatch
import glob
import os
from pathlib import Path
import re
import time

# third party packages
import h5py
import matplotlib.pyplot as plt
import numpy as np

# local packages
from .filesrw import *
from .h5chunk_shape_3D import chunk_shape_3D
from ..utils import (
    checkhostname,
    convert_to_delta,
    convert_to_beta,
    padarray_bothsides,
    progbar,
    ShowProjections,
    plot_checkangles,
    replace_bad,
)

__all__ = [
    "remove_extraprojs",
    "PathName",
    "LoadProjections",
    "SaveData",
    "LoadData",
    "SaveTomogram",
    "LoadTomogram",
]


def remove_extraprojs(stack_projs, theta):
    """
    Remove extra projections of tomographic scans with projections at
    180, 90 and 0 degrees at the end

    Parameters
    ----------
    stack_projs : array_like
        Stack of projections with the first index correspoding to the
        projection number
    theta : array_like
        Array of theta values

    Returns
    -------
    stack_projs : array_like
        Stack of projections after the removal
    theta : array_like
        Array of theta values after the removal
    """

    print(theta[-5:])
    a = str(input("Do you want to remove extra thetas?([y]/n)")).lower()
    if a == "" or a == "y":
        a1 = eval(input("How many to remove?"))
        # the 3 last angles are 180, 90 and 0 degrees
        stack_projs = stack_projs[:-a1]
        theta = theta[:-a1]  # the 3 last angles are 180, 90 and 0 degrees
        print(theta[-5:])
    return stack_projs, theta


class SliceMaker(object):
    """
    Class to make array slices more efficient
    """

    def __getitem__(self, item):
        return item


class PathName:
    """
    Class to manage file location and paths
    """

    def __init__(self, **params):
        """
        Parameters
        ----------
        **params
            Dictionary of parameters
        params["pathfilename"] : str
            Path to first file and filename
        params["account"] : str
            User account number
        params["samplename"] : str
            Samplename
        params["regime"] : str
            Regime of imaging
        
        """
        self.pathfilename = os.path.abspath(params["pathfilename"])
        self.useraccount = params["account"]
        self.samplename = params["samplename"]
        self.scanprefix = params["scanprefix"]
        self.regime = params["regime"]
        self.filename = os.path.basename(self.pathfilename)  # data filename
        self.dirname = os.path.dirname(self.pathfilename)  # data filename
        self.fileprefix, self.fileext = os.path.splitext(
            self.filename
        )  # filename and extension
        # metadata filename
        if self.fileext == ".edf":  # edf
            self.rootpath = Path(self.pathfilename).parents[2]
        else:
            self.rootpath = Path(self.pathfilename).parents[4]
        self.icath5file = "{}-id16a.h5".format(self.useraccount)
        self.icath5path = os.path.join(
            self.rootpath, self.icath5file
        )  # metadata filename path

    def datafilewcard(self):
        """
        Create file wildcard to search for files
        """

        file_wcard = re.sub(
            self.samplename + r"\w*", self.samplename + "*", self.fileprefix
        )  # file_wcard
        # ~ file_wcard = re.sub(self.samplename+'\w*',self.samplename+'*_ML',fileprefix)
        return file_wcard

    def metadatafilewcard(self):
        """
        Create file wildcard to search for metafiles
        """

        if not os.path.isfile(self.icath5path):
            raise IOError("File {} not found".format(self.icath5file))
        if self.fileext == ".ptyr":  # Ptypy
            metafile_wcard = re.sub(
                r"_subtomo\d{3}_\d{4}_\w+",
                "_subtomo*",
                os.path.splitext(self.filename)[0],
            )
        elif self.fileext == ".cxi":  # PyNX
            metafile_wcard = re.sub(
                r"_subtomo\d{3}_\d{4}", "_subtomo*", os.path.splitext(self.filename)[0]
            )
        elif self.fileext == ".edf":  # edf
            metafile_wcard = re.sub(r"_\d{4}$", "*", os.path.splitext(self.filename)[0])
        else:
            raise IOError(
                "File {} is not a .ptyr, nor a .cxi, nor a .edf file. Please, load a compatible file.".format(
                    self.filename
                )
            )
        return metafile_wcard

    def search_projections(self):
        """
        Search for projection given the filenames
        """
        print("Path: {}".format(self.dirname))
        print("First projection file: {}".format(self.filename))
        if self.fileext == ".edf":
            scan_wcard = re.sub(r"_\d{4}.edf", "*.edf", self.pathfilename)
        elif self.fileext == ".ptyr":  # Ptypy
	    # TODO: correct the path name if the calculations are done in cuda or not
            # ~ scan_wcard = os.path.join(
                # ~ re.sub(self.samplename + r"_\w*", self.samplename + "_*", self.dirname),
                # ~ self.metadatafilewcard() + "_ML" + self.fileext,
            # ~ )
            scan_wcard = os.path.join(
                re.sub(self.scanprefix + r"_\w*", self.scanprefix + "_*", self.dirname),
                self.metadatafilewcard() + "_ML_pycuda" + self.fileext,
            )
        elif self.fileext == ".cxi":  # PyNX
            scan_wcard = os.path.join(
                re.sub(self.samplename + r"_\w*", self.samplename + "_*", self.dirname),
                self.metadatafilewcard() + self.fileext,
            )
        else:
            raise IOError(
                "File {} is not a .ptyr, nor a .cxi, nor a .edf file. Please, load a compatible file.".format(
                    self.filename
                )
            )
        return scan_wcard

    def results_folder(self):
        """
        create path for the result folder
        """
        aux_wcard = re.sub(r"\*", "", self.datafilewcard())
        if self.regime == "nearfield":
            foldername = aux_wcard + "_nfpxct"
        elif self.regime == "farfield":
            foldername = aux_wcard + "_pxct"
        elif self.regime == "holoct":
            foldername = aux_wcard + "_hxct"
        else:
            raise ValueError("Unrecognized regime")
        results_path = os.path.join(os.path.dirname(self.dirname), foldername)
        if not os.path.isdir(results_path):
            print("Directory does not exist. Creating the directory...")
            os.makedirs(results_path)
        return results_path

    def results_datapath(self, h5name):
        """
        create path for the h5file in result folder
        """
        aux_path = self.results_folder()
        h5file = os.path.join(aux_path, h5name)
        return h5file


class Variables(object):
    """
    Auxiliary class to initialize some variables
    """

    showrecons = False
    phaseonly = False
    amponly = False
    autosave = False
    checkextraprojs = True
    cxientry = None
    missingprojs = False
    missingnum = None
    border_crop_x = None
    border_crop_y = None
    shiftmeth = "fourier"
    derivatives = True
    calc_derivatives = False
    correct_bad = False
    bad_projs = []
    opencl = True
    load_previous_shiftstack = False
    algorithm = "FBP"


class LoadProjections(PathName, Variables):
    """
    Load the reconstructed projections from the ptyr files
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.params = params
        try:
            self.showrecons = params["showrecons"]
        except:
            pass
        try:
            self.border_crop_x = params["border_crop_x"]
        except:
            pass
        try:
            self.border_crop_y = params["border_crop_y"]
        except:
            pass
        try:
            self.checkextraprojs = params["checkextraprojs"]
        except:
            pass
        try:
            self.missingprojs = params["missingprojs"]
        except:
            pass
        try:
            self.missingnum = params["missingnum"]
        except:
            pass
        try:
            self.cxientry = params["cxientry"]
        except:
            pass

        if self.showrecons:
            self.SP = ShowProjections()

        if self.fileext == ".ptyr":  # Ptypy
            self.read_reconfile = read_ptyr
        elif self.fileext == ".cxi":  # PyNX
            self.read_reconfile = read_cxi
        elif self.fileext == ".edf":  # edf projections
            self.read_reconfile = read_edf
        else:
            raise IOError(
                "File {} is not a .ptyr, nor a .cxi, nor a .edf file. Please, load a compatible file.".format(
                    self.filename
                )
            )

        # create_paramsh5(**params)

        # get the list of files to load
        self.proj_files = sorted(glob.glob(self.search_projections()))

    def __call__(self):
        return self._load_projections()

    @classmethod
    def load(cls, **params):
        """
        Load the reconstructed projections from phase-retrieved files.

        Parameters
        ----------
        **params
            Container with parameters to load the files.
        params["account"] : str
            User experiment number at ESRF.
        params["samplename"] : str
            Sample name
        params["pathfilename"] : str
            Path to the first projection file.
        params["regime"] : str
            Imaging regime. The options are: `nearfield`, `farfield`,
            `holoct`.
        params["showrecons"] : bool
            To show or not the projections once loaded
        params["autosave"] : bool
            Save the projections once load without asking
        params["phaseonly"] : bool
            Load only phase projections. Used when the projections are
            complex-valued.
        params["amponly"] : bool
            Load only amplitude projections. Used when the projections are
            complex-valued.
        params["border_crop_x"] : int, None
            Amount of pixels to crop at each border in x.
        params["border_crop_y"] : int, None
            Amount of pixels to crop at each border in y.
        params["checkextraprojs"] : bool
            Check for the projections acquired at and over 180 degrees.
        params["missingprojs"] : bool
            Allow to interpolate for missing projections. The numbers of
            the projections need to be provided in params["missingnum"].
        params["missingnum"] : list of ints
            Numbers of the missing projections to be interpolated.

        Returns
        -------
        stack_objs : array_like
            Array containing the projections
        stack_angles : array_like
            Array containing the thetas
        pxsize : list of floats
            List containing the pixel size in the vertical and horizontal
            directions. Typically, the resolution is isotropic and the
            two values are the same
        paramsload : dict
            Parameters of the loading
        """
        return cls(**params)._load_projections()

    @classmethod
    def loadedf(cls, **params):
        """
        Load the reconstructed projections from the edf files
        This is adapted for the phase-contrast imaging generating
        projections as edf files

        Parameters
        ----------
        **params
            Container with parameters to load the files.
        params["account"] : str
            User experiment number at ESRF.
        params["samplename"] : str
            Sample name
        params["pathfilename"] : str
            Path to the first projection file.
        params["regime"] : str
            Imaging regime. The options are: `nearfield`, `farfield`,
            `holoct`.
        params["showrecons"] : bool
            To show or not the projections once loaded
        params["autosave"] : bool
            Save the projections once load without asking

        Returns
        -------
        stack_objs : array_like
            Array containing the projections
        stack_angles : array_like
            Array containing the thetas
        pxsize : list of floats
            List containing the pixel size in the vertical and horizontal
            directions. Typically, the resolution is isotropic and the
            two values are the same
        paramsload : dict
            Parameters of the loading
        """
        return cls(**params)._load_edfprojections()

    def check_angles(self):
        """
        Find the angles of the projections and plot them to be checked
        Specific to ID16A beamline (ESRF)
        """
        thetas = {}
        with h5py.File(self.icath5path, "r") as fid:
            sorted_keys = sorted(list(fid.keys()))
            for keys in sorted_keys:
                if fnmatch.fnmatch(keys, "*" + self.metadatafilewcard()):
                    try:
                        # old style at ID16A beamline
                        positioners = fid[keys + "/sample/positioner/value"][()]
                    except KeyError:
                        # new style at ID16A beamline
                        positioners = fid[keys + "/sample/positioners/value"][()]
                    thetas[keys] = np.float(positioners.split()[0])
        # remove additional angles
        if self.checkextraprojs:
            theta_keys = sorted(list(thetas.keys()))
            thetas_array = np.array([ii for ii in thetas.values()])
            thetas_array -= thetas_array.min()
            idxend = int(np.where(thetas_array == 180)[0])
            print(theta_keys[idxend:])
            if theta_keys[idxend:] != []:
                print(
                    "Removing projections at the end of the scan (180,90, and 0 degrees)"
                )
                [thetas.pop(keyrm) for keyrm in theta_keys[idxend:]]
                rmkeys = [ii.split()[-1] for ii in theta_keys[idxend:]]
                for ii in rmkeys:
                    [self.proj_files.remove(s) for s in self.proj_files if ii in s]

        # checking the angles
        print("Checking the angles")
        angles = []
        deltaidx = 0  # in case of repeated values
        sorted_thetakeys = sorted(thetas.keys())
        for idx, keys in enumerate(sorted_thetakeys):
            th = np.float(thetas[keys])
            if th == np.float(thetas[sorted_thetakeys[idx - 1]]):
                print("Found repeated value of theta. Discarding it")
                deltaidx += 1
                continue
            print("Projection {}: {} degrees".format(idx + 1 - deltaidx, thetas[keys]))
            angles.append(th)

        # plot the angles for verification
        plot_checkangles(angles)
        a = input("Are the angles ok?([Y]/n)").lower()
        if a == "" or a == "y":
            print("Continuing...")
        else:
            raise SystemExit("Exiting")
        return angles, thetas

    def _remove_extraprojs(self, thetas, proj_files):
        """
        Remove extra projections of tomographic scans with projections at
        180, 90 and 0 degrees at the end

        Parameters
        ----------
        theta : array_like
            Array of theta values
        proj_files : list of str
            List of projection files

        Returns
        -------
        stack_projs : array_like
            Stack of projections after the removal
        theta : array_like
            Array of theta values after the removal
        """
        print("The final 5 angles are: {}".format(list(thetas[-5:])))
        a = str(input("Do you want to remove extra thetas?([y]/n)")).lower()
        if a == "" or a == "y":
            a1 = input("How many to remove?(default=3) ")
            if a1 == "":
                rmnum = 3
            else:
                rmnum = eval(a1)
            # the 3 last angles are 180, 90 and 0 degrees
            proj_files = proj_files[:-rmnum]
            # the 3 last angles are 180, 90 and 0 degrees
            thetas = thetas[:-rmnum]
            print("The final 5 angles are now: {}".format(list(thetas[-5:])))
        plot_checkangles(thetas)  # re-ploting for checking
        return proj_files

    @staticmethod
    def insert_missing(stack_objs, theta, missingnum):
        """
        Insert missing projections by interpolation of neighbours
        """
        # special: insert the information of the missing projections
        print("Inserting the missing projections:{}".format(missingnum))
        delta_theta = theta[1] - theta[0]
        for ii in missingnum:
            print("Projection: {}".format(ii), end="\r")
            theta = np.insert(theta, ii, theta[ii - 1] + delta_theta)
            stack_objs = np.insert(stack_objs, ii, stack_objs[ii - 1], axis=0)
        print("\r")
        return stack_objs, theta

    @checkhostname
    def _load_projections(self):
        """
        Load the reconstructed projections from the ptyr files
        """
        # get the angles
        angles, thetas = self.check_angles()

        # count the number of available projections
        num_projections = len(self.proj_files)
        a = input(
            "I have found {} projections. Do you want to continue?([Y]/n)".format(
                num_projections
            )
        ).lower()
        if a == "" or a == "y":
            print("Continuing...")
            plt.close("all")
        else:
            raise SystemExit("Exiting the script")

        # Read the first projection to check size and reconstruction parameters
        objs0, probe0, pxsize, energy = self.read_reconfile(self.pathfilename)
        # add the information of pixelsize and energy to params
        paramsload = dict()
        paramsload.update(self.params)
        paramsload["pixelsize"] = pxsize
        paramsload["energy"] = energy
        # crop image if requested
        objs0 = crop_array(objs0, self.border_crop_x, self.border_crop_y)
        nr, nc = objs0.shape
        # ~ print(objs0.shape)
        if pxsize[0] != pxsize[1]:
            raise SystemExit("Pixel size is not symmetric. Exiting the script")
        print(
            "the pixelsize of the first projection is {:.2f} nm".format(pxsize[0] * 1e9)
        )

        # initialize the array for the stack objects
        stack_objs = np.empty((num_projections, nr, nc), dtype=np.complex64)
        stack_angles = np.empty(num_projections, dtype=np.float32)

        # reads the ptyr or cxi files and get object and probe in a stack
        for idxp, proj in enumerate(self.proj_files):
            print("\nProjection: {}".format(idxp))
            print("Reading: {}".format(proj))
            objs, probes, pxsize, energy = self.read_reconfile(proj)  # reading file
            # crop image if requested
            if self.border_crop_x is not None:
                if self.border_crop_y is not None:
                    objs = crop_array(objs, self.border_crop_x, self.border_crop_y)
            # check if same size, otherwise pad
            if objs.shape != objs0.shape:
                print("########################")
                objs = padarray_bothsides(objs, (nr, nc), padmode="edge")
                print("File {} has different shape and was padded".format(proj))
                print("########################")

            # update stack_objs
            stack_objs[idxp] = objs

            # compare projection name with thetas dictionary and associate angles
            if self.fileext == ".ptyr":
                key_finder = os.path.basename(os.path.dirname(proj))
            elif self.fileext == ".cxi":
                key_finder = os.path.splitext(os.path.basename(proj))[0]
            # compare projection name with thetas dictionary and associate angles
            for keys in sorted(thetas.keys()):
                if keys.find(key_finder) != -1:
                    stack_angles[idxp] = thetas[keys]
                    print("Angle: {}".format(thetas[keys]))
                    break
            if self.showrecons:
                print("Showing projection {}".format(idxp + 1))
                self.SP.show_projections(objs, probes, idxp)

        nprojs, nr, nc = stack_objs.shape
        print("\nNumber of projections loaded: {}".format(nprojs))

        if self.missingprojs:
            stack_objs, stack_angles = self.insert_missing(
                stack_objs, stack_angles, self.missingnum
            )
            nprojs, nr, nc = stack_objs.shape
            print("New number of projections: {}".format(nprojs))
        print("Dimensions {} x {} pixels".format(nr, nc))
        print("All projections loaded\n")
        return stack_objs, stack_angles, pxsize, paramsload

    @checkhostname
    def _load_edfprojections(self):
        """
        Load the reconstructed projections from the edf files
        This is adapted for the phase-contrast imaging generating
        projections as edf files
        """
        ## to be tested
        # notuseful = sorted(glob.glob(self.samplename+'*_[0-4].edf'))
        # self.proj_files = [ii for ii in self.proj_files if ii not in notuseful]

        # remove the last projection, which is 180 degrees
        self.proj_files = self.proj_files[:-1]

        # count the number of available projections
        num_projections = len(self.proj_files)

        a = input(
            "I have found {} projections. Do you want to continue?([Y]/n)".format(
                num_projections
            )
        ).lower()
        if a == "" or a == "y":
            print("Continuing...")
            plt.close("all")
        else:
            raise SystemExit("Exiting the script")

        # Read the first projection to check size and reconstruction parameters
        objs0, pxsize, energy, nvue = self.read_reconfile(self.pathfilename)
        if nvue != num_projections:
            raise ValueError("The number of projections is different from nvue in file")
        nr, nc = objs0.shape
        # add the information of pixelsize and energy to params
        paramsload = dict()
        paramsload.update(self.params)
        paramsload["pixelsize"] = pxsize
        paramsload["energy"] = energy
        print("the pixelsize of the first projection is {:.2f} nm".format(pxsize * 1e9))

        # initialize the array for the stack objects
        stack_objs = np.empty((num_projections, nr, nc), dtype=np.float32)
        stack_angles = np.arange(0, 180, 180 / nvue, dtype=np.float32)
        if stack_angles.shape[0] != num_projections:
            raise ValueError(
                "The number of projections is different from number of thetas"
            )

        # reads the edf files and get object and probe in a stack
        for idxp, proj in enumerate(self.proj_files):
            print("\nProjection: {}".format(idxp))
            print("Reading: {}".format(proj))
            objs, _, _, _ = self.read_reconfile(proj)  # reading file
            # update stack_objs
            stack_objs[idxp] = objs
            if self.showrecons:
                print("Showing projection {}".format(idxp + 1))
                self.SP.show_projections(objs, probes, idxp)

        nprojs, nr, nc = stack_objs.shape
        print("\nNumber of projections loaded: {}".format(nprojs))
        print("Dimensions {} x {} pixels".format(nr, nc))
        print("All projections loaded\n")
        return stack_objs, stack_angles, pxsize, paramsload


class SaveData(PathName, Variables):
    """
    Save projections to HDF5 file
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.params = params
        try:
            self.cxientry = params["cxientry"]
        except:
            pass
        try:
            self.autosave = params["autosave"]
        except:
            pass

    def __call__(self, *args):  # h5name,stack_projs,theta,shiftstack):
        return self._save_data(*args)

    @classmethod
    def save(cls, *args, **params):
        """
        Save data to HDF5 File

        Parameters
        ----------
        *args
            positional arguments
        args[0] : str
            H5 file name
        args[1] : array_like
            Array containing the stack of projections
        args[2] : array_like
            Values of theta
        args[3] : array_like
            Array containing the shifts for each projection in the
            stack. If not provided, it will be initialized with zeros
        args[4] : array_like or None
            Array containing the projection masks
        """
        return cls(**params)._save_data(*args)

    @classmethod
    def saveFSC(cls, *args, **params):
        """
        Save FSC data to HDF5 file

        Parameters
        ----------
        *args
            positional arguments
        args[0] : str
            H5 file name
        args[1] : array_like
            Normalized frequencies
        args[2] : array_like
            Value of the threshold for each frequency
        args[3] : array_like
            The FSC curve
        args[4] : array_like
            The first tomogram
        args[5] : array_like
            The second tomogram
        args[6] : array_like
            The array of theta values
        args[7] : float
            Pixel size
        """
        return cls(**params)._save_FSC(*args)

    @classmethod
    def savemasks(cls, *args, **params):
        return cls(**params)._save_masks(*args)

    def _save_masks(self, h5name, masks):
        """
        Save masks for the linear phase ramp removal of the phase
        contrast image or the air removal from the amplitude images
        """
        print("Saving {}".format(h5name))
        h5file = self.results_datapath(h5name)
        if os.path.isfile(h5file):
            os.remove(h5file)
        with h5py.File(h5file, "a") as fid:
            fid.create_dataset(
                "masks/stack", data=masks, dtype=np.bool
            )  # air/vacuum mask
        print("Done")

    def savecheck(func):
        """
        Decorator for save data
        """

        @functools.wraps(func)
        def new_func(self, *args, **params):
            if self.autosave:
                ansuser = "y"
                func(self, *args)
            else:
                while True:
                    ansuser = input(
                        "Do you want to save the data to HDF5 file? ([y]/n) "
                    ).lower()
                    if ansuser == "" or ansuser == "y":
                        func(self, *args)
                        break
                    elif ansuser == "n":
                        print("The data have NOT been saved yet. Please, be careful")
                        break
                    else:
                        print("You have to answer y or n")

        return new_func

    @savecheck
    def _save_data(self, *args):
        """
        Save data to HDF5 File

        Parameters
        ----------
        *args
            positional arguments
        args[0] : str
            H5 file name
        args[1] : array_like
            Array containing the stack of projections
        args[2] : array_like
            Values of theta
        args[3] : array_like
            Array containing the shifts for each projection in the
            stack. If not provided, it will be initialized with zeros
        args[4] : array_like or None
            Array containing the projection masks
        """
        h5name = args[0]
        stack_projs = args[1]
        nprojs, nr, nc = stack_projs.shape
        theta = args[2]

        if len(args) == 4:
            shiftstack = args[3]
        else:
            shiftstack = np.zeros((2, nprojs), dtype=np.float32)

        if len(args) == 5:
            masks = args[4]
        else:
            masks = None

        if np.iscomplexobj(stack_projs[0]):
            array_dtype = np.complex64
        else:
            array_dtype = np.float32

        # calculate the chunk size for writing the HDF5 files
        chunk_size = chunk_shape_3D(stack_projs.shape)

        print("Saving {}".format(h5name))
        h5file = self.results_datapath(h5name)
        if os.path.isfile(h5file):
            print("File {} already exists and will be overwritten".format(h5name))
            os.remove(h5file)
        print("\rSaving metadata...", end="")
        write_paramsh5(h5file, **self.params)
        create_paramsh5(**self.params)
        print("\b\b Done")
        print("Saving data. This takes time, please wait...")
        with h5py.File(h5file, "a") as fid:
            fid.create_dataset(
                "shiftstack/shiftstack", data=shiftstack, dtype=np.float32
            )  # shiftstack
            fid.create_dataset("angles/thetas", data=theta, dtype=np.float32)  # thetas
            dset = fid.create_dataset(
                "projections/stack",
                shape=(nprojs, nr, nc),
                dtype=array_dtype,
                chunks=chunk_size,
            )
            p0 = time.time()
            for ii in range(nprojs):
                strbar = "Projection: {} out of {}".format(ii + 1, nprojs)
                dset[ii : ii + 1, :, :] = stack_projs[ii]  # avoid fancy slicing
                progbar(ii + 1, nprojs, strbar)
            print("\r")
            if masks is not None:
                fid.create_dataset(
                    "masks/stack", data=masks, dtype=np.bool
                )  # air/vacuum mask
            print("Done. Time elapsed = {:.03f} s".format(time.time() - p0))
        print("Data saved to file {}".format(h5name))
        print("In the folder {}".format(self.results_folder()))

    def _save_FSC(self, *args):
        """
        Save FSC data to HDF5 file

        Parameters
        ----------
        *args
            positional arguments
        args[0] : str
            H5 file name
        args[1] : array_like
            Normalized frequencies
        args[2] : array_like
            Value of the threshold for each frequency
        args[3] : array_like
            The FSC curve
        args[4] : array_like
            The first tomogram
        args[5] : array_like
            The second tomogram
        args[6] : array_like
            The array of theta values
        args[7] : float
            Pixel size
        """
        h5name = args[0]
        normfreqs = args[1]
        T = args[2]
        FSCcurve = args[3]
        tomogram1 = args[4]
        tomogram2 = args[5]
        theta = args[6]
        pxsize = args[7]

        print("Saving {}".format(h5name))
        h5file = self.results_datapath(h5name)
        if os.path.isfile(h5file):
            print("File {} already exists and will be overwritten".format(h5name))
            os.remove(h5file)
        print("\rSaving metadata...", end="")
        write_paramsh5(h5file, **self.params)
        create_paramsh5(**self.params)
        print(". Done")
        print("Saving data. This takes time, please wait...")
        p0 = time.time()
        with h5py.File(h5file, "a") as fid:
            fid.create_dataset(
                "angles/thetas", data=theta, dtype=np.float32
            )  # add the thetas
            fid.create_dataset(
                "FSC", data=FSCcurve, dtype=np.float32
            )  # add the FSC curve
            # add the threshold
            fid.create_dataset("T", data=T, dtype=np.float32)
            fid.create_dataset(
                "normfreqs", data=normfreqs, dtype=np.float32
            )  # add the normalized freqs
            if tomogram1.ndim == 2:
                fid.create_dataset("tomogram1", data=tomogram1, dtype=np.float32)
                fid.create_dataset("tomogram2", data=tomogram2, dtype=np.float32)
            elif tomogram1.ndim == 3:
                # calculate the chunk size for writing the HDF5 files
                chunk_size = chunk_shape_3D(tomogram1.shape)
                print("Saving tomogram1 and tomogram2. This takes time, please wait...")
                dset1 = fid.create_dataset(
                    "tomogram1",
                    shape=tomogram1.shape,
                    dtype=np.float32,
                    chunks=chunk_size,
                )
                dset2 = fid.create_dataset(
                    "tomogram2",
                    shape=tomogram2.shape,
                    dtype=np.float32,
                    chunks=chunk_size,
                )
                nslices, nr, nc = tomogram1.shape
                for ii in range(nslices):
                    print(" Slice: {} out of {}".format(ii + 1, nslices), end="\r")
                    dset1[ii, :, :] = tomogram1[ii]
                    dset2[ii, :, :] = tomogram2[ii]
                    progbar(ii + 1, nslices)
                # ~ print('\r')
            print("Done. Time elapsed = {:.03f} s".format(time.time() - p0))
        print("FSC data saved to file {}".format(h5name))
        print("In the folder {}".format(self.results_folder()))


class LoadData(PathName, Variables):
    """
    Load projections from HDF5 file
    """

    def __init__(self, **params):
        paramsh5 = load_paramsh5(**params)
        super().__init__(**paramsh5)
        self.params = paramsh5
        self.params.update(params)
        try:
            self.amponly = params["amponly"]
        except:
            pass
        try:
            self.phaseonly = params["phaseonly"]
        except:
            pass
        try:
            self.loadroi = params["loadroi"]
        except:
            self.loadroi = False
        if self.loadroi:
            self.roi = params["roi"]
            if self.roi == []:
                print("ROI not specified. Loading entire dataset")
                self.loadroi = False

    def __call__(self, h5name):
        return self._load_data(h5name)

    @classmethod
    def load(cls, *args, **params):
        """
        Load data from h5 file

        Parameters
        ----------
        h5name: str
            File name from which data is loaded
        **params
            Dictionary of additonal parameters
        params["autosave"] : bool
            Save the projections once load without asking
        params["phaseonly"] : bool
            Load only phase projections. Used when the projections are
            complex-valued.
        params["amponly"] : bool
            Load only amplitude projections. Used when the projections are
            complex-valued.
        params["pixtol"] : float
            Tolerance for alignment, which is also used as a search step
        params["alignx"] : bool
            True or False to activate align x using center of mass
            (default= False, which means align y only)
        params["shiftmeth"] : str
            Shift images with fourier method (default). The options are
            `linear` ->  Shift images with linear interpolation (default);
            `fourier` -> Fourier shift or `spline` -> Shift images with spline
            interpolation.
        params["circle"] : bool
            Use a circular mask to eliminate corners of the tomogram
        params["filtertype"] : str
            Filter to use for FBP
        params["freqcutoff"] : float
            Frequency cutoff for tomography filter (between 0 and 1)
        params["cliplow"] : float
            Minimum value in tomogram
        params["cliphigh"] : float
            Maximum value in tomogram
        params["correct_bad"] : bool
            If true, it will interpolate bad projections. The numbers of
            projections to be corrected is given by `params["bad_projs"]`.
        params["bad_projs"] : list of ints
            List of projections to be interpolated. It starts at 0.

        Returns
        -------
        stack_projs: array_like
            Stack of projections
        theta: array_like
            Stack of thetas
        shiftstack : array_like
            Shifts in vertical (1st dimension) and horizontal (2nd dimension)
        datakwargs : dict
            Dictionary with metadata information
        """

        return cls(**params)._load_data(*args)

    @classmethod
    def load_olddata(cls, *args, **params):
        """
        Load old data from h5 file. It should disappear soon. 

        Parameters
        ----------
        h5name: str
            File name from which data is loaded
        **params
            Dictionary of additonal parameters
        params["autosave"] : bool
            Save the projections once load without asking
        params["phaseonly"] : bool
            Load only phase projections. Used when the projections are
            complex-valued.
        params["amponly"] : bool
            Load only amplitude projections. Used when the projections are
            complex-valued.
        params["pixtol"] : float
            Tolerance for alignment, which is also used as a search step
        params["alignx"] : bool
            True or False to activate align x using center of mass
            (default= False, which means align y only)
        params["shiftmeth"] : str
            Shift images with fourier method (default). The options are
            `linear` ->  Shift images with linear interpolation (default);
            `fourier` -> Fourier shift or `spline` -> Shift images with spline
            interpolation.
        params["circle"] : bool
            Use a circular mask to eliminate corners of the tomogram
        params["filtertype"] : str
            Filter to use for FBP
        params["freqcutoff"] : float
            Frequency cutoff for tomography filter (between 0 and 1)
        params["cliplow"] : float
            Minimum value in tomogram
        params["cliphigh"] : float
            Maximum value in tomogram
        params["correct_bad"] : bool
            If true, it will interpolate bad projections. The numbers of
            projections to be corrected is given by `params["bad_projs"]`.
        params["bad_projs"] : list of ints
            List of projections to be interpolated. It starts at 0.

        Returns
        -------
        stack_projs: array_like
            Stack of projections
        theta: array_like
            Stack of thetas
        shiftstack : array_like
            Shifts in vertical (1st dimension) and horizontal (2nd dimension)
        datakwargs : dict
            Dictionary with metadata information
        """

        return cls(**params)._load_olddata(*args)

    @classmethod
    def loadshiftstack(cls, *args, **params):
        """
        Load shitstack from previous h5 file

        Parameters
        ----------
        h5name : str
            File name from which data is loaded

        Returns
        -------
        shiftstack : array_like
            Shifts in vertical (1st dimension) and horizontal (2nd dimension)
        """
        return cls(**params)._load_shiftstack(*args)

    @classmethod
    def loadtheta(cls, *args, **params):
        """
        Load shitstack from previous h5 file

        Parameters
        ----------
        h5name : str
            File name from which data is loaded

        Returns
        -------
        shiftstack : array_like
            Shifts in vertical (1st dimension) and horizontal (2nd dimension)
        """
        return cls(**params)._load_theta(*args)

    @classmethod
    def loadmasks(cls, *args, **params):
        """
        Load masks from previous h5 file

        Parameters
        ----------
        h5name: str
            File name from which data is loaded

        Returns
        -------
        masks: array_like
            Array with the masks
        """
        return cls(**params)._load_masks(*args)

    def _load_shiftstack(self, h5name):
        """
        Load shitstack from previous h5 file

        Parameters
        ----------
        h5name : str
            File name from which data is loaded

        Returns
        -------
        shiftstack : array_like
            Shifts in vertical (1st dimension) and horizontal (2nd dimension)
        """
        print("Loading shiftstack from file {}".format(h5name))
        h5file = self.results_datapath(h5name)
        with h5py.File(h5file, "r") as fid:
            shiftstack = fid["shiftstack/shiftstack"][()]
        return shiftstack

    def _load_theta(self, h5name):
        """
        Load shitstack from previous h5 file

        Parameters
        ----------
        h5name : str
            File name from which data is loaded

        Returns
        -------
        shiftstack : array_like
            Shifts in vertical (1st dimension) and horizontal (2nd dimension)
        """
        print("Loading thetas from file {}".format(h5name))
        h5file = self.results_datapath(h5name)
        with h5py.File(h5file, "r") as fid:
            theta = fid["angles/thetas"][()]
        return theta

    def _load_masks(self, h5name):
        """
        Load masks from previous h5 file

        Parameters
        ----------
        h5name: str
            File name from which data is loaded

        Returns
        -------
        masks: array_like
            Array with the masks
        """
        print("Loading the projections from file {}".format(h5name))
        h5file = self.results_datapath(h5name)
        with h5py.File(h5file, "r") as fid:
            masks = fid["masks/stack"][()]
        return masks

    @checkhostname
    def _load_data(self, h5name):
        """
        Load data from h5 file

        Parameters
        ----------
        h5name: str
            File name from which data is loaded

        Returns
        -------
        stack_projs: array_like
            Stack of projections
        theta: array_like
            Stack of thetas
        shiftstack : array_like
            Shifts in vertical (1st dimension) and horizontal (2nd dimension)
        datakwargs : dict
            Dictionary with metadata information
        """
        h5file = self.results_datapath(h5name)
        shiftstack = self._load_shiftstack(h5name)
        theta = self._load_theta(h5name)
        it0 = time.time()
        with h5py.File(h5file, "r") as fid:
            # read the inputkwargs dict
            datakwargs = dict()
            print("\rLoading metadata...", end="")
            for keys in sorted(list(fid["info"].keys())):
                datakwargs[keys] = fid["info/{}".format(keys)][()]
            datakwargs.update(self.params)  # add/update with new values
            print("\b\b Done")
            print("Loading the projections from file {}".format(h5name))
            dset = fid["projections/stack"]
            # ~ stack_projs = dset[()]
            if self.loadroi:
                roi = self.roi
                nprojs, nr, nc = [roi[ii + 1] - roi[ii] for ii in range(0, len(roi), 2)]
                print("\rInitializing array...", end="")
                stack_projs = np.empty((nprojs, nr, nc), dtype=dset.dtype)
                print("\b\b Done")
                print("Loading. This takes time, please wait...")
                for ii in [projs]:
                    strbar = "Projection: {} out of {}".format(ii + 1, nprojs)
                    stack_projs[ii, roi[2] : roi[3], roi[4] : roi[5]] = dset[
                        ii, roi[2] : roi[3], roi[4] : roi[5]
                    ]
                    progbar(ii + 1, nprojs, strbar)
                print("\r")
            else:
                nprojs = dset.shape[0]
                print("\rInitializing array...", end="")
                stack_projs = np.empty(dset.shape, dtype=dset.dtype)
                print("\b\b Done")
                print("Loading. This takes time, please wait...")
                for ii in range(nprojs):
                    strbar = "Projection: {} out of {}".format(ii + 1, nprojs)
                    stack_projs[ii, :, :] = dset[ii, :, :]
                    progbar(ii + 1, nprojs, strbar)
                print("\r")
        if self.amponly and np.iscomplexobj(stack_projs):
            print("\rTaking only amplitudes...", end="")
            stack_projs = np.abs(stack_projs)
            print("\b\b Done")
        elif self.phaseonly and np.iscomplexobj(stack_projs):
            print("\rTaking only phases...", end="")
            stack_projs = np.angle(stack_projs)
            print("\b\b Done")
        if self.params["correct_bad"]:
            stack_projs = replace_bad(stack_projs,
                list_bad = self.params["bad_projs"],
                temporary=True
                )
        print("Projections loaded from file {}".format(h5name))
        print("Time elapsed = {:.03f} s".format(time.time() - it0))
        return stack_projs, theta, shiftstack, datakwargs

    @checkhostname
    def _load_olddata(h5name, **params):
        """
        Load data from the old-format h5 file.

        Parameters
        ----------
        h5name: str
            File name from which data is loaded

        Returns
        -------
        stack_projs : array_like
            Stack of projections
        theta : array_like
            Stack of thetas
        shiftstack : array_like
            Shifts in vertical (1st dimension) and horizontal (2nd dimension)
        datakwargs : dict
            Dictionary with metadata information

        Note
        ----
        May be deprecated soon.
        """
        h5file = self.results_datapath(h5name)
        shiftstack = self._load_shiftstack(h5name)
        theta = self._load_theta(h5name)
        it0 = time.time()
        print("The file format is old.")
        with h5py.File(h5file, "r") as fid:
            # read the inputkwargs dict
            datakwargs = dict()
            print("\rLoading metadata...", end="")
            for keys in sorted(list(fid["info"].keys())):
                datakwargs[keys] = fid["info/{}".format(keys)][()]
            datakwargs.update(self.params)  # add/update with new values
            print("\b\b Done")
            datakwargs["pixelsize"] = fid["pixelsize"][()]
            dset0 = fid["aligned_projections_proj/projection_000"]
            nr, nc = dset0.shape
            key_list = list(fid["aligned_projections_proj"].keys())
            nprojs = len(key_list)
            stack_projs = np.empty((nprojs, nr, nc), dtype=dset.dtype)
            print("Loading projections. This takes time, please wait...")
            for ii in range(nprojs):
                strbar = "Projection: {} out of {}".format(ii + 1, nprojs)
                dset = fid["aligned_projections_proj/{}".format(key_list[ii])]
                stack_projs[ii] = dset[()]
                progbar(ii + 1, nprojs, strbar)
            print("\r")

        # sorting theta
        print("Sorting theta...")
        stack_projs, theta = sort_array(stack_projs, theta)

        print("Projections loaded from file {}".format(h5name))
        print("Time elapsed = {:.03f} s".format(time.time() - it0))
        return stack_projs, theta, shiftstack, datakwargs


class SaveTomogram(SaveData):
    """
    Save tomogram to HDF5 file
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.params = params

    def savecheck(func):
        """
        Decorator for save data
        """

        @functools.wraps(func)
        def new_func(self, *args, **params):
            if self.autosave:
                ansuser = "y"
                func(self, *args)
            else:
                while True:
                    ansuser = input(
                        "Do you want to save the data to HDF5 file? ([y]/n) "
                    ).lower()
                    if ansuser == "" or ansuser == "y":
                        func(self, *args)
                        break
                    elif ansuser == "n":
                        print("The data have NOT been saved yet. Please, be careful")
                        break
                    else:
                        print("You have to answer y or n")

        return new_func

    def __call__(self, *args):
        return self._save_tomogram(*args)

    @classmethod
    def save(cls, *args, **params):
        return cls(**params)._save_tomogram(*args)

    @classmethod
    def save_vol_to_h5(cls, *args, **params):
        return cls(**params)._save_vol_to_h5(*args)

    @classmethod
    def save(cls, *args, **params):
        """
        Parameters
        ----------
        *args 
            positional arguments
        args[0] : str
            H5 file name
        args[1] : array_like
            Array containing the stack of slices (tomogram)
        args[2] : array_like
            Values of theta
        args[3] : array_like
            Array containing the shifts for each projection in the stack
        """
        return cls(**params)._save_tomogram(*args)

    @classmethod
    def convert_to_tiff(cls, *args, **params):
        """
        Convert the HDF5 file with the tomogram to tiff

        Parameters
        ----------
        *args
            positional arguments
        args[0] : str
            H5 file name
        args[1] : array_like
            Array containing the stack of slices (tomogram)
        args[2] : array_like
            Values of theta
        args[3] : array_like
            Array containing the shifts for each projection in the stack
        """
        return cls(**params)._convert_to_tiff(*args)

    @savecheck
    def _save_tomogram(self, *args):
        """
        Parameters
        ----------
        *args
            positional arguments
        args[0] : str
            H5 file name
        args[1] : array_like
            Array containing the stack of slices (tomogram)
        args[2] : array_like
            Values of theta
        args[3] : array_like
            Array containing the shifts for each projection in the stack
        """
        h5name = args[0]
        tomogram = args[1]
        nslices, nr, nc = tomogram.shape
        theta = args[2]

        if len(args) == 4:
            shiftstack = args[3]
        else:
            shiftstack = np.zeros((2, nprojs), dtype=np.float32)

        # calculate the chunk size for writing the HDF5 files
        chunk_size = chunk_shape_3D(tomogram.shape)

        print("Saving {}".format(h5name))
        h5file = self.results_datapath(h5name)
        if os.path.isfile(h5file):
            print("File {} already exists and will be overwritten".format(h5name))
            os.remove(h5file)
        print("\rSaving metadata...", end="")
        write_paramsh5(h5file, **self.params)
        create_paramsh5(**self.params)
        print("\b\b Done")
        print("Saving data. This takes time, please wait...")
        with h5py.File(h5file, "a") as fid:
            # add the shiftstack
            fid.create_dataset(
                "shiftstack/shiftstack", data=shiftstack, dtype=np.float32
            )
            fid.create_dataset(
                "angles/thetas", data=theta, dtype=np.float32
            )  # add the thetas
            # ,compression='lzf')#, compression='gzip', compression_opts=9)
            dset = fid.create_dataset(
                "tomogram/slices",
                shape=(nslices, nr, nc),
                dtype=np.float32,
                chunks=chunk_size,
            )
            print("Saving tomographic slices. This takes time, please wait...")
            p0 = time.time()
            for ii in range(nslices):
                strbar = "Slice: {} out of {}".format(ii + 1, nslices)
                # ~ print(' Slice: {} out of {}'.format(ii+1, nslices), end='\r')
                dset[ii, :, :] = tomogram[ii]
                progbar(ii + 1, nslices, strbar)
            print("\r")
            print("Done. Time elapsed = {:.03f} s".format(time.time() - p0))
        print("Tomogram saved to file {}".format(h5name))
        print("In the folder {}".format(self.results_folder()))

    def tiff_folderpath(self, foldername):
        """
        Create the path to the folder in which the tiff files will be
        stored.
        """
        aux_path = self.results_folder()
        folderpath = os.path.join(aux_path, foldername)
        if not os.path.isdir(folderpath):
            print("Folder {} does not exists and will be created.".format(folderpath))
            os.makedirs(folderpath)
        else:
            print("Folder exists:{}".format(folderpath))
            userans = input(
                "Do you want to overwrite TIFFs in this folder ([y]/n)?"
            ).lower()
            if userans == "" or userans == "y":
                print("Overwritting")
            else:
                print("Writting of TIFFs aborted")
                raise SystemExit("Writting of TIFFs aborted")

        return folderpath

    def _convert_to_tiff(self, *args):
        """
        Convert the HDF5 file with the tomogram to tiff

        Parameter
        ---------
        *args
            positional arguments
        args[0] : str
            H5 file name
        args[1] : array_like
            Array containing the stack of slices (tomogram)
        args[2] : array_like
            Values of theta
        args[3] : array_like
            Array containing the shifts for each projection in the stack
    """
        tomogram = args[0]
        nslices, nr, nc = tomogram.shape

        try:
            voxelsize = self.params["voxelsize"]
        except KeyError:
            voxelsize = self.params["pixelsize"]

        energy = self.params["energy"]

        print("The total number of slides is {}".format(nslices))
        print("The voxel size is {} nm".format(voxelsize[0] * 1e9))

        create_paramsh5(**self.params)

        # create the TIFF folder
        tiff_subfolder_name = "TIFF_{}_{}_freqscl_{:0.2f}_{:d}bits".format(
            self.params["tomo_type"],
            self.params["filtertype"],
            self.params["freqcutoff"],
            self.params["bits"],
        )

        if self.params["tomo_type"] == "delta":
            # Conversion from phase-shifts tomogram to delta
            print("Converting from phase-shifts values to delta values")
            for ii in range(nslices):
                strbar = "Slice {} out of {}".format(ii + 1, nslices)
                tomogram[ii], factor = convert_to_delta(tomogram[ii], energy, voxelsize)
                progbar(ii + 1, nslices, strbar)
        elif self.params["tomo_type"] == "beta":
            # Conversion from amplitude to beta
            print("Converting from amplitude to beta values")
            for ii in range(slices):
                strbar = "Slice {} out of {}".format(ii + 1, nslices)
                tomogram[ii], factor = convert_to_beta(tomogram[ii], energy, voxelsize)
                progbar(ii + 1, nslices, strbar)
        print("\r")
        # low and high cutoff for Tiff normalization
        low_cutoff = np.min(tomogram)
        high_cutoff = np.max(tomogram)

        # writing the tiffs
        print("Writing the tiff files...")
        tiff_path = self.tiff_folderpath(tiff_subfolder_name)
        for ii in range(nslices):
            strbar = "Writing slice {:>5.0f} out of {:>5.0f}".format(ii + 1, nslices)
            if self.params["bits"] == 16:
                imgtiff = convertimageto16bits(tomogram[ii], low_cutoff, high_cutoff)
            elif self.params["bits"] == 8:
                imgtiff = convertimageto8bits(tomogram[ii], low_cutoff, high_cutoff)
            filename = "tomo_{}_filter_{}_cutoff_{:0.2f}_{:04d}.tif".format(
                self.params["tomo_type"],
                self.params["filtertype"],
                self.params["freqcutoff"],
                ii,
            )
            pathfilename = os.path.join(tiff_path, filename)
            write_tiff(imgtiff, pathfilename)
            progbar(ii + 1, nslices, strbar)
        print("\r")

        # writing the metadata
        filename = tiff_subfolder_name + "_cutoffs.txt"
        metadatafile = os.path.join(tiff_path, filename)
        write_tiffmetadata(metadatafile, low_cutoff, high_cutoff, factor, **self.params)

    @savecheck
    def _save_vol_to_h5(self, *args):
        """
        Save .vol files into HDF5 file
        """
        h5name = args[0]
        theta = args[1]

        print("Saving .vol into HDF5")
        filename = "volfloat/{}.vol".format(self.params["samplename"])
        tomogram = read_volfile(filename)
        nslices = tomogram.shape[0]
        print("Found {} slices in the volume.".format(nslices))
        self._save_tomogram(self, h5name, tomogram, theta, **self.params)


class LoadTomogram(LoadData):
    """
    Load projections from HDF5 file
    """

    def __init__(self, **params):
        paramsh5 = load_paramsh5(**params)
        super().__init__(**paramsh5)
        self.params = paramsh5
        self.params.update(params)

    def __call__(self, h5name):
        return self._load_tomogram(h5name)

    @classmethod
    def load(cls, *args, **params):
        """
        Load tomographic data from h5 file

        Parameters
        ----------
        args[0] : str
            HDF5 file name from which data is loaded

        Returns
        -------
        tomogram : array_like
            Stack of tomographic slices
        theta : array_like
            Stack of thetas
        shiftstack : array_like
            Shifts in vertical (1st dimension) and horizontal (2nd dimension)
        datakwargs : dict
            Dictionary with metadata information
        """
        return cls(**params)._load_tomogram(*args)

    @checkhostname
    def _load_tomogram(self, h5name):
        """
        Load tomographic data from h5 file

        Parameters
        ----------
        h5name : str
            File name from which data is loaded

        Returns
        --------
        tomogram : array_like
            Stack of tomographic slices
        theta : array_like
            Stack of thetas
        shiftstack : array_like
            Shifts in vertical (1st dimension) and horizontal (2nd dimension)
        datakwargs : dict
            Dictionary with metadata information
        """
        print("Loading tomogram from file {}".format(h5name))
        h5file = self.results_datapath(h5name)
        shiftstack = self._load_shiftstack(h5name)
        p0 = time.time()
        with h5py.File(h5file, "r") as fid:
            theta = fid["angles/thetas"][()]
            # read the inputkwargs dict
            datakwargs = dict()
            print("\rLoading metadata...", end="")
            for keys in sorted(list(fid["info"].keys())):
                datakwargs[keys] = fid["info/{}".format(keys)][()]
            datakwargs.update(self.params)
            print("\b\b Done")
            print("Loading tomogram. This takes time, please wait...")
            dset = fid["tomogram/slices"]
            nslices = dset.shape[0]
            tomogram = np.empty(dset.shape, dtype=dset.dtype)
            # ~ tomogram = fid[u'tomogram/slices'][()]
            for ii in range(nslices):
                strbar = "Slice: {} out of {}".format(ii + 1, nslices)
                tomogram[ii : ii + 1, :, :] = dset[ii, :, :]
                progbar(ii + 1, nslices, strbar)
            print("\r")
        print("Tomogram loaded from file {}".format(h5name))
        print("Time elapsed = {:.03f} s".format(time.time() - p0))
        return tomogram, theta, shiftstack, datakwargs
