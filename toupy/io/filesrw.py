#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 2015

Updated on Tue Mar 25 2019:
    - Fully compatible to Python 3
    - No more backcompatibility to Python 2

@author: jdasilva
"""
# Standard library imports
import fnmatch
import glob
import os
from pathlib import Path
import re
import time

# third party packages
import libtiff
import fabio
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

__all__ = [
    "read_info_file",
    "read_ptyr",
    "read_cxi",
    "crop_array",
    "write_edf",
    "read_edf",
    "read_edffilestack",
    "create_paramsh5",
    "load_paramsh5",
    "write_paramsh5",
    "read_tiff",
    "write_tiff",
    "write_tiffmetadata",
    "convertimageto16bits",
    "convertimageto8bits",
    "convert16bitstiff",
    "convert8bitstiff",
]


def read_info_file(tiff_info_file):
    """
    Read info file from tiff slices of the reconstructed tomographic 
    volume

    Parameters
    ----------
    tiff_info_file : str
        Filename

    Returns
    -------
    low_cutoff : float
        Low cutoff of the gray level
    high_cutoff : float
        High cutoff of the gray level
    pixelsize : float
        Pixelsize    
    """
    # read info file
    # with open(tiff_info_beta,'r') as ff:
    with open(tiff_info_file, "r") as ff:
        info_file = ff.readlines()
        print(info_file)
    # separate the infos
    low_cutoff = np.float(info_file[0].strip().split("=")[1])
    high_cutoff = np.float(info_file[1].strip().split("=")[1])
    factor = np.float(info_file[2].strip().split("=")[1])
    # ~ pixelsize_beta = np.array([np.float(x) for x in (info_beta[3].strip().split('=')[1]).strip().lstrip('[').rstrip(']').split()])
    pixelsize = (
        (info_file[3].strip().split("=")[1]).strip().lstrip("[").rstrip("]").split()[0]
    )

    return low_cutoff, high_cutoff, factor, pixelsize


def _reorient_ptyrimg(input_array):
    """

    Auxiliary function to corrects the orientation of the image and 
    probe from the arrays in ptyr file
    """
    # reorienting the probe
    if input_array.ndim == 3:
        output_array = np.empty_like(input_array)
        for ii in range(len(input_array)):
            output_array[ii] = np.fliplr(np.transpose(input_array[ii]))
    elif input_array.ndim == 2:
        output_array = np.fliplr(np.transpose(input_array))
    else:
        raise ValueError(u"Wrong dimensions for the array")
    return output_array


metaptyr = dict()
def _print_attrs_ptyr(name):
    """
    Auxiliary function to indentify from where the data must be read in
    the ptyr files
    """
    global metaptyr
    if "obj" in name:
        if "_psize" in name:
            metaptyr["psize_h5path"] = name
        if "data" in name:
            metaptyr["obj_h5path"] = name
    if "probe" in name:
        if "data" in name:
            metaptyr["probe_h5path"] = name


def _findh5paths(filename):
    """
    Auxiliary function to indentify from where the data must be read in
    the HDF5 files
    """
    global metaptyr
    with h5py.File(filename, "r") as fid:
        fid.visit(_print_attrs_ptyr)
    print(metaptyr)


def read_ptyr(pathfilename):
    """
    Read reconstruction files .ptyr from Ptypy

    Parameters
    ----------
    pathfilename : str
        Path to file

    Returns
    -------
    data1 : ndarray (complex)
        Object image
    probe1 : ndarray (complex)
        Probe images
    pixelsize : float
        List with pixelsizes in vertical and horizontal directions
    """
    global metaptyr
    if metaptyr == {}:
        print("meta is empty")
        _findh5paths(pathfilename)

    with h5py.File(pathfilename, "r") as fid:
        # get the data from the object
        data0 = np.squeeze(fid[metaptyr["obj_h5path"]]).astype(np.complex64)
        # get the data from the probe
        probe0 = np.squeeze(fid[metaptyr["probe_h5path"]]).astype(np.complex64)
        # get the pixel size
        pixelsize = (fid[metaptyr["psize_h5path"]][()]).astype(np.float32)

    # reorienting the object
    data1 = _reorient_ptyrimg(data0)
    probe1 = _reorient_ptyrimg(probe0)

    return data1, probe1, pixelsize


def _h5py_dataset_iterator(g, prefix=""):
    """
    Auxiliary function to iterate over the h5 file datasets
    """
    for key in g.keys():
        item = g[key]
        path = "{}/{}".format(prefix, key)
        if isinstance(item, h5py.Dataset):  # test for dataset
            yield (path, item[()])
        elif isinstance(item, h5py.Group):  # test for group (go down)
            yield from _h5py_dataset_iterator(item, path)


metacxi = dict()
def _h5pathcxi(filename):
    """
    h5py visititems does not find links
    """
    global metacxi
    with h5py.File(filename, "r") as fid:
        listpath = [
            path
            for path, dset in _h5py_dataset_iterator(fid)
            if "object" in path or "probe" in path
        ]
    metacxi["obj_h5path"] = [
        ii for ii in sorted(listpath) if "object/data" in ii and ii.endswith("data")
    ][-1]
    metacxi["probe_h5path"] = [
        ii for ii in sorted(listpath) if "probe/data" in ii and ii.endswith("data")
    ][-1]
    metacxi["xpsize_h5path"] = [
        ii
        for ii in sorted(listpath)
        if "object/x_pixel_size" in ii and ii.endswith("x_pixel_size")
    ][-1]
    metacxi["ypsize_h5path"] = [
        ii
        for ii in sorted(listpath)
        if "object/y_pixel_size" in ii and ii.endswith("y_pixel_size")
    ][-1]


def read_cxi(pathfilename):
    """
    Read reconstruction files .cxi from PyNX

    Parameters
    ----------
    pathfilename : str
        Path to file

    Returns
    -------
    data1 : ndarray (complex)
        Object image
    probe1 : ndarray (complex)
        Probe images
    pixelsize : list float
        List with pixelsizes in vertical and horizontal directions
    """
    global metacxi
    if metacxi == {}:
        print("meta is empty")
        _h5pathcxi(pathfilename)

    with h5py.File(pathfilename, "r") as fid:
        # get the data from the object
        data0 = np.squeeze(fid[metacxi["obj_h5path"]]).astype(np.complex64)
        # get the data from the probe
        probe0 = np.squeeze(fid[metacxi["probe_h5path"]]).astype(np.complex64)
        # get the pixel size
        pixelsizex = fid[metacxi["xpsize_h5path"]][()]
        pixelsizey = fid[metacxi["ypsize_h5path"]][()]
    pixelsize = np.array([pixelsizex, pixelsizey]).astype(np.float32)

    # reorienting the object
    data1 = _reorient_ptyrimg(data0)
    probe1 = _reorient_ptyrimg(probe0)

    return data1, probe1, pixelsize


def crop_array(input_array, delcropx, delcropy):
    """
    Crop borders from 2D arrays

    Parameters
    ----------
    input_array : ndarray
        Input array to be cropped
    delcropx, delcropy : int
        Number of pixels to be crop from borders in x and y directions

    Returns
    -------
    cropped_array : ndarray
        Cropped array
    """
    if delcropx is not None or delcropy is not None:
        print("Cropping ROI of data")
        print("Before: " + input_array.shape)
        print(input_array[delcropy:-delcropy, delcropx:-delcropx].shape)
        if input_array.ndim == 2:
            return input_array[delcropy:-delcropy, delcropx:-delcropx]
        elif input_array.ndim == 3:
            return input_array[:, delcropy:-delcropy, delcropx:-delcropx]
        print("After: " + input_array.shape)
    else:
        print("No cropping of data")
        return input_array


def write_edf(fname, data_array, hd=None):
    """
    Write EDF files

    Parameters
    ----------
    fname : str
        File name
    data_array : ndarray
        Data to be saved as edf
    hd : dict
        Dictionary with header information
    """
    with fabio.edfimage.edfimage() as fid:
        fid.data = data_array
        if hd is not None:  # if header information
            fid.header = hd
        fid.write(fname)  # writing the file


def read_edf(fname):
    """
    read EDF files

    Parameters
    ----------
    fname : str
        Path to file

    Returns
    -------
    projs: ndarray
        Array of projections
    pixelsize : list of float
        List with pixelsizes in vertical and horizontal directions
    """
    imgobj = fabio.open(fname)
    imgdata = imgobj.data
    try:
        pixelsize = imgobj.header["pixel_size"]
    except:
        pixelsize = 1
    imgobj.close()
    return imgdata, pixelsize


def read_edffilestack(**params):
    """
    Read projection stack

    Parameters
    ----------
    inputkwargs : dict
        dict with parameters

    Returns
    -------
    projs: ndarray
        Array of projections
    pixelsize : list of float
        List with pixelsizes in vertical and horizontal directions
    """
    # create the file wildcard
    file_wcard = re.sub(r"_\d{4}.edf", "*.edf", params[u"pathfilename"])
    # glob the list of files in sorted order
    listfiles = sorted(glob.glob(file_wcard))
    nfiles = len(listfiles)
    print("{} files found".format(nfiles))
    # read one file to obtain array shape
    img0, px = read_edf(listfiles[0])
    print("The pixel size is: {:.02f} nm".format(eval(px) * 1e9))
    nr, nc = img0.shape
    # initializing the array of projections
    projs = np.empty((nfiles, nr, nc))
    for ii in range(nfiles):
        print("Reading projection {}:".format(ii))
        print("File: {}".format(listfiles[ii]))
        projs[ii], _ = read_edf(listfiles[ii])
    return projs, eval(px)


def load_paramsh5(**params):
    """
    Load parameters from HDF5 file of parameters
    """
    # read parameter file
    paramsh5file = params["samplename"] + "_params.h5"
    with h5py.File(paramsh5file, "r") as fid:
        # read the inputkwargs dict
        out_params = dict()
        for keys in sorted(list(fid["info"].keys())):
            out_params[keys] = fid["info/{}".format(keys)][()]
    out_params.update(params)  # add/update with new values
    return out_params


def create_paramsh5(*args, **params):
    """
    Create parameter file in HDF5 format
    """
    # create a parameter file
    print("Creating the h5 parameter file")
    if len(args) == 0:
        filename = params["samplename"] + "_params.h5"
    # print(pathparamsh5)
    # paramsh5file = os.path.join(pathparamsh5,filename)
    write_paramsh5(filename, **params)


def write_paramsh5(h5filename, **params):
    """
    Writes params to HDF5 file
    """
    # check if file already exists and overwritte it if so
    if os.path.isfile(h5filename):
        print("File {} already exists and will be overwritten".format(h5filename))
    # writing the file
    with h5py.File(h5filename, "w") as ff:
        dt = h5py.special_dtype(vlen=str)  # special type for str for h5py
        for k, v in sorted(params.items()):
            if v is None:
                v = "none"
                ff.create_dataset("info/{}".format(k), data=v, dtype=dt)
            elif isinstance(v, str):  # string
                ff.create_dataset("info/{}".format(k), data=v, dtype=dt)
            elif isinstance(v, bool) or isinstance(v, np.bool_):  # boolean
                ff.create_dataset("info/{}".format(k), data=v, dtype=bool)
            elif isinstance(v, np.ndarray):  # float array
                ff.create_dataset("info/{}".format(k), data=v, dtype=np.float32)
            elif (
                isinstance(v, np.float32)
                or isinstance(v, float)
                or isinstance(v, np.float)
            ):  # float
                ff.create_dataset("info/{}".format(k), data=v, dtype=np.float32)
            elif (
                isinstance(v, np.int32) or isinstance(v, np.int) or isinstance(v, int)
            ):  # integer
                ff.create_dataset("info/{}".format(k), data=v, dtype=np.int32)
            else:
                ff.create_dataset("info/{}".format(k), data=v)  # other


def read_tiff(imgpath):
    """
    Read tiff files using libtiff

    Parameters
    ----------
    imgpath : str
        Path to tiff file with extension

    Returns
    -------
    imgout : ndarray
        Array containing the image
    Examples:
    ---------
    >>> imgpath = 'libtiff.tiff'
    >>> tiff = read_tiff(imgpath)
    >>> ar = tiff.read_image()
    >>> tiff.close()
    >>> ar.dtype
    dtype('uint16')
    >>> np.max(ar)
    65535
    """
    tiff = libtiff.TIFF.open(imgpath, mode="r")
    imgout = tiff.read_image()
    tiff.close()
    return imgout

def write_tiff(input_array,pathfilename):
    """
    Write tiff files using libtiff
    """
    # Writing to file
    tiff = libtiff.TIFF.open(pathfilename,"w")
    tiff.write_image(input_array)
    tiff.close()

def write_tiffmetadata(filename, low_cutoff, high_cutoff, factor, **params):
    """
    Creates a txt file with the information about the Tiff normalization
    """
    try:
        voxelsize = params["voxelsize"] * 1e9 # in nm
    except KeyError:
        voxelsize = params["pixelsize"] * 1e9
    filtertype = params["filtertype"]
    freqcutoff = params["filtertomo"]
    nbits = params["bits"]

    # writing
    fid = open(filename,"w")
    fid.write("# Tomo filter = {}\n".format(filtertype))
    fid.write("# Tomo filter cutoff = {}\n".format(freqcutoff))
    fid.write("# low_cutoff = {}\n".format(low_cutoff))
    fid.write("# high_cutoff = {}\n".format(high_cutoff))
    fid.write("# factor = {}\n".format(factor))
    fid.write("# voxel size = {} nm\n".format(voxelsize))
    fid.write("# to convert back to quantitative values:\n")
    fid.write(
        "# low_cutoff + [(high_cutoff-low_cutoff)*tiff_image/(2^{:d}-1)] \n".format(
            nbits
        )
    )
    fid.close()


def convertimageto16bits(input_image, low_cutoff, high_cutoff):
    """
    Convert image gray-level to 16 bits with normalization
    
    Parameters
    ----------
    input_image : ndarray
        Input image to be converted
    low_cutoff : float
        Low cutoff of the gray level
    high_cutoff : float
        High cutoff of the gray level

    Returns
    -------
    tiffimage : ndarray
        Array containing the image at 16 bits
    """
    # Tiff normalization - 16 bits
    imgtiff = input_image-low_cutoff
    imgtiff /= high_cutoff - low_cutoff
    imgtiff *= 2 ** 16 - 1  # 16 bits
    return np.uint16(imgtiff)

def convertimageto8bits(input_image, low_cutoff, high_cutoff):
    """
    Convert image gray-level to 8 bits with normalization
    
    Parameters
    ----------
    input_image : ndarray
        Input image to be converted
    low_cutoff : float
        Low cutoff of the gray level
    high_cutoff : float
        High cutoff of the gray level

    Returns
    -------
    tiffimage : ndarray
        Array containing the image at 8 bits
    """
    # Tiff normalization - 8 bits
    imgtiff = input_image - low_cutoff
    imgtiff /= high_cutoff - low_cutoff
    imgtiff *= 2 ** 8 - 1  # 8 bits
    return np.uint8(imgtiff)


def convert16bitstiff(tiffimage, low_cutoff, high_cutoff):
    """
    Convert 16 bits tiff files back to quantitative values

    Parameters
    ----------
    imgpath : ndarray
        Image read from 16 bits tiff file
    low_cutoff : float
        Low cutoff of the gray level
    high_cutoff : float
        High cutoff of the gray level

    Returns
    -------
    tiffimage : ndarray
        Array containing the image with quantitative values
    """
    tiffimage = tiffimage.astype(np.float)
    # Convert to 16 bits
    tiffimage /= 2 ** 16 - 1
    tiffimage *= high_cutoff - low_cutoff
    tiffimage += low_cutoff

    return tiffimage

def convert8bitstiff(filename, low_cutoff, high_cutoff):
    """
    Convert 8bits tiff files back to quantitative values

    Parameters
    ----------
    imgpath : ndarray
        Image read from 8 bits tiff file
    low_cutoff : float
        Low cutoff of the gray level
    high_cutoff : float
        High cutoff of the gray level

    Returns
    -------
    tiffimage : ndarray
        Array containing the image with quantitative values
    """
    tiffimage = tiffimage.astype(np.float)
    # Convert to 8 bits
    tiffimage /= 2 ** 8 - 1
    tiffimage *= high_cutoff - low_cutoff
    tiffimage += low_cutoff

    return tiffimage


def saveh5file(h5file, stack_projs, theta, shiftstack, **kwargs):
    """
    Save data to h5 file
    """
    nprojs, nr, nc = stack_projs.shape
    # check if file exists
    if os.path.isfile(h5file):
        print("File {} already exists and will be overwritten".format(h5name))
        os.remove(h5file)
    # save metadata first
    print("Saving metadata")
    write_paramsh5(h5file, **kwargs)
    # save the data
    print("Saving data. This takes time, please wait...")
    if np.iscomplexobj(stack_projs[0]):
        array_dtype = np.complex64
    else:
        array_dtype = np.float32
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
            print("Projection: {} out of {}".format(ii + 1, nprojs), end="\r")
            dset[ii, :, :] = stack_projs[ii]
        print("\r")
        if masks is not None:
            fid.create_dataset(
                "masks/stack", data=masks, dtype=np.bool
            )  # air/vacuum mask
        print("Done. Time elapsed = {:.03f} s".format(time.time() - p0))
    print("Data saved to file {}".format(h5name))
