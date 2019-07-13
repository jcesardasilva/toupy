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
import fabio
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

__all__ = [
          'read_ptyr',
          'read_cxi',
          'crop_array',
          'write_edf',
          'read_edf',
          'read_edffilestack',
          'create_paramsh5',
          'load_paramsh5',
          'write_paramsh5'
          ]

def _reorient_ptyrimg(input_array):
    """
    Corrects the orientation of the image and probe from the arrays
    in ptyr file
    """
    # reorienting the probe
    if input_array.ndim==3:
        output_array = np.empty_like(input_array)
        for ii in range(len(input_array)):
            output_array[ii]= np.fliplr(np.transpose(input_array[ii]))
    elif input_array.ndim==2:
        output_array = np.fliplr(np.transpose(input_array))
    else:
        raise ValueError(u"Wrong dimensions for the array")
    return output_array

def read_ptyr(pathfilename):
    """
    Read reconstruction files .ptyr from Ptypy
    inputs:
        pathfilename = path to file
    @author: Julio C. da Silva (jdasilva@esrf.fr)
    """
    with h5py.File(pathfilename,'r') as fid:
        # get the root entry
        content0 = list(fid.keys())[0]
        # get the data from the object
        data0 = np.squeeze(fid[content0+"/obj/S00G00/data"]).astype(np.complex64)
        # get the data from the probe
        probe0 = np.squeeze(fid[content0+"/probe/S00G00/data"]).astype(np.complex64)
        # get the pixel size
        pixelsize = (fid[content0+"/obj/S00G00/_psize"][()]).astype(np.float32)

    # reorienting the object
    data1 = _reorient_ptyrimg(data0)
    probe1 = _reorient_ptyrimg(probe0)

    return data1,probe1,pixelsize

def read_cxi(pathfilename):
    """
    Read reconstruction files .cxi from PyNX
    inputs:
        pathfilename = path to file
    @author: Julio C. da Silva (jdasilva@esrf.fr)
    """
    cxientry = 'entry_last' #params['cxientry'] # to know where the data is
    with h5py.File(pathfilename,'r') as fid:
        # get the data from the object
        data0 = np.squeeze(fid[cxientry+'/object/data']).astype(np.complex64)
        # get the data from the probe
        probe0 = np.squeeze(fid[cxientry+'/probe/data']).astype(np.complex64)
        # get the pixel size
        pixelsizex = fid[cxientry+'/object/x_pixel_size'][()]
        pixelsizey = fid[cxientry+'/object/y_pixel_size'][()]
    pixelsize = np.array([pixelsizex,pixelsizey]).astype(np.float32)

    # reorienting the object
    data1 = _reorient_ptyrimg(data0)
    probe1 = _reorient_ptyrimg(probe0)

    return data1,probe1,pixelsize

def crop_array(input_array,delcropx,delcropy):
    """
    Crop images
    Inputs:
        input_array: input image to be cropped
        **params: dict of parameters
    @author: Julio C. da Silva (jdasilva@esrf.fr)
    """
    if delcropx is not None or delcropy is not None:
        print('Cropping ROI of data')
        print('Before: '+input_array.shape)
        print(input_array[delcropy:-delcropy,delcropx:-delcropx].shape)
        if input_array.ndim == 2:
            return input_array[delcropy:-delcropy,delcropx:-delcropx]
        elif input_array.ndim == 3:
            return input_array[:,delcropy:-delcropy,delcropx:-delcropx]
        print('After: '+input_array.shape)
    else:
        print('No cropping of data')
        return input_array

def write_edf(fname, data_array,hd=None):
    """
    Write EDF files
    Inputs:
        fname = file name
        data_array = data to be saved as edf
        hd = dictionary with header information
    @author: Julio C. da Silva (jdasilva@esrf.fr)
    """
    with fabio.edfimage.edfimage() as fid:
        fid.data = data_array
        if hd is not None: # if header information
            fid.header = hd
        fid.write(fname) # writing the file

def read_edf(fname):
    """
    read EDF files
    fname = file name
    """
    imgobj = fabio.open(fname)
    imgdata = imgobj.data
    try: pixelsize = imgobj.header['pixel_size']
    except: pixelsize = 1
    imgobj.close()
    return imgdata, pixelsize

def read_edffilestack(**params):
    """
    Read projection stack
    Inputs:
        inputkwargs: dict with parameters
    Output:
        projs: ndarray with the projections
    """
    # create the file wildcard
    file_wcard = re.sub(r"_\d{4}.edf","*.edf",params[u'pathfilename'])
    # glob the list of files in sorted order
    listfiles = sorted(glob.glob(file_wcard))
    nfiles = len(listfiles)
    print('{} files found'.format(nfiles))
    # read one file to obtain array shape
    img0,px = read_edf(listfiles[0])
    print('The pixel size is: {:.02f} nm'.format(eval(px)*1e9))
    nr,nc = img0.shape
    # initializing the array of projections
    projs = np.empty((nfiles,nr,nc))
    for ii in range(nfiles):
        print('Reading projection {}:'.format(ii))
        print('File: {}'.format(listfiles[ii]))
        projs[ii],_ = read_edf(listfiles[ii])
    return projs, eval(px)

def load_paramsh5(**params):
    """
    Load parameters from HDF5 file of parameters
    @author: Julio C. da Silva (jdasilva@esrf.fr)
    """
    # read parameter file
    paramsh5file = params['samplename']+'_params.h5'
    with h5py.File(paramsh5file,'r') as fid:
        # read the inputkwargs dict
        out_params = dict()
        for keys in sorted(list(fid['info'].keys())):
            out_params[keys]=fid['info/{}'.format(keys)][()]
    out_params.update(params) # add/update with new values
    return out_params

def create_paramsh5(**params):
    """
    Create parameter file in HDF5 format
    @author: Julio C. da Silva (jdasilva@esrf.fr)
    """
    # create a parameter file
    print('Creating the h5 parameter file')
    paramsh5file = params['samplename']+'_params.h5'
    write_paramsh5(paramsh5file,**params)

def write_paramsh5(h5filename,**params):
    """
    Writes params to HDF5 file
    """
    # check if file already exists and overwritte it if so
    if os.path.isfile(h5filename):
        print('File {} already exists and will be overwritten'.format(h5filename))
    # writing the file
    with h5py.File(h5filename,'w') as ff:
        dt = h5py.special_dtype(vlen = str) # special type for str for h5py
        for k,v in sorted(params.items()):
            if v is None:
                v = 'none'
                ff.create_dataset('info/{}'.format(k),data = v,dtype=dt)
            elif isinstance(v,str): # string
                ff.create_dataset('info/{}'.format(k),data = v,dtype=dt)
            elif isinstance(v,bool) or isinstance(v,np.bool_): # boolean
                ff.create_dataset('info/{}'.format(k),data = v,dtype=bool)
            elif isinstance(v,np.ndarray): # float array
                ff.create_dataset('info/{}'.format(k),data = v,dtype=np.float32)
            elif isinstance(v,np.float32) or isinstance(v,float) or isinstance(v,np.float): # float
                ff.create_dataset('info/{}'.format(k),data = v,dtype=np.float32)
            elif isinstance(v,np.int32) or isinstance(v,np.int) or isinstance(v,int): # integer
                ff.create_dataset('info/{}'.format(k),data = v,dtype=np.int32)
            else:
                ff.create_dataset('info/{}'.format(k),data = v) # other
