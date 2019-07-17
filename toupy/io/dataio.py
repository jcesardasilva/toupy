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
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

# local packages
from .filesrw import *

__all__ = [
          'remove_extraprojs',
          'PathName',
          'LoadProjections',
          'SaveData',
          'LoadData',
          'SaveTomogram',
          'LoadTomogram',
          ]

def remove_extraprojs(stack_projs,theta):
    """
    Remove extra projections of tomographic scans with projections at
    180, 90 and 0 degrees at the end

    Parameters
    ----------
    stack_projs : ndarray
        Stack of projections with the first index correspoding to the
        projection number
    theta : ndarray
        Array of theta values

    Returns
    -------
        stack_projs : ndarray
            Stack of projections after the removal
        theta : ndarray
            Array of theta values after the removal
    """
    print(theta[-5:])
    a = str(input('Do you want to remove extra thetas?([y]/n)')).lower()
    if a == '' or a == 'y':
        a1 = eval(input('How many to remove?'))
        stack_projs = stack_projs[:-a1] # the 3 last angles are 180, 90 and 0 degrees
        theta = theta[:-a1] # the 3 last angles are 180, 90 and 0 degrees
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
    def __init__(self,**params):
        self.pathfilename = os.path.abspath(params['pathfilename'])
        self.useraccount = params['account']
        self.samplename = params['samplename']
        self.regime = params['regime']
        #~ self.rootpath = os.path.split(os.path.split(os.path.split(os.path.split(os.path.dirname(self.pathfilename))[0])[0])[0])[0]
        self.rootpath = str(Path(self.pathfilename).parents[3])#.parents[3]) # root path
        self.filename = os.path.basename(self.pathfilename) # data filename
        self.dirname = os.path.dirname(self.pathfilename) # data filename
        self.fileprefix, self.fileext = os.path.splitext(self.filename) # filename and extension
        self.icath5file = "{}-id16a.h5".format(self.useraccount) # metadata filename
        self.icath5path = os.path.join(self.rootpath, self.icath5file) # metadata filename path

    def datafilewcard(self):
        """
        Create file wildcard to search for files
        """
        file_wcard = re.sub(self.samplename+r'\w*',self.samplename+'*',self.fileprefix) # file_wcard
        #~ file_wcard = re.sub(self.samplename+'\w*',self.samplename+'*_ML',fileprefix)
        return file_wcard

    def metadatafilewcard(self):
        """
        Create file wildcard to search for files
        """
        if not os.path.isfile(self.icath5path):
            raise IOError('File {} not found'.format(self.icath5file))
        if self.fileext=='.ptyr': # Ptypy
            metafile_wcard=re.sub(r'_subtomo\d{3}_\d{4}_\w+','_subtomo*', os.path.splitext(self.filename)[0])
        elif self.fileext=='.cxi': # PyNX
            metafile_wcard=re.sub(r'_subtomo\d{3}_\d{4}','_subtomo*', os.path.splitext(self.filename)[0])
        else:
            raise IOError("File {} is not a .ptyr or a .cxi file. Please, load a .ptyr or a .cxi file.".format(filename))
        return metafile_wcard

    def search_projections(self):
        """
        Search for projection given the filenames
        """
        print(u'Path: {}'.format(self.dirname))
        print(u'First projection file: {}'.format(self.filename))
        scan_wcard = os.path.join(
                     re.sub(
                     self.samplename+r'_\w*',
                     self.samplename+'_*',
                     self.dirname),
                     self.metadatafilewcard()+'_ML'+self.fileext)
        return scan_wcard

    def results_folder(self):
        """
        create path for the result folder
        """
        aux_wcard = re.sub(r'\*','',self.datafilewcard())
        if self.regime == 'nearfield':
            foldername = aux_wcard+'_nfpxct'
        elif self.regime == 'farfield':
            foldername = aux_wcard+'_pxct'
        else:
            raise ValueError('Unrecognized regime')
        aux_path = os.path.join(os.path.dirname(self.dirname),foldername)
        if not os.path.isdir(aux_path):
            print('Directory does not exist. Creating the directory...')
            os.makedirs(aux_path)
        return aux_path

    def results_datapath(self,h5name):
        aux_path = self.results_folder()
        h5file = os.path.join(aux_path,h5name)
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

class LoadProjections(PathName,Variables):
    """
    Load the reconstructed projections from the ptyr files
    """
    def __init__(self,**params):
        super().__init__(**params)
        self.showrecons = params['showrecons']
        self.border_crop_x = params['border_crop_x']
        self.border_crop_y = params['border_crop_y']
        try: self.checkextraprojs = params[u'checkextraprojs']
        except: pass
        try: self.cxientry = params['cxientry']
        except: pass

        if self.fileext=='.ptyr': # Ptypy
            findh5paths(self.pathfilename)
            self.read_reconfile = read_ptyr
        elif self.fileext=='.cxi': # PyNX
            self.read_reconfile = read_cxi
        else:
            raise IOError("File {} is not a .ptyr or a .cxi file. Please, load a .ptyr or a .cxi file.".format(self.filename))

    def __call__(self):
        return self.load_projections()

    def check_angles(self):
        """
        Find the angles of the projections and plot them to be checked
        Specific to ID16A beamline (ESRF)
        """
        thetas={}
        with h5py.File(self.icath5path,'r') as fid:
            sorted_keys = sorted(list(fid.keys()))
            for keys in sorted_keys:
                if fnmatch.fnmatch(keys,'*'+self.metadatafilewcard()):
                    try:
                        positioners=fid[keys+u'/sample/positioner/value'][()] # old style at ID16A beamline
                    except KeyError:
                        positioners=fid[keys+u'/sample/positioners/value'][()] #new style at ID16A beamline
                    thetas[keys] = np.float(positioners.split()[0])
        # checking the angles
        print(u'Checking the angles')
        angles = []
        deltaidx = 0 # in case of repeated values
        sorted_thetakeys = sorted(thetas.keys())
        for idx, keys in enumerate(sorted_thetakeys):
            th = np.float(thetas[keys])
            if th == np.float(thetas[sorted_thetakeys[idx-1]]):
                print('Found repeated value of theta. Discarding it')
                deltaidx+=1
                continue
            print('Projection {}: {} degrees'.format(idx-deltaidx,thetas[keys]))
            angles.append(th)

        # plot the angles for verification
        plt.close('all')
        fig, (ax1,ax2) = plt.subplots(num=1,nrows=2,ncols=1)
        pltangles = ax1.plot(angles,'ro')
        ax1.set_xlabel('projection')
        ax1.set_ylabel('Theta angles')
        ax1.axis('tight')
        pltdiffangles = ax2.plot(np.diff(sorted(angles)),'ro-')
        ax2.set_xlabel('Sorted projections')
        ax2.set_ylabel('Angular spacing')
        ax2.axis('tight')
        plt.tight_layout()
        plt.show(block=False)
        a = input(u'Are the angles ok?[Y/n]').lower()
        if a=='' or a=='y':
            print('Starting to load the projections')
        else:
            raise SystemExit('Exiting')
        return angles, thetas

    def _remove_extraprojs(self,thetas,proj_files):
        """
        Remove extra projections of tomographic scans with projections at
        180, 90 and 0 degrees at the end

        Parameters
        ----------
        theta : ndarray
            Array of theta values
        proj_files : list of str
            List of projection files

        Returns
        -------
            stack_projs : ndarray
                Stack of projections after the removal
            theta : ndarray
                Array of theta values after the removal
        """
        print('The final 5 angles are: {}'.format(list(thetas[-5:])))
        a = str(input('Do you want to remove extra thetas?([y]/n)')).lower()
        if a == '' or a == 'y':
            a1 = eval(input('How many to remove?'))
            proj_files = proj_files[:-a1] # the 3 last angles are 180, 90 and 0 degrees
            thetas = thetas[:-a1] # the 3 last angles are 180, 90 and 0 degrees
            print('The final 5 angles are now: {}'.format(list(thetas[-5:])))
        return proj_files

    def load_projections(self):
        """
        Load the reconstructed projections from the ptyr files
        """
        # get the angles
        angles, thetas = self.check_angles()

        # get the list of files to load
        proj_files = sorted(glob.glob(self.search_projections()))
        if self.checkextraprojs:
            proj_files = self._remove_extraprojs(angles,proj_files)

        # count the number of available projections
        num_projections = len(proj_files)
        a = input("I have found {} projections. Do you want to continue?[Y/n]".format(num_projections)).lower()
        if a == '' or a == 'y':
            print('Continuing...')
            plt.close('all')
        else:
            raise SystemExit('Exiting the script')

        # Read the first projection to check size and reconstruction parameters
        objs0,probe0,pixelsize = self.read_reconfile(self.pathfilename)
        objs0 = crop_array(objs0,self.border_crop_x,self.border_crop_y) # crop image if requested
        nr,nc = objs0.shape
        if pixelsize[0] != pixelsize[1]:
            raise SystemExit("Pixel size is not symmetric. Exiting the script")
        print("the pixelsize of the first projection is {:.2f} nm".format(pixelsize[0]*1e9))

        # initialize the array for the stack objects
        stack_objs = np.empty((num_projections,nr,nc),dtype=np.complex64)
        stack_angles = np.empty(num_projections,dtype=np.float32)

        if self.showrecons:
            print(u'Showing reconstructions for each angle')
            probe = probe0[0]
            if objs0.shape[0]<objs0.shape[1]:
                plotgrid=(3,1)
                plotsize=(6,20)
            else:
                plotgrid=(1,3)
                plotsize=(20,6)
            vabsmean = np.abs(objs0).mean()
            perabsmean = 0.2*vabsmean
            # Special tricks for the probe display
            H = np.angle(probe)/(2*np.pi)+0.5
            S = np.ones_like(H).astype(int)
            V = np.abs(probe)/np.max(np.abs(probe))
            probe_hsv = np.dstack((H,S,V))
            # display first image
            plt.close('all')
            plt.ion()
            fig, (ax1,ax2,ax3) = plt.subplots(num=1,nrows=plotgrid[0],ncols=plotgrid[1],figsize=plotsize)
            im1 = ax1.imshow(np.abs(objs0),interpolation='none',cmap='gray',vmin=vabsmean-perabsmean,vmax=vabsmean+perabsmean)
            ax1.set_title(u'Object magnitude - projection {}'.format(1))
            im2 = ax2.imshow(np.angle(objs0),interpolation='none',cmap='bone')#,vmin=vphasemean-perabsmean,vmax=vphasemean+perabsmean)
            ax2.set_title(u'Object Phase - projection {}'.format(1))
            im3 = ax3.imshow(hsv_to_rgb(probe_hsv),interpolation='none')
            ax3.set_title(u'Probe - projection {}'.format(1))
            im1.axes.figure.canvas.draw()
            im2.axes.figure.canvas.draw()
            im3.axes.figure.canvas.draw()

        # reads the ptyr or cxi files and get object and probe in a stack
        for idxp, proj in enumerate(proj_files):
            print(u'\nProjection: {}'.format(idxp))
            print(u'Reading: {}'.format(proj))
            objs,probe_allmodes,pixelsize = self.read_reconfile(proj) # reading file
            # crop image if requested
            if self.border_crop_x is not None:
                if self.border_crop_y is not None:
                    objs = crop_array(objs,self.border_crop_x,self.border_crop_y)
            # check if same size, otherwise pad
            if objs.shape != objs0.shape:
                print('########################')
                print('File {} has different shape and was padded'.format(proj))
                padwr, padwc = [int((nr-objs.shape[0])/2.), int((nc-objs.shape[1])/2.)]
                objs = np.pad(objs,((padwr,padwr),(padwc,padwc)),'edge')
                print('########################')

            # update stack_objs
            stack_objs[idxp] = objs

            # compare projection name with thetas dictionary and associate angles
            if self.fileext=='.ptyr':
                key_finder = os.path.basename(os.path.dirname(proj))
            elif self.fileext=='.cxi':
                key_finder = os.path.splitext(os.path.basename(proj))[0]
            # compare projection name with thetas dictionary and associate angles
            for keys in sorted(thetas.keys()):
                if keys.find(key_finder)!=-1:
                    stack_angles[idxp] = thetas[keys]
                    print(u'Angle: {}'.format(thetas[keys]))
                    break
            if self.showrecons:
                # Special tricks for the probe display
                probe = probe_allmodes[0]
                H = np.angle(probe)/(2*np.pi)+0.5
                S = np.ones_like(H).astype(int)
                V = np.abs(probe)/np.max(np.abs(probe))
                probe_hsv = np.dstack((H,S,V))

                im1.set_data(np.abs(objs))
                im1.set_cmap('gray')
                im1.set_clim((vabsmean-perabsmean,vabsmean+perabsmean))
                im1.set_interpolation(u'none')
                ax1.set_title(u'Object magnitude - projection {}'.format(idxp))
                im2.set_data(np.angle(objs))
                im1.set_cmap('bone')
                im2.set_interpolation(u'none')
                ax2.set_title(u'Object Phase - projection {}'.format(idxp))
                im3.set_data(hsv_to_rgb(probe_hsv))
                im3.set_interpolation(u'none')
                ax3.set_title(u'Probe (1st mode) - projection {}'.format(idxp))
                im1.axes.figure.canvas.draw()
                im2.axes.figure.canvas.draw()
                im3.axes.figure.canvas.draw()

        if self.showrecons:
            plt.ioff()

        obj_shape=stack_objs.shape
        print(u"Number of projections loaded: {}".format(obj_shape[0]))
        print(u"Dimensions {} x {} pixels".format(obj_shape[1],obj_shape[2]))
        print('All projections loaded\n')
        return stack_objs, stack_angles, pixelsize

class SaveData(PathName,Variables):
    """
    Save projections to HDF5 file
    """
    def __init__(self,**params):
        super().__init__(**params)
        self.params = params
        try: self.cxientry = params['cxientry']
        except: pass
        try: self.autosave = params['autosave']
        except: pass

    def __call__(self,*args):#h5name,stack_projs,theta,shiftstack):
        if len(args)<3:
            print('Wrong number of arguments')
            print('Usage: SaveData(stack_projs,theta,shiftstack[optional])')
            raise Exception('At least 3 arguments are needed and cannot be more than 4')
        if self.autosave:
            ansuser = 'y'
            self.save_data(*args)
        else:
            while True:
                ansuser = input("Do you want to save the data to HDF5 file? ([y]/n) ").lower()
                if ansuser == '' or ansuser =='y':
                    self.save_data(*args)#stack_projs,theta,shiftstack)
                    break
                elif ansuser == 'n':
                    print("The projections have NOT been saved yet. Please, be careful")
                    break
                else:
                    print("You have to answer y or n")

    def round_to_even(self):
        return lambda x: int(2*np.floor(x/2))

    def save_masks(self,h5name, masks):
        """
        Save masks for the linear phase ramp removal of the phase
        contrast image or the air removal from the amplitude images
        """
        print('Saving {}'.format(h5name))
        h5file = self.results_datapath(h5name)
        if os.path.isfile(h5file): os.remove(h5file)
        with h5py.File(h5file,'a') as fid:
            fid.create_dataset('masks/stack', data = masks, dtype = np.bool) # air/vacuum mask
        print('Done')

    def save_data(self,*args):
        """
        Save data to HDF5 File

        Parameters:
        -----------
        *args: positional arguments
            args[0] : str
                H5 file name
            args[1] : ndarray
                Array containing the stack of projections
            args[2] : ndarray
                Values of theta
            args[3] : ndarray
                Array containing the shifts for each projection in the
                stack. If not provided, it will be initialized with zeros
            args[4] : ndarray or None
                Array containing the projection masks
        """
        h5name = args[0]
        stack_projs = args[1]
        nprojs, nr, nc = stack_projs.shape
        theta = args[2]

        if len(args)==4: shiftstack = args[3]
        else: shiftstack = np.zeros((2,nprojs))

        if len(args)==5: masks = args[4]
        else: masks = None

        round_to_even = self.round_to_even() #lambda x: int(2*np.floor(x/2))
        chunk_size = (round_to_even(nprojs/4),round_to_even(nr/20),round_to_even(nc/20))

        if np.iscomplexobj(stack_projs[0]): array_dtype = np.complex64
        else: array_dtype = np.float32

        print('Saving {}'.format(h5name))
        h5file = self.results_datapath(h5name)
        if os.path.isfile(h5file):
            print('File {} already exists and will be overwritten'.format(h5name))
            os.remove(h5file)
        print('Saving metadata')
        write_paramsh5(h5file,**self.params)
        print('Saving data. This takes time, please wait...')
        with h5py.File(h5file,'a') as fid:
            fid.create_dataset('shiftstack/shiftstack', data = shiftstack, dtype = np.float32) # shiftstack
            fid.create_dataset('angles/thetas', data = theta, dtype = np.float32) # thetas
            dset = fid.create_dataset('projections/stack', shape= (nprojs,nr,nc),dtype=array_dtype, chunks=chunk_size)#,compression='lzf')#, compression='gzip', compression_opts=9)
            p0 = time.time()
            for ii in range(nprojs):
                print('Projection: {} out of {}'.format(ii+1,nprojs), end='\r')
                dset[ii,:,:]= stack_projs[ii]
            print('\r')
            if masks is not None:
                fid.create_dataset('masks/stack', data = masks, dtype = np.bool) # air/vacuum mask
            print('Done. Time elapsed = {:.03f} s'.format(time.time()-p0))
        print('Data saved to file {}'.format(h5name))
        print('In the folder {}'.format(self.results_folder()))

    def save_FSC(self, *args):
        """
        Save FSC data to HDF5 file
        Parameters:
        -----------
        *args: positional arguments
            args[0] : str
                H5 file name
            args[1] : ndarray
                Normalized frequencies
            args[2] : ndarray
                Value of the threshold for each frequency
            args[3] : ndarray
                The FSC curve
            args[4] : ndarray
                The first tomogram
            args[5] : ndarray
                The second tomogram
            args[6] : ndarray
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
        pixelsize = args[7]

        round_to_even = self.round_to_even() #lambda x: int(2*np.floor(x/2))

        print('Saving {}'.format(h5name))
        h5file = self.results_datapath(h5name)
        if os.path.isfile(h5file):
            print('File {} already exists and will be overwritten'.format(h5name))
            os.remove(h5file)
        print('Saving metadata')
        write_paramsh5(h5file,**self.params)
        print('Saving data. This takes time, please wait...')
        p0 = time.time()
        with h5py.File(h5file,'a') as fid:
            fid.create_dataset('angles/thetas',data = theta,dtype=np.float32) # add the thetas
            #fid.create_dataset('pixelsize',data=pixelsize,dtype=np.float32)
            fid.create_dataset('FSC',data = FSCcurve ,dtype=np.float32) # add the FSC curve
            fid.create_dataset('T',data = T ,dtype=np.float32) # add the threshold
            fid.create_dataset('normfreqs',data = normfreqs ,dtype=np.float32) # add the normalized freqs
            if tomogram1.ndim == 2:
                fid.create_dataset('tomogram1', data= tomogram1,dtype=np.float32)
                fid.create_dataset('tomogram2', data= tomogram2,dtype=np.float32)
            elif tomogram1.ndim == 3:
                nslices,nr,nc = tomogram1.shape
                chunk_size = (round_to_even(nslices/4),round_to_even(nr/20),round_to_even(nc/20))
                print('Saving tomogram1 and tomogram2. This takes time, please wait...')
                dset1 = fid.create_dataset('tomogram1', shape= tomogram1.shape,dtype=np.float32, chunks=chunk_size)#,compression='lzf')#, compression='gzip', compression_opts=9)
                dset2 = fid.create_dataset('tomogram2', shape= tomogram2.shape,dtype=np.float32, chunks=chunk_size)#,compression='lzf')#, compression='gzip', compression_opts=9)
                for ii in range(nslices):
                    print('Slice: {} out of {}'.format(ii+1,nprojs), end='\r')
                    dset1[ii,:,:] = tomogram1[ii]
                    dset2[ii,:,:] = tomogram2[ii]
                print('\r')
            print('Done. Time elapsed = {:.03f} s'.format(time.time()-p0))
        print('FSC data saved to file {}'.format(h5name))
        print('In the folder {}'.format(self.results_folder()))

class LoadData(PathName,Variables):
    """
    Load projections from HDF5 file
    """
    def __init__(self,**params):
        super().__init__(**params)
        self.params = params
        try: self.amponly = params['amponly']
        except: pass
        try: self.phaseonly = params['phaseonly']
        except: pass

    def __call__(self,h5name):
        return self.load_data(h5name)

    def load_shiftstack(self,h5name):
        """
        Load shitstack from previous h5 file

        Parameters:
        ---------
        h5name: str
            File name from which data is loaded

        Returns:
        --------
        shiftstack: ndarray
            Shifts in vertical (1st dimension) and horizontal
                    (2nd dimension)
        """
        print('Loading shiftstack from file {}'.format(h5name))
        h5file = self.results_datapath(h5name)
        with h5py.File(h5file,'r') as fid:
            shiftstack= fid[u'shiftstack/shiftstack'][()]
        return shiftstack

    def load_masks(self,h5name):
        """
        Load masks from previous h5 file

        Parameters:
        ---------
        h5name: str
            File name from which data is loaded

        Returns:
        --------
        masks: ndarray
            Array with the masks
        """
        print('Loading the projections from file {}'.format(h5name))
        h5file = self.results_datapath(h5name)
        with h5py.File(h5file,'r') as fid:
            masks= fid[u'masks/stack'][()]
        return masks

    def load_data(self,h5name):
        """
        Load data from h5 file

        Parameters:
        ---------
        h5name: str
            File name from which data is loaded

        Returns:
        --------
        stack_projs: ndarray
            Stack of projections
        theta: ndarray
            Stack of thetas
        shiftstack: ndarray
            Shifts in vertical (1st dimension) and horizontal
                    (2nd dimension)
        datakwargs : dict
            Dictionary with metadata information
        """
        print('Loading the projections from file {}'.format(h5name))
        h5file = self.results_datapath(h5name)
        shiftstack = self.load_shiftstack(h5name)
        with h5py.File(h5file,'r') as fid:
            theta = fid['angles/thetas'][()]
            #shiftstack= fid[u'shiftstack/shiftstack'][()]
            # read the inputkwargs dict
            datakwargs = dict()
            print('Loading metadata')
            for keys in sorted(list(fid['info'].keys())):
                datakwargs[keys]=fid['info/{}'.format(keys)][()]
            datakwargs.update(self.params) # add/update with new values
            print('Loading projections. This takes time, please wait...')
            p0 = time.time()
            stack_projs = fid[u'projections/stack'][()]
            if self.amponly and np.iscomplexobj(stack_projs):
                print('Taking only amplitudes')
                stack_projs = np.abs(stack_projs)
            elif self.phaseonly and np.iscomplexobj(stack_projs):
                print('Taking only phases')
                stack_projs = np.angle(stack_projs)
            print('Done. Time elapsed = {:.03f} s'.format(time.time()-p0))
        print('Projections loaded from file {}'.format(h5name))
        return stack_projs, theta, shiftstack, datakwargs

class SaveTomogram(SaveData):
    """
    Save tomogram to HDF5 file
    """
    def __init__(self,**params):
        super().__init__(**params)
        self.params = params

    def __call__(self,*args):
        if self.autosave:
            ansuser = 'y'
            self.save_tomogram(*args)
        else:
            while True:
                ansuser = input("Do you want to save the data to HDF5 file? ([y]/n) ").lower()
                if ansuser == '' or ansuser =='y':
                    self.save_tomogram(*args)
                    break
                elif ansuser == 'n':
                    print("The projections have NOT been saved yet. Please, be careful")
                    break
                else:
                    print("You have to answer y or n")

    def save_tomogram(self,*args):
        """
        Parameters:
        -----------
        *args: positional arguments
            args[0] : str
                H5 file name
            args[1] : ndarray
                Array containing the stack of slices (tomogram)
            args[2] : ndarray
                Values of theta
            args[3] : ndarray
                Array containing the shifts for each projection in the stack
        """
        h5name = args[0]
        tomogram = args[1]
        nslices, nr, nc = tomogram.shape
        theta = args[2]
        shiftstack = args[3]

        round_to_even = self.round_to_even()
        chunk_size = (round_to_even(nslices/4),round_to_even(nr/20),round_to_even(nc/20))

        print('Saving {}'.format(h5name))
        h5file = self.results_datapath(h5name)
        if os.path.isfile(h5file):
            print('File {} already exists and will be overwritten'.format(h5name))
            os.remove(h5file)
        print('Saving metadata')
        write_paramsh5(h5file,**kwargs)
        print('Saving data. This takes time, please wait...')
        with h5py.File(h5file,'a') as fid:
            fid.create_dataset('shiftstack/shiftstack',data = shiftstack,dtype=np.float32) # add the shiftstack
            fid.create_dataset('angles/thetas',data = theta,dtype=np.float32) # add the thetas
            dset = fid.create_dataset('tomogram/slices', shape= (nslices,nr,nc),dtype=np.float32, chunks=chunk_size)#,compression='lzf')#, compression='gzip', compression_opts=9)
            print('Saving tomographic slices. This takes time, please wait...')
            p0 = time.time()
            for ii in range(nslices):
                print('Slice: {} out of {}'.format(ii+1,nslices),end='\r')
                dset[ii,:,:] = tomogram[ii]
            print('\r')
            print('Done. Time elapsed = {:.03f} s'.format(time.time()-p0))
        print('Tomogram saved to file {}'.format(h5name))
        print('In the folder {}'.format(self.results_folder()))

class LoadTomogram(LoadData):
    """
    Load projections from HDF5 file
    """
    def __init__(self,**params):
        super().__init__(**params)
        self.params = params

    def __call__(self,h5name):
        return self.load_tomogram(h5name)

    def load_tomogram(self,h5name):
        """
        Load tomographic data from h5 file

        Parameters:
        ---------
        h5name: str
            File name from which data is loaded

        Returns:
        --------
        tomogram: ndarray
            Stack of tomographic slices
        theta: ndarray
            Stack of thetas
        shiftstack: ndarray
            Shifts in vertical (1st dimension) and horizontal
                    (2nd dimension)
        datakwargs : dict
            Dictionary with metadata information
        """
        print('Loading tomogram from file {}'.format(h5name))
        h5file = self.results_datapath(h5name)
        shiftstack = self.load_shiftstack(h5name)
        with h5py.File(h5file,'r') as fid:
            theta = fid['angles/thetas'][()]
            # read the inputkwargs dict
            datakwargs = dict()
            for keys in sorted(list(ff['info'].keys())):
                datakwargs[keys]=ff['info/{}'.format(keys)][()]
            datakwargs.update(self.params)
            print('Loading tomogram. This takes time, please wait...')
            p0=time.time()
            tomogram = ff[u'tomogram/slices'][()]
            print('Done. Time elapsed = {:.03f} s'.format(time.time()-p0))
        print('Tomogram loaded from file {}'.format(h5name))
        return tomogram,theta,shiftstack,datakwargs
