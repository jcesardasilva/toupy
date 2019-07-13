# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:20:36 2015

@author: Julio Cesar da Silva (mailto: jdasilva@esrf.fr)
"""

#from __future__ import division, print_function
import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os, re, glob
import sys, time
import libtiff
#from ptypy.utils import rmphaseramp
sys.tracebacklimit = 0

savetif = False
remove_ramp = False#True
creg = 1024 # how many pixels are cropped from the borders of the image
smin = -1.2
smax = 0.5
proj0 = eval(sys.argv[1])
projf = eval(sys.argv[2])

if len(sys.argv) == 4:
    interp_type = str(sys.argv[3])
    print('Using {} interpolation'.format(interp_type))
else:
    interp_type = 'none'
   
import Tkinter,tkFileDialog

root = Tkinter.Tk()
root.withdraw()

print('%%%%%% Load the first recon HDF5 file %%%%%%')
filename = tkFileDialog.askopenfilename(parent=root,initialdir=".",title='Please select the file')
print(filename)

scan_wcard = re.sub("subtomo001_\d{4}","subtomo001_*",filename)

filelist = sorted(glob.glob(scan_wcard))
print("Found {} files".format(len(filelist)))
frames = filelist[proj0:projf+1]

def read_ptyr(filename):
    # Read the HDF5 file .ptyr
    with h5py.File(filename,'r') as file1:#'id16siemensstar_2_ML.ptyr','r')
        # get the root entry
        content1 = file1.keys()[0]

        # get the data from the object
        img1 = np.squeeze(np.array(file1[content1+"/obj/S00G00/data"]))

        # get the pixel size
        pixelsize1 = file1[content1+"/obj/S00G00/_psize"][()]
    return img1, pixelsize1

#%% Auxiliary functions
def rmphaseramp(a, weight=None, return_phaseramp=False):
    """
    Attempts to remove the phase ramp in a two-dimensional complex array 
    ``a``.

    Parameters
    ----------
    a : ndarray
        Input image as complex 2D-array.
    
    weight : ndarray, str, optional
        Pass weighting array or use ``'abs'`` for a modulus-weighted 
        phaseramp and ``Non`` for no weights.
    
    return_phaseramp : bool, optional
        Use True to get also the phaseramp array ``p``.
    
    Returns
    -------
    out : ndarray
        Modified 2D-array, ``out=a*p``
    p : ndarray, optional
        Phaseramp if ``return_phaseramp = True``, otherwise omitted 
    
    Examples
    --------
    >>> b = rmphaseramp(image)
    >>> b, p = rmphaseramp(image , return_phaseramp=True)
    """

    useweight = True
    if weight is None:
        useweight = False
    elif weight == 'abs':
        weight = np.abs(a)
        
    ph = np.exp(1j*np.angle(a))
    [gx, gy] = np.gradient(ph)
    gx = -np.real(1j*gx/ph)
    gy = -np.real(1j*gy/ph)

    if useweight:
        nrm = weight.sum()
        agx = (gx*weight).sum() / nrm
        agy = (gy*weight).sum() / nrm
    else:
        agx = gx.mean()
        agy = gy.mean()

    (xx, yy) = np.indices(a.shape)
    p = np.exp(-1j*(agx*xx + agy*yy))

    if return_phaseramp:
        return a*p, p
    else:
        return a*p


# read the first image to give an idea
img1,pixelsize1  = read_ptyr(filelist[0])
print("The pixelsize of the first recons is {:04.2f} nm".format(pixelsize1[0]*1e9))
print('The dimensions in pixels of the data are {}'.format(img1.shape))
print('Therefore, the dimensions in microns are ({:.2f},{:.2f})'.format(1e6*pixelsize1[0]*img1.shape[0],1e6*pixelsize1[1]*img1.shape[1]))

def remove_phaseramp(img1):
    return rmphaseramp(img1, weight=None, return_phaseramp=False)

def flip_crop(img1,reg=0):
    imgout = np.fliplr(np.transpose(img1[0+reg:-1-reg,0+reg:-1-reg]))
    return np.abs(imgout), np.angle(imgout)

#x = 1
#print "Script started."
#stored_exception=None

#while True:
    #try:
        #print "Processing file #",x,"started...",
        ## do something time-cosnuming
        #time.sleep(1)
        #print " finished."
        #if stored_exception:
            #break
        #x += 1
    #except KeyboardInterrupt:
        #stored_exception=sys.exc_info()

#print "Bye"
#print "x=",x

#if stored_exception:
    #raise stored_exception[0], stored_exception[1], stored_exception[2]

#sys.exit()
stored_exception=None


# Display the images
plt.ion()
fig = plt.figure(1, figsize=(10,10))
for ff in frames:
    try:
        print('File: {}'.format(ff))
        img, pixelsize2 = read_ptyr(ff)
        # cropping the image to an useful area
        if remove_ramp:
            img = remove_phaseramp(img)
        
        image2, image1 = flip_crop(img,creg)
        #image1 = flip_crop(np.angle(remove_phaseramp(img1)),creg)#np.fliplr(np.transpose(np.angle(img1[0+reg:-1-reg,0+reg:-1-reg])))
        #image2 = flip_crop(np.abs(remove_phaseramp(img1)),creg)#np.fliplr(np.transpose(np.abs(img1[0+reg:-1-reg,0+reg:-1-reg])))
            
        vmean = image2.mean()
        permean = 0.2*vmean
        
        #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,6))
        plt.clf()
        ax1 = fig.add_subplot(221)
        im1=ax1.imshow(image1,cmap='bone',interpolation=interp_type,vmin = smin, vmax = smax)#vmin=-0.2,vmax=0.2)#,cmap='bone')#origin='lower',
        fig.colorbar(im1,ax=ax1)
        ax1.set_axis_off()
        ax1.set_title('Phase')
        
        ax3 = fig.add_subplot(223)
        ax3.plot(image1[image1.shape[0]//2,:])
        ax3.set_title('Profile Phase')
        ax3.axis('auto')
        
        ax2 = fig.add_subplot(222)
        #im2=ax2.imshow(image2,origin='lower',vmin=1.16,vmax=1.63, cmap='gray')
        im2=ax2.imshow(image2,vmin=vmean-permean,vmax=vmean+permean, cmap='gray',interpolation=interp_type)#origin='lower',
        #im2=ax2.imshow(image2,cmap='gray')#origin='lower',
        fig.colorbar(im2,ax=ax2)
        ax2.set_axis_off()
        ax2.set_title('Amplitude')
        
        ax3 = fig.add_subplot(224)
        ax3.plot(image2[image2.shape[0]//2,:])
        ax3.set_title('Profile Amplitude')
        ax3.axis('auto')
        
        plt.suptitle(os.path.basename(ff))
        #plt.savefig('screenshot.png',bbox_inches='tight')
        #fig.draw()
        fig.canvas.draw()
        plt.pause(0.01)
        #plt.clf()
        if stored_exception:
            time.sleep(1)
            #plt.close('all')
            break
    except KeyboardInterrupt:
        print("[CTRL+C detected]"),
        stored_exception=sys.exc_info()
        #break
        #traceback.print_exc(file=sys.stdout)
if stored_exception:
    plt.close('all')
    plt.ioff()
    raise stored_exception[0], stored_exception[1], stored_exception[2]

sys.exit()
if savetif: 
    #fig.savefig('ptycho_4x4microns.png', bbox_inches='tight')
    tifffilenamep = os.path.splitext(os.path.basename(filename))[0]+'_phase.tif'
    tiffp = libtiff.TIFF.open(tifffilenamep,mode='w')
    tiffp.write_image(image1)
    tiffp.close()

    tifffilenamea = os.path.splitext(os.path.basename(filename))[0]+'_amplitude.tif'
    tiffa = libtiff.TIFF.open(tifffilenamea,mode='w')
    tiffa.write_image(image2)
    tiffa.close()

#a=raw_input("\n<Hit Return to close images>")
#plt.close('all')