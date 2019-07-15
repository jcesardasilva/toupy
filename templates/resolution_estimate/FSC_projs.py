# -*- coding: utf-8 -*-
"""
@author: jdasilva
"""
In [122]: pathfilename1                                                                  
Out[122]: '/mntdirect/_data_id16a_inhouse4/visitor/ma4351/id16a/v97_h_nfptomo/v97_h_nfptomo_15nm_analysis/recons/v97_h_nfptomo_15nm_subtomo001_0000/v97_h_nfptomo_15nm_subtomo001_0000_ML.ptyr'

In [123]: pathfilename2                                                                  
Out[123]: '/mntdirect/_data_id16a_inhouse4/visitor/ma4351/id16a/v97_h_nfptomo/v97_h_nfptomo_15nm_analysis/recons/v97_h_nfptomo_15nm_subtomo004_0200/v97_h_nfptomo_15nm_subtomo004_0200_ML.ptyr'



# import of standard libraries
import os
import re
import socket
import sys
import time
if re.search('gpu', socket.gethostname()) or re.search('gpid16a', socket.gethostname()):
    import pyfftw # has to be imported first to avoid ImportError: dlopen: cannot load any more object with static TLS
else:
    import pyfftw

# import of third party packages
import h5py
import matplotlib.pyplot as plt
import numpy as np
#from skimage.measure import profile_line
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from skimage.restoration import unwrap_phase
from scipy.ndimage.fourier import fourier_shift
#from scipy.special import erf
#from FSC import *

# import of local packages
from FSC import FSCPlot, checkhostname, load_data_FSC
from io_utils import read_ptyr, save_or_load_data, checkhostname, create_paramsh5, load_paramsh5, save_or_load_FSCdata

if sys.version_info<(3,0):
    import Tkinter,tkFileDialog
    input = raw_input
else:
    import tkinter as Tkinter
    import tkinter.filedialog as tkFileDialog

#initializing params
params = dict()
#=========================
# Edit session
#=========================
params[u'transv_apod'] = 250 #transverse apodization
params[u'axial_apod'] = 200 # axial apodization
params[u'thick_ring'] = 4 # number of pixel to average each FRC ring
params[u'crop'] = [450,1150,350,-350]#[448:2150,350:-350]#[550,1800,550,1800]#[580] #640
params[u'vmin_plot'] = None#0.5e-5
params[u'vmax_plot'] = -0.5e-4#None
params[u'colormap'] = 'bone' # colormap to show images
#=========================

# open GUI to choose file
root = Tkinter.Tk()
root.withdraw()
pathfilename1 = tkFileDialog.askopenfilename(initialdir='.', title='Please, load the reference frame ...')#'/data/id16a/inhouse1/commissioning/')
print('File 1: {}'.format(pathfilename1))
pathfilename2 = tkFileDialog.askopenfilename(initialdir='.', title='Please, load the frame for comparison...')
print('File 2: {}'.format(pathfilename2))

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

# Read the HDF5 file .ptyr
data1,probe1,pixelsize1 = read_ptyr(pathfilename1) # file1
data2,probe2,pixelsize2 = read_ptyr(pathfilename2) # file2

# flip one of the images
data2 = np.fliplr(data2)

print("the pixelsize of data1 is {:g} nm".format(pixelsize1[0]*1e9))
print("the pixelsize of data2 is {:g} nm".format(pixelsize2[0]*1e9))

# cropping the image to an useful area
reg = params[u'crop']  # how many pixels are cropped from the borders of the image
img1 = (data1[reg[0]:reg[1],reg[2]:reg[3]])
#~ img1 = (data1[reg:-reg,reg:-reg])
img2 = (data2[reg[0]:reg[1],reg[2]:reg[3]])
#~ img2 = (data2[reg:-reg,reg:-reg])

print("Removing the ramp")
#~ image1 = rmphaseramp(img1, weight=None, return_phaseramp=False)
#~ image2 = rmphaseramp(img2, weight=None, return_phaseramp=False)
image1 = np.angle(rmphaseramp(np.exp(1j*img1),weight=None,return_phaseramp=False))
image2 = np.angle(rmphaseramp(np.exp(1j*img2),weight=None,return_phaseramp=False))

# remove also wrapping
print("Unwrapping the phase")
image1 = unwrap_phase(image1)
image2 = unwrap_phase(image2)

# normalizing the images
image1=(image1-image1.min())/(image1.max()-image1.min())
image2=(image2-image2.min())/(image2.max()-image2.min())

# Display the images
plt.close('all')
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14,6))
ax1.imshow(image1,interpolation='none',cmap='bone')
ax1.set_axis_off()
ax1.set_title('Image 1 (ref.)')
ax2.imshow(image2,interpolation='none',cmap='bone')
ax2.set_axis_off()
ax2.set_title('Image 2')

# View the output of a cross-correlation to show what the algorithm is
# doing behind the scenes
image_product = np.fft.fft2(image1) * np.fft.fft2(image2).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
#ax3.set_axis_off()
ax3.set_title("Cross-correlation")
plt.show(block=False)

# Choose between pixel or subpixel precision image registration. By default, it is pixel precision.
precision = (input("Do you want to use pixel(1) or subpixel(2) precision registration?[1] "))
if (precision==str(1) or precision ==''):
    # pixel precision
    print("\nCalculating the pixel precision image registration ...")
    start = time.time()
    shift, error, diffphase = register_translation(image1, image2)
    print(diffphase)
    end = time.time()
    print("Time elapsed: {:g} s".format(end-start))
    print("Detected pixel offset [y,x]: [{:g}, {:g}]".format(shift[0],shift[1]))
elif precision == str(2):
    # subpixel precision
    print("\nCalculating the subpixel image registration ...")
    start = time.time()
    shift, error, diffphase = register_translation(image1, image2, 100)
    print(diffphase)
    end = time.time()
    print("Time elapsed: {:g} s".format(end-start))
    print("Detected subpixel offset [y,x]: [{:g}, {:g}]".format(shift[0],shift[1]))
else:
    print("You must choose between 1 and 2")
    raise SystemExit

print("Shift = [%g, %g], Error= %g, DiffPhase = %g" %(shift[0],shift[1],error,diffphase))

print("\nCorrecting the shift of image2 by using subpixel precision...")
offset_image2 = np.fft.ifftn(fourier_shift(np.fft.fftn(image2),shift))
offset_image2 *= np.exp(1j*diffphase)

# cropping the images beyond the shift amplitude
regfsc = int(np.ceil(np.max(shift)))
if regfsc !=0:
    image1 = image1[regfsc:-regfsc,regfsc:-regfsc]
    offset_image2 = offset_image2[regfsc:-regfsc,regfsc:-regfsc]

print("Removing the ramp again")
#~ image1 = np.real(rmphaseramp(image1, weight=None, return_phaseramp=False))
#~ offset_image2 = np.real(rmphaseramp(offset_image2, weight=None, return_phaseramp=False))

#~ image1 = np.angle(rmphaseramp(np.exp(1j*image1),weight=None,return_phaseramp=False))
#~ offset_image2 = np.angle(rmphaseramp(np.exp(1j*offset_image2),weight=None,return_phaseramp=False))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))
ax1.imshow(image1,interpolation='none',cmap='bone')
ax1.set_axis_off()
ax1.set_title('Image1 (ref.)')
ax2.imshow(offset_image2.real,interpolation='none',cmap='bone')
ax2.set_axis_off()
ax2.set_title('Offset corrected image2')
plt.show(block=False)

print("Estimating the resolution by FSC...")
startfsc = time.time()
FSC2D=FSCPlot(image1,offset_image2.real,'onebit',params['thick_ring'],transv_apod=params['transv_apod'],axial_apod=params['axial_apod'])
normfreqs, T, FSC2Dcurve, intersec = FSC2D.plot()
endfsc = time.time()
print("Time elapsed: {:g} s".format(endfsc-startfsc))

print("The pixelsize of the data is %.2f nm" %(pixelsize1[0]*1e9))

a = input("\nPlease, input the value of the intersection: ")
#a = intersection
print("------------------------------------------")
print("| Resolution is estimated to be %.2f nm |" %(pixelsize1[0]*1e9/float(a)))
print("------------------------------------------")

raw_input("\n<Hit Return to close images>")
plt.close('all')

print("Your program has finished!")
