#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Resolution estimate of projections using
Fourier Ring correlation
@author: jdasilva
"""
# import of standard libraries
import os
import re
import socket
import sys
import time

# import of third party packages
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from skimage.restoration import unwrap_phase
from scipy.ndimage.fourier import fourier_shift

# import of local packages
from FSC import FSCPlot, read_ptyr, rmphaseramp

#-------------------------------------------------------
# still keep this block, but it should disappear soon
if sys.version_info<(3,0): # backcompatibility
    import Tkinter,tkFileDialog
    input = raw_input
else:
    import tkinter as Tkinter
    import tkinter.filedialog as tkFileDialog
#-------------------------------------------------------

#initializing params
params = dict()
#=========================
# Edit session
#=========================
params[u'apod_width'] = 100 # apodization width in pixels
params[u'thick_ring'] = 8 # number of pixel to average each FRC ring
params[u'crop'] = [200,-370,300,-300] # cropping [top,bottom,left,right]
params[u'vmin_plot'] = None
params[u'vmax_plot'] = -0.5e-4#None
params[u'colormap'] = 'bone' # colormap to show images
params[u'unwrap'] = False # unwrap the phase
params[u'flip2ndimage'] = False # flip the 2nd image
params[u'normalizeimage'] = False # normalize the images
#=========================

# open GUI to choose file
root = Tkinter.Tk()
root.withdraw()
print('You need to load two files for the FSC evalution')
print('Please, load the first file...')
pathfilename1 = tkFileDialog.askopenfilename(initialdir='.', title='Please, load the first file...')
print('File 1: {}'.format(pathfilename1))
print('Please, load the second file...')
pathfilename2 = tkFileDialog.askopenfilename(initialdir='.', title='Please, load the second file...')
print('File 2: {}'.format(pathfilename2))

# Read the HDF5 file .ptyr
data1,probe1,pixelsize1 = read_ptyr(pathfilename1) # file1
data2,probe2,pixelsize2 = read_ptyr(pathfilename2) # file2

if params[u'flip2ndimage']: # flip one of the images
    print('Flipping 2nd image')
    data2 = np.fliplr(data2)

print("the pixelsize of data1 is {:0.02f} nm".format(pixelsize1[0]*1e9))
print("the pixelsize of data2 is {:0.02f} nm".format(pixelsize2[0]*1e9))

# cropping the image to an useful area
reg = params[u'crop']  # how many pixels are cropped from the borders of the image
img1 = (data1[reg[0]:reg[1],reg[2]:reg[3]])
img2 = (data2[reg[0]:reg[1],reg[2]:reg[3]])

# remove phase ramp
print("Removing the ramp")
image1 = rmphaseramp(img1, weight=None, return_phaseramp=False)
image2 = rmphaseramp(img2, weight=None, return_phaseramp=False)

if params[u'unwrap']:
    # remove also wrapping
    print("Unwrapping the phase image1")
    image1 = unwrap_phase(np.angle(image1))
    print("Unwrapping the phase image2")
    image2 = unwrap_phase(np.angle(image2))
else:
    image1 = np.angle(image1)
    image2 = np.angle(image2)

if params[u'normalizeimage']:
    image1=(image1-image1.min())/(image1.max()-image1.min())
    image2=(image2-image2.min())/(image2.max()-image2.min())

# Display the images
plt.close('all')
fig, (ax1, ax2, ax3) = plt.subplots(num=1, ncols=3)
ax1.imshow(image1,interpolation='none',cmap='bone')
ax1.set_axis_off()
ax1.set_title('Image 1 (ref.)')
ax2.imshow(image2,interpolation='none',cmap='bone')
ax2.set_axis_off()
ax2.set_title('Image 2')
# View the output of a cross-correlation
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

print("\nCorrecting the shift of image2 by using subpixel precision...")
offset_image2 = np.fft.ifftn(fourier_shift(np.fft.fftn(image2),shift))#(shift[0],-shift[1])))
offset_image2 *= np.exp(1j*diffphase)

# cropping the images beyond the shift amplitude
regfsc = np.ceil(np.abs(shift)).astype(np.int)
if regfsc[0] != 0 and regfsc[1] != 0:
    image1FSC = image1[regfsc[0]:-regfsc[0],regfsc[1]:-regfsc[1]]
    offset_image2FSC = offset_image2[regfsc[0]:-regfsc[0],regfsc[1]:-regfsc[1]]

# display aligned images
plt.close('all')
fig, (ax1, ax2) = plt.subplots(num=1, ncols=2)
ax1.imshow(image1FSC,interpolation='none',cmap='bone')
ax1.set_axis_off()
ax1.set_title('Image1 (ref.)')
ax2.imshow(offset_image2FSC.real,interpolation='none',cmap='bone')
ax2.set_axis_off()
ax2.set_title('Offset corrected image2')
plt.show(block=False)

print("Estimating the resolution by FSC. Press <Enter> to continue")
a = input()
plt.close('all')

startfsc = time.time()
FSC2D=FSCPlot(image1FSC,offset_image2FSC.real,'onebit',params['thick_ring'],apod_width=params['apod_width'])#transv_apod=params['transv_apod'],axial_apod=params['axial_apod'])
normfreqs, T, FSC2Dcurve = FSC2D.plot()
endfsc = time.time()
print("Time elapsed: {:g} s".format(endfsc-startfsc))

print("The pixelsize of the data is {:.02f} nm".format(pixelsize1[0]*1e9))

a = input("\nPlease, input the value of the intersection: ")
print("------------------------------------------")
print("| Resolution is estimated to be {:.02f} nm |".format(pixelsize1[0]*1e9/float(a)))
print("------------------------------------------")

input("\n<Hit Return to close images>")
plt.close('all')
