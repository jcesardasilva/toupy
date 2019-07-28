# coding: utf-8
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import h5py
from ..restoration.ramptools import rmphaseramp
import time
import shutil
import os

overwrite_h5 = True
#filename = '/data/visitor/ma3495/id16a/analysis/recons/O2_LSCF_CGO_25nm_subtomo001_0000/O2_LSCF_CGO_25nm_subtomo001_0000_ML.ptyr'
filename = '/data/id16a/inhouse2/staff/ap/ihls2664/id16a/analysis/dumps/bone2_hoptyfluo_nfptomo_subtomo001_0057/bone2_hoptyfluo_nfptomo_subtomo001_0057_DM_0500.ptyr'

# only to keep a copy of the file and prevent overwritting
if not os.path.isfile(filename+'.vort'):
    shutil.copy(filename, filename+'.vort')

# Open the file and extract the image data
with h5py.File(filename, 'r') as fid:
    probe = (fid['content/probe/S00G00/data'][()])
    objs = (fid['content/obj/S00G00/data'][()])

# get object phases
obj_phase = np.angle(objs[0])

# get the phase of the modes
p1_phase = np.angle(probe[0])
p2_phase = np.angle(probe[1])
p3_phase = np.angle(probe[2])

# function definition


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def remove_vortices(img_in, to_ignore=100):
    # remove phase ramp
    img_pr = rmphaseramp(img_in)  # [101:-100,93:-93] # to get square image
    # find residues
    residues, residues_charge = phaseresidues(np.angle(img_pr))

    # ignore borders
    if to_ignore != 0:
        print('Ignoring border')
        residues[:to_ignore, :] = 0
        residues[:, :to_ignore] = 0
        residues[-to_ignore:, :] = 0
        residues[:, -to_ignore:] = 0

    img_res = np.zeros_like(np.angle(img_pr))
    img_res[1:-1, 1:-1] = residues.copy()

    # get array of vortice positions
    yres, xres = np.where(np.abs(residues) > 0.1)
    print('Getting vortice positions...')
    print('Found {}'.format(len(xres)))
    # remove the vortices
    img_phase_novort = img_pr.copy().astype(np.complex64)
    n, m = img_res.shape
    x = np.arange(m)
    y = np.arange(n)
    for idx, ii in enumerate(xrange(len(xres))):
        print('{} residues out of {}'.format(idx, len(xres)))
        s0 = time.time()
        X, Y = np.meshgrid(x-xres[ii], y-yres[ii])
        R, T = cart2pol(X, Y)
        #img_phase_novort *= np.exp(-1j*T*residues[yres[ii],xres[ii]])
        img_phase_novort *= np.exp(1j*T*residues[yres[ii], xres[ii]])
        print('Time elapsed = {} s'.format(time.time()-s0))
    return img_phase_novort, xres, yres

# Removing vortices

# def processInput(ii):
    # return remove_vortices(probe[ii],to_ignore = 100)


#num_cores = multiprocessing.cpu_count()
#probe_phase_novort =np.empty_like(probe)
##results = Parallel(n_jobs = num_cores)(delayed(processInput)(ii) for ii in (xrange(probe.shape[0])))
print('Removing vortices of probe mode 1')
p1_phase_novort, p1_xres, p1_yres = remove_vortices(probe[0], to_ignore=100)
print('Removing vortices of probe mode 2')
p2_phase_novort, p2_xres, p2_yres = remove_vortices(probe[1], to_ignore=100)
print('Removing vortices of probe mode 2')
p3_phase_novort, p3_xres, p3_yres = remove_vortices(probe[2], to_ignore=100)

# remove phase ramp again
p1_phase_novort2, ramp_obj = rmphaseramp(
    p1_phase_novort, return_phaseramp=True)
p2_phase_novort2, ramp_obj = rmphaseramp(
    p2_phase_novort, return_phaseramp=True)
p3_phase_novort2, ramp_obj = rmphaseramp(
    p3_phase_novort, return_phaseramp=True)

# feed the new array
probe_novort = np.empty_like(probe)
probe_novort[0] = p1_phase_novort2
probe_novort[1] = p2_phase_novort2
probe_novort[2] = p3_phase_novort2

# display the probes with the residues
plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(131)
# plt.imshow(residues,cmap='jet')
ax1.imshow(p1_phase, cmap='bone')
ax1.axis('image')
ax1.set_title('probe 1')
ax1.plot(p1_xres, p1_yres, 'or')
# plt.show(block=False)

#fig2 = plt.figure(1)
ax2 = fig1.add_subplot(132)
# plt.imshow(residues,cmap='jet')
ax2.imshow(p2_phase, cmap='bone')
ax2.axis('image')
ax2.set_title('probe 2')
ax2.plot(p2_xres, p2_yres, 'or')
# plt.show(block=False)

#fig3 = plt.figure(1)
ax3 = fig1.add_subplot(133)
# plt.imshow(residues,cmap='jet')
ax3.imshow(p3_phase, cmap='bone')
ax3.axis('image')
ax3.set_title('probe 3')
ax3.plot(p3_xres, p3_yres, 'or')
plt.show(block=False)
#a =raw_input()

# display the probes after vortices removal
# plt.close('all')
fig2 = plt.figure(2)
ax4 = fig2.add_subplot(131)
ax4.imshow(np.angle(p1_phase_novort2), cmap='bone')
ax4.axis('image')
ax4.set_title('probe 1 with no vortices')

ax5 = fig2.add_subplot(132)
ax5.imshow(np.angle(p2_phase_novort2), cmap='bone')
ax5.axis('image')
ax5.set_title('probe 2 with no vortices')

ax6 = fig2.add_subplot(133)
ax6.imshow(np.angle(p3_phase_novort2), cmap='bone')
ax6.axis('image')
ax6.set_title('probe 3 with no vortices')
plt.show(block=False)
#a =raw_input()

if overwrite_h5:
    print("Overwritting object information in the h5 file")
    with h5py.File(filename, 'r+') as fid:
        probe_new = (fid['content/probe/S00G00/data'])
        probe_new[...] = probe_novort
