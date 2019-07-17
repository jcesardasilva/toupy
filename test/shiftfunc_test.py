import numpy as np
import matplotlib.pyplot as plt
import toupy

filenamec = '../H2int_15000h_outlet_25nm_repeatnfptomo_subtomo001_0000.cxi'
obj,probe,px = toupy.io.filesrw.read_cxi(filenamec)

shift = (200.5,500.2)
params=dict()

params['shiftmeth'] = 'linear'
S = toupy.registration.shift.ShiftFunc(**params)
shiftedobj = S(np.angle(obj),shift)

params['shiftmeth'] = 'spline'
S = toupy.registration.shift.ShiftFunc(**params)
shiftedobj = S(np.angle(obj),shift)

params['shiftmeth'] = 'sinc'
S = toupy.registration.shift.ShiftFunc(**params)
shiftedobj = S(np.angle(obj),shift)

# Display figures
plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(121)
ax1.imshow(np.angle(obj),cmap='bone',vmin=-np.pi,vmax=np.pi)
ax2 = fig1.add_subplot(122)
ax2.imshow(shiftedobj,cmap='bone',vmin=-np.pi,vmax=np.pi)
fig1.suptitle('linear')
plt.tight_layout()
fig1.subplots_adjust(top=0.88)
plt.show(block=False)

fig2 = plt.figure(2)
ax1 = fig2.add_subplot(121)
ax1.imshow(np.angle(obj),cmap='bone',vmin=-np.pi,vmax=np.pi)
ax2 = fig2.add_subplot(122)
ax2.imshow(shiftedobj,cmap='bone',vmin=-np.pi,vmax=np.pi)
fig2.suptitle('spline')
plt.tight_layout()
fig2.subplots_adjust(top=0.88)
plt.show(block=False)

fig3 = plt.figure(3)
ax1 = fig3.add_subplot(121)
ax1.imshow(np.angle(obj),cmap='bone',vmin=-np.pi,vmax=np.pi)
ax2 = fig3.add_subplot(122)
ax2.imshow(shiftedobj,cmap='bone',vmin=-np.pi,vmax=np.pi)
fig3.suptitle('sinc')
plt.tight_layout()
fig3.subplots_adjust(top=0.88)
plt.show(block=False)
