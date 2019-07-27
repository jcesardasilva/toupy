import numpy as np
import matplotlib.pyplot as plt
import toupy

obj = np.zeros((1025,801))
nr,nc = obj.shape
obj[300:-300,200:-200]=1

cmin = None
cmax = None
shift = (150.2,-50)
params=dict()


params['shiftmeth'] = 'linear'
S = toupy.registration.ShiftFunc(**params)
%timeit S(obj,shift)
shiftedobjl = S(obj,shift)

params['shiftmeth'] = 'spline'
S = toupy.registration.ShiftFunc(**params)
%timeit S(obj,shift)
shiftedobjs = S(obj,shift)

params['shiftmeth'] = 'fourier'
S = toupy.registration.ShiftFunc(**params)
%timeit S(obj,shift,'constant')
shiftedobjf = S(obj,shift,'constant')

# Display figures
plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(121)
ax1.imshow(obj,cmap='bone',vmin=cmin,vmax=cmax)
ax2 = fig1.add_subplot(122)
ax2.imshow(shiftedobjl,cmap='bone',vmin=cmin,vmax=cmax)
fig1.suptitle('linear')
plt.tight_layout()
fig1.subplots_adjust(top=0.88)
plt.show(block=False)

fig2 = plt.figure(2)
ax1 = fig2.add_subplot(121)
ax1.imshow(obj,cmap='bone',vmin=cmin,vmax=cmax)
ax2 = fig2.add_subplot(122)
ax2.imshow(shiftedobjs,cmap='bone',vmin=cmin,vmax=cmax)
fig2.suptitle('spline')
plt.tight_layout()
fig2.subplots_adjust(top=0.88)
plt.show(block=False)

fig3 = plt.figure(3)
ax1 = fig3.add_subplot(121)
ax1.imshow(obj,cmap='bone',vmin=cmin,vmax=cmax)
ax2 = fig3.add_subplot(122)
ax2.imshow(shiftedobjf,cmap='bone',vmin=cmin,vmax=cmax)
fig3.suptitle('fourier')
plt.tight_layout()
fig3.subplots_adjust(top=0.88)
plt.show(block=False)
