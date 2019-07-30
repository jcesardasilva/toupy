import numpy as np
import matplotlib.pyplot as plt
import toupy

Fs = 150.0  # sampling rate
Ts = 1.0 / Fs  # sampling interval
t = np.arange(0, 1, Ts)  # time vector

ff = 5  # frequency of the signal

# let's generate a sine signal
y = np.sin(2 * np.pi * ff * t)

shift = 5
params = dict()

params["shiftmeth"] = "linear"
S = toupy.registration.ShiftFunc(**params)
# ~ %timeit S(y, shift)
shiftedlin = S(y, shift)

params["shiftmeth"] = "spline"
S = toupy.registration.ShiftFunc(**params)
# ~ %timeit S(y, shift)
shiftedspl = S(y, shift)

params["shiftmeth"] = "fourier"
S = toupy.registration.ShiftFunc(**params)
# ~ %timeit S(y, shift)
shiftedfour = S(y, shift)

# Display figures
plt.close("all")
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(y, "r-")
ax1.plot(shiftedlin, "bo-")
ax1.set_title("linear")
plt.tight_layout()
fig1.subplots_adjust(top=0.88)
plt.show(block=False)

fig2 = plt.figure(2)
ax1 = fig2.add_subplot(111)
ax1.plot(y, "r-")
ax1.plot(shiftedspl, "bo-")
ax1.set_title("spline")
plt.tight_layout()
fig2.subplots_adjust(top=0.88)
plt.show(block=False)

fig3 = plt.figure(3)
ax1 = fig3.add_subplot(111)
ax1.plot(y, "r-")
ax1.plot(shiftedfour, "bo-")
ax1.set_title("fourier")
plt.tight_layout()
fig3.subplots_adjust(top=0.88)
plt.show(block=False)

a = input('Press Enter to close')
