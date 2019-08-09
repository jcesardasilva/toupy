#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from scipy.special import erf, erfc

# local packages
from toupy.utils import (
    model_erf,
    model_tanh,
    residuals_erf,
    residuals_tanh,
)

# filename containing 
filename = "Values.txt" 
values = np.loadtxt(filename)

# normalization 
values[:, 1] = (values[:, 1] - values[:, 1].min()) / (
    values[:, 1].max() - values[:, 1].min()
)

# correct by the pixel size
values[:, 0] = 12e-9 * values[:, 0] * 1e9


# Start the fitting

# Fitting with erf function
# ===========================

sign_of_function = np.sign(values[:, 1][0] - values[:, 1][-1])
x0_1 = np.array(
    [0.01, -0.01, values[:, 1].max(), values[:, 0].mean(), sign_of_function * 20.0],
    dtype=float,
)
popt1, pcov1, infodict1, errmsg1, ier1 = leastsq(
    residuals_erf, x0_1, full_output=1, args=(values[:, 1], values[:, 0])
)
x1_1 = popt1
FWHMerf = np.sqrt(2 * np.log(2)) * np.abs(x1_1[-1])
#FWHMerf = 2 * np.sqrt(2 * np.log(2)) * np.abs(x1_1[-1])

# display plots
plt.close("all")
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
# motorscan-factor1,counterscan,'ro',mfc='none',mec='red',mew=2)
linescan1, = ax1.plot(
    values[:, 0] - x1_1[3], values[:, 1], "ro-", mfc="none", mec="red", mew=2
)
fitlinescan1, = ax1.plot(
    values[:, 0] - x1_1[3], model_erf(values[:, 0], *x1_1), "b--", linewidth=3
)
ax1.legend(["Data", "Fit (erf)"], loc="best")
ax1.axis("tight")
ax1.tick_params(labelsize="large")
ax1.set_ylabel("$gray-level\,\, counts\, [a.u.]$", fontsize="xx-large")
ax1.set_xlabel("$Position \,[nm]$", fontsize="xx-large")
plt.show(block=False)
plt.savefig("model_erf.png", dpi=200, bbox_inches="tight")

print("=========== Fit with erf =============")
print("spot size (1/e^2 radius): {:.02f} nm".format(np.abs(x1_1[-1])))
print("FWHM: {:.02f} nm".format(FWHMerf))
print("Average position: {:.02f} nm".format(x1_1[3]))


# Fitting with tanh function
# ===========================

sign_of_function = np.sign(values[:, 1][0] - values[:, 1][-1])
x0_2 = np.array(
    [0.01, -0.01, values[:, 1].max(), values[:, 0].mean(), sign_of_function * 20.0],
    dtype=float,
)
popt2, pcov2, infodict2, errmsg2, ier2 = leastsq(
    residuals_tanh, x0_2, full_output=1, args=(values[:, 1], values[:, 0])
)
x1_2 = popt2
FWHMtanh = (
    np.arccosh(1 / np.sqrt(1 / 2.0 - x1_2[1] * np.abs(x1_2[4]) / x1_2[2]))
    * 2 * np.abs(x1_2[4])
)
FWHMtanh_back = np.arccosh(1 / np.sqrt(1 / 2.0)) * 2 * np.abs(x1_2[4])

# display plots
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
# motorscan-factor1,counterscan,'ro',mfc='none',mec='red',mew=2)
linescan2, = ax2.plot(
    values[:, 0] - x1_2[3], values[:, 1], "ro-", mfc="none", mec="red", mew=2
)
fitlinescan2, = ax2.plot(
    values[:, 0] - x1_2[3], model_tanh(values[:, 0], *x1_2), "b--", linewidth=3
)
ax2.legend(["Data", "Fit (tanh)"], loc="best")
ax2.axis("tight")
ax2.tick_params(labelsize="large")
ax2.set_ylabel("$gray-level\,\, counts\, [a.u.]$", fontsize="xx-large")
ax2.set_xlabel("$Position \,[nm]$", fontsize="xx-large")
plt.show(block=False)
plt.savefig("model_tanh.png", dpi=200, bbox_inches="tight")

print("=========== Fit with tanh =============")
print("FWHM: {:.02f} nm".format(FWHMtanh))
print("FWHM (background removed): {:.02f} nm".format(FWHMtanh_back))
print("Average position: {:.02f} nm".format(x1_2[3]))
