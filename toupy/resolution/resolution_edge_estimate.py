import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from scipy.special import erf,erfc

def model1(t,*coeffs):
    """
    Model for the erf fitting
    
    P0 + P1*t + (P2/2)*(1-erf(sqrt(2)*(x-P3)/(P4)))
    where:
    coeffs[0] is P0 (noise)
    coeffs[1] is P1 (linear term)
    coeffs[2] is P2 (Maximum amplitude)
    coeffs[3] is P3 (center)
    coeffs[4] is P4 (width)
    
    """
    return coeffs[0]+coeffs[1]*t + (coeffs[2]/2.)*( 1 + erf( np.sqrt(2)*(t - coeffs[3])/(coeffs[4]) ) )
    #~ return coeffs[0]+(coeffs[2]/2.)*( 1 + erf( np.sqrt(2)*(t - coeffs[3])/(coeffs[4]) ) )

def model2(t,*coeffs):
    """
    Model for the erf fitting
    
    P0 + P1*t + (P2/2)*(1-tanh(sqrt(2)*(x-P3)/P4))
    where:
    coeffs[0] is P0 (noise)
    coeffs[1] is P1 (linear term)
    coeffs[2] is P2 (Maximum amplitude)
    coeffs[3] is P3 (center)
    coeffs[4] is P4 (width)
    
    """
    return coeffs[0]+coeffs[1]*t + (coeffs[2]/2.)*( 1 - np.tanh( (t - coeffs[3])/coeffs[4] ) )
    #~ return coeffs[0]+ (coeffs[2]/2.)*( 1 - np.tanh( (t - coeffs[3])/coeffs[4] ) )

def residuals1(coeffs, y, t):
    """
    Residuals for the least-squares optimization
    coeffs as the ones for the function models
    y = the data
    t = coordinates
    """
    return (y-model1(t,*coeffs))

def residuals2(coeffs, y, t):
    """
    Residuals for the least-squares optimization
    coeffs as the ones for the function models
    y = the data
    t = coordinates
    """
    return (y-model2(t,*coeffs))

np.loadtxt('Values.txt')
values = np.loadtxt('Values.txt')
values = values[11:-6,:]
#~ values = values[:-6,:]
values[:,1] = (values[:,1]-values[:,1].min())/(values[:,1].max()-values[:,1].min())
values[:,0]=12e-9*values[:,0]*1e9 # correct by the pixel size

# model 1
sign_of_function = np.sign(values[:,1][0]-values[:,1][-1])
x0_1 = np.array([0.01,-0.01,values[:,1].max(),values[:,0].mean(),sign_of_function*20.],dtype=float)
popt1, pcov1, infodict1, errmsg1, ier1 = leastsq(residuals1,x0_1, full_output=1, args=(values[:,1],values[:,0]))
x1_1 = popt1

plt.close('all')
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
linescan1, = ax1.plot(values[:,0]-x1_1[3],values[:,1],'ro-',mfc='none',mec='red',mew=2) #motorscan-factor1,counterscan,'ro',mfc='none',mec='red',mew=2)
fitlinescan1, = ax1.plot(values[:,0]-x1_1[3],model1(values[:,0],*x1_1),'b--',linewidth=3)
ax1.legend(['Data','Fit (erf)'],loc='best')
ax1.axis('tight')
ax1.tick_params(labelsize='large')
ax1.set_ylabel('$gray-level\,\, counts\, [a.u.]$',fontsize='xx-large')
ax1.set_xlabel('$Position \,[nm]$',fontsize='xx-large')
plt.show(block=False)
plt.savefig('model1.png',dpi=200,bbox_inches='tight')

print('=========== Fit with erf =============')
#print('FWHM: {:.02f} nm'.format(2*np.sqrt(2*np.log(2))*np.abs(x1_1[-1])*1e9))
print('spot size (1/e^2 radius): {:.02f} nm'.format(np.abs(x1_1[-1])))#*1e9))
print('FWHM: {:.02f} nm'.format(np.sqrt(2*np.log(2))*np.abs(x1_1[-1])))#*1e9))
print('Average position: {:.02f} nm'.format(x1_1[3]))#*1e6))

# model 2
sign_of_function = np.sign(values[:,1][0]-values[:,1][-1])
x0_2 = np.array([0.01,-0.01,values[:,1].max(),values[:,0].mean(),sign_of_function*20.],dtype=float)
popt2, pcov2, infodict2, errmsg2, ier2 = leastsq(residuals2,x0_2, full_output=1, args=(values[:,1],values[:,0]))
x1_2 = popt2

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
linescan2, = ax2.plot(values[:,0]-x1_2[3],values[:,1],'ro-',mfc='none',mec='red',mew=2) #motorscan-factor1,counterscan,'ro',mfc='none',mec='red',mew=2)
fitlinescan2, = ax2.plot(values[:,0]-x1_2[3],model2(values[:,0],*x1_2),'b--',linewidth=3)
ax2.legend(['Data','Fit (tanh)'],loc='best')
ax2.axis('tight')
ax2.tick_params(labelsize='large')
ax2.set_ylabel('$gray-level\,\, counts\, [a.u.]$',fontsize='xx-large')
ax2.set_xlabel('$Position \,[nm]$',fontsize='xx-large')
plt.show(block=False)
plt.savefig('model2.png',dpi=200,bbox_inches='tight')

print('=========== Fit with tanh =============')
print('FWHM: {:.02f} nm'.format(np.arccosh(1/np.sqrt(1/2.-x1_2[1]*np.abs(x1_2[4])/x1_2[2]))*2*np.abs(x1_2[4])))
print('FWHM (background removed): {:.02f} nm'.format(np.arccosh(1/np.sqrt(1/2.))*2*np.abs(x1_2[4])))
print('Average position: {:.02f} nm'.format(x1_2[3]))
