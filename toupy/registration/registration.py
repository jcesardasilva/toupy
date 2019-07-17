#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import time

# third party packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, interpolation
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d


# local packages
from .register_translation_fast import register_translation
from .shift import ShiftFunc
from ..tomo.iradon import backprojector
from ..tomo.radon import projector
from ..utils.funcutils import deprecated
from ..utils.array_utils import projectpoly1d, fract_hanning_pad

__all__=['compute_aligned_stack',
         'compute_aligned_sino',
         'derivatives',
         'derivatives_sino',
         'center_of_mass_stack',
         'vertical_fluctuations',
         'vertical_shift',
         'alignprojections_vertical',
         'alignprojections_horizontal',
         'cc_align',
         ]

def compute_aligned_stack(input_stack,deltastack,params):
    """
    Compute the aligned stack given the correction for object positions
    """
    # Initialize shift class
    S = ShiftFunc(shiftmeth = params['shiftmeth'])
    # array shape
    nstack = input_stack.shape[0]
    print('Using {} shift method (function {})'.format(params['shiftmeth'],S.shiftmeth.__name__))
    output_stack = np.empty_like(input_stack)
    for ii in range(nstack):
        deltashift = (deltastack[0,ii],deltastack[1,ii])
        output_stack[ii] = S(input_stack[ii],deltashift)
        print('Image {} of {}'.format(ii+1,nstack),end="\r")
    return output_stack

def compute_aligned_sino(input_sino,deltaslice,params):
    """
    Compute the aligned stack given the correction for object positions
    """
    # Initialize shift class
    S = ShiftFunc(shiftmeth = params['shiftmeth'])
    # array shape
    nprojs = input_sino.shape[1]
    print('Using {} shift method (function {})'.format(params['shiftmeth'],S.shiftmeth.__name__))
    output_sino = np.empty_like(input_sino)
    for ii in range(nprojs):
        deltashift = deltaslice[0,ii]
        output_sino[:,ii] = S(input_sino[:,ii],deltashift)
        print('Image {} of {}'.format(ii+1,nprojs),end="\r")
    return output_sino

def derivatives(input_array,shift_method='fourier'):
    """
    Calculate the derivative of an image

    Parameters
    ----------
    input_array: ndarray
        Input image to calculate the derivatives
    shift_method: str
        Name of the shift method to use. Available options:
        'sinc', 'linear'

    Returns
    -------
    diffimg : ndarray
        Derivatives of the images along the row direction
    """
    S = ShiftFunc(shiftmeth = shift_method)
    rshift = [0,0.5]
    lshift = [0,-0.5]
    diffimg = S(input_array,rshift) - S(input_array,lshift)
    # ~ if shift_method == 'phasor':
        # ~ diffimg = np.angle(S(np.exp(1j*input_array),rshift,'reflect',True)*S(np.exp(-1j*input_array),lshift,'reflect',True))
    return diffimg

def derivatives_sino(input_sino,shift_method='fourier'):
    """
    Calculate the derivative of the sinogram

    Parameters
    ----------
    input_array : ndarray
        Input sinogram to calculate the derivatives
    shift_method : str
        Name of the shift method to use. Available options:
        'fourier', 'linear', 'spline'

    Returns
    -------
    diffsino : ndarray
        Derivatives of the sinogram along the radial direction
    """
    rollsino = np.rollaxis(input_sino,1) # same as np.transpose(input_sino,1)
    rolldiff = derivatives(rollsino,shift_method)
    diffsino = np.rollaxis(rolldiff,1) # same as np.transpose(rolldiff,1)
    return diffsino

def center_of_mass_stack(input_stack,params,**kwargs):
    """
    Calculates the center of the mass for each projection in the stack and
    returns a stack of centers of mass (row, col) i.e., returns deltastack[1]
    If the array is zero, it return the center of mass at 0.
    @author: jdasilva
    """
    if not isinstance(input_stack,np.ndarray):
        input_stack = np.asarray(input_stack).copy()

    for ii in kwargs:
        if ii=='deltastack':
            deltastack=kwargs[ii]
        elif ii=='limrow':
            limrow = kwargs[ii]
        elif ii=='limcol':
            limcol = kwargs[ii]
        elif ii=='stack_type':
            stack_type = kwargs[ii]
    try:
        stack_type
    except NameError:
        stack_type='images'

    try:
        deltastack
    except NameError:
        deltastack = np.zeros((2,input_stack.shape[0]))

    # initialize shift class
    S = ShiftFunc(shiftmeth = params['shiftmeth'])

    print('Calculating center-of-mass with pixel precision')
    if stack_type =='images':
        try:
            limrow
        except NameError:
            limrow=np.array([0,input_stack.shape[1]])
        try:
            limcol
        except NameError:
            limcol=np.array([0,input_stack.shape[2]])

        # create array positions
        stack_roi = input_stack[0,limrow[0]:limrow[-1],limcol[0]:limcol[-1]].copy()
        ind_roi= np.indices(stack_roi.shape)
        # create array Xp of horizontal of positions
        ind_roi[1]-=np.floor(ind_roi[1].mean(axis=1)).reshape((ind_roi.shape[1],1)).astype('int')
        Xp = ind_roi[1].copy().astype('float')

        # create array Xp of horizontal of positions
        ind_roi[0]-=np.floor(ind_roi[0].mean(axis=0)).reshape((ind_roi.shape[2],1)).T.astype('int')
        Yp = ind_roi[0].copy().astype('float')

        # initializing the arrays
        mass_sum = np.empty(input_stack.shape[0])
        centerx = np.empty(input_stack.shape[0])
        centery = np.empty(input_stack.shape[0])

        for ii in range(input_stack.shape[0]):
            stack_aux = S(input_stack[ii],(deltastack[0,ii],deltastack[1,ii]))
            mass_sum[ii] = np.sum(stack_aux[limrow[0]:limrow[-1],limcol[0]:limcol[-1]])
            centerx[ii] = np.sum(Xp*stack_aux[limrow[0]:limrow[-1],limcol[0]:limcol[-1]])
            centery[ii] = np.sum(Yp*stack_aux[limrow[0]:limrow[-1],limcol[0]:limcol[-1]])
        centerx[np.nonzero(mass_sum)]= centerx[np.nonzero(mass_sum)]/mass_sum[np.nonzero(mass_sum)]
        centerx[np.where(mass_sum==0)] = 0
        centery = np.asarray(centery)
        centery[np.nonzero(mass_sum)]= centery[np.nonzero(mass_sum)]/mass_sum[np.nonzero(mass_sum)]
        centery[np.where(mass_sum==0)] = 0
        com=np.asarray([centerx,centery])
    elif stack_type =='mass':
        try:
            limcol
        except NameError:
            limcol=np.array([0,input_stack.shape[1]])

        # create array Xp of horizontal positions
        # create array positions
        stack_roi = input_stack[0,limcol[0]:limcol[-1]].copy()
        ind_roi= np.indices(stack_roi.shape)
        ind_roi-=np.floor(ind_roi.mean()).astype('int')
        Xp = ind_roi.copy().astype('float')

        # initializing the arrays
        mass_sum = np.empty(len(input_stack))
        centerx = np.empty(len(input_stack))

        for ii in range(len(input_stack)):
            stack_aux = S(input_stack[ii],deltastack[1,ii])
            mass_sum[ii] = np.sum(stack_aux[limcol[0]:limcol[-1]])
            centerx[ii] = np.sum(Xp*stack_aux[limcol[0]:limcol[-1]])
        centerx[np.nonzero(mass_sum)]= centerx[np.nonzero(mass_sum)]/mass_sum[np.nonzero(mass_sum)]
        centerx[np.where(mass_sum==0)] = 0
        com=np.asarray(centerx)
    return com

def vertical_fluctuations(input_stack,lims,params,**kwargs):
    """
    Calculate the vertical fluctuation functions of a stack
    """
    for ii in kwargs:
        if ii=='deltastack':
            print('Using current deltastack')
            deltastack=kwargs[ii]
    for ii in params:
        if ii=='shift_gaussian_filter':
            use_filter = params[ii]
            print('Use gaussian filter to shifts')
    try:
        deltastack
    except NameError:
        deltastack = np.zeros((2,input_stack.shape[0]))

    # Initialize shift class
    S = ShiftFunc(shiftmeth = params['shiftmeth'])
    # array shape
    _, nr, nc = input_stack.shape
    # separate the lims
    rows,cols = lims
    # get the maximum shift value
    max_vshift = int(np.ceil(np.max(np.abs(deltastack[0,:]))))+1
    if np.any((rows-max_vshift)<0) or np.any((rows+max_vshift)>nr):
        max_vshift = 1

    # initializing array
    vert_fluct = np.empty((input_stack.shape[0],rows[-1]-rows[0])) #+2*max_vshift))
    for ii in range(input_stack.shape[0]):
        print('Calculating for projection: {}'.format(ii+1),end="\r")
        proj = input_stack[ii,rows[0]-max_vshift:rows[-1]+max_vshift,cols[0]:cols[-1]]
        stack_shift = S(proj,(deltastack[0,ii],0.))
        shift_calc = stack_shift[max_vshift:-max_vshift].sum(axis=1) # the max_vshift has to be subtracted
        # to remove possible bias
        if params['bias']:
            shift_calc = projectpoly1d(shift_calc,params['maxorder'],1)
        if use_filter:
            shift_calc = gaussian_filter1d(shift_calc,params[u'shift_gaussian_sigma'])
        vert_fluct[ii] = shift_calc
    return vert_fluct

def vertical_shift(input_array,lims,vstep,params):
    """
    Calculate the vertical shift of an array and remove bias if needed
    It is used by _search_shift_direction
    """
    if not isinstance(input_array,np.ndarray):
        input_array = np.asarray(input_array).copy()
    # Initialize shift class
    S = ShiftFunc(shiftmeth = params['shiftmeth'])
    # array shape
    nr, nc = input_array.shape

    # Max vertical shift + 1. At least one for a margin. Had to take the int of vstep.
    max_vshift = params['max_vshift']+int(np.abs(vstep))#+1

    # get the maximum shift value
    rows,cols = lims
    if np.any((rows-max_vshift)<0) or np.any((rows+max_vshift)>nr):
        max_vshift = 1
    stack_shift = S(input_array[rows[0]-max_vshift:rows[-1]+max_vshift,cols[0]:cols[-1]],(vstep,0.))
    # Integration because stack_shift is 2D
    shift_calc = stack_shift[max_vshift:-max_vshift].sum(axis=1)
    # to remove possible bias
    if params['bias']:
        shift_calc = projectpoly1d(shift_calc,params['maxorder'],1)

    return shift_calc

def alignprojections_vertical(input_stack,limrow,limcol,deltastack,params):
    """
    Vertical alignment of projections using mass fluctuation approach.
    It relies on having air on both sides of the sample (non local tomography).
    It performs a local search in y, so convergence issues can be addressed by
    giving an approximate initial guess for a possible drift via deltastack
    
    Parameters
    ----------
    input_stack : ndarray
        Stack of projections
    limrow : list of ints
        Limits of window of interest in y
    limcol : list of ints
        Limits of window of interest in x
    deltastack : ndarray
        Array of initial estimates for object motion (2,n)
    Extra parameters in the dictionary params:
    params['pixtol'] : float
        Tolerance for alignment, which is also used as a search step
    params['rembias'] : bool
        True -> removal of bias and lower order terms (for y registration).
    params['maxorder'] : int
        If params['rembias']=True, specify the polynomial order of bias 
        removal (e.g. = 1 mean, = 2 linear).
    params['disp'] : int
        = 0 Display no images
        = 1 Final diagnostic images
        = 2 Diagnostic images per iteration
    params['alignx'] : bool
        True or False to activate align x using center of mass 
        (default= False, which means align y only)
    params['shiftmeth'] : str
        Shift images with sinc interpolation (default). The options are:
           'linear' - Shift images with linear interpolation (default)
           'fourier' - Fourier shift
           'spline' - Shift images with spline interpolation
    
    Returns
    -------
    deltastack : ndarray
        Object positions
    input_stack : ndarray
        Aligned stack of the projections
    """

    if not isinstance(input_stack,np.ndarray):
        input_stack = np.asarray(input_stack).copy()

    if not isinstance(params['maxit'],int):
        print('Using default number of iteration: 10')
        params['maxit']=10
    if not isinstance(limrow,np.ndarray) or not isinstance(limcol,np.ndarray):
        limrow = np.asarray(limrow)
        limcol = np.asarray(limcol)
    lims = (limrow,limcol)

    print('\n============================================')
    print('Vertical Mass fluctuation pixel alignment')

    print('Initializing the shifts arrays')
    # display one projection with limits
    plt.close('all')
    fig1 = plt.figure(num=1)#,figsize=(15,6))
    plt.clf()
    ax11 = fig1.add_subplot(111)
    im11 = ax11.imshow(input_stack[0],cmap='bone')
    ax11.set_title('Projection')
    ax11.axis('image')
    ax11.plot([limcol[0],limcol[-1]],[limrow[0],limrow[0]],'r-')
    ax11.plot([limcol[0],limcol[-1]],[limrow[-1],limrow[-1]],'r-')
    ax11.plot([limcol[0],limcol[0]],[limrow[0],limrow[-1]],'r-')
    ax11.plot([limcol[-1],limcol[-1]],[limrow[0],limrow[-1]],'r-')
    plt.show(block=False)
    plt.pause(0.01)

    if params[u'2Dgaussian_filter']:
        for ii in range(input_stack.shape[0]):
            print('Applying 2D gaussian filter projection: {}'.format(ii+1),end="\r")
            input_stack[ii] = gaussian_filter(input_stack[ii],params[u'2Dgaussian_sigma'])

    # horizontal alignement with center of mass if requested
    if params['alignx'] and count == 1:
        print('Estimating the changes in x using center-of-mass:')
        centerx=center_of_mass_stack(input_stack,params,limrow=limrow,limcol=limcol,deltastack=deltastack)[0]#[1]
        # Correction with mass center
        deltastack[1] = -centerx.round()
        deltastack[1] -= deltastack[1].mean().round()
        changex = np.abs(deltaprev[1] - deltastack[1])
    else:
        changex = 0

    print('Maximum correction of center of mass in x = {:.02f} pixels'.format(np.max(changex)))

    # first iteration only correcting for the limrow and limcol and in case deltastack is already no zero
    vert_fluct_init = vertical_fluctuations(input_stack,(limrow,limcol),params,deltastack=deltastack)
    avg_init = vert_fluct_init.mean(axis=0)
    deltastack_init = deltastack.copy()
    nr,nc = vert_fluct_init.shape # for the image display

    # Store initial states
    metric_error = [] #initialize metrics
    error_init = np.zeros(vert_fluct_init.shape[0])
    error_reg = np.zeros_like(error_init)
    for ii in range(vert_fluct_init.shape[0]):
        error_init[ii] = np.sum(np.abs(vert_fluct_init[ii]-avg_init)**2)
    print('Initial error metric for y, E = {:.02e}'.format(np.sum(error_init)))
    metric_error.append(np.sum(error_init))
    #metric_error = np.sum(error_init)

    #figures display
    if nc>nr:
        figsize = (np.round(6*nc/nr),6)
    else:
        figsize = (6,np.round(6*nr/nc))

    fig2 = plt.figure(num=2,figsize=figsize)
    plt.clf()
    ax21 = fig2.add_subplot(211)
    im21 = ax21.imshow(vert_fluct_init.T,cmap='jet',interpolation='none')
    ax21.axis('tight')
    ax21.set_title('Initial Integral in x')
    ax21.set_xlabel('Projection')
    ax21.set_ylabel('y [pixels]')
    ax22 = fig2.add_subplot(212)
    im22 = ax22.imshow(vert_fluct_init.T,cmap='jet',interpolation='none')
    ax22.axis('tight')
    ax22.set_title('Current Integral in x')
    ax22.set_xlabel('Projection')
    ax22.set_ylabel('y [pixels]')
    plt.tight_layout()
    fig2.canvas.draw()
    plt.pause(0.1)

    fig3 = plt.figure(num=3,figsize=figsize)
    plt.clf()
    ax31 = fig3.add_subplot(211)
    im31 = ax31.plot(vert_fluct_init.T)
    im31_a1 = ax31.plot(vert_fluct_init.mean(axis=0),'r',linewidth=2.5)
    im31_a2 = ax31.plot(vert_fluct_init.mean(axis=0),'--w',linewidth=1.5)
    ax31.axis('tight')
    ax31.set_title('Initial Integral in x')
    ax31.set_xlabel('Projection')
    ax31.set_ylabel('y [pixels]')
    ax32 = fig3.add_subplot(212)
    im32 = ax32.plot(vert_fluct_init.T)
    im32_a1 = ax32.plot(vert_fluct_init.mean(axis=0),'r',linewidth=2.5)
    im32_a2 = ax32.plot(vert_fluct_init.mean(axis=0),'--w',linewidth=1.5)
    ax32.axis('tight')
    ax32.set_title('Current Integral in x')
    ax32.set_xlabel('Projection')
    ax32.set_ylabel('y [pixels]')
    plt.tight_layout()
    fig3.canvas.draw()
    plt.pause(0.1)

    # Single pixel precision
    print('\n================================================')
    print('Registration of projections with pixel precision')
    print('================================================')

    plt.ion()

    # Initialize the counter
    count = 0
    while True:
        count += 1
        print('\n============================================')
        print('Iteration {}'.format(count))
        deltaprev = deltastack.copy()

        # Mass distribution registration in y
        if count ==1:
            vert_fluct = vert_fluct_init.copy()
        else:
            print('Updating the vertical fluctuations')
            vert_fluct = vertical_fluctuations(input_stack,(limrow,limcol),params,deltastack=deltastack)

        # Average the vertical fluctuation functions
        print('Calculating the average of the vertical fluctuation function')
        vert_fluct_mean = vert_fluct.mean(axis=0)

        # Search for shifts with respect to mean
        print('Search for the shifts with respect to the mean vertical fluctuations...')
        deltastack_aux, vert_fluct_temp = _search_shift_direction_stack(input_stack,lims,deltastack,vert_fluct_mean,params,subpixel=False)
        deltastack[0] = deltastack_aux[0].copy()
        deltastack[0] -= deltastack_aux[0].mean().round() # recentering

        # Error calculation
        vert_fluct_mean_temp = vert_fluct_temp.mean(axis=0)#keep temporarily the vertical fluctuations
        print('\nCalculating the error metric')
        for ii in range(vert_fluct_temp.shape[0]):
            error_reg[ii] = np.sum(np.abs(vert_fluct_temp[ii]-vert_fluct_mean_temp)**2)
        print('Final error metric for y, E = {:.04e}'.format(np.sum(error_reg)))
        metric_error.append(np.sum(error_reg))

        # Maximum changes in y
        print('Estimating the changes in y:')
        changey = np.abs(deltaprev[0] - deltastack[0])
        print('Maximum correction in y = {:.0f} pixels'.format(np.max(changey)))

        if params['disp']>1:
            # Figure 2
            im21.set_data(vert_fluct_init.T)#,interpolation='none')
            ax21.set_title('Initial Integral in x')
            im22.set_data(vert_fluct_temp.T)#,interpolation='none')
            ax22.set_title('Current Integral in x')
            plt.tight_layout()
            fig2.canvas.draw()
            plt.pause(0.1)

            # Figure 3
            fig3 = plt.figure(num=3,figsize=figsize)
            plt.clf()
            ax31 = fig3.add_subplot(211)
            im31 = plt.plot(vert_fluct_init.T)
            ax31.plot(avg_init,'r',linewidth=2.5)
            ax31.plot(avg_init,'--w',linewidth=1.5)
            ax31.axis('tight')
            ax31.set_title('Initial Integral in x')
            ax31.set_xlabel('Projection')
            ax31.set_ylabel('y [pixels]')
            ax32 = fig3.add_subplot(212)
            ax32.plot(vert_fluct_temp.T)
            ax32.plot(vert_fluct_mean_temp,'r',linewidth=2.5)
            ax32.plot(vert_fluct_mean_temp,'--w',linewidth=1.5)
            ax32.axis('tight')
            ax32.set_title('Current Integral in x')
            ax32.set_xlabel('Projection')
            ax32.set_ylabel('y [pixels]')
            plt.tight_layout()
            fig3.canvas.draw()
            plt.pause(0.1)

            # Figure 4
            fig4 = plt.figure(num=4)
            plt.clf()
            ax41 = fig4.add_subplot(111)
            im41 = ax41.plot(np.transpose(deltastack))
            ax41.axis('tight')
            ax41.set_title('Object position')
            plt.tight_layout()
            fig4.canvas.draw()
            plt.pause(0.1)

            # Figure 5
            fig5 = plt.figure(num=5)#,figsize=figsize)
            plt.clf()
            ax51 = fig5.add_subplot(111)
            ax51.plot(metric_error,'bo-')
            ax51.axis('tight')
            ax51.set_title('Error metric')
            plt.tight_layout()
            fig5.canvas.draw()
            plt.pause(0.1)

        # We then check if the error increases
        if metric_error[-1] > metric_error[-2]: # compare the last with the before last value
            print('Last iteration increased error.')
            print('Before -> {:.04e}, current -> {:.04e}'.format(metric_error[-2],metric_error[-1]))
            print('Keeping previous shifts.')
            deltastack = deltaprev.copy()
            metric_error.pop()
            break

        # We check if the changes is larger than 1
        if np.max(changey) < 1:# and isinstance(params['pixtol'],int):#max(params['pixtol'],1)):
            print('Changes are smaller than one pixel.')
            break

        # we check if the number of iteration is reached
        if count >= params['maxit']:
            print('Maximum number of iterations reached.')
            break


    print('\n================================================')
    print('Switching to subpixel precision alignement')
    print('================================================')

    print('\n================================================')
    print('Registration of projections with subpixel precision')
    print('================================================')
    # Initialize the counter
    count = 0
    while True:
        count += 1
        print('\n============================================')
        print('Iteration {}'.format(count))
        deltaprev = deltastack.copy()

        # Mass distribution registration in y
        if count !=1:
            print('Updating the vertical fluctuations')
            vert_fluct = vertical_fluctuations(input_stack,(limrow,limcol),params,deltastack=deltastack)
            # Average the vertical fluctuation functions
            print('Calculating the average of the vertical fluctuation function')
            vert_fluct_mean = vert_fluct.mean(axis=0)

        # Search for shifts with respect to mean
        print('Search for the shifts with respect to the mean vertical fluctuations...')
        deltastack_aux, vert_fluct_temp = _search_shift_direction_stack(input_stack,lims,deltastack,vert_fluct_mean,params,subpixel=True)
        deltastack[0] = deltastack_aux[0].copy()
        deltastack[0] -= deltastack_aux[0].mean().round() # recentering

        # Error calculation
        vert_fluct_mean_temp = vert_fluct_temp.mean(axis=0)#keep temporarily the vertical fluctuations
        print('\nCalculating the error metric')
        for ii in range(vert_fluct_temp.shape[0]):
            error_reg[ii] = np.sum(np.abs(vert_fluct_temp[ii]-vert_fluct_mean_temp)**2)
        print('Final error metric for y, E = {:.04e}'.format(np.sum(error_reg)))
        metric_error.append(np.sum(error_reg))

        # Maximum changes in y
        print('Estimating the changes in y:')
        changey = np.abs(deltaprev[0] - deltastack[0])
        print('Maximum correction in y = {:.04f} pixels'.format(np.max(changey)))

        if params['disp']>1:
            # Figure 2
            im21.set_data(vert_fluct_init.T)#,interpolation='none')
            ax21.set_title('Initial Integral in x')
            im22.set_data(vert_fluct_temp.T)#,interpolation='none')
            ax22.set_title('Current Integral in x')
            plt.tight_layout()
            fig2.canvas.draw()
            plt.pause(0.1)

            # Figure 3
            fig3 = plt.figure(num=3,figsize=figsize)
            plt.clf()
            ax31 = fig3.add_subplot(211)
            im31 = plt.plot(vert_fluct_init.T)
            ax31.plot(avg_init,'r',linewidth=2.5)
            ax31.plot(avg_init,'--w',linewidth=1.5)
            ax31.axis('tight')
            ax31.set_title('Initial Integral in x')
            ax31.set_xlabel('Projection')
            ax31.set_ylabel('y [pixels]')
            ax32 = fig3.add_subplot(212)
            ax32.plot(vert_fluct_temp.T)
            ax32.plot(vert_fluct_mean_temp,'r',linewidth=2.5)
            ax32.plot(vert_fluct_mean_temp,'--w',linewidth=1.5)
            ax32.axis('tight')
            ax32.set_title('Current Integral in x')
            ax32.set_xlabel('Projection')
            ax32.set_ylabel('y [pixels]')
            plt.tight_layout()
            fig3.canvas.draw()
            plt.pause(0.1)

            # Figure 4
            fig4 = plt.figure(num=4)
            plt.clf()
            ax41 = fig4.add_subplot(111)
            im41 = ax41.plot(np.transpose(deltastack))
            ax41.axis('tight')
            ax41.set_title('Object position')
            plt.tight_layout()
            fig4.canvas.draw()
            plt.pause(0.1)

            # Figure 5
            fig5 = plt.figure(num=5)#,figsize=figsize)
            plt.clf()
            ax51 = fig5.add_subplot(111)
            ax51.plot(metric_error,'bo-')
            ax51.axis('tight')
            ax51.set_title('Error metric')
            plt.tight_layout()
            fig5.canvas.draw()
            plt.pause(0.1)

        # We then check if the error increases
        if metric_error[-1] > metric_error[-2]: # index starts at 0 and count at 1
            print('Last iteration increased error.')
            print('Before -> {:.04e}, current -> {:.04e}'.format(metric_error[-2],metric_error[-1]))
            print('Keeping previous shifts.')
            deltastack = deltaprev.copy()
            metric_error.pop()
            break

        # We check if the changes is larger than 1
        if np.max(changey) < params['pixtol']:# and isinstance(params['pixtol'],int):#max(params['pixtol'],1)):
            print('Changes are smaller than {} pixel.'.format(params['pixtol']))
            break

        # we check if the number of iteration is reached
        if count >= params['maxit']:
            print('Maximum number of iterations reached.')
            break

    # Compute the shifted images
    print('Computing aligned images')
    output_stack = compute_aligned_stack(input_stack,deltastack.copy(),params)

    return deltastack,output_stack

def alignprojections_horizontal(sinogram,theta,deltaslice,params):
    """
    Function to align projections. It relies on having already aligned the
    vertical direction. The code aligns using the consistency before and
    after tomographic combination of projections.

    Parameters
    ----------
    sinogram      Sinogram derivative, the second index should be the angle
    deltaslice    Row array with initial estimates of positions
    Extra parameters in the dictionary params:
    params['pixtol'] : float
        Tolerance for alignment, which is also used as a search step
    params['disp'] : int
        = 0 Display no images
        = 1 Final diagnostic images
        = 2 Diagnostic images per iteration
    params['alignx'] : bool
        True or False to activate align x using center of mass 
        (default= False, which means align y only)
    params['shiftmeth'] : str
        Shift images with sinc interpolation (default). The options are:
           'linear' - Shift images with linear interpolation (default)
           'fourier' - Fourier shift
           'spline' - Shift images with spline interpolation
    params['circle'] : bool
        Use a circular mask to eliminate corners of the tomogram
    params['filtertomo'] : float
        Frequency cutoff for tomography filter
    params['cliplow'] : float
        Minimum value in tomogram
    params['cliphigh'] : float
        Maximum value in tomogram
    
    Returns
    -------
    deltastack : ndarray
        Corrected object positions
    alinedsinogram : ndarray
        Aligned sinogram derivatives
    """
    print('\n=====================================')
    print('\nStarting the horizontal alignment')

    # parsing of the parameters
    try:
        isinstance(params,dict)
    except NameError:
        raise SystemExit('Undefined params')
    try:
        params['circle']
    except KeyError:
        params['circle'] = True

    try:
        params[u'sinohigh']
    except KeyError:
        params[u'sinohigh'] = 0.6
    cmax = params[u'sinohigh']

    try:
        params[u'sinolow']
    except KeyError:
        params[u'sinolow'] = -0.6
    cmin = params[u'sinolow']

    try:
        params['opencl']
    except KeyError:
        params['opencl']=False

    if params['circle']:
        print('Using a circle')

    if not isinstance(params['maxit'],int):
        print('Using default number of iteration: 10')
        params['maxit'] = 10

    if not isinstance(deltaslice, np.ndarray):
        print('Deltaslice is not a numpy ndarray. Converting')
        deltaslice = np.asarray(deltaslice)

    print('Using a frequency cutoff of {}'.format(params['filtertomo']))

    try:
        params['cliplow']
        print('Low limit for tomo values = {}'.format(params['cliplow']))
    except:
        params['cliplow'] = None

    try:
        params['cliphigh']
        print('High limit for tomo values = {}'.format(params['cliphigh']))
    except:
        params['cliphigh'] = None

    # appropriate keeping of variable
    alignedsinogram = np.asarray(sinogram).copy()

    # pad sinogram of derivatives #TODO: check if we only need this for derivative (if params['derivatives']:) or not!
    padval = int(2*np.round(1/params['filtertomo']))
    sinogram = np.pad(sinogram,((padval,padval),(0,0)),'constant',constant_values=0).copy()
    N = sinogram.shape[0]

    # applying a filter to the sinogram
    filteraux = fract_hanning_pad(N,N,np.round(N*(1-params['filtertomo'])))#1- at the beginning
    filteraux = np.tile(np.fft.fftshift(filteraux[0,:]),(len(theta),1))
    sino_orig = np.real(np.fft.ifft(np.fft.fft(sinogram)*filteraux.T))

    # Shifting projection according to the initial deltaslice
    if not np.all(deltaslice==0):
        print('Shifting sinogram.')
        sinogram = compute_aligned_sino(sino_orig,deltaslice,params)
        print('Done.')
    else:
        print('Initializing deltaslice with zeros')

    # filtered and unfiltered sinogram
    unfilt_sino_orig = sinogram.copy()
    unfilt_sino = sinogram.copy()

    # initial reconstruction
    print('Computing initial tomographic slice. This takes time. Please, be patient.')
    # Filtered back projection
    print("Backprojecting")
    t0 = time.time()
    recons = backprojector(sinogram,theta=theta,
                           output_size=sinogram.shape[0],
                           filter_type=params['filtertype'],
                           derivative=params[u'derivatives'],
                           freqcutoff=params['filtertomo'])
    print('Done. Time elapsed: {} s'.format(time.time()-t0))
    print('Slice standard deviation = {:0.04e}'.format(recons.std()))
    
    # clipping gray level if needed
    recons = _clipping_tomo(recons,**params)
    if params['circle']:
        circle = _create_circle(recons)#center,N) # only need to calculate once
    #if params['circle']:
        recons = recons*circle

    # initial synthetic sinogram
    print('Computing synthetic sinogram. This takes time. Please, be patient.')
    t0 = time.time()
    sinogramcomp = FBP_projector(recons,theta,P,**params)
    print('Done. Time elapsed: {} s'.format(time.time()-t0))

    # store initial error metric
    metric_error = []
    print('Store initial error metric')
    errorinit = _sino_error_metric(sinogram,sinogramcomp,params)
    sumerrorinit = np.sum(errorinit)
    print('Initial error metric, E= {}'.format(sumerrorinit))
    metric_error.append(sumerrorinit)

    # Preparing the canvas for the figures
    plt.close('all')
    fig1 = plt.figure(num=1)
    plt.clf()
    ax11 = fig1.add_subplot(111)
    im11 = ax11.imshow(recons,cmap='jet')
    ax11.axis('image')
    ax11.set_title('Initial slice')
    ax11.set_xlabel('x [pixels]')
    ax11.set_ylabel('y [pixels]')
    fig1.canvas.draw()
    plt.pause(0.001)

    fig2 = plt.figure(num=2)
    plt.clf()
    ax21 = fig2.add_subplot(211)
    im21 = ax21.imshow(sino_orig,cmap='bone',vmin=cmin,vmax=cmax)
    ax21.axis('tight')
    ax21.set_title('Initial sinogram')
    ax21.set_xlabel('Projection')
    ax21.set_ylabel('x [pixels]')

    ax22 = fig2.add_subplot(212)
    im22 = ax22.imshow(sinogram,cmap='bone',vmin=cmin,vmax=cmax)
    ax22.axis('tight')
    ax22.set_title('Current sinogram')
    ax22.set_xlabel('Projection')
    ax22.set_ylabel('x [pixels]')
    plt.tight_layout()
    fig2.canvas.draw()
    plt.pause(0.001)

    fig3 = plt.figure(num=3)
    plt.clf()
    ax31 = fig3.add_subplot(111)
    im31 = ax31.imshow(sinogramcomp,cmap='bone',vmin=cmin,vmax=cmax)
    ax31.axis('tight')
    ax31.set_title('Synthetic sinogram')
    ax31.set_xlabel('Projection')
    ax31.set_ylabel('x [pixels]')
    plt.tight_layout()
    fig3.canvas.draw()
    plt.pause(0.001)

    #===========================#
    # single pixel registration #
    #===========================#

    print('\nRegistration of projections - Single pixel precision')
    # main loop
    plt.ion()
    count = 0
    while True:
        count += 1
        print('\n=====================================')
        print('Iteration {}'.format(count))
        it0 = time.time()
        print('Keeping previous sinogram before iteration')
        deltaprev = deltaslice.copy() # keep deltaprev in case the iteration does not decrease the error
        sinoprev = sinogram.copy()

        # Compute tomogram with current sinogram
        print('Computing tomographic slice. This takes time. Please, be patient.')
        t0 = time.time()
        recons = backprojector(sinogram,theta=theta,
                           output_size=sinogram.shape[0],
                           filter_type=params['filtertype'],
                           derivative=params['derivatives'],
                           freqcutoff=params['filtertomo'])
        print('Done. Time elapsed: {} s'.format(time.time()-t0))
        print('Slice standard deviation = {:0.04e}'.format(recons.std()))

        # clipping gray level if needed
        recons = _clipping_tomo(recons,**params)
        if params['circle']:
            recons = recons*circle

        # Show slice images
        im11.set_data(recons)#,cmap='jet')
        ax11.set_title('Slice - iteration {}'.format(count))
        fig1.canvas.draw()
        plt.pause(0.001)

        # Compute synthetic sinogram
        print('Computing synthetic sinogram. This takes time. Please, be patient.')
        sinogramcomp = FBP_projector(recons,theta,**params)
        #sinogramcomp = np.flipud(FBP_projector(recons,theta,**params)) # fliplr only for fluo data

        # Start searching for shift relative to synthetic sinogram
        sinotempreg,deltaslice = _search_sino_shifts(sinogram,sinogramcomp,deltaslice,subpixel=False,**params)

        # calculate the error:
        errorxreg = _sino_error_metric(sinotempreg,sinogramcomp,params)
        sumerrorxreg = errorxreg.sum()
        print('Final error metric for x, E = {}'.format(sumerrorxreg))
        metric_error.append(sumerrorxreg)

        #Evaluate if pixel tolerance is already met
        print('Estimating the changes in x:')
        changex = np.abs(deltaprev - deltaslice)
        print('Maximum correction in x = {} pixels'.format(np.max(changex)))

        # updating sinogram
        sinogram = sinotempreg.copy()

        print('Elapsed time = {} s'.format(time.time()-it0))

        if params['disp']>1:
            im21.set_data(sino_orig)
            ax21.set_title('Initial sinogram')
            im22.set_data(sinogram)
            ax22.set_title('Current sinogram')
            plt.tight_layout()
            fig2.canvas.draw()
            plt.pause(0.001)

            im31.set_data(sinogramcomp)
            plt.tight_layout()
            fig3.canvas.draw()
            plt.pause(0.001)

            fig4 = plt.figure(num=4)
            plt.clf()
            ax41 = fig4.add_subplot(111)
            ax41.plot(deltaslice.T)
            ax41.axis('tight')
            ax41.set_title('Object position')
            plt.tight_layout()
            fig4.canvas.draw()
            plt.pause(0.001)

            fig5 = plt.figure(num=5)
            plt.clf()
            ax51 = fig5.add_subplot(111)
            ax51.plot(metric_error,'bo-')
            ax51.axis('tight')
            ax51.set_title('Error metric')
            plt.tight_layout()
            fig5.canvas.draw()
            plt.pause(0.001)

        if np.max(changex) < 1:
            print('\nChanges are smaller than one pixel')
            break

        if count >= params['maxit']:
            print('\nMaximum number of iterations exceeded, increase maxit')
            break

        if metric_error[-1] > metric_error[-2]:
            print('\nLast iteration increases error. Keeping previous positions')
            print('Before -> {}, current -> {}'.format(metric_error[-2],metric_error[-1]))
            print('Keeping previous shifts.')
            deltaslice = deltaprev.copy() # return deltaslice one step before
            sinogram = sinoprev.copy() # return to previous sinogram
            metric_error.pop() # remove the last value from the metric_error
            count -=1
            break

    # Sinogram alignment with subpixel precision
    if not isinstance(params['pixtol'],int):
        print('\nSwitch to supixel registration')
        print('\n=====================================')

        print('Registration of the projection with subpixel precision')

        plt.ion()
        #### Uses 'count', 'metric_error' and 'errorxreg' from the alignement with single pixel precision
        countpix = count
        while True: #dosubpix == 1,
            count += 1
            print('\n=====================================')
            print('Iteration {}'.format(count-countpix))
            it0 = time.time()
            print('Keeping previous sinogram before iteration')
            deltaprev = deltaslice.copy()
            sinoprev = sinogram.copy()

            # Compute tomogram with current sinogram
            print('Computing tomographic slice. This takes time. Please, be patient.')
            t0 = time.time()
            recons = backprojector(sinogram,theta=theta,
                           output_size=sinogram.shape[0],
                           filter_type=params['filtertype'],
                           derivative=params['derivatives'],
                           freqcutoff=params['filtertomo'])
            print('Done. Time elapsed: {} s'.format(time.time()-t0))
            print('Slice standard deviation = {:0.04e}'.format(recons.std()))

            # clipping gray level if needed
            recons = _clipping_tomo(recons,**params)
            if params['circle']:
                recons = recons*circle

            # Show slice images
            im11.set_data(recons)
            ax11.set_title('Slice - iteration {}'.format(count))
            fig1.canvas.draw()
            plt.pause(0.001)

            # Compute synthetic sinogram
            print('Computing synthetic sinogram. This takes time. Please, be patient.')
            sinogramcomp = FBP_projector(recons,theta,P,**params)
            #sinogramcomp = np.flipud(radon(recons,theta)) # fliplr only for fluo data

            # Start searching for shift relative to synthetic sinogram
            sinotempreg,deltaslice = _search_sino_shifts(sinogram,sinogramcomp,deltaslice,subpixel=True,**params)

            # calculate the error:
            errorxreg = _sino_error_metric(sinotempreg,sinogramcomp,params)
            sumerrorxreg = errorxreg.sum()
            print('Final error metric for x, E = {}'.format(sumerrorxreg))
            metric_error.append(sumerrorxreg)

            #Evaluate if pixel tolerance is already met
            print('Estimating the changes in x:')
            changex = np.abs(deltaprev - deltaslice)
            print('Maximum correction in x = {:0.03f} pixels'.format(np.max(changex)))

            # updating the sinogram
            sinogram = sinotempreg.copy()

            print('Elapsed time = {} s'.format(time.time()-it0))

            if params['disp']>1:
                im21.set_data(sino_orig)
                ax21.set_title('Initial sinogram')
                im22.set_data(sinogram)
                ax22.set_title('Current sinogram')
                plt.tight_layout()
                fig2.canvas.draw()
                plt.pause(0.001)

                im31.set_data(sinogramcomp)
                plt.tight_layout()
                fig3.canvas.draw()
                plt.pause(0.001)

                fig4 = plt.figure(num=4)
                plt.clf()
                ax41 = fig4.add_subplot(111)
                ax41.plot(deltaslice.T)
                ax41.axis('tight')
                ax41.set_title('Object position')
                plt.tight_layout()
                fig4.canvas.draw()
                plt.pause(0.001)

                fig5 = plt.figure(num=5)
                plt.clf()
                ax51 = fig5.add_subplot(111)
                ax51.plot(metric_error,'bo-')
                ax51.axis('tight')
                ax51.set_title('Error metric')
                plt.tight_layout()
                fig5.canvas.draw()
                plt.pause(0.001)

            if (np.max(changex) < params['pixtol']):
                print('\nChanges are smaller than {} pixel'.format(params['pixtol']))
                break

            if count-countpix >= params['maxit']:
                print('\nMaximum number of iterations exceeded, increase maxit')
                break

            if metric_error[-1] > metric_error[-2]:
                print('\nLast iteration increases error. Keeping previous positions')
                print('Before -> {}, current -> {}'.format(metric_error[-2],metric_error[-1]))
                print('Keeping previous shifts.')
                deltaslice = deltaprev.copy() # return deltaslice one step before
                sinogram = sinoprev.copy()
                metric_error.pop() # remove the last value from the metric_error
                #count -=1
                break

    if params['disp']>1:
        print('Calculating one slice for display')
        p0 = time.time()
        recons = backprojector(sinogram,theta=theta,
                           output_size=sinogram.shape[0],
                           filter_type=params['filtertype'],
                           derivative=params['derivatives'],
                           freqcutoff=params['filtertomo'])
        # clipping gray level if needed
        recons = _clipping_tomo(recons,**params)
        if params['circle']:
            recons = recons*circle
        print('Done. Time elapsed: {} s'.format(time.time()-p0))

        fig = plt.figure(num=10)
        plt.clf()
        ax1 = fig.add_subplot(111)
        ax1.imshow(recons,cmap='bone')
        ax1.axis('image')
        ax1.set_title('Aligned tomographic slice')
        ax1.set_xlabel('x [pixels]')
        ax1.set_ylabel('y [pixels]')
        plt.show(block=False)
        plt.pause(0.01)

    # Compute the shifted images
    if params[u'apply_alignement']:
        print('\nComputing aligned images')
        alignedsinogram = compute_aligned_sino(alignedsinogram,deltaslice,params)

    return deltaslice, alignedsinogram

def _clipping_tomo(recons,**params):
    """
    Clip gray level of tomographic images
    """
    if params['cliplow'] is not None:
        recons = recons*(recons>=params['cliplow'])+params['cliplow']*(recons<params['cliplow'])
    if params['cliphigh'] is not None:
        recons = recons*(recons<=params['cliphigh'])+params['cliphigh']*(recons>params['cliphigh'])
        recons = recons - params['cliphigh']
    return recons

def _create_circle(inputimg):
    """
    Create circle with apodized edges
    """
    bordercrop = 10
    nr,nc = inputimg.shape
    Y,X = np.indices((nr,nc))
    Y -= np.round(nr/2).astype(int)
    X -= np.round(nc/2).astype(int)
    R = np.sqrt(self.X**2+self.Y**2)
    Rmax = np.round(np.max(R.shape)/2.)
    maskout = R < Rmax
    t = maskout*(1-np.cos(np.pi*(R-Rmax-2*bordercrop)/bordercrop))/2.
    t[np.where(R < (Rmax - bordercrop))]=1
    return t

def _search_shift_direction_stack(input_stack,lims,input_delta,avg_vert_fluct,params,**kwargs):
    """
    Search for the shifts directions
    @author: jdasilva
    """
    #lims = (limrow,limcol)
    shift_params=params.copy()
    if isinstance(params['pixtol'],int) or kwargs['subpixel']==False:
        shift_params['pixtol'] = 1
        shift_params['interpmeth'] = 'pseudo_linear'
    elif not isinstance(params['pixtol'],int) or kwargs['subpixel']==True:
        pixtol = params['pixtol']
        shift_params['interpmeth']=params['interpmeth']

    # separate the lims
    rows,cols = lims
    # get the maximum shift value
    _, nr, nc = input_stack.shape
    max_vshift = int(np.ceil(np.max(np.abs(input_delta[0,:]))))+1 # plus 1 for a margin
    if np.any((rows-max_vshift)<0) or np.any((rows+max_vshift)>nr):
        max_vshift = 1 # at least one for a margin
    shift_params['max_vshift'] = max_vshift

    # initializing array
    vert_fluct_stack = np.empty((input_stack.shape[0],rows[-1]-rows[0]))
    output_deltastack = np.empty_like(input_delta)#np.zeros_like(input_delta)

    if not isinstance(input_stack,np.ndarray):
        input_stack = np.asarray(input_stack).copy()

    for ii in range(input_stack.shape[0]):
        print('Searching the shifts for projection: {}'.format(ii+1),end="\r")
        shift_delta = input_delta[0,ii]
        output_deltastack[0,ii], vert_fluct_stack[ii]=_search_shift_direction(input_stack[ii], \
                                                    lims,shift_delta,avg_vert_fluct,shift_params)

    return output_deltastack, vert_fluct_stack

def _search_shift_direction(input_array,lims,shift_delta,avg_vert_fluct,shift_params):
    """
    Search for the shifts directions
    It is used by _search_shift_direction_stack
    @author: jdasilva
    """
    # Search for shifts with respect to mean
    dir_shift=dict() # dictionary shift directions
    shifts=dict() # dictionary shifts arrays

    # pixel tolerance
    pixtol = shift_params['pixtol']

    # compute current shift error
    shifts['current'] = vertical_shift(input_array,lims,shift_delta-0,shift_params)
    # compute shift forward error
    shifts['forward'] = vertical_shift(input_array,lims,shift_delta+pixtol,shift_params)
    # compute shift backward error
    shifts['backward'] = vertical_shift(input_array,lims,shift_delta-pixtol,shift_params)

    # directional shift error calculation
    dir_shift['current']=np.sum(np.abs(shifts['current']-avg_vert_fluct)**2)
    dir_shift['forward']=np.sum(np.abs(shifts['forward']-avg_vert_fluct)**2)
    dir_shift['backward']=np.sum(np.abs(shifts['backward']-avg_vert_fluct)**2)

    # sort the dict dir_shift by value
    sort_error = sorted(dir_shift.items(), key=lambda x: x[1])
    # get the smallest shift error, which is the first in sort_error dict
    min_error = sort_error[0][0]
    # calculate the increment to be shifted
    if min_error == u'current':
        dir_inc = 0
    elif min_error == u'backward':
        dir_inc = -1*pixtol
    elif min_error == u'forward':
        dir_inc = 1*pixtol
    # update shift_delta
    shift_delta += dir_inc

    # keep shifting in the direction that minimizes errors.
    shift = shift_delta.copy() # will return this value if dir_inc = 0
    if dir_inc !=0:
        shift += dir_inc
        while True:
            # shift the stack once more in the same direction
            shifted_stack = vertical_shift(input_array,lims,shift,shift_params)
            nexterror = np.sum(np.abs(shifted_stack-avg_vert_fluct)**2)
            if nexterror < dir_shift['current']: #if error is minimized
                dir_shift['current'] = nexterror
                shift += dir_inc
            else:
                shift -= dir_inc # subtract once dir_inc in case of no sucess in the previous iteraction
                break
    else:
        shifted_stack = shifts['current']
    return shift, shifted_stack

def _search_sino_shifts(sinogram,sinogramcomp,deltaslice,subpixel=False,**params):
    """
    Wrapper to search for sinogram shifts
    """
    # initializing temporary sinogram and error function
    sinotempreg = sinogram.copy()#np.zeros_like(sinogram)
    errorxreg = np.zeros(sinogram.shape[1])
    # Initialize shift class
    S = ShiftFunc(shiftmeth = params['shiftmeth'])
    if subpixel:
        pixshift = params['pixtol']
    else:
        pixshift = 1
    for ii in range(sinogram.shape[1]):
        shifts = dict() # dictionary shifts arrays
        dir_shift = dict() # dictionary shifts direction

        # looking both ways
        shifts['current'] = S(sinogram[:,ii],0) # compute current shift error
        shifts['forwards'] = S(sinogram[:,ii],+1*pixshift) # compute shift forward error
        shifts['backwards'] = S(sinogram[:,ii],-1*pixshift) # compute shift backward error

        # directional shift error calculation
        dir_shift['current'] = np.sum(np.abs(shifts['current']-sinogramcomp[:,ii])**2)
        dir_shift['forward'] = np.sum(np.abs(shifts['forwards']-sinogramcomp[:,ii])**2)
        dir_shift['backward'] = np.sum(np.abs(shifts['backwards']-sinogramcomp[:,ii])**2)

        # sort the dict dir_shift by value
        sort_error = sorted(dir_shift.items(), key=lambda x: x[1])
        # get the smallest shift error, which is the first in sort_error dict
        min_error = sort_error[0][0]
        # calculate the increment to be shifted
        if min_error == u'current':
            sinotempreg[:,ii] = shifts['current'].copy()
            errorxreg[ii] = dir_shift['current']
            dir_inc = 0
            #continue
        else:
            if min_error == u'backward':
                dir_inc = -1*pixshift
            elif min_error == u'forward':
                dir_inc = 1*pixshift
            # update shift delta
            shift = dir_inc

            # keep shifting in the direction that minimizes errors.
            shift += dir_inc
            while True:
                # shift the sino according to shift
                shifted_sino = S(sinogram[:,ii],shift)
                nexterror = np.sum(np.abs(shifted_sino - sinogramcomp[:,ii])**2)
                if nexterror < dir_shift['current']: #if error is minimized
                    dir_shift['current'] = nexterror
                    shift += dir_inc # shift the sino once more in the same direction
                else:
                    shift -= dir_inc # subtract once dir_inc in case of no sucess in the previous iteraction
                    #errorxreg[ii] = dir_shift['current'].copy()#currenterror
                    break
            deltaslice[0,ii] += shift # update deltaslice
            sinotempreg[:,ii] = S(sinogram[:,ii],shift)#shifted_sino.copy()

    return sinotempreg, deltaslice

def _sino_error_metric(sinogramexp,sinogramcomp,params):
    """
    Estimate the error metric between the experimental sinogram and
    the synthetic one.
    @author: jdasilva
    """
    errorxreg = np.zeros(sinogramexp.shape[1])
    for ii in range(sinogramexp.shape[1]):
        errorxreg[ii] = np.sum(np.abs(sinogramexp[:,ii]-sinogramcomp[:,ii])**2)
    return errorxreg

@deprecated
def cc_align(input_stack,limrow,limcol,params):
    """
    Cross-correlation alignment (DEPRECATED)
    FIXME: IT IS NOT WORKING PROPERLY
    """
    shift_values = np.empty((len(input_stack),2))
    # The cross-correlation compares to the first projections, which does not move
    shift_values[0] = np.array([0,0])

    for ii in range(1,len(input_stack)):
        print("\nCalculating the subpixel image registration...")
        print("Projection: {}".format(ii-1))
        image1 = input_stack[ii-1,limrow[0]:limrow[-1],limcol[0]:limcol[-1]]
        print("Projection: {}".format(ii))
        image2 = input_stack[ii,limrow[0]:limrow[-1],limcol[0]:limcol[-1]]
        start = time.time()
        if params['gaussian_filter']:
            image1 = gaussian_filter(image1,params['gaussian_sigma'])
            image2 = gaussian_filter(image2,params['gaussian_sigma'])
        shift, error, diffphase = register_translation(image1, image2,100)
        shift_values[ii] = shift
        print(diffphase)
        end = time.time()
        print("Time elapsed: {} s".format(end-start))
        print("Detected subpixel offset [y,x]: [{}, {}]".format(shift[0],shift[1]))

    shift_vert_aux = np.array(shift_values)[:,0]
    shift_hor_aux = np.array(shift_values)[:,1]
    # Cumulative sum of the shifts minus the average
    shift_vert = np.cumsum(shift_vert_aux - shift_vert_aux.mean())
    shift_hor = np.cumsum(shift_hor_aux - shift_hor_aux.mean())

    # smoothing the shifts is needed
    if params['smooth_shifts'] is not None:
        shift_vert = snf.gaussian_filter1d(shift_vert,params['smooth_shifts'])
        shift_hor = snf.gaussian_filter1d(shift_hor,params['smooth_shifts'])

    # display shifts
    plt.close('all')
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(211)
    ax1.plot(np.array(shift_vert),'ro-')
    ax1.set_title('Vertical shifts')
    ax2 = fig1.add_subplot(212)
    ax2.plot(np.array(shift_hor),'ro-')
    ax2.set_title('Horizontal shifts')
    plt.show()

    #updating the deltastack
    deltastack = np.zeros((2,input_stack.shape[0]))
    deltastack[0]=shift_vert
    deltastack[1]=shift_hor

    ## Compute the shifted images
    #print('Computing aligned images')
    #if not params['expshift']:
        #output_stack = compute_aligned_stack(input_stack,deltastack.copy(),params)
    #else:
        #print('Computing aligned images in phase space')
        #output_stack = np.angle(compute_aligned_stack(np.exp(1j*input_stack),deltastack.copy(),params))

    #return deltastack,output_stack

    plt.close('all')
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)#(ncols=1, figsize=(14, 6))
    im1=ax1.imshow(stack_unwrap[1,limrow[0]:limrow[-1],limcol[0]:limcol[-1]],interpolation='none',cmap='bone')
    ax1.set_axis_off()
    ax1.set_title('Offset corrected image2')

    #offset_stack_unwrap = np.empty_like(stack_unwrap[:,80:-80,80:-80])
    #aligned = np.empty_like(stack_unwrap[:,80:-80,80:-80])
    aligned = compute_aligned_stack(input_stack,deltastack.copy(),params)
    plt.ion()
    for ii in range(0,len(stack_unwrap)):
        #img = stack_unwrap[ii,80:-80,80:-80]
        shift = np.array([shift_vert[ii],shift_hor[ii]])
        print(shift)
        print("\nCorrecting the shift of projection {} by using subpixel precision.".format(ii))
        #offset_stack_unwrap[ii] = np.fft.ifftn(fourier_shift(np.fft.fftn(img),shift))#
        #aligned[ii] = np.fft.ifftn(fourier_shift(np.fft.fftn(img),shift))#
        #im1.set_data(offset_stack_unwrap[ii])
        im1.set_data(aligned[ii])
        ax1.set_title(u'Projection {}'.format(ii))
        fig1.canvas.draw()
        plt.pause(0.001)
    plt.ioff()

    # Display the images
    fig, (ax1, ax2, ax3) = plt.subplots(num=3,ncols=3, figsize=(14,6))
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
    return deltastack, aligned
