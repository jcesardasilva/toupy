#!/usr/bin/env python
# -*- coding: utf-8 -*-

#third party packages
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import numpy as np

__all__=['autoscale_y',
         'RegisterPlot',
         'show_projections',
         'show_linearphase']

def autoscale_y(ax,margin=0.1):
    """
    This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax : a matplotlib axes object
    margin : the fraction of the total height of the y-data to pad the upper and lower ylims
    """

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

def _plotdelimiters(ax, limrow, limcol):
    """
    Create ROI limits in image

    Parameters
    ----------
    ax : Matplotlib object
        axes
    limrow : list of ints
        Limits of rows in the format [begining, end]
    limcol : list of ints
        Limits of cols in the format [begining, end]
    """
    ax.plot([limcol[0],limcol[-1]],[limrow[0],limrow[0]],'r-')
    ax.plot([limcol[0],limcol[-1]],[limrow[-1],limrow[-1]],'r-')
    ax.plot([limcol[0],limcol[0]],[limrow[0],limrow[-1]],'r-')
    ax.plot([limcol[-1],limcol[-1]],[limrow[0],limrow[-1]],'r-')
    return ax

def _createcanvashorizontal(recons, sinoorig, sinocurr, sinocomp,
                            deltaslice, metric_error,**params):
    """
    Create canvas for the plots during horizontal alignement
    """
    slicenum = params['slicenum']
    cmax = params['sinohigh']
    cmin = params['sinolow']

    # Display one reconstructed slice
    fig1 = plt.figure(num=1)
    plt.clf()
    ax11 = fig1.add_subplot(111)
    im11 = ax11.imshow(recons,cmap='jet')
    ax11.axis('image')
    ax11.set_title('Initial slice number: {}'.format(slicenum))
    ax11.set_xlabel('x [pixels]')
    ax11.set_ylabel('y [pixels]')
    fig1.tight_layout()
    fig1.show()

    # Display initial, current and synthetic sinograms
    fig2 = plt.figure(num=2,figsize=(6,10))
    plt.clf()
    ax21 = fig2.add_subplot(311)
    im21 = ax21.imshow(sinoorig,cmap='bone',vmin=cmin,vmax=cmax)
    ax21.axis('tight')
    ax21.set_title('Initial sinogram')
    ax21.set_xlabel('Projection')
    ax21.set_ylabel('x [pixels]')
    ax22 = fig2.add_subplot(312)
    im22 = ax22.imshow(sinocurr,cmap='bone',vmin=cmin,vmax=cmax)
    ax22.axis('tight')
    ax22.set_title('Current sinogram')
    ax22.set_xlabel('Projection')
    ax22.set_ylabel('x [pixels]')
    ax23 = fig2.add_subplot(313)
    im23 = ax23.imshow(sinocomp,cmap='bone',vmin=cmin,vmax=cmax)
    ax23.axis('tight')
    ax23.set_title('Synthetic sinogram')
    ax23.set_xlabel('Projection')
    ax23.set_ylabel('x [pixels]')
    fig2.tight_layout()
    fig2.show()

    # Display deltaslice and metric_error
    fig3 = plt.figure(num=3)
    plt.clf()
    ax31 = fig3.add_subplot(211)
    im31 = ax31.plot(deltaslice)
    ax31.axis('tight')
    ax31.set_title('Object position')
    ax32 = fig3.add_subplot(212)
    im32 = ax32.plot(metric_error,'bo-')
    ax32.axis('tight')
    ax32.set_title('Error metric')
    fig3.tight_layout()
    fig3.show()

    plt.pause(0.001)

    fig_array = [fig1,fig2,fig3]
    im_array = [im11,im21,im22,im23,im31,im32]
    ax_array = [ax11,ax21,ax22,ax23,ax31,ax32]

    return(fig_array, im_array,ax_array)

def _createcanvasvertical(proj, lims, vertfluctinit, vertfluctcurr,
                           deltastack, metric_error, **params):
    """
    Create canvas for the plots during vertical alignement
    """
    limrow,limcol = lims

    #figures display
    nr,nc = vertfluctinit.shape # for the image display
    if nc > nr: figsize = (np.round(6*nc/nr),6)
    else: figsize = (6,np.round(6*nr/nc))

    # display one projection with limits
    fig1 = plt.figure(num=1)
    plt.clf()
    ax11 = fig1.add_subplot(111)
    im11 = ax11.imshow(proj,cmap='bone')
    ax11.set_title('Projection')
    ax11.axis('image')
    ax11 = _plotdelimiters(ax11, limrow, limcol)
    fig1.tight_layout()
    fig1.show()

    # display vertical fluctuations as 2D images
    fig2 = plt.figure(num=2,figsize=figsize)
    plt.clf()
    ax21 = fig2.add_subplot(211)
    im21 = ax21.imshow(vertfluctinit,cmap='jet',interpolation='none')
    ax21.axis('tight')
    ax21.set_title('Initial Integral in x')
    ax21.set_xlabel('Projection')
    ax21.set_ylabel('y [pixels]')
    ax22 = fig2.add_subplot(212)
    im22 = ax22.imshow(vertfluctcurr,cmap='jet',interpolation='none')
    ax22.axis('tight')
    ax22.set_title('Current Integral in x')
    ax22.set_xlabel('Projection')
    ax22.set_ylabel('y [pixels]')
    fig2.tight_layout()
    fig2.show()

    # display vertical fluctuations as plots
    fig3 = plt.figure(num=3,figsize=figsize)
    plt.clf()
    ax31 = fig3.add_subplot(211)
    im31 = ax31.plot(vertfluctinit)
    im31a, = ax31.plot(vertfluctinit.mean(axis=1),'r',linewidth=2.5)
    im31b, = ax31.plot(vertfluctinit.mean(axis=1),'--w',linewidth=1.5)
    ax31.axis('tight')
    ax31.set_title('Initial Integral in x')
    ax31.set_xlabel('Projection')
    ax31.set_ylabel('y [pixels]')
    ax32 = fig3.add_subplot(212)
    im32 = ax32.plot(vertfluctcurr)
    im32a, = ax32.plot(vertfluctcurr.mean(axis=1),'r',linewidth=2.5)
    im32b, = ax32.plot(vertfluctcurr.mean(axis=1),'--w',linewidth=1.5)
    ax32.axis('tight')
    ax32.set_title('Current Integral in x')
    ax32.set_xlabel('Projection')
    ax32.set_ylabel('y [pixels]')
    fig3.tight_layout()
    fig3.show()

    # shifts
    fig4 = plt.figure(num=4)
    plt.clf()
    ax41 = fig4.add_subplot(211)
    im41 = ax41.plot(deltastack)
    ax41.axis('tight')
    ax41.set_title('Object position')
    # metric_error
    ax42 = fig4.add_subplot(212)
    im42, = ax42.plot(metric_error,'bo-')
    ax42.axis('tight')
    ax42.set_title('Error metric')
    fig4.tight_layout()
    fig4.show()

    plt.pause(0.001)

    fig_array = [fig1,fig2,fig3,fig4]
    im_array = [im11,im21,im22,im31,im31a,im31b,im32,im32a,im32b,im41,im42]
    ax_array = [ax11,ax21,ax22,ax31,ax32,ax41,ax42]

    return(fig_array, im_array,ax_array)

class RegisterPlot:
    """
    Display plots during registration
    """
    def __init__(self, **params):
        self.count = 0
        self.params = params
        #self.vmin = params['vmin']
        #self.vmax = params['vmax']
        plt.close('all')

    def plotsvertical(self, proj, lims, vertfluctinit, vertfluctcurr,
                      deltastack, metric_error, count):
        """
        Display plots during the vertical registration
        """
        self.proj = proj
        self.lims = lims
        self.vertfluctinit = vertfluctinit.T
        self.vertfluctinit_avg = self.vertfluctinit.mean(axis=1)
        self.vertfluctcurr = vertfluctcurr.T
        self.vertfluctcurr_avg = self.vertfluctcurr.mean(axis=1)
        self.deltastack = deltastack.T
        self.metric_error = metric_error
        self.count = count

        #figures display
        nr,nc = self.vertfluctinit.shape # for the image display
        if nc>nr:
            figsize = (np.round(6*nc/nr),6)
        else:
            figsize = (6,np.round(6*nr/nc))

        if self.count == 0:
            # Preparing the canvas for the figures
            fig_array, im_array, ax_array = _createcanvasvertical(self.proj,
                                    self.lims,
                                    self.vertfluctinit,
                                    self.vertfluctcurr,
                                    self.deltastack,
                                    self.metric_error,
                                    **self.params)
            self.fig1 = fig_array[0]
            self.fig2 = fig_array[1]
            self.fig3 = fig_array[2]
            self.fig4 = fig_array[3]

            self.im11 = im_array[0] #im11
            self.im21 = im_array[1] #im21
            self.im22 = im_array[2] #im22
            self.im31 = im_array[3] #im31
            self.im31a = im_array[4] #im31a
            self.im31b = im_array[5] #im31b
            self.im32 = im_array[6] #im32
            self.im32a = im_array[7] #im32a
            self.im32b = im_array[8] #im32b
            self.im41 = im_array[9] #im41
            self.im42 = im_array[10] #im51

            self.ax11 = ax_array[0] #ax11
            self.ax21 = ax_array[1] #ax21
            self.ax22 = ax_array[2] #ax22
            self.ax31 = ax_array[3] #ax31
            self.ax32 = ax_array[4] #ax32
            self.ax41 = ax_array[5] #ax41
            self.ax42 = ax_array[6] #ax51
            self.updatevertical()
        else:
            self.updatevertical()

    def updatevertical(self):
        """
        Update the plot canvas during vertical registration
        """
        self.im21.set_data(self.vertfluctinit)
        self.im22.set_data(self.vertfluctcurr)

        self.ax21.axes.figure.canvas.draw()
        self.ax22.axes.figure.canvas.draw()

        # display vertical fluctuations as plots
        fig3 = plt.figure(num=3)
        plt.clf()
        ax31 = fig3.add_subplot(211)
        im31 = ax31.plot(self.vertfluctinit)
        im31a, = ax31.plot(self.vertfluctinit_avg,'r',linewidth=2.5)
        im31b, = ax31.plot(self.vertfluctinit_avg,'--w',linewidth=1.5)
        ax31.axis('tight')
        ax31.set_title('Initial Integral in x')
        ax31.set_xlabel('Projection')
        ax31.set_ylabel('y [pixels]')
        ax32 = fig3.add_subplot(212)
        im32 = ax32.plot(self.vertfluctcurr)
        im32a, = ax32.plot(self.vertfluctcurr_avg,'r',linewidth=2.5)
        im32b, = ax32.plot(self.vertfluctcurr_avg,'--w',linewidth=1.5)
        ax32.axis('tight')
        ax32.set_title('Current Integral in x')
        ax32.set_xlabel('Projection')
        ax32.set_ylabel('y [pixels]')
        fig3.tight_layout()
        fig3.show()

        # shifts
        fig4 = plt.figure(num=4)
        plt.clf()
        ax41 = fig4.add_subplot(211)
        im41 = ax41.plot(self.deltastack)
        ax41.axis('tight')
        ax41.set_title('Object position')
        # metric_error
        ax42 = fig4.add_subplot(212)
        im42, = ax42.plot(self.metric_error,'bo-')
        ax42.axis('tight')
        ax42.set_title('Error metric')
        fig4.tight_layout()
        fig4.show()

        # TOCHECK: Find out why this does not work
        #~ for lnum,line in enumerate(self.im32):
            #~ line.set_ydata(self.vertfluctcurr[lnum])
        #~ autoscale_y(self.ax41)
        #~ self.im32a.set_ydata(self.vertfluctcurr.mean(axis=0))
        #~ self.im32b.set_ydata(self.vertfluctcurr.mean(axis=0))
        #~ for lnum,line in enumerate(self.im41):
            #~ line.set_ydata(self.deltastack[ii])
        #~ autoscale_y(self.ax41)
        plt.pause(0.001)

    def plotshorizontal(self, recons, sinoorig, sinocurr, sinocomp,
                        deltaslice, metric_error, count):
        """
        Display plots during the horizontal registration
        """
        self.recons = recons
        self.sinoorig = sinoorig
        self.sinocurr = sinocurr
        self.sinocomp = sinocomp
        self.deltaslice = deltaslice.T
        self.metric_error = metric_error
        self.count = count

        if self.count == 0:
            # Preparing the canvas for the figures
            fig_array, im_array, ax_array = _createcanvashorizontal(self.recons,
                                            self.sinoorig,
                                            self.sinocurr,
                                            self.sinocomp,
                                            self.deltaslice,
                                            self.metric_error,
                                            **self.params)

            self.fig1 = fig_array[0]
            self.fig2 = fig_array[1]
            self.fig3 = fig_array[2]

            self.im11 = im_array[0] #im11
            self.im21 = im_array[1] #im21
            self.im22 = im_array[2] #im22
            self.im23 = im_array[3] #im23
            self.im31 = im_array[4] #im31
            self.im32 = im_array[5] #im32

            self.ax11 = ax_array[0] #ax11
            self.ax21 = ax_array[1] #ax21
            self.ax22 = ax_array[2] #ax22
            self.ax23 = ax_array[3] #ax23
            self.ax31 = ax_array[4] #ax31
            self.ax32 = ax_array[5] #ax32
        else:
            self.updatehorizontal()

    def updatehorizontal(self):
        """
        Update the plot canvas during horizontal registration
        """
        self.im11.set_data(self.recons)
        self.im22.set_data(self.sinocurr)
        self.im23.set_data(self.sinocomp)

        self.ax11.set_title('Reconstruced slice. Iteration {}'.format(self.count))
        self.ax11.axes.figure.canvas.draw()
        self.ax22.axes.figure.canvas.draw()
        self.ax23.axes.figure.canvas.draw()

        # shifts and error metric
        fig3 = plt.figure(num=3)
        plt.clf()
        ax31 = fig3.add_subplot(211)
        im31 = ax31.plot(self.deltaslice)
        ax31.axis('tight')
        ax31.set_title('Object position')
        ax32 = fig3.add_subplot(212)
        im32 = ax32.plot(self.metric_error,'bo-')
        ax32.axis('tight')
        ax32.set_title('Error metric')
        fig3.tight_layout()
        fig3.show()

        #~ self.im31.set_ydata(deltaslice.T)
        #~ self.im32.set_ydata(metric_error)
        #~ autoscale_y(self.im31)
        #~ autoscale_y(self.im32)
        #~ plt.draw()

        #~ self.ax11.axes.figure.canvas.draw()
        #~ self.ax21.axes.figure.canvas.draw()
        #~ self.ax22.axes.figure.canvas.draw()
        #~ self.ax23.axes.figure.canvas.draw()
        #~ self.ax31.axes.figure.canvas.draw()
        #~ self.ax32.axes.figure.canvas.draw()

def iterative_show(stack_array,*args):
    """
    Iterative plot of the images
    """
    nproj,nr,nc = stack_array.shape
    if len(args) == 0:
        limrow = [0,nr]
        limcol = [0,nc]
    elif len(args) == 2:
        limrow = args[0]
        limcol = args[1]
    else:
        raise ValueError('This function accepts only two args')

    # display
    plt.close('all')
    plt.ion()
    fig = plt.figure(4)#,figsize=(14,6))
    ax1 = fig.add_subplot(111)
    im1 = ax1.imshow(stack_array[0,limrow[0]:limrow[-1],limcol[0]:limcol[-1]],cmap='bone' )
    ax1.set_title("Projection: {}".format(1))
    fig.show()
    plt.pause(0.001)
    for ii in range(nproj):
        print("Projection: {}".format(ii+1),end='\r')
        projection = stack_array[ii,limrow[0]:limrow[-1],limcol[0]:limcol[-1]]
        im1.set_data(projection)
        ax1.set_title('Projection {}'.format(ii+1))
        fig.show()
        plt.pause(0.001)

def _animated_image(stack_array,*args):
    """
    Iterative plot of the images using pyplot text for the title
    """
    nproj,nr,nc = stack_array.shape
    if len(args) == 0:
        limrow = [0,nr]
        limcol = [0,nc]
    elif len(args) == 2:
        limrow = args[0]
        limcol = args[1]
    else:
        raise ValueError('This function accepts only two args')

    # display
    plt.close('all')
    #plt.ion()
    fig = plt.figure(4)#,figsize=(14,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(stack_array[0,limrow[0]:limrow[-1],limcol[0]:limcol[-1]],cmap='bone',animated=True)
    #~ title = ax.text(0.5,1.05,"",fontsize=20,bbox={'facecolor':'w','alpha':0.5,'pad':5},
                #~ transform=ax.transAxes,ha='center')
    title = ax.text(0.5,1.05,"",fontsize=20,transform=ax.transAxes,ha='center')
    #~ plt.tight_layout()
    def updatefig(ii):
        global stack_array, limrow, limcol
        imgi = stack_array[ii,limrow[0]:limrow[-1],limcol[0]:limcol[-1]]
        im.set_data(imgi)
        title.set_text("Projection: {}".format(ii+1))
        return im,title,

    return fig, updatefig,nproj

def _animated_image2(stack_array,*args):
    """
    Iterative plot of the images using pyplot title
    """
    nproj,nr,nc = stack_array.shape
    if len(args) == 0:
        limrow = [0,nr]
        limcol = [0,nc]
    elif len(args) == 2:
        limrow = args[0]
        limcol = args[1]
    else:
        raise ValueError('This function accepts only two args')

    # display
    plt.close('all')
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    im = ax.imshow(stack_array[0,limrow[0]:limrow[-1],limcol[0]:limcol[-1]],cmap='bone',animated=True)
    plt.tight_layout()
    arr1 = [None]
    def updatefig(ii):
        global stack_array, limrow, limcol
        ax.set_title("Projection: {}".format(ii+1),fontsize=20)
        if arr1[0]: arr1[0].remove()
        arr1[0] = im.set_data(stack_array[ii,limrow[0]:limrow[-1],limcol[0]:limcol[-1]])

    return fig, updatefig,nproj

def animated_image(stack_array,*args):
    """
    Iterative plot of the images using animation module of Matplotlib

    Parameters
    ----------
    stack_array : ndarray
        Array containing the stack of images to animate. The first index
        corresponds to the image number in the sequence of images.
    args[0] : list of ints
        Row limits to display
    args[1] : list of ints
        Column limits to display
    """
    fig, updatefig, nproj = _animated_image(stack_array,*args)
    ani = animation.FuncAnimation(fig, updatefig, frames=nproj, interval=50, blit=False, repeat=False)
    plt.show()

def show_projections(objs,probe,idxproj):
    """
    Show projections and probe
    """
    if objs.shape[0]<objs.shape[1]:
        plotgrid=(3,1)
        plotsize=(6,20)
    else:
        plotgrid=(1,3)
        plotsize=(20,6)
    vabsmean = np.abs(objs).mean()
    perabsmean = 0.2*vabsmean
    #vphasemean = np.angle(objs).mean()
    #perphasemean = 0.3*vphasemean
    plt.clf()
    fig, (ax1,ax2,ax3) = plt.subplots(num=1,nrows=plotgrid[0],ncols=plotgrid[1],figsize=plotsize)
    im1 = ax1.imshow(np.abs(objs),interpolation='none',cmap='gray',vmin=vabsmean-perabsmean,vmax=vabsmean+perabsmean)
    fig.colorbar(im1,ax=ax1)
    ax1.set_title(u'Object magnitude - projection {}'.format(idxproj))
    im2 = ax2.imshow(np.angle(objs),interpolation='none',cmap='bone')#,vmin=vphasemean-perabsmean,vmax=vphasemean+perabsmean)
    fig.colorbar(im2,ax=ax2)
    ax2.set_title(u'Object Phase - projection {}'.format(idxproj))
    # Special tricks for the probe display
    H = np.angle(probe)/(2*np.pi)+0.5
    S = np.ones_like(H).astype(int)
    V = np.abs(probe)/np.max(np.abs(probe))
    probe_hsv = np.dstack((H,S,V))
    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    norm = mpl.colors.Normalize(-np.pi,np.pi)
    cmap = mpl.cm.colors.hsv_to_rgb # TO BE FIXED
    #fig.subplots_adjust(hspace=plotgrid[0]/3.,wspace=plotgrid[1]/3.)
    #fig.tight_layout()
    im3 = ax3.imshow(hsv_to_rgb(probe_hsv),interpolation='none')
    fig.colorbar(im3,ax=ax3,cmap=mpl.cm.get_cmap('hsv'),norm=norm) # TO BE FIXED
    #cb = mpl.colorbar.ColorbarBase(ax3,cmap=mpl.cm.get_cmap('hsv'),norm=norm,orientation = 'horizontal')
    ax3.set_title('Probe - projection {}'.format(idxproj))
    #plt.tight_layout()
    plt.draw()
    #plt.clf()

def show_linearphase(image,mask,*args):
    """
    Show projections and probe
    """
    try:
        idxproj=args[0]
    except:
        idxproj=''

    linecut = np.round(image.shape[0]/2.)

    fig, (ax1,ax2) = plt.subplots(num=3,nrows=2,ncols=1,figsize=(14,10))
    im1=ax1.imshow(image+mask,cmap='bone')
    ax1.set_title('Projection {}'.format(idxproj))
    im2=ax2.plot(image[linecut,:])
    ax2.plot([0,image.shape[1]],[0,0])
    ax2.axis('tight')
    plt.draw()
    #ax2.cla()

# ~ fig1 = plt.figure(num=1)
# ~ self.ax11 = fig1.add_subplot(111)
# ~ self.im11 = self.ax11.imshow(recons,cmap='jet')
# ~ self.ax11.axis('image')
# ~ self.ax11.set_title('Initial slice')
# ~ self.ax11.set_xlabel('x [pixels]')
# ~ self.ax11.set_ylabel('y [pixels]')
# ~ fig1.tight_layout()
# ~ fig1.show()

# ~ fig2 = plt.figure(num=2)
# ~ self.ax21 = fig2.add_subplot(311)
# ~ self.im21 = self.ax21.imshow(sinoorig,cmap='bone',vmin=cmin,vmax=cmax)
# ~ self.ax21.axis('tight')
# ~ self.ax21.set_title('Initial sinogram')
# ~ self.ax21.set_xlabel('Projection')
# ~ self.ax21.set_ylabel('x [pixels]')
# ~ self.ax22 = self.fig2.add_subplot(312)
# ~ self.im22 = self.ax22.imshow(sinocurr,cmap='bone',vmin=cmin,vmax=cmax)
# ~ self.ax22.axis('tight')
# ~ self.ax22.set_title('Current sinogram')
# ~ self.ax22.set_xlabel('Projection')
# ~ self.ax22.set_ylabel('x [pixels]')
# ~ self.ax23 = fig2.add_subplot(313)
# ~ self.im23 = self.ax23.imshow(sinocomp,cmap='bone',vmin=cmin,vmax=cmax)
# ~ self.ax23.axis('tight')
# ~ self.ax23.set_title('Synthetic sinogram')
# ~ self.ax23.set_xlabel('Projection')
# ~ self.ax23.set_ylabel('x [pixels]')
# ~ fig2.tight_layout()
# ~ fig2.show()

# ~ fig3 = plt.figure(num=3)
# ~ self.ax31 = fig3.add_subplot(211)
# ~ self.im31, = self.ax31.plot(deltaslice.T)
# ~ self.ax31.axis('tight')
# ~ self.ax31.set_title('Object position')
# ~ self.ax32 = fig3.add_subplot(212)
# ~ self.im32, = ax32.plot(metric_error,'bo-')
# ~ self.ax32.axis('tight')
# ~ self.ax32.set_title('Error metric')
# ~ fig3.tight_layout()
# ~ fig3.show()
