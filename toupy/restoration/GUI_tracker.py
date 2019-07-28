#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from matplotlib.widgets import Button  # , RectangleSelector
from matplotlib.widgets import TextBox
import numpy as np
from numpy.fft import fftfreq
#from roipoly import RoiPoly, MultiRoi
from skimage.restoration import unwrap_phase

# local packages
from ..io.dataio import LoadData, SaveData
from .ramptools import rmphaseramp, rmlinearphase, rmair
from .roipoly import roipoly

__all__ = ['gui_plotamp',
           'gui_plotphase',
           'AmpTracker',
           'PhaseTracker']


def _crop_stack(stack_images, cropreg):
    """
    Crop stack of images for the phase ramp removal

    Parameters
    ----------
    stack_images : ndarray
        Stack of images
    cropreg : sequence of ints
        List of number of pixel to cut from border. The order is
        [left, bottom, right, top]

    Returns
    -------
    crop_stack : ndarray
        Cropped stack
    """
    return stack_objs[:, cropreg[0]:-cropreg[0], cropreg[1]:-cropreg[1]]


def gui_plotamp(stack_objs, **params):
    """
    GUI for the air removal from amplitude projections
    """
    if params['crop_reg'] is not None:
        if params['crop_reg'] != []:
            # cropping the image for the phase ramp removal
            stack_objs = _crop_stack(stack_objs, params['crop_reg'])
    plt.close('all')
    fig = plt.figure(4)
    gs = gridspec.GridSpec(3, 3,  # figure=4,
                           width_ratios=[8, 3, 2],
                           height_ratios=[8, 4.5, 0.5]
                           )
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[3])
    tracker = AmpTracker(
        fig, ax1, ax2, stack_objs, **params)
    # Button draw mask
    axdraw = plt.axes([0.58, 0.82, 0.19, 0.06])
    bdraw = Button(axdraw, 'draw mask')
    bdraw.on_clicked(tracker.draw_mask)
    # Button Close figure
    axclose = plt.axes([0.78, 0.82, 0.19, 0.06])
    bclose = Button(axclose, 'close figure')
    bclose.on_clicked(tracker.onclose)
    # Button add mask
    axadd = plt.axes([0.58, 0.72, 0.19, 0.06])
    badd = Button(axadd, 'add mask')
    badd.on_clicked(tracker.add_mask)
    # Button Apply mask
    axapply = plt.axes([0.78, 0.72, 0.19, 0.06])
    bapply = Button(axapply, 'apply mask')
    bapply.on_clicked(tracker.apply_mask)
    # Button mask to all
    axmaskall = plt.axes([0.58, 0.62, 0.19, 0.06])
    bmaskall = Button(axmaskall, 'mask all')
    bmaskall.on_clicked(tracker.mask_all)
    # Button Apply all mask
    axapplyall = plt.axes([0.78, 0.62, 0.19, 0.06])
    bapplyall = Button(axapplyall, 'apply all masks')
    bapplyall.on_clicked(tracker.apply_all_masks)
    # Button remove mask
    axremove = plt.axes([0.58, 0.52, 0.19, 0.06])
    bremove = Button(axremove, 'remove mask')
    bremove.on_clicked(tracker.remove_mask)
    # Button remove all mask
    axremoveall = plt.axes([0.58, 0.42, 0.19, 0.06])
    bremoveall = Button(axremoveall, 'remove all mask')
    bremoveall.on_clicked(tracker.remove_all_mask)
    # Button Save masks
    axsave = plt.axes([0.58, 0.32, 0.19, 0.06])
    bsave = Button(axsave, 'save masks')
    bsave.on_clicked(tracker.save_masks)
    # Button load mask from file
    axload = plt.axes([0.58, 0.22, 0.19, 0.06])
    bload = Button(axload, 'load masks')
    bload.on_clicked(tracker.load_masks)
    # text boxes
    axboxprojn = plt.axes([0.125, 0.05, 0.1, 0.06])
    text_box = TextBox(axboxprojn, 'Goto #', initial="1")
    text_box.on_submit(tracker.submit)
    #~ text_box.on_text_change(tracker.updateval)

    # Colormap boxes
    axboxvmin = plt.axes([0.67, 0.05, 0.1, 0.06])
    textboxvmin = TextBox(axboxvmin, 'vmin', initial="None")
    textboxvmin.on_submit(tracker.cmvmin)
    axboxvmax = plt.axes([0.87, 0.05, 0.1, 0.06])
    textboxvmax = TextBox(axboxvmax, 'vmax', initial="None")
    textboxvmax.on_submit(tracker.cmvmax)
    # give the name for colormap boxes
    cmap_title = plt.axes([0.72, 0.14, 0.1, 0.06])
    cmap_title.set_axis_off()
    cmap_title.text(0, 0, 'Colormap', fontsize=14)
    # axboxvmin.set_title('Colormap')

    # Buttons Prev/Next
    axprev = plt.axes([0.28, 0.05, 0.05, 0.06])
    axnext = plt.axes([0.35, 0.05, 0.05, 0.06])
    bnext = Button(axnext, '>')
    bnext.on_clicked(tracker.up)
    bprev = Button(axprev, '<')
    bprev.on_clicked(tracker.down)
    # Button Play
    axplay = plt.axes([0.445, 0.05, 0.1, 0.06])
    bplay = Button(axplay, 'play')
    bplay.on_clicked(tracker.play)
    # Connect scroll and key_press events
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.key_event)
    multi = MultiCursor(fig.canvas, (ax1, ax2), color='r', lw=1)
    plt.show(block=False)
    a = input('Press Enter to finish\n')
    # output of the linear phase removal
    stack_ampcorr = tracker.X1.copy()  # tracker.X1 is where the data is stored
    plt.close('all')
    return stack_ampcorr


def gui_plotphase(stack_objs, **params):
    """
    GUI for the phase ramp removal from phase projections
    """
    if params['crop_reg'] is not None:
        if params['crop_reg'] != []:
            # cropping the image for the phase ramp removal
            stack_objs = _crop_stack(stack_objs, params['crop_reg'])

    plt.close('all')
    fig = plt.figure(4)
    gs = gridspec.GridSpec(3, 3,  # figure=4,
                           width_ratios=[8, 3, 2],
                           height_ratios=[8, 4.5, 0.5]
                           )
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[3])
    tracker = PhaseTracker(
        fig, ax1, ax2, stack_objs, **params)
    # Button draw mask
    axdraw = plt.axes([0.58, 0.82, 0.19, 0.06])
    bdraw = Button(axdraw, 'draw mask')
    bdraw.on_clicked(tracker.draw_mask)
    # Button Close figure
    axclose = plt.axes([0.78, 0.82, 0.19, 0.06])
    bclose = Button(axclose, 'close figure')
    bclose.on_clicked(tracker.onclose)
    # Button add mask
    axadd = plt.axes([0.58, 0.72, 0.19, 0.06])
    badd = Button(axadd, 'add mask')
    badd.on_clicked(tracker.add_mask)
    # Button Apply mask
    axapply = plt.axes([0.78, 0.72, 0.19, 0.06])
    bapply = Button(axapply, 'apply mask')
    bapply.on_clicked(tracker.apply_mask)
    # Button mask to all
    axmaskall = plt.axes([0.58, 0.62, 0.19, 0.06])
    bmaskall = Button(axmaskall, 'mask all')
    bmaskall.on_clicked(tracker.mask_all)
    # Button Apply all mask
    axapplyall = plt.axes([0.78, 0.62, 0.19, 0.06])
    bapplyall = Button(axapplyall, 'apply all masks')
    bapplyall.on_clicked(tracker.apply_all_masks)
    # Button remove mask
    axremove = plt.axes([0.58, 0.52, 0.19, 0.06])
    bremove = Button(axremove, 'remove mask')
    bremove.on_clicked(tracker.remove_mask)
    # Button remove phase ramp
    axrmramp = plt.axes([0.78, 0.52, 0.19, 0.06])
    brmramp = Button(axrmramp, 'remove ramp')
    brmramp.on_clicked(tracker.remove_ramp)
    # Button remove all mask
    axremoveall = plt.axes([0.58, 0.42, 0.19, 0.06])
    bremoveall = Button(axremoveall, 'remove all mask')
    bremoveall.on_clicked(tracker.remove_all_mask)
    # Button remove all phase ramp
    axrmrampall = plt.axes([0.78, 0.42, 0.19, 0.06])
    brmrampall = Button(axrmrampall, 'remove all ramp')
    brmrampall.on_clicked(tracker.remove_rampall)
    # Button Save masks
    axsave = plt.axes([0.58, 0.32, 0.19, 0.06])
    bsave = Button(axsave, 'save masks')
    bsave.on_clicked(tracker.save_masks)
    # Button unwrap
    axunwrap = plt.axes([0.78, 0.32, 0.19, 0.06])
    bunwrap = Button(axunwrap, 'unwrap')
    bunwrap.on_clicked(tracker.unwrapping_phase)
    # Button load mask from file
    axload = plt.axes([0.58, 0.22, 0.19, 0.06])
    bload = Button(axload, 'load masks')
    bload.on_clicked(tracker.load_masks)
    # Button unwrap all
    axunwrapall = plt.axes([0.78, 0.22, 0.19, 0.06])
    bunwrapall = Button(axunwrapall, 'unwrap all')
    bunwrapall.on_clicked(tracker.unwrapping_all)
    # text boxes
    axboxprojn = plt.axes([0.125, 0.05, 0.1, 0.06])
    text_box = TextBox(axboxprojn, 'Goto #', initial="1")
    text_box.on_submit(tracker.submit)

    # Colormap boxes
    axboxvmin = plt.axes([0.67, 0.05, 0.1, 0.06])
    textboxvmin = TextBox(axboxvmin, 'vmin', initial="None")
    textboxvmin.on_submit(tracker.cmvmin)
    axboxvmax = plt.axes([0.87, 0.05, 0.1, 0.06])
    textboxvmax = TextBox(axboxvmax, 'vmax', initial="None")
    textboxvmax.on_submit(tracker.cmvmax)
    # give the name for colormap boxes
    cmap_title = plt.axes([0.72, 0.14, 0.1, 0.06])
    cmap_title.set_axis_off()
    cmap_title.text(0, 0, 'Colormap', fontsize=14)
    # axboxvmin.set_title('Colormap')

    # Buttons Prev/Next
    axprev = plt.axes([0.28, 0.05, 0.05, 0.06])
    axnext = plt.axes([0.35, 0.05, 0.05, 0.06])
    bnext = Button(axnext, '>')
    bnext.on_clicked(tracker.up)
    bprev = Button(axprev, '<')
    bprev.on_clicked(tracker.down)
    # Button Play
    axplay = plt.axes([0.445, 0.05, 0.1, 0.06])
    bplay = Button(axplay, 'play')
    bplay.on_clicked(tracker.play)
    # Connect scroll and key_press events
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.key_event)
    multi = MultiCursor(fig.canvas, (ax1, ax2), color='r', lw=1)
    plt.show(block=False)
    a = input('Press Enter to finish\n')
    # output of the linear phase removal
    stack_phasecorr = tracker.X1.copy()  # tracker.X1 is where the data is stored
    plt.close('all')
    return stack_phasecorr


class PhaseTracker(object):
    def __init__(self, fig, ax1, ax2, X1, **params):
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        ax1.set_title(
            'Use scroll wheel or \n left/right arrows to navigate images')

        self.X1 = X1.copy()
        if np.iscomplexobj(self.X1):
            raise ValueError('The array is complex')

        self.X2 = self.X1[:, np.int(X1.shape[1]/2.), :].copy()
        self.mask = np.zeros_like(X1, dtype=np.bool)
        self.slices, rows, cols = X1.shape
        self.ind = 0
        self.params = params
        self.vmin = params['vmin']
        self.vmax = params['vmax']

        self.im1 = self.ax1.imshow(
            self.X1[self.ind, :, :], cmap='bone', vmin=self.vmin, vmax=self.vmax)
        # ~ self.vmin,self.vmax = self.im1.get_clim() # get colormap limits
        self.ax1.axis('tight')
        self.im2, = self.ax2.plot(self.X2[self.ind, :])
        self.ax2.plot([0, self.X1.shape[2]], [0, 0])
        self.pmin, self.pmax = self.ax2.get_ylim()  # get plot limits
        # ax2.axis('tight')
        self.ax2.set_ylim([2*self.vmin, 2*self.vmax])
        self.ax2.set_xlim([0, cols])
        self.update()
        print('Done. When finished, close figure and press <Enter> to exit')

    def cmvmin(self, val):
        """
        Set the vmin on colormap
        """
        if eval(val) >= self.vmax:
            print('vmin is equal or larger than vmax. Choose smaller value')
        else:
            self.vmin = eval(val)  # np.clip(eval(val), 0, self.vmax - 1)
            self.pmin = self.vmin
            self.update()

    def cmvmax(self, val):
        """
        Set the vmax on colormap
        """
        if eval(val) <= self.vmin:
            print('vmax is equal or smaller than vmin. Choose larger value')
        else:
            self.vmax = eval(val)
            self.pmax = self.vmax
            self.update()

    def onscroll(self, event):
        """
        Move projection number up/down using the mouse scroll wheel
        """
        if event.button == 'up':
            if self.ind == self.slices:  # X1.shape[0]:
                print('You reached the maximum number of projections')
            else:
                self.ind = np.clip(self.ind + 1, 0, self.slices - 1)
                print("{} {} - projection {}".format(event.button,
                                                     event.step, self.ind+1))
        else:
            if self.ind == 0:
                print('You reached the first projection')
            else:
                self.ind = np.clip(self.ind - 1, 0, self.slices - 1)
                print("{} {} - projection {}".format(event.button,
                                                     event.step, self.ind+1))
        self.update()

    def key_event(self, event):
        """
        Move projection number up/down using right/left arrows in the keyboard
        """
        if event.key == 'right':
            if self.ind == self.slices:  # self.X1.shape[0]:
                print('You reached the maximum number of projections')
            else:
                self.ind = np.clip(self.ind + 1, 0, self.slices - 1)
                print("{} - projection {}".format(event.key, self.ind+1))
        elif event.key == 'left':
            if self.ind == 0:
                print('You reached the first projection')
            else:
                self.ind = np.clip(self.ind - 1, 0, self.slices - 1)
                print("{} - projection {}".format(event.key, self.ind+1))
        else:
            return
        self.update()

    def down(self, event):
        """
        Move projection number down using button Prev
        """
        self.ind = np.clip(self.ind - 1, 0, self.slices - 1)
        print("Projection {}".format(self.ind+1))
        self.update()

    def up(self, event):
        """
        Move projection number up using button Next
        """
        self.ind = np.clip(self.ind + 1, 0, self.slices - 1)
        print("Projection {}".format(self.ind+1))
        self.update()

    def draw_mask(self, event):
        """
        Draw the mask using roipoly
        """
        print('\nDrawing the poly')
        self.img_mask = self.X1[self.ind, :, :]+self.mask[self.ind, :, :]
        # create another fig in order to close later
        fig_mask = plt.figure()
        ax_mask = fig_mask.add_subplot(111)
        ax_mask.imshow(self.img_mask, cmap='bone')
        self.ROI_draw = roipoly(ax=ax_mask)
        # ~ self.ROI_draw = RoiPoly(color='b', fig = fig_mask, close_fig=True) # has to close to validate
        # ~ self.ROI_draw = MultiRoi_mod(fig=fig_mask,ax=ax_mask)#(color='b', fig = self.fig)
        #~ print(self.ROI_draw.rois.items())
        #~ print('Done')

    def add_mask(self, event):
        """
        Add the mask to the plot
        """
        print('\nAdding mask')
        self.mask[self.ind, :, :] |= self.ROI_draw.getMask(self.img_mask)
        self.ROI_draw.displayROI()
        self.update()

    def mask_all(self, event):
        """
        Use the same mask for all projections
        """
        print('\nRepeating the same mask for all projections')
        print('Please wait...')
        mask = self.ROI_draw.getMask(self.img_mask)
        self.mask |= np.array([mask for _ in range(self.slices)])
        print('Done')
        self.update()

    def remove_mask(self, event):
        """
        Remove the current selected area from the mask
        """
        print('\nRemoving mask')
        #self.mask[self.mask_ind,:,:] &= ~self.ROI_draw.getMask(self.img_mask)
        self.mask[self.ind, :, :] &= ~self.ROI_draw.getMask(self.img_mask)
        self.update()

    def remove_all_mask(self, event):
        """
        Remove all the masks
        """
        print('\nRemoving all mask')
        print('Please wait...')
        mask = self.ROI_draw.getMask(self.img_mask)
        #self.mask[self.mask_ind,:,:] &= ~self.ROI_draw.getMask(self.img_mask)
        self.mask &= ~np.array([mask for _ in range(self.slices)])
        print('Done')
        self.update()

    def apply_mask(self, event):
        """
        Apply the linear phase correction using current mask
        """
        print('\nApply the linear phase correction using current mask')
        imgin = np.exp(1j*self.X1[self.ind, :, :]).copy()
        mask = self.mask[self.ind, :, :].copy()
        # ~ self.X1[self.ind,:,:]=np.angle(remove_linearphase(imgin,mask,100)).copy()
        self.X1[self.ind, :, :] = np.angle(rmlinearphase(imgin, mask)).copy()
        self.X2[self.ind, :] = self.X1[self.ind,
                                       np.int(self.X1.shape[1]/2.), :].copy()
        self.update()

    def apply_all_masks(self, event):
        """
        Apply the linear phase correction using current mask to all projections
        """
        print('\nApply the linear phase correction using current mask to all projections')
        for ii in range(self.slices):
            self.ind = ii
            print('Projection {} out of {}'.format(
                ii+1, self.slices), end='\r')
            imgin = np.exp(1j*self.X1[ii, :, :]).copy()
            mask = self.mask[ii, :, :].copy()
            # ~ self.X1[ii,:,:]=np.angle(remove_linearphase(imgin,mask,100)).copy()
            self.X1[ii, :, :] = np.angle(rmlinearphase(imgin, mask)).copy()
            self.X2[ii, :] = self.X1[ii, np.int(self.X1.shape[1]/2.), :].copy()
            self.update()
        print('\r')
        print('Done')

    def remove_ramp(self, event):
        """
        Remove linear phase ramp
        """
        print('\nRemove linear ramp')
        self.X1[self.ind, :, :] = np.angle(rmphaseramp(
            np.exp(1j*self.X1[self.ind, :, :]), weight=None, return_phaseramp=False))
        #mask = self.mask[self.ind,:,:]*self.X1[self.ind,:,:]
        #self.X1[self.ind,:,:] -= mask.astype(np.float).mean()
        self.X2[self.ind, :] = self.X1[self.ind,
                                       np.int(self.X1.shape[1]/2.), :].copy()
        self.update()

    def remove_rampall(self, event):
        """
        Remove linear phase ramp of all
        """
        print('\nRemove linear phase ramp of all projections')
        for ii in range(self.slices):
            self.ind = ii
            print('Projection {} out of {}'.format(
                ii+1, self.slices), end='\r')
            self.X1[self.ind, :, :] = np.angle(rmphaseramp(
                np.exp(1j*self.X1[self.ind, :, :]), weight=None, return_phaseramp=False))
            #mask = self.mask[self.ind,:,:]*self.X1[self.ind,:,:]
            #self.X1[self.ind,:,:] -= mask.astype(np.float).mean()
            self.X2[self.ind, :] = self.X1[self.ind,
                                           np.int(self.X1.shape[1]/2.), :].copy()
            self.update()
        print('\r')
        print('Done')

    def unwrapping_phase(self, event):
        """
        Unwrap phase
        """
        print('\nUnwrapping phase')
        self.X1[self.ind, :, :] = unwrap_phase(self.X1[self.ind, :, :])
        if np.any(self.mask[self.ind, :, :] == True):
            vals = self.X1[self.ind, :, :][np.where(
                self.mask[self.ind, :, :] == True)].mean()
            self.X1[self.ind, :, :] -= 2*np.pi*np.round(vals/(2*np.pi))
        self.X2[self.ind, :] = self.X1[self.ind,
                                       np.int(self.X1.shape[1]/2.), :].copy()
        self.update()

    def unwrapping_all(self, event):
        """
        Unwrap phase of all projections
        """
        print('\nRemove linear phase ramp of all projections')
        for ii in range(self.slices):
            self.ind = ii
            print('Projection {} out of {}'.format(
                ii+1, self.slices), end='\r')
            self.X1[self.ind, :, :] = unwrap_phase(self.X1[self.ind, :, :])
            if np.any(self.mask[self.ind, :, :] == True):
                vals = self.X1[self.ind, :, :][np.where(
                    self.mask[self.ind, :, :] == True)].mean()
                self.X1[self.ind, :, :] -= 2*np.pi*np.round(vals/(2*np.pi))
            self.X2[self.ind, :] = self.X1[self.ind,
                                           np.int(self.X1.shape[1]/2.), :].copy()
            self.update()
        print('\r')
        print('Done')

    def load_masks(self, event):
        """
        Load masks from file
        """
        print('\nLoad masks from file')
        self.mask = LoadData.load('masks.h5', **params)
        # ~ Lm = LoadData(**self.params)
        # ~ self.mask = Lm.load_masks('masks.h5')
        self.update()

    def save_masks(self, event):
        """
        Save mask to file
        """
        print('\nSave masks to file')
        SaveData.save('masks.h5', self.mask, **self.params)
        # ~ Sm = SaveData(**self.params)
        #~ Sm.save_masks('masks.h5',self.mask)

    def submit(self, text):
        """
        Textbox submit
        """
        self.ind = np.clip(eval(text)-1, 0, self.slices - 1)
        print("Projection {}".format(self.ind+1))
        self.update()

    def play(self, event):
        """
        Plot one project after the other (play)
        """
        for ii in range(self.ind, self.slices):
            self.ind = ii  # np.clip(self.ind + ii, 0, self.slices - ii)
            print("Projection {}".format(self.ind+1))
            self.update()

    def update(self):
        """
        Update the plot canvas
        """
        self.im1.set_data(self.X1[self.ind, :, :]+self.mask[self.ind, :, :])
        self.im1.set_clim(self.vmin, self.vmax)
        self.im2.set_ydata(self.X2[self.ind, :])
        self.im2.axes.set_ylim([self.pmin, self.pmax])
        self.ax1.set_ylabel('Projection {}'.format(self.ind+1))
        self.ax2.set_ylabel('Projection {}'.format(self.ind+1))
        self.ax1.axes.figure.canvas.draw()
        self.ax2.axes.figure.canvas.draw()
        # self.im1.axes.figure.canvas.draw()
        # self.im2.axes.figure.canvas.draw()

    def onclose(self, event):
        """
        Close the figure
        """
        print('\nImage closed. Press Enter to exit and save projections')
        plt.close(event.canvas.figure)


class AmpTracker(PhaseTracker):
    def __init__(self, ax1, ax2, X1, **params):
        super().__init__(self, ax1, ax2, X1,  **params)
        self.done = []  # flag to keep the already corrected projections

    def apply_mask(self, event):
        """
        Apply the air correction using current mask and apply the log
        """
        print('\nApply the air correctioncorrection using current mask and the logarithm')
        if self.ind in self.done:
            print('Projection {} was already corrected'.format(self.ind+1))
        else:
            imgin = self.X1[self.ind, :, :].copy()
            mask = self.mask[self.ind, :, :].copy()
            self.X1[self.ind, :, :] = np.log(
                rmair(imgin, mask))  # remove air and apply log
            self.X2[self.ind, :] = self.X1[self.ind,
                                           np.int(self.X1.shape[1]/2.), :].copy()
            self.done.append(self.ind)
            self.vmin = -0.5
            self.vmax = 0.1
        self.update()

    def apply_all_masks(self, event):
        """
        Apply the linear air correction using current mask and log to all projections
        """
        print('\nApply the air correction using current mask and the logarithm to all projections')
        for ii in range(self.slices):
            self.ind = ii
            print('Projection {}'.format(ii+1))
            if self.ind in self.done:
                print('Projection {} was already corrected'.format(self.ind+1))
            else:
                # np.exp(1j*self.X1[ii,:,:]).copy()
                imgin = self.X1[ii, :, :].copy()
                mask = self.mask[ii, :, :].copy()
                # remove air and apply log
                self.X1[ii, :, :] = np.log(rmair(imgin, mask))
                self.X2[ii, :] = self.X1[ii, np.int(
                    self.X1.shape[1]/2.), :].copy()
                self.done.append(self.ind)
            self.update()
        print('Done')
