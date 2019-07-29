# -*- coding: utf-8 -*-
from __future__ import division, print_function  # ,unicode_literals

# Standard library imports
import re
import sys

# third party packages
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, Button  # , RectangleSelector
from matplotlib.widgets import TextBox

# local packages
from io_utils import checkhostname, create_paramsh5, load_paramsh5, save_or_load_data
from amplitude_utils import IndexTracker

inputkwargs = dict()
params = dict()

### Edit section ###
inputkwargs[u"crop_reg"] = 10
inputkwargs[u"samplename"] = u"v97_h_nfptomo_15nm"
# params[u'remove_last']=True
# params[u'proj_to_remove']=2
params[u"correct_bad"] = True
params[u"bad_projs"] = [117]  # starting at zero

### End of edit section ###
# =============================================================================#
# Don't edit below this line, please                                          #
# =============================================================================#
if __name__ == "__main__":
    # check machine
    host_machine = checkhostname()

    if sys.version_info > (3, 0):
        raw_input = input

    inputkwargs[u"amponly"] = True
    inputkwargs[u"phaseonly"] = False
    # auxiliary dictionary to avoid overwriting of variables
    inputparams = dict()

    # loading parameters from h5file
    kwargs = load_paramsh5(**inputkwargs)
    inputparams.update(kwargs)
    # as second to update to the most recent values
    inputparams.update(inputkwargs)

    # load the reconstructed phase projections
    stack_objs, theta, deltastack, pixelsize, kwargs = save_or_load_data(
        "reconstructed_projections.h5", **inputparams
    )
    inputparams.update(kwargs)  # updating the inputkwargs
    # as second to update to the most recent values
    inputparams.update(inputkwargs)

    ###inputparams['pathfilename'] = re.sub('/data/visitor/','/data/id16a/inhouse4/visitor/',inputparams['pathfilename'])
    # updating parameter h5 file
    create_paramsh5(**inputparams)

    # removing extra projections over 180-\Delta\theta degrees
    # print(theta)
    print(theta[-5:])
    a = str(input("Do you want to remove extra thetas?([y]/n)")).lower()
    if a == "" or a == "y":
        a1 = eval(input("How many to remove?"))
        # the 3 last angles are 180, 90 and 0 degrees
        stack_objs = stack_objs[:-a1]
        theta = theta[:-a1]  # the 3 last angles are 180, 90 and 0 degrees
    print(theta[-5:])

    # correcting bad projections after unwrapping
    if params[u"correct_bad"]:
        # bad_proj = [156, 226, 363, 371, 673, 990]
        for ii in params[u"bad_projs"]:
            print("Temporary replacement of bad projection: {}".format(ii))
            stack_objs[ii] = stack_objs[ii - 1]

    # cropping the image for the phase ramp removal
    crop_reg = inputkwargs["crop_reg"]
    if crop_reg != 0:
        stack_objs = stack_objs[:, crop_reg:-crop_reg, crop_reg:-crop_reg]

    # Convinient copy to prevent overwritting accidents
    # X1=np.angle(stack_objs).copy()
    # X1=np.angle(X1).copy()

    plt.close("all")
    fig = plt.figure(4)
    gs = gridspec.GridSpec(
        3, 3, width_ratios=[8, 3, 2], height_ratios=[8, 4.5, 0.5]  # figure=4,
    )
    # plt.subplots_adjust(bottom=0.25)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[3])
    # ~ ax1 = fig.add_subplot(211)
    # ~ ax2 = fig.add_subplot(212)
    tracker = IndexTracker(ax1, ax2, stack_objs, inputparams["pathfilename"])
    # Button draw mask
    axdraw = plt.axes([0.58, 0.82, 0.19, 0.06])
    bdraw = Button(axdraw, "draw mask")
    bdraw.on_clicked(tracker.draw_mask)
    # Button Close figure
    axclose = plt.axes([0.78, 0.82, 0.19, 0.06])
    bclose = Button(axclose, "close figure")
    bclose.on_clicked(tracker.onclose)
    # Button add mask
    axadd = plt.axes([0.58, 0.72, 0.19, 0.06])
    badd = Button(axadd, "add mask")
    badd.on_clicked(tracker.add_mask)
    # Button Apply mask
    axapply = plt.axes([0.78, 0.72, 0.19, 0.06])
    bapply = Button(axapply, "apply mask")
    bapply.on_clicked(tracker.apply_mask)
    # Button mask to all
    axmaskall = plt.axes([0.58, 0.62, 0.19, 0.06])
    bmaskall = Button(axmaskall, "mask all")
    bmaskall.on_clicked(tracker.mask_all)
    # Button Apply all mask
    axapplyall = plt.axes([0.78, 0.62, 0.19, 0.06])
    bapplyall = Button(axapplyall, "apply all masks")
    bapplyall.on_clicked(tracker.apply_all_masks)
    # Button remove mask
    axremove = plt.axes([0.58, 0.52, 0.19, 0.06])
    bremove = Button(axremove, "remove mask")
    bremove.on_clicked(tracker.remove_mask)
    # ~ #### Button remove phase ramp
    # ~ axrmramp = plt.axes([0.78,0.52,0.19,0.06])
    # ~ brmramp = Button(axrmramp,'remove ramp')
    # ~ brmramp.on_clicked(tracker.remove_ramp)
    # Button remove all mask
    axremoveall = plt.axes([0.58, 0.42, 0.19, 0.06])
    bremoveall = Button(axremoveall, "remove all mask")
    bremoveall.on_clicked(tracker.remove_all_mask)
    # ~ #### Button remove all phase ramp
    # ~ axrmrampall = plt.axes([0.78,0.42,0.19,0.06])
    # ~ brmrampall = Button(axrmrampall,'remove all ramp')
    # ~ brmrampall.on_clicked(tracker.remove_rampall)
    # Button Save masks
    axsave = plt.axes([0.58, 0.32, 0.19, 0.06])
    bsave = Button(axsave, "save masks")
    bsave.on_clicked(tracker.save_masks)
    # ~ #### Button unwrap
    # ~ axunwrap = plt.axes([0.78,0.32,0.19,0.06])
    # ~ bunwrap = Button(axunwrap,'unwrap')
    # ~ bunwrap.on_clicked(tracker.unwrapping_phase)
    # Button load mask from file
    axload = plt.axes([0.58, 0.22, 0.19, 0.06])
    bload = Button(axload, "load masks")
    bload.on_clicked(tracker.load_masks)
    # ~ #### Button unwrap all
    # ~ axunwrapall = plt.axes([0.78,0.22,0.19,0.06])
    # ~ bunwrapall = Button(axunwrapall,'unwrap all')
    # ~ bunwrapall.on_clicked(tracker.unwrapping_all)

    # text boxes
    axboxprojn = plt.axes([0.125, 0.05, 0.1, 0.06])
    text_box = TextBox(axboxprojn, "Goto #", initial="1")
    text_box.on_submit(tracker.submit)
    # ~ text_box.on_text_change(tracker.updateval)

    # Colormap boxes
    axboxvmin = plt.axes([0.67, 0.05, 0.1, 0.06])
    textboxvmin = TextBox(axboxvmin, "vmin", initial="None")
    textboxvmin.on_submit(tracker.cmvmin)
    axboxvmax = plt.axes([0.87, 0.05, 0.1, 0.06])
    textboxvmax = TextBox(axboxvmax, "vmax", initial="None")
    textboxvmax.on_submit(tracker.cmvmax)

    # give the name for colormap boxes
    cmap_title = plt.axes([0.72, 0.14, 0.1, 0.06])
    cmap_title.set_axis_off()
    cmap_title.text(0, 0, "Colormap", fontsize=14)
    # axboxvmin.set_title('Colormap')

    # Buttons Prev/Next
    axprev = plt.axes([0.28, 0.05, 0.05, 0.06])
    axnext = plt.axes([0.35, 0.05, 0.05, 0.06])
    bnext = Button(axnext, ">")
    bnext.on_clicked(tracker.up)
    bprev = Button(axprev, "<")
    bprev.on_clicked(tracker.down)
    # Button Play
    axplay = plt.axes([0.445, 0.05, 0.1, 0.06])
    bplay = Button(axplay, "play")
    bplay.on_clicked(tracker.play)
    # Connect scroll and key_press events
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)
    fig.canvas.mpl_connect("key_press_event", tracker.key_event)
    multi = MultiCursor(fig.canvas, (ax1, ax2), color="r", lw=1)
    plt.show(block=False)
    a = raw_input("Press Enter to finish\n")
    # output of the linear phase removal
    stack_phasecorr = tracker.X1.copy()  # tracker.X1 is where the data is stored
    plt.close("all")

    # ~ if params[u'remove_last']:
    # ~ rm_proj = params[u'proj_to_remove']
    # ~ stack_phasecorr = stack_phasecorr[:-rm_proj]
    # ~ theta = theta[:-rm_proj]

    # Save reconstructed phase projections
    a = raw_input(
        "Are you happy with the correction of the amplitude projections?([y]/n) "
    ).lower()
    if a == "" or a == "y":
        save_or_load_data(
            "air_corrected_amplitude.h5",
            stack_phasecorr,
            theta,
            pixelsize,
            **inputparams
        )
    else:
        print("The results of the air correction have not been saved")
    # next step
    print("You should run " "amplitude_alignment.py" " now")
# =============================================================================#
