#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import warnings

import matplotlib.gridspec as gridspec
from ..utils.plot_utils import plt
from matplotlib.widgets import MultiCursor
from ..utils import tqdm
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import numpy as np
from numpy.fft import fftfreq

from skimage.restoration import unwrap_phase

# local packages
from ..io.dataio import LoadData, SaveData
from .ramptools import rmphaseramp, rmair
from .roipoly import roipoly
from ..utils import isnotebook

__all__ = ["gui_plotamp", "gui_plotphase", "AmpTracker", "PhaseTracker"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_ipympl():
    """Return True if the active matplotlib backend is ipympl (widget)."""
    try:
        import matplotlib
        backend = matplotlib.get_backend().lower()
        return "ipympl" in backend or "widget" in backend
    except Exception:
        return False


def _unwrapping_phase(imgin, mask):
    """Unwrap phase and shift so that the air/vacuum mean is a multiple of 2π."""
    unwrapimg = unwrap_phase(imgin)
    if np.any(mask):
        vals = unwrapimg[mask].mean()
        unwrapimg -= 2 * np.pi * np.round(vals / (2 * np.pi))
    return unwrapimg


def _removing_phaseramp(imgin, mask):
    """
    Remove the linear phase ramp using a boolean air mask.

    Parameters
    ----------
    imgin : ndarray, float
        2-D phase image (radians).
    mask : ndarray, bool
        Air/vacuum mask (``True`` = air).

    Returns
    -------
    ndarray, float
        Phase image with linear ramp removed and air phase zeroed.
    """
    return np.angle(
        rmphaseramp(np.exp(1j * imgin), weight=mask, zero_air_phase=True)
    )


def _crop_stack(stack_images, cropreg):
    """
    Crop a stack of images symmetrically.

    Parameters
    ----------
    stack_images : array_like
        Stack of images, shape (n, nr, nc).
    cropreg : sequence of int
        ``[left, bottom, right, top]`` pixels to crop from each border.

    Returns
    -------
    ndarray
        Cropped stack.
    """
    return stack_images[:, cropreg[0]:-cropreg[0], cropreg[1]:-cropreg[1]]


def _setup_and_run(fig, tracker, notebook_hint):
    """
    Display the figure and block (scripts) or return immediately (notebooks).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    tracker : PhaseTracker or AmpTracker
    notebook_hint : str
        Variable name the user should access in the next notebook cell,
        e.g. ``"tracker.X1"``.

    Returns
    -------
    tracker
        Always returns the tracker object.  In script mode,
        ``plt.show(block=True)`` has already returned (i.e. the user
        has closed the figure) before this function returns.
    """
    if isnotebook():
        if _is_ipympl():
            # ipympl renders the figure as a live interactive widget.
            # Force a synchronous draw so the browser receives the fully
            # populated figure (axes, image data, buttons) before the
            # cell finishes — draw_idle() may not fire in time.
            fig.canvas.draw()
            plt.show(block=False)
            print(
                "Interact with the GUI above.\n"
                f"When done, access the corrected stack in the next cell via:\n"
                f"    stack_corrected = {notebook_hint}"
            )
        else:
            # No interactive backend: force a draw, show a static
            # preview and warn the user about the missing backend.
            fig.canvas.draw()
            from IPython import display as ipy_display
            ipy_display.display(fig)
            warnings.warn(
                "The interactive GUI requires the ipympl backend.\n"
                "Add  %matplotlib widget  at the top of your notebook "
                "and restart the kernel.",
                UserWarning,
                stacklevel=3,
            )
    else:
        # Script / terminal mode: block until the figure is closed.
        # The user clicks 'close figure' (which calls plt.close) or
        # closes the window directly; plt.show(block=True) then returns.
        plt.show(block=True)
        plt.close("all")

    return tracker


# ---------------------------------------------------------------------------
# Public GUI entry points
# ---------------------------------------------------------------------------

def gui_plotamp(stack_objs, **params):
    """
    GUI for air removal from amplitude projections.

    Parameters
    ----------
    stack_objs : array_like, shape (nprojs, nr, nc)
        Stack of amplitude projections.
    **params : dict
        ``crop_reg`` : list or None
            ``[left, bottom, right, top]`` pixels to crop.
        ``vmin``, ``vmax`` : float or None
            Colormap limits.
        ``colormap`` : str
            Matplotlib colourmap name.
        ``autosave`` : bool
            Save projections automatically on load.
        ``correct_bad`` : bool
            Interpolate bad projections listed in ``bad_projs``.
        ``bad_projs`` : list of int
            Projections to interpolate (0-indexed).

    Returns
    -------
    tracker : AmpTracker
        Tracker object holding the corrected stack in ``tracker.X1``.

        * **Script mode** — ``plt.show(block=True)`` has already
          returned before this function returns; ``tracker.X1`` is
          ready immediately::

              tracker = gui_plotamp(stack_objs, **params)
              stack_ampcorr = tracker.X1.copy()

        * **Notebook mode** (requires ``%matplotlib widget``) — the
          function returns immediately; interact with the GUI in the
          cell output, then access results in the **next** cell::

              tracker = gui_plotamp(stack_objs, **params)
              # ── next cell ──
              stack_ampcorr = tracker.X1.copy()
    """
    if params.get("crop_reg") not in (None, []):
        stack_objs = _crop_stack(stack_objs, params["crop_reg"])

    plt.close("all")
    fig = plt.figure(4)
    gs = gridspec.GridSpec(
        3, 3, width_ratios=[8, 3, 2], height_ratios=[8, 4.5, 0.5]
    )
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[3])
    tracker = AmpTracker(fig, ax1, ax2, stack_objs, **params)

    # --- Buttons ---
    axdraw     = plt.axes([0.58, 0.82, 0.19, 0.06])
    axclose    = plt.axes([0.78, 0.82, 0.19, 0.06])
    axadd      = plt.axes([0.58, 0.72, 0.19, 0.06])
    axapply    = plt.axes([0.78, 0.72, 0.19, 0.06])
    axmaskall  = plt.axes([0.58, 0.62, 0.19, 0.06])
    axapplyall = plt.axes([0.78, 0.62, 0.19, 0.06])
    axremove   = plt.axes([0.58, 0.52, 0.19, 0.06])
    axremoveall= plt.axes([0.58, 0.42, 0.19, 0.06])
    axsave     = plt.axes([0.58, 0.32, 0.19, 0.06])
    axload     = plt.axes([0.58, 0.22, 0.19, 0.06])

    Button(axdraw,      "draw mask"       ).on_clicked(tracker.draw_mask)
    Button(axclose,     "close figure"    ).on_clicked(tracker.onclose)
    Button(axadd,       "add mask"        ).on_clicked(tracker.add_mask)
    Button(axapply,     "apply mask"      ).on_clicked(tracker.apply_mask)
    Button(axmaskall,   "mask all"        ).on_clicked(tracker.mask_all)
    Button(axapplyall,  "apply all masks" ).on_clicked(tracker.apply_all_masks)
    Button(axremove,    "remove mask"     ).on_clicked(tracker.remove_mask)
    Button(axremoveall, "remove all mask" ).on_clicked(tracker.remove_all_mask)
    Button(axsave,      "save masks"      ).on_clicked(tracker.save_masks)
    Button(axload,      "load masks"      ).on_clicked(tracker.load_masks)

    # --- Text boxes ---
    axboxprojn = plt.axes([0.125, 0.05, 0.1,  0.06])
    axboxvmin  = plt.axes([0.67,  0.05, 0.1,  0.06])
    axboxvmax  = plt.axes([0.87,  0.05, 0.1,  0.06])
    TextBox(axboxprojn, "Goto #", initial="1"   ).on_submit(tracker.submit)
    TextBox(axboxvmin,  "vmin",   initial="None").on_submit(tracker.cmvmin)
    TextBox(axboxvmax,  "vmax",   initial="None").on_submit(tracker.cmvmax)

    cmap_title = plt.axes([0.72, 0.14, 0.1, 0.06])
    cmap_title.set_axis_off()
    cmap_title.text(0, 0, "Colormap", fontsize=14)

    # --- Prev / Next / Play ---
    axprev = plt.axes([0.28, 0.05, 0.05, 0.06])
    axnext = plt.axes([0.35, 0.05, 0.05, 0.06])
    axplay = plt.axes([0.445, 0.05, 0.1, 0.06])
    Button(axprev, "<"    ).on_clicked(tracker.down)
    Button(axnext, ">"    ).on_clicked(tracker.up)
    Button(axplay, "play" ).on_clicked(tracker.play)

    fig.canvas.mpl_connect("scroll_event",    tracker.onscroll)
    fig.canvas.mpl_connect("key_press_event", tracker.key_event)
    MultiCursor(fig.canvas, (ax1, ax2), color="r", lw=1)

    return _setup_and_run(fig, tracker, "tracker.X1")


def gui_plotphase(stack_objs, **params):
    """
    GUI for phase-ramp removal from phase projections.

    Parameters
    ----------
    stack_objs : array_like, shape (nprojs, nr, nc)
        Stack of phase projections (float, radians).
    **params : dict
        ``crop_reg`` : list or None
            ``[left, bottom, right, top]`` pixels to crop.
        ``hcen`` : int
            Row index of the horizontal profile displayed below the image.
        ``vmin``, ``vmax`` : float or None
            Colormap limits.
        ``colormap`` : str
            Matplotlib colourmap name.
        ``autosave`` : bool
            Save projections automatically on load.

    Returns
    -------
    tracker : PhaseTracker
        Tracker object holding the corrected stack in ``tracker.X1``.

        * **Script mode** — ``plt.show(block=True)`` has already
          returned before this function returns; ``tracker.X1`` is
          ready immediately::

              tracker = gui_plotphase(stack_objs, **params)
              stack_phasecorr = tracker.X1.copy()

        * **Notebook mode** (requires ``%matplotlib widget``) — the
          function returns immediately; interact with the GUI in the
          cell output, then access results in the **next** cell::

              tracker = gui_plotphase(stack_objs, **params)
              # ── next cell ──
              stack_phasecorr = tracker.X1.copy()
    """
    if params.get("crop_reg") not in (None, []):
        stack_objs = _crop_stack(stack_objs, params["crop_reg"])

    plt.close("all")
    fig = plt.figure(1)
    gs = gridspec.GridSpec(
        3, 3, width_ratios=[8, 3, 2], height_ratios=[8, 4.5, 0.5]
    )
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[3])
    tracker = PhaseTracker(fig, ax1, ax2, stack_objs, **params)

    # --- Buttons ---
    axdraw      = plt.axes([0.58, 0.82, 0.19, 0.06])
    axclose     = plt.axes([0.78, 0.82, 0.19, 0.06])
    axadd       = plt.axes([0.58, 0.72, 0.19, 0.06])
    axapply     = plt.axes([0.78, 0.72, 0.19, 0.06])
    axmaskall   = plt.axes([0.58, 0.62, 0.19, 0.06])
    axapplyall  = plt.axes([0.78, 0.62, 0.19, 0.06])
    axremove    = plt.axes([0.58, 0.52, 0.19, 0.06])
    axrmramp    = plt.axes([0.78, 0.52, 0.19, 0.06])
    axremoveall = plt.axes([0.58, 0.42, 0.19, 0.06])
    axrmrampall = plt.axes([0.78, 0.42, 0.19, 0.06])
    axsave      = plt.axes([0.58, 0.32, 0.19, 0.06])
    axunwrap    = plt.axes([0.78, 0.32, 0.19, 0.06])
    axload      = plt.axes([0.58, 0.22, 0.19, 0.06])
    axunwrapall = plt.axes([0.78, 0.22, 0.19, 0.06])

    Button(axdraw,      "draw mask"       ).on_clicked(tracker.draw_mask)
    Button(axclose,     "close figure"    ).on_clicked(tracker.onclose)
    Button(axadd,       "add mask"        ).on_clicked(tracker.add_mask)
    Button(axapply,     "apply mask"      ).on_clicked(tracker.apply_mask)
    Button(axmaskall,   "mask all"        ).on_clicked(tracker.mask_all)
    Button(axapplyall,  "apply all masks" ).on_clicked(tracker.apply_all_masks)
    Button(axremove,    "remove mask"     ).on_clicked(tracker.remove_mask)
    Button(axrmramp,    "remove ramp"     ).on_clicked(tracker.remove_ramp)
    Button(axremoveall, "remove all mask" ).on_clicked(tracker.remove_all_mask)
    Button(axrmrampall, "remove all ramp" ).on_clicked(tracker.remove_rampall)
    Button(axsave,      "save masks"      ).on_clicked(tracker.save_masks)
    Button(axunwrap,    "unwrap"          ).on_clicked(tracker.unwrapping_phase)
    Button(axload,      "load masks"      ).on_clicked(tracker.load_masks)
    Button(axunwrapall, "unwrap all"      ).on_clicked(tracker.unwrapping_all)

    # --- Text boxes ---
    axboxprojn = plt.axes([0.125, 0.05, 0.1,  0.06])
    axboxvmin  = plt.axes([0.67,  0.05, 0.1,  0.06])
    axboxvmax  = plt.axes([0.87,  0.05, 0.1,  0.06])
    TextBox(axboxprojn, "Goto #", initial="1"   ).on_submit(tracker.submit)
    TextBox(axboxvmin,  "vmin",   initial="None").on_submit(tracker.cmvmin)
    TextBox(axboxvmax,  "vmax",   initial="None").on_submit(tracker.cmvmax)

    cmap_title = plt.axes([0.72, 0.14, 0.1, 0.06])
    cmap_title.set_axis_off()
    cmap_title.text(0, 0, "Colormap", fontsize=14)

    # --- Prev / Next / Play ---
    axprev = plt.axes([0.28, 0.05, 0.05, 0.06])
    axnext = plt.axes([0.35, 0.05, 0.05, 0.06])
    axplay = plt.axes([0.445, 0.05, 0.1,  0.06])
    Button(axprev, "<"    ).on_clicked(tracker.down)
    Button(axnext, ">"    ).on_clicked(tracker.up)
    Button(axplay, "play" ).on_clicked(tracker.play)

    fig.canvas.mpl_connect("scroll_event",    tracker.onscroll)
    fig.canvas.mpl_connect("key_press_event", tracker.key_event)
    MultiCursor(fig.canvas, (ax1, ax2), color="r", lw=1)

    return _setup_and_run(fig, tracker, "tracker.X1")


# ---------------------------------------------------------------------------
# Tracker classes
# ---------------------------------------------------------------------------

class PhaseTracker(object):
    """
    Widgets for interactive phase-ramp removal.

    This class is not normally instantiated directly — use
    :func:`gui_plotphase` instead.
    """

    def __init__(self, fig, ax1, ax2, X1, **params):
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        ax1.set_title(
            "Use scroll wheel or left/right arrows to navigate images"
        )

        self.X1 = X1.copy()
        if np.iscomplexobj(self.X1):
            raise ValueError("The input array must be real (phase values).")

        self.projs, self.rows, self.cols = X1.shape
        self.hcen = params.get("hcen", int(self.rows / 2.0))
        self.X2 = self.X1[:, self.hcen, :].copy()
        self.mask = np.zeros_like(X1, dtype=bool)
        self.ind = 0
        self.params = params
        self.vmin = params["vmin"]
        self.vmax = params["vmax"]
        self.colormap = params["colormap"]

        # Initialise canvas
        plt.ion()
        self.im1 = self.ax1.imshow(
            self.X1[self.ind],
            cmap=self.colormap,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        self.ax1.plot([1, self.cols - 1], [self.hcen, self.hcen], "b--")
        self.ax1.axis("tight")
        (self.im2,) = self.ax2.plot(self.X2[self.ind])
        self.ax2.plot([0, self.cols], [0, 0])
        self.pmin, self.pmax = self.ax2.get_ylim()
        self.ax2.set_ylim([2 * self.vmin, 2 * self.vmax])
        self.ax2.set_xlim([0, self.cols])
        self.update()
        print("Done. When finished, close the figure window to exit.")

    # ------------------------------------------------------------------ nav
    def cmvmin(self, val):
        """Set colormap vmin."""
        v = eval(val)
        if v >= self.vmax:
            print("vmin must be smaller than vmax.")
        else:
            self.vmin = v
            self.pmin = v
            self.update()

    def cmvmax(self, val):
        """Set colormap vmax."""
        v = eval(val)
        if v <= self.vmin:
            print("vmax must be larger than vmin.")
        else:
            self.vmax = v
            self.pmax = v
            self.update()

    def onscroll(self, event):
        """Scroll wheel navigation."""
        if event.button == "up":
            self.ind = np.clip(self.ind + 1, 0, self.projs - 1)
        else:
            self.ind = np.clip(self.ind - 1, 0, self.projs - 1)
        print("Projection {}".format(self.ind + 1))
        self.update()

    def key_event(self, event):
        """Left/right arrow key navigation."""
        if event.key == "right":
            self.ind = np.clip(self.ind + 1, 0, self.projs - 1)
        elif event.key == "left":
            self.ind = np.clip(self.ind - 1, 0, self.projs - 1)
        else:
            return
        print("Projection {}".format(self.ind + 1))
        self.update()

    def down(self, event):
        """Previous projection button."""
        self.ind = np.clip(self.ind - 1, 0, self.projs - 1)
        print("Projection {}".format(self.ind + 1))
        self.update()

    def up(self, event):
        """Next projection button."""
        self.ind = np.clip(self.ind + 1, 0, self.projs - 1)
        print("Projection {}".format(self.ind + 1))
        self.update()

    def submit(self, text):
        """Jump to a projection number via the text box."""
        self.ind = np.clip(eval(text) - 1, 0, self.projs - 1)
        print("Projection {}".format(self.ind + 1))
        self.update()

    def play(self, event):
        """Play through projections from the current one."""
        for ii in range(self.ind, self.projs):
            self.ind = ii
            print("Projection {}".format(self.ind + 1))
            self.update()

    # ------------------------------------------------------------------ mask
    def draw_mask(self, event):
        """Open a separate figure to draw a polygon mask."""
        print("\nDrawing polygon mask — left-click to add vertices, "
              "click first vertex (or press Enter) to finish.")
        self.img_mask = self.X1[self.ind] + self.mask[self.ind]
        fig_mask = plt.figure()
        ax_mask = fig_mask.add_subplot(111)
        ax_mask.imshow(
            self.img_mask, cmap=self.colormap,
            vmin=self.vmin, vmax=self.vmax,
        )
        # roipoly now wraps PolygonSelector; closes fig_mask on completion
        self.ROI_draw = roipoly(fig=fig_mask, ax=ax_mask)

    def add_mask(self, event):
        """Apply the drawn polygon to the current projection's mask."""
        print("\nAdding mask")
        self.mask[self.ind] |= self.ROI_draw.getMask(self.img_mask)
        # Pass ax1 explicitly so the overlay goes on the main image axes
        self.ROI_draw.displayROI(ax=self.ax1)
        self.update()

    def mask_all(self, event):
        """Copy the current polygon to every projection."""
        print("\nCopying mask to all projections — please wait...")
        mask = self.ROI_draw.getMask(self.img_mask)
        self.mask |= np.broadcast_to(mask, self.mask.shape).copy()
        print("Done")
        self.update()

    def remove_mask(self, event):
        """Remove the drawn polygon from the current projection's mask."""
        print("\nRemoving mask from current projection")
        self.mask[self.ind] &= ~self.ROI_draw.getMask(self.img_mask)
        self.update()

    def remove_all_mask(self, event):
        """Remove the drawn polygon from every projection's mask."""
        print("\nRemoving mask from all projections — please wait...")
        mask = self.ROI_draw.getMask(self.img_mask)
        self.mask &= ~np.broadcast_to(mask, self.mask.shape).copy()
        print("Done")
        self.update()

    # ------------------------------------------------------------------ correction
    def apply_mask(self, event):
        """Remove phase ramp from the current projection using its mask."""
        print("\nApplying phase ramp correction to projection", self.ind + 1)
        self.X1[self.ind] = _removing_phaseramp(
            self.X1[self.ind], self.mask[self.ind]
        )
        self.X2[self.ind] = self.X1[self.ind, self.hcen, :].copy()
        self.update()

    def apply_all_masks(self, event):
        """Remove phase ramp from all projections using their masks."""
        print("\nApplying phase ramp correction to all projections")
        for ii in tqdm(range(self.projs), desc="Applying phase correction"):
            self.ind = ii
            self.X1[ii] = _removing_phaseramp(
                self.X1[ii], self.mask[ii]
            )
            self.X2[ii] = self.X1[ii, self.hcen, :].copy()
            self.update()
        print("Done")

    def remove_ramp(self, event):
        """Remove linear phase ramp from the current projection (no mask)."""
        print("\nRemoving ramp from projection", self.ind + 1)
        self.X1[self.ind] = np.angle(
            rmphaseramp(np.exp(1j * self.X1[self.ind]), weight=None)
        )
        self.X2[self.ind] = self.X1[
            self.ind, int(self.X1.shape[1] // 2), :
        ].copy()
        self.update()

    def remove_rampall(self, event):
        """Remove linear phase ramp from all projections (no mask)."""
        print("\nRemoving ramp from all projections")
        for ii in tqdm(range(self.projs), desc="Removing phase ramp"):
            self.ind = ii
            self.X1[ii] = np.angle(
                rmphaseramp(np.exp(1j * self.X1[ii]), weight=None)
            )
            self.X2[ii] = self.X1[
                ii, int(self.X1.shape[1] // 2), :
            ].copy()
            self.update()
        print("Done")

    def unwrapping_phase(self, event):
        """Unwrap phase of the current projection."""
        print(f"\nUnwrapping projection {self.ind + 1}")
        self.X1[self.ind] = _unwrapping_phase(
            self.X1[self.ind], self.mask[self.ind]
        )
        self.X2[self.ind] = self.X1[self.ind, self.hcen, :].copy()
        self.update()

    def unwrapping_all(self, event):
        """Unwrap phase of all projections."""
        print("\nUnwrapping all projections")
        for ii in tqdm(range(self.projs), desc="Unwrapping projections"):
            self.ind = ii
            self.X1[ii] = _unwrapping_phase(
                self.X1[ii], self.mask[ii]
            )
            self.X2[ii] = self.X1[ii, self.hcen, :].copy()
            self.update()
        print("Done")

    # ------------------------------------------------------------------ I/O
    def load_masks(self, event):
        """Load masks from ``masks.h5``."""
        print("\nLoading masks from file")
        self.mask = LoadData.loadmasks("masks.h5", **self.params)
        self.update()

    def save_masks(self, event):
        """Save masks to ``masks.h5``."""
        print("\nSaving masks to file")
        SaveData.savemasks("masks.h5", self.mask, **self.params)

    # ------------------------------------------------------------------ display
    def update(self):
        """Refresh the canvas after any state change.

        Uses the synchronous ``canvas.draw()`` rather than
        ``draw_idle()`` so that the browser receives the updated frame
        immediately — critical for ipympl in Jupyter where the async
        variant may not fire before the cell finishes executing.
        """
        self.im1.set_data(self.X1[self.ind] + self.mask[self.ind])
        self.im1.set_clim(self.vmin, self.vmax)
        self.im2.set_ydata(self.X2[self.ind])
        self.im2.axes.set_ylim([self.pmin, self.pmax])
        self.im2.axes.set_xlim([0, self.cols])
        self.ax1.set_ylabel("Projection {}".format(self.ind + 1))
        self.ax2.set_ylabel("Projection {}".format(self.ind + 1))
        self.ax1.figure.canvas.draw()
        self.ax2.figure.canvas.draw()

    def onclose(self, event):
        """Close the GUI figure."""
        print("\nFigure closed.")
        plt.close(event.canvas.figure)


class AmpTracker(PhaseTracker):
    """
    Widgets for air removal from amplitude projections.

    Inherits navigation, mask drawing, I/O, and display from
    :class:`PhaseTracker`.  Overrides the correction methods to apply
    :func:`~toupy.restoration.ramptools.rmair` followed by a logarithm
    instead of phase-ramp removal.
    """

    def __init__(self, fig, ax1, ax2, X1, **params):
        super().__init__(fig, ax1, ax2, X1, **params)
        self.done = []  # projections that have already been corrected

    def apply_mask(self, event):
        """Normalise air, apply log; guards against double-correction."""
        print("\nApplying air correction + log to projection", self.ind + 1)
        if self.ind in self.done:
            print("  (already corrected — skipping)")
        else:
            imgin = self.X1[self.ind].copy()
            mask  = self.mask[self.ind].copy()
            self.X1[self.ind] = np.log(rmair(imgin, mask))
            self.X2[self.ind] = self.X1[
                self.ind, int(self.X1.shape[1] // 2), :
            ].copy()
            self.done.append(self.ind)
            self.vmin = -0.5
            self.vmax =  0.1
        self.update()

    def apply_all_masks(self, event):
        """Normalise air + log for all uncorrected projections."""
        print("\nApplying air correction + log to all projections")
        for ii in tqdm(range(self.projs), desc="Applying air correction"):
            self.ind = ii
            if self.ind in self.done:
                print(f"\r  Projection {self.ind + 1} already corrected — skipping",
                      end="")
            else:
                imgin = self.X1[ii].copy()
                mask  = self.mask[ii].copy()
                self.X1[ii] = np.log(rmair(imgin, mask))
                self.X2[ii] = self.X1[
                    ii, int(self.X1.shape[1] // 2), :
                ].copy()
                self.done.append(self.ind)
            self.update()
        print("\nDone")
