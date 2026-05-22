#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third party packages
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.path as mplPath
from ..utils.plot_utils import plt
from matplotlib.widgets import Button, MultiCursor, PolygonSelector, TextBox
from ..utils import tqdm
import numpy as np
from numpy.fft import fftfreq

from skimage.restoration import unwrap_phase

# local packages
from ..io.dataio import LoadData, SaveData
from .ramptools import rmphaseramp, rmair
from ..utils import isnotebook

__all__ = ["gui_plotamp", "gui_plotphase", "AmpTracker", "PhaseTracker",
           "make_air_mask"]


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


def _is_interactive_notebook():
    """
    Return True when running inside Jupyter with an *interactive* backend.

    Covers both the built-in ``%matplotlib notebook`` (NbAgg, no extra
    package required) and ``%matplotlib widget`` (ipympl).  Returns
    False for the static ``%matplotlib inline`` backend and for
    non-notebook environments.
    """
    if not isnotebook():
        return False
    try:
        import matplotlib
        return "inline" not in matplotlib.get_backend().lower()
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
    # IMPORTANT: Button and TextBox objects must be kept alive for their
    # callbacks to remain active.  Matplotlib's CallbackRegistry stores
    # bound methods via WeakMethod, so any widget that is not referenced
    # elsewhere is immediately garbage-collected and its click/submit
    # handlers silently stop working.  We store all widgets on the
    # tracker so they live as long as the tracker does.
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

    bdraw      = Button(axdraw,      "draw mask"       ); bdraw.on_clicked(tracker.draw_mask)
    bclose     = Button(axclose,     "close figure"    ); bclose.on_clicked(tracker.onclose)
    badd       = Button(axadd,       "add mask"        ); badd.on_clicked(tracker.add_mask)
    bapply     = Button(axapply,     "apply mask"      ); bapply.on_clicked(tracker.apply_mask)
    bmaskall   = Button(axmaskall,   "mask all"        ); bmaskall.on_clicked(tracker.mask_all)
    bapplyall  = Button(axapplyall,  "apply all masks" ); bapplyall.on_clicked(tracker.apply_all_masks)
    bremove    = Button(axremove,    "remove mask"     ); bremove.on_clicked(tracker.remove_mask)
    bremoveall = Button(axremoveall, "remove all mask" ); bremoveall.on_clicked(tracker.remove_all_mask)
    bsave      = Button(axsave,      "save masks"      ); bsave.on_clicked(tracker.save_masks)
    bload      = Button(axload,      "load masks"      ); bload.on_clicked(tracker.load_masks)

    # --- Text boxes ---
    axboxprojn = plt.axes([0.125, 0.05, 0.1,  0.06])
    axboxvmin  = plt.axes([0.67,  0.05, 0.1,  0.06])
    axboxvmax  = plt.axes([0.87,  0.05, 0.1,  0.06])
    tbprojn = TextBox(axboxprojn, "Goto #", initial="1"   ); tbprojn.on_submit(tracker.submit)
    tbvmin  = TextBox(axboxvmin,  "vmin",   initial="None"); tbvmin.on_submit(tracker.cmvmin)
    tbvmax  = TextBox(axboxvmax,  "vmax",   initial="None"); tbvmax.on_submit(tracker.cmvmax)

    cmap_title = plt.axes([0.72, 0.14, 0.1, 0.06])
    cmap_title.set_axis_off()
    cmap_title.text(0, 0, "Colormap", fontsize=14)

    # --- Prev / Next / Play / Stop ---
    axprev = plt.axes([0.28, 0.05, 0.05, 0.06])
    axnext = plt.axes([0.35, 0.05, 0.05, 0.06])
    axplay = plt.axes([0.42, 0.05, 0.08, 0.06])
    axstop = plt.axes([0.51, 0.05, 0.08, 0.06])
    bprev = Button(axprev, "<"   ); bprev.on_clicked(tracker.down)
    bnext = Button(axnext, ">"   ); bnext.on_clicked(tracker.up)
    bplay = Button(axplay, "play"); bplay.on_clicked(tracker.play)
    bstop = Button(axstop, "stop"); bstop.on_clicked(tracker.stop_play)

    # Store all widget references on the tracker to prevent GC
    tracker._widgets = [
        bdraw, bclose, badd, bapply, bmaskall, bapplyall,
        bremove, bremoveall, bsave, bload,
        tbprojn, tbvmin, tbvmax, bprev, bnext, bplay, bstop,
    ]

    fig.canvas.mpl_connect("scroll_event",    tracker.onscroll)
    fig.canvas.mpl_connect("key_press_event", tracker.key_event)
    tracker._multicursor = MultiCursor(fig.canvas, (ax1, ax2), color="r", lw=1)

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
    # IMPORTANT: Button and TextBox objects must be kept alive for their
    # callbacks to remain active.  Matplotlib's CallbackRegistry stores
    # bound methods via WeakMethod, so any widget that is not referenced
    # elsewhere is immediately garbage-collected and its click/submit
    # handlers silently stop working.  We store all widgets on the
    # tracker so they live as long as the tracker does.
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

    bdraw      = Button(axdraw,      "draw mask"      ); bdraw.on_clicked(tracker.draw_mask)
    bclose     = Button(axclose,     "close figure"   ); bclose.on_clicked(tracker.onclose)
    badd       = Button(axadd,       "add mask"       ); badd.on_clicked(tracker.add_mask)
    bapply     = Button(axapply,     "apply mask"     ); bapply.on_clicked(tracker.apply_mask)
    bmaskall   = Button(axmaskall,   "mask all"       ); bmaskall.on_clicked(tracker.mask_all)
    bapplyall  = Button(axapplyall,  "apply all masks"); bapplyall.on_clicked(tracker.apply_all_masks)
    bremove    = Button(axremove,    "remove mask"    ); bremove.on_clicked(tracker.remove_mask)
    brmramp    = Button(axrmramp,    "remove ramp"    ); brmramp.on_clicked(tracker.remove_ramp)
    bremoveall = Button(axremoveall, "remove all mask"); bremoveall.on_clicked(tracker.remove_all_mask)
    brmrampall = Button(axrmrampall, "remove all ramp"); brmrampall.on_clicked(tracker.remove_rampall)
    bsave      = Button(axsave,      "save masks"     ); bsave.on_clicked(tracker.save_masks)
    bunwrap    = Button(axunwrap,    "unwrap"         ); bunwrap.on_clicked(tracker.unwrapping_phase)
    bload      = Button(axload,      "load masks"     ); bload.on_clicked(tracker.load_masks)
    bunwrapall = Button(axunwrapall, "unwrap all"     ); bunwrapall.on_clicked(tracker.unwrapping_all)

    # --- Text boxes ---
    axboxprojn = plt.axes([0.125, 0.05, 0.1,  0.06])
    axboxvmin  = plt.axes([0.67,  0.05, 0.1,  0.06])
    axboxvmax  = plt.axes([0.87,  0.05, 0.1,  0.06])
    tbprojn = TextBox(axboxprojn, "Goto #", initial="1"   ); tbprojn.on_submit(tracker.submit)
    tbvmin  = TextBox(axboxvmin,  "vmin",   initial="None"); tbvmin.on_submit(tracker.cmvmin)
    tbvmax  = TextBox(axboxvmax,  "vmax",   initial="None"); tbvmax.on_submit(tracker.cmvmax)

    cmap_title = plt.axes([0.72, 0.14, 0.1, 0.06])
    cmap_title.set_axis_off()
    cmap_title.text(0, 0, "Colormap", fontsize=14)

    # --- Prev / Next / Play / Stop ---
    axprev = plt.axes([0.28, 0.05, 0.05, 0.06])
    axnext = plt.axes([0.35, 0.05, 0.05, 0.06])
    axplay = plt.axes([0.42, 0.05, 0.08, 0.06])
    axstop = plt.axes([0.51, 0.05, 0.08, 0.06])
    bprev = Button(axprev, "<"   ); bprev.on_clicked(tracker.down)
    bnext = Button(axnext, ">"   ); bnext.on_clicked(tracker.up)
    bplay = Button(axplay, "play"); bplay.on_clicked(tracker.play)
    bstop = Button(axstop, "stop"); bstop.on_clicked(tracker.stop_play)

    # Store all widget references on the tracker to prevent GC
    tracker._widgets = [
        bdraw, bclose, badd, bapply, bmaskall, bapplyall,
        bremove, brmramp, bremoveall, brmrampall,
        bsave, bunwrap, bload, bunwrapall,
        tbprojn, tbvmin, tbvmax, bprev, bnext, bplay, bstop,
    ]

    fig.canvas.mpl_connect("scroll_event",    tracker.onscroll)
    fig.canvas.mpl_connect("key_press_event", tracker.key_event)
    tracker._multicursor = MultiCursor(fig.canvas, (ax1, ax2), color="r", lw=1)

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
        # Centered hint line at the top of the *figure* (not ax1).
        # ax1.set_title() would centre over ax1 only (~60 % of the figure
        # width), making the text appear left-shifted.  fig.text() with
        # x=0.5 centres it over the full figure canvas.
        fig.text(
            0.5, 0.975,
            "Scroll wheel or ← / → arrows to navigate   |   "
            "< / > buttons or 'Goto #' to jump",
            ha="center", va="top", fontsize=9, color="0.45",
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
        # Polygon state — populated by draw_mask / _on_poly_select
        self._poly_verts = []
        self._poly_selector = None
        # Play-loop stop flag — set True by stop_play(); checked each frame
        self._stop_play = False
        self.vmin = params["vmin"]
        self.vmax = params["vmax"]
        self.colormap = params["colormap"]

        # Initialise canvas.
        # Do NOT call plt.ion() here: with ipympl it installs a REPL
        # display hook that fires draw_all() at unpredictable times,
        # causing scroll/key callbacks to trigger on startup (appearing
        # as an automatic play-through of all projections).
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
        """Play through all projections from the current index.

        Click **stop** (or press the stop button) to halt the animation at
        the current frame.

        Each frame is rendered synchronously so the animation is actually
        visible.  ``draw_idle()`` (used by ``update()``) schedules an
        async repaint that never fires inside a tight Python loop; we
        force a synchronous ``draw()`` + ``flush_events()`` after each
        frame to push the data to the screen.  ``flush_events()`` also
        pumps the GUI event queue so the stop button callback can fire
        mid-loop.
        """
        self._stop_play = False
        print("Playing from projection {} …  (click 'stop' to halt)".format(self.ind + 1),
              flush=True)
        canvas = self.ax1.figure.canvas
        for ii in range(self.ind, self.projs):
            if self._stop_play:
                print("Stopped at projection {}.".format(self.ind + 1), flush=True)
                return
            self.ind = ii
            self.update()
            canvas.draw()           # synchronous render
            canvas.flush_events()   # process GUI events so the frame shows
        print("Play finished at projection {}.".format(self.ind + 1), flush=True)

    def stop_play(self, event):
        """Request that the play loop stop at the current frame."""
        self._stop_play = True

    # ------------------------------------------------------------------ mask
    def draw_mask(self, event):
        """
        Attach a PolygonSelector to the main image axes for mask drawing.

        Left-click to add vertices; click the first vertex again (or press
        **Enter** in matplotlib ≥ 3.7) to close and finalise the polygon.
        Then click **add mask** (or another mask button) to apply it.

        The selector is drawn directly on ``ax1`` — no separate figure is
        opened.  Any previously unfinished selector is discarded first.
        """
        print("\nDrawing polygon mask directly on the image — "
              "left-click to add vertices, click first vertex (or Enter) to finish.\n"
              "When done, click 'add mask' to apply.")
        # Discard any unfinished selector from a previous call
        if self._poly_selector is not None:
            try:
                self._poly_selector.disconnect_events()
            except Exception:
                pass
        self._poly_verts = []
        _props = dict(color="r", linewidth=1.5, alpha=0.8)
        try:
            self._poly_selector = PolygonSelector(
                self.ax1, self._on_poly_select, props=_props
            )
        except TypeError:               # matplotlib < 3.5
            self._poly_selector = PolygonSelector(
                self.ax1, self._on_poly_select, lineprops=_props
            )
        self.ax1.figure.canvas.draw_idle()

    def _on_poly_select(self, verts):
        """Store vertices when the PolygonSelector polygon is finalised."""
        self._poly_verts = list(verts)
        self._poly_selector.disconnect_events()
        self._poly_selector = None
        self.ax1.figure.canvas.draw_idle()
        print(f"Polygon with {len(self._poly_verts)} vertices recorded — "
              "click 'add mask' (or another mask button) to apply.")

    def _get_roi_mask(self):
        """
        Rasterise the last completed polygon onto the current image grid.

        Returns
        -------
        ndarray of bool, shape (rows, cols)
            ``True`` for pixels inside the polygon.  All-False if no
            polygon has been drawn yet.
        """
        ny, nx = self.rows, self.cols
        if len(self._poly_verts) < 3:
            print("No completed polygon — use 'draw mask' first.")
            return np.zeros((ny, nx), dtype=bool)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.ravel(), y.ravel())).T
        path = mplPath.Path(self._poly_verts)
        return path.contains_points(points).reshape(ny, nx)

    def add_mask(self, event):
        """Apply the drawn polygon to the current projection's mask."""
        roi = self._get_roi_mask()
        print("Adding mask to projection {} …".format(self.ind + 1), flush=True)
        self.mask[self.ind] |= roi
        # Overlay the closed polygon outline on ax1
        if len(self._poly_verts) >= 2:
            xs = [v[0] for v in self._poly_verts] + [self._poly_verts[0][0]]
            ys = [v[1] for v in self._poly_verts] + [self._poly_verts[0][1]]
            self.ax1.add_line(plt.Line2D(xs, ys, color="r"))
        self.update()
        print("→ done  ({} masked pixels).".format(int(roi.sum())), flush=True)

    def mask_all(self, event):
        """Copy the current polygon to every projection."""
        print("Copying mask to all {} projections …".format(self.projs), flush=True)
        mask = self._get_roi_mask()
        self.mask |= np.broadcast_to(mask, self.mask.shape).copy()
        self.update()
        print("→ done.", flush=True)

    def remove_mask(self, event):
        """Remove the drawn polygon from the current projection's mask."""
        print("Removing mask from projection {} …".format(self.ind + 1), flush=True)
        self.mask[self.ind] &= ~self._get_roi_mask()
        self.update()
        print("→ done.", flush=True)

    def remove_all_mask(self, event):
        """Remove the drawn polygon from every projection's mask."""
        print("Removing mask from all {} projections …".format(self.projs), flush=True)
        mask = self._get_roi_mask()
        self.mask &= ~np.broadcast_to(mask, self.mask.shape).copy()
        self.update()
        print("→ done.", flush=True)

    # ------------------------------------------------------------------ correction
    def apply_mask(self, event):
        """Remove phase ramp from the current projection using its mask."""
        print("Applying phase ramp correction to projection {} …".format(self.ind + 1),
              flush=True)
        self.X1[self.ind] = _removing_phaseramp(
            self.X1[self.ind], self.mask[self.ind]
        )
        self.X2[self.ind] = self.X1[self.ind, self.hcen, :].copy()
        self.update()
        print("→ done.", flush=True)

    def apply_all_masks(self, event):
        """Remove phase ramp from all projections using their masks."""
        print("Applying phase ramp correction to all {} projections:".format(self.projs))
        for ii in tqdm(range(self.projs), desc="  phase ramp removal", unit="proj"):
            self.X1[ii] = _removing_phaseramp(self.X1[ii], self.mask[ii])
            self.X2[ii] = self.X1[ii, self.hcen, :].copy()
        # Single canvas refresh at the end — draw_idle() inside the loop
        # never fires because the Python thread never yields to the event loop.
        self.ind = self.projs - 1
        self.update()
        print("All projections corrected.")

    def remove_ramp(self, event):
        """Remove linear phase ramp from the current projection (no mask)."""
        print("Removing ramp from projection {} …".format(self.ind + 1), flush=True)
        self.X1[self.ind] = np.angle(
            rmphaseramp(np.exp(1j * self.X1[self.ind]), weight=None)
        )
        self.X2[self.ind] = self.X1[self.ind, int(self.X1.shape[1] // 2), :].copy()
        self.update()
        print("→ done.", flush=True)

    def remove_rampall(self, event):
        """Remove linear phase ramp from all projections (no mask)."""
        print("Removing ramp from all {} projections:".format(self.projs))
        for ii in tqdm(range(self.projs), desc="  ramp removal", unit="proj"):
            self.X1[ii] = np.angle(
                rmphaseramp(np.exp(1j * self.X1[ii]), weight=None)
            )
            self.X2[ii] = self.X1[ii, int(self.X1.shape[1] // 2), :].copy()
        self.ind = self.projs - 1
        self.update()
        print("All ramps removed.")

    def unwrapping_phase(self, event):
        """Unwrap phase of the current projection."""
        print("Unwrapping projection {} …".format(self.ind + 1), flush=True)
        self.X1[self.ind] = _unwrapping_phase(
            self.X1[self.ind], self.mask[self.ind]
        )
        self.X2[self.ind] = self.X1[self.ind, self.hcen, :].copy()
        self.update()
        print("→ done.", flush=True)

    def unwrapping_all(self, event):
        """Unwrap phase of all projections."""
        print("Unwrapping all {} projections:".format(self.projs))
        for ii in tqdm(range(self.projs), desc="  phase unwrapping", unit="proj"):
            self.X1[ii] = _unwrapping_phase(self.X1[ii], self.mask[ii])
            self.X2[ii] = self.X1[ii, self.hcen, :].copy()
        self.ind = self.projs - 1
        self.update()
        print("All projections unwrapped.")

    # ------------------------------------------------------------------ I/O
    def load_masks(self, event):
        """Load masks from ``masks.h5``."""
        print("Loading masks from masks.h5 …", flush=True)
        self.mask = LoadData.loadmasks("masks.h5", **self.params)
        self.update()
        print("→ done.", flush=True)

    def save_masks(self, event):
        """Save masks to ``masks.h5``."""
        print("Saving masks to masks.h5 …", flush=True)
        SaveData.savemasks("masks.h5", self.mask, **self.params)
        print("→ done.", flush=True)

    # ------------------------------------------------------------------ display
    def update(self):
        """Refresh the canvas after any state change."""
        self.im1.set_data(self.X1[self.ind] + self.mask[self.ind])
        self.im1.set_clim(self.vmin, self.vmax)
        self.im2.set_ydata(self.X2[self.ind])
        self.im2.axes.set_ylim([self.pmin, self.pmax])
        self.im2.axes.set_xlim([0, self.cols])
        self.ax1.set_ylabel("Projection {}".format(self.ind + 1))
        self.ax2.set_ylabel("Projection {}".format(self.ind + 1))
        self.ax1.figure.canvas.draw_idle()
        self.ax2.figure.canvas.draw_idle()

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
        if self.ind in self.done:
            print("Projection {} already corrected — skipping.".format(self.ind + 1), flush=True)
        else:
            print("Applying air correction + log to projection {} …".format(self.ind + 1),
                  flush=True)
            imgin = self.X1[self.ind].copy()
            mask  = self.mask[self.ind].copy()
            self.X1[self.ind] = np.log(rmair(imgin, mask))
            self.X2[self.ind] = self.X1[self.ind, int(self.X1.shape[1] // 2), :].copy()
            self.done.append(self.ind)
            self.vmin = -0.5
            self.vmax =  0.1
            print("→ done.", flush=True)
        self.update()

    def apply_all_masks(self, event):
        """Normalise air + log for all uncorrected projections."""
        already = [i for i in range(self.projs) if i in self.done]
        todo    = [i for i in range(self.projs) if i not in self.done]
        if already:
            print("  ({} projection(s) already corrected — skipping them)".format(len(already)))
        print("Applying air correction + log to {} projection(s):".format(len(todo)))
        for ii in tqdm(todo, desc="  air correction", unit="proj"):
            imgin = self.X1[ii].copy()
            mask  = self.mask[ii].copy()
            self.X1[ii] = np.log(rmair(imgin, mask))
            self.X2[ii] = self.X1[ii, int(self.X1.shape[1] // 2), :].copy()
            self.done.append(ii)
        self.vmin = -0.5
        self.vmax =  0.1
        self.ind = self.projs - 1
        self.update()
        print("All projections corrected.")


# ---------------------------------------------------------------------------
# Lightweight notebook-friendly mask helper
# ---------------------------------------------------------------------------

class _MaskPainter:
    """
    Internal state holder for :func:`make_air_mask`.

    Not intended for direct use — call :func:`make_air_mask` instead.

    Vertices are collected via a raw ``button_press_event`` listener on the
    figure canvas, **not** via :class:`~matplotlib.widgets.PolygonSelector`.
    ``PolygonSelector`` relies on keyboard events (Enter) and internal state
    that changed across matplotlib versions; neither is reliable in Jupyter
    where keyboard focus stays in the cell.  A plain canvas click listener
    is routed through ipympl's comm layer without any of those issues.

    Attributes
    ----------
    mask : ndarray of bool, shape (ny, nx)
        Cumulative boolean mask — union of every region added with
        **Add region**.  All-False until at least one region is added.
    n_regions : int
        Number of polygon regions added so far.
    """

    def __init__(self, image, cmap="gray", vmin=None, vmax=None,
                 figsize=(8, 7)):
        self.image = np.asarray(image, dtype=float)
        ny, nx = self.image.shape[:2]
        self._ny, self._nx = ny, nx
        self.mask = np.zeros((ny, nx), dtype=bool)
        self.n_regions = 0

        # Vertices for the polygon currently being drawn
        self._xs = []
        self._ys = []
        self._live_line = None   # Line2D for the in-progress polygon

        fig, ax = plt.subplots(figsize=figsize)
        self.fig = fig
        self.ax  = ax

        ax.imshow(self.image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.axis("tight")
        fig.text(
            0.5, 0.99,
            "Left-click on image to add vertices  |  "
            "'Add region' to store polygon  |  'Finish' when done",
            ha="center", va="top", fontsize=9, color="0.4",
        )

        # Raw canvas event — works identically in terminal and Jupyter/ipympl.
        # Store the connection id so we can disconnect cleanly in _on_finish.
        self._cid = fig.canvas.mpl_connect(
            "button_press_event", self._on_click)

        # Two buttons — stored on self to prevent GC
        ax_add    = fig.add_axes([0.25, 0.01, 0.25, 0.055])
        ax_finish = fig.add_axes([0.55, 0.01, 0.2,  0.055])
        self._btn_add    = Button(ax_add,    "Add region")
        self._btn_finish = Button(ax_finish, "Finish")
        self._btn_add.on_clicked(self._on_add)
        self._btn_finish.on_clicked(self._on_finish)

    # ------------------------------------------------------------------
    def _on_click(self, event):
        """
        Left-click inside the image axes: record a vertex.

        Button clicks (which are also mouse events) arrive here too, but
        their ``inaxes`` is the button's own axes object, not ``self.ax``,
        so the guard below ignores them automatically.
        """
        if event.button != 1:
            return
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._xs.append(event.xdata)
        self._ys.append(event.ydata)
        self._redraw_live()
        print("  vertex {}: ({:.1f}, {:.1f})".format(
            len(self._xs), event.xdata, event.ydata), flush=True)

    # ------------------------------------------------------------------
    def _redraw_live(self):
        """Redraw the in-progress polygon after each new vertex."""
        if self._live_line is not None:
            try:
                self._live_line.remove()
            except Exception:
                pass
            self._live_line = None
        n = len(self._xs)
        if n == 0:
            self.ax.figure.canvas.draw_idle()
            return
        if n == 1:
            self._live_line, = self.ax.plot(
                self._xs, self._ys, "ro", markersize=6, zorder=5)
        else:
            # Close the polygon visually once ≥ 3 points exist
            xs = self._xs + [self._xs[0]] if n >= 3 else list(self._xs)
            ys = self._ys + [self._ys[0]] if n >= 3 else list(self._ys)
            self._live_line, = self.ax.plot(
                xs, ys, "r-o", linewidth=1.5, markersize=4,
                alpha=0.8, zorder=5)
        self.ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _rasterise(self, verts):
        """Rasterise *verts* onto the image grid; return a bool mask."""
        x, y = np.meshgrid(np.arange(self._nx), np.arange(self._ny))
        pts  = np.vstack((x.ravel(), y.ravel())).T
        path = mplPath.Path(verts)
        return path.contains_points(pts).reshape(self._ny, self._nx)

    # ------------------------------------------------------------------
    def _on_add(self, event):
        """
        Commit the current polygon to the cumulative mask.

        Leaves a permanent dashed outline on the image, clears the live
        drawing, and resets the vertex list so the next polygon can be
        drawn immediately.
        """
        verts = list(zip(self._xs, self._ys))
        if len(verts) < 3:
            print("Need ≥ 3 vertices — keep clicking on the image.",
                  flush=True)
            return
        region = self._rasterise(verts)
        self.mask |= region
        self.n_regions += 1
        # Remove the live line and replace with a permanent dashed outline
        if self._live_line is not None:
            try:
                self._live_line.remove()
            except Exception:
                pass
            self._live_line = None
        xs_closed = self._xs + [self._xs[0]]
        ys_closed = self._ys + [self._ys[0]]
        self.ax.add_line(plt.Line2D(
            xs_closed, ys_closed, color="r", linewidth=1.5, linestyle="--"))
        self.ax.figure.canvas.draw_idle()
        print(
            "Region {} added: {} px.  Total mask: {} px.  "
            "Draw next region or click 'Finish'.".format(
                self.n_regions, int(region.sum()), int(self.mask.sum())),
            flush=True,
        )
        # Reset for the next polygon
        self._xs = []
        self._ys = []

    # ------------------------------------------------------------------
    def _on_finish(self, event):
        """
        Finalise the mask and close the figure.

        Any in-progress polygon (≥ 3 vertices, not yet added) is committed
        automatically so the user does not have to click 'Add region' last.
        """
        verts = list(zip(self._xs, self._ys))
        if len(verts) >= 3:
            region = self._rasterise(verts)
            self.mask |= region
            self.n_regions += 1
            print("Auto-added pending region {}: {} px.".format(
                self.n_regions, int(region.sum())), flush=True)
        # Disconnect the click listener before closing
        try:
            self.fig.canvas.mpl_disconnect(self._cid)
        except Exception:
            pass
        print(
            "Mask finalised: {} region(s), {} pixels total.  "
            "Access via  painter.mask".format(
                self.n_regions, int(self.mask.sum())),
            flush=True,
        )
        plt.close(self.fig)


def make_air_mask(image, cmap="gray", vmin=None, vmax=None, figsize=(8, 7)):
    """
    Interactively draw one or more polygon regions on *image* and return
    a cumulative boolean mask.

    Opens a figure with the image and two buttons:

    * **Add region** — commits the current polygon to the mask; a
      permanent dashed outline marks it; you can immediately draw the
      next polygon.
    * **Finish** — closes the figure (any in-progress polygon with ≥ 3
      vertices is committed automatically).

    Draw each polygon by **left-clicking** vertices directly on the image.
    The polygon closes visually once you have ≥ 3 points.  Click
    **Add region** to store it, then draw the next one.  No Enter key or
    precise "click first vertex to close" is needed.

    This is the lightweight notebook alternative to the full
    :func:`gui_plotphase` / :func:`gui_plotamp` GUI (which is better
    suited to terminal use).

    .. important::

       **Jupyter notebook — two-cell workflow.**
       ``make_air_mask`` returns *immediately* (non-blocking) so that the
       figure widget can be displayed.  Do **not** put ``painter.mask``
       access in the same cell — the cell will have finished executing
       before you interact with the figure, and the mask will be empty.
       Always use two separate cells::

           # Cell 1 — show the figure and draw
           painter = make_air_mask(stack[0], vmin=-1.6, vmax=1.6)

           # Cell 2 — run AFTER clicking Finish in the figure above
           air_mask = painter.mask.copy()

    Backend requirements
    --------------------
    * **Terminal / IPython** — works with any backend; blocks until the
      figure is closed.
    * **JupyterLab** — requires ``%matplotlib widget``
      (``pip install ipympl``).  Put it as the first cell of your
      notebook and restart the kernel.
    * **Classic Jupyter Notebook** — ``%matplotlib widget`` (preferred)
      or ``%matplotlib notebook`` (built-in, no install).
    * **``%matplotlib inline``** — not interactive; a
      :class:`UserWarning` explains how to switch.

    Parameters
    ----------
    image : array_like, shape (ny, nx)
        2-D image to display (e.g. one slice from the projection stack).
    cmap : str, optional
        Colourmap name.  Default ``'gray'``.
    vmin, vmax : float or None, optional
        Colormap limits.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches.  Default ``(8, 7)``.

    Returns
    -------
    painter : _MaskPainter
        Object whose ``.mask`` attribute (``ndarray`` of bool, shape
        ``(ny, nx)``) is the union of all added regions.
        ``.n_regions`` counts how many polygons were added.

    Examples
    --------
    **Terminal / IPython** (single cell, blocks until Finish is clicked)::

        from toupy.restoration import make_air_mask, rmphaseramp
        import numpy as np

        painter = make_air_mask(stack[0], vmin=-1.6, vmax=1.6)
        air_mask = painter.mask.copy()   # available immediately after return

        corrected = np.stack([
            np.angle(rmphaseramp(np.exp(1j * proj),
                                 weight=air_mask, zero_air_phase=True))
            for proj in stack
        ])

    **JupyterLab** (``%matplotlib widget`` + ``pip install ipympl`` once,
    **two separate cells**)::

        # ── Cell 1 ── draw the mask (returns immediately)
        painter = make_air_mask(stack[0], vmin=-1.6, vmax=1.6)

        # ── Cell 2 ── run after clicking Finish in the figure above
        air_mask = painter.mask.copy()
    """
    painter = _MaskPainter(image, cmap=cmap, vmin=vmin, vmax=vmax,
                           figsize=figsize)

    if isnotebook():
        if _is_interactive_notebook():
            plt.show(block=False)
            # draw() must come *after* plt.show(): the WebAgg/ipympl backend
            # creates canvas.manager only when plt.show() is called; calling
            # draw() before that raises AttributeError: 'NoneType'.refresh_all
            painter.fig.canvas.draw()
            print(
                "Left-click on the image to add vertices — polygon closes at 3+.\n"
                "Click 'Add region' to store it, then draw the next one.\n"
                "Click 'Finish' when done.\n"
                "\n"
                "IMPORTANT: access the mask in the NEXT cell (not this one):\n"
                "    air_mask = painter.mask.copy()",
                flush=True,
            )
        else:
            from IPython import display as ipy_display
            ipy_display.display(painter.fig)
            warnings.warn(
                "Interactive mask drawing requires a non-inline backend.\n"
                "For JupyterLab add  %matplotlib widget  as the first cell\n"
                "and run  pip install ipympl  once, then restart the kernel.\n"
                "For classic Jupyter Notebook you can also use "
                "%matplotlib notebook  (no install needed).",
                UserWarning,
                stacklevel=2,
            )
    else:
        plt.show(block=True)

    return painter
