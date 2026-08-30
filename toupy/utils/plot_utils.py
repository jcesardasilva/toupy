#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import functools
import io as _io
import re
import sys

# third party packages
import matplotlib
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import numpy as np


class _LazyPlt:
    """Proxy for matplotlib.pyplot — defers the heavy pyplot import until first plot call."""
    def __getattr__(self, name):
        global plt
        import matplotlib.pyplot as _real_plt
        plt = _real_plt  # replace proxy with the real module for all future lookups
        return getattr(_real_plt, name)


plt = _LazyPlt()


class _LazyIPythonDisplay:
    """Proxy for IPython.display — defers the actual import until first attribute access."""
    def __getattr__(self, name):
        from IPython import display as _d
        return getattr(_d, name)


display = _LazyIPythonDisplay()

__all__ = [
    "isnotebook",
    "autoscale_y",
    "show_figure",
    "show_fsc_images",
    "show_fsc_curve",
    "show_ssnr_curve",
    "show_random_fsc_curve",
    "show_resolution_map",
    "RegisterPlot",
    "ShowProjections",
    "plot_checkangles",
    "show_linearphase",
    "iterative_show",
    "animated_image",
    "display_slice",
]

def show_figure(fig=None, close=True):
    """Display a figure in the current environment and optionally close it.

    In Jupyter: switches the figure's canvas to a plain Agg renderer, renders
    to a PNG via savefig, and shows it as a static ``display.Image``.
    Switching to Agg before savefig is necessary because the default ipympl
    canvas (``FigureCanvasWebAggCore``) routes savefig through
    ``canvas.draw()`` → ``manager.refresh_all()``, which raises
    ``AttributeError`` when ``manager`` is ``None`` (e.g. when ``plt.ion()``
    was never called).  The Agg canvas has no manager dependency.

    In terminal: calls plt.show(block=False).

    Parameters
    ----------
    fig : matplotlib Figure, optional
        Figure to display. Defaults to the current figure (plt.gcf()).
    close : bool, optional
        If True (default), close the figure after displaying it in Jupyter.
        Set to False for figures that will be reused or updated.
    """
    if fig is None:
        fig = plt.gcf()
    if isnotebook():
        from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCAgg
        _FCAgg(fig)       # swap to Agg canvas — savefig no longer needs manager
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        display.display(display.Image(buf.read()))
        if close:
            plt.close(fig)
    else:
        plt.show(block=False)


def show_fsc_images(img1_apod, img2_apod):
    """Display the two apodized images used in the FSC computation.

    Parameters
    ----------
    img1_apod : ndarray
        First apodized image (or sagittal slice for 3D).
    img2_apod : ndarray
        Second apodized image (or sagittal slice for 3D).
    """
    if isnotebook():
        # Use the OO matplotlib API (not pyplot) so the figure is created with
        # a plain Agg canvas instead of the ipympl webagg canvas.  With the
        # webagg canvas, fig.savefig() internally calls
        # FigureCanvasWebAggCore.draw() → self.manager.refresh_all(), which
        # raises AttributeError when manager is None (e.g. when plt.ion() was
        # never called).  The Agg canvas path has no manager dependency at all.
        import matplotlib.figure as _mf
        from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCAgg
        fig = _mf.Figure(figsize=(10, 5))
        _FCAgg(fig)           # attach Agg canvas — no pyplot/manager involved
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img1_apod, cmap="bone", interpolation="none")
        ax1.set_title("image1")
        ax1.set_axis_off()
        ax2.imshow(img2_apod, cmap="bone", interpolation="none")
        ax2.set_title("image2")
        ax2.set_axis_off()
        fig.tight_layout()
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        display.display(display.Image(buf.read()))
    else:
        fig = plt.figure()
        plt.clf()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img1_apod, cmap="bone", interpolation="none")
        ax1.set_title("image1")
        ax1.set_axis_off()
        ax2.imshow(img2_apod, cmap="bone", interpolation="none")
        ax2.set_title("image2")
        ax2.set_axis_off()
        fig.tight_layout()
        plt.show(block=False)


def show_fsc_curve(fn, FSC, T, snrt, ndim):
    """Plot the FSC and threshold curves, save to disk, and display.

    Draws the FSC curve, the threshold, a vertical dashed line at the
    estimated resolution crossing, and an informative title.  The figure
    is saved to ``FSC_2D.png`` or ``FSC_3D.png`` and then displayed.

    Parameters
    ----------
    fn : ndarray
        Spatial frequencies normalised by the Nyquist frequency.
    FSC : ndarray
        Fourier Shell Correlation curve (real part).
    T : ndarray
        Threshold curve.
    snrt : float
        SNR threshold value used to select the threshold label:
        ``0.2071`` → ``"1/2 bit threshold"``,
        ``0.5``    → ``"1 bit threshold"``,
        anything else → ``f"Threshold SNR = {snrt:g}"``.
    ndim : int
        Number of dimensions of the original data (2 or 3).

    Returns
    -------
    None
    """
    suffix = "2D" if ndim == 2 else "3D"
    if isnotebook():
        fig = plt.figure(figsize=(8, 6))
    else:
        fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.plot(fn, FSC, "-b", label="FSC")
    if snrt == 0.2071:
        thr_label = "1/2 bit threshold"
    elif snrt == 0.5:
        thr_label = "1 bit threshold"
    else:
        thr_label = f"Threshold SNR = {snrt:g}"
    ax.plot(fn, T, "--r", label=thr_label)

    # Resolution crossing: last index where FSC > T
    above = np.asarray(FSC) > np.asarray(T)
    if np.any(above):
        last_above = int(np.where(above)[0][-1])
        fn_res = float(fn[last_above])
        ax.axvline(fn_res, color="k", linestyle="--", alpha=0.7,
                   label=f"Resolution ≈ {fn_res:.3f} × Nyquist")

    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Spatial frequency/Nyquist")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Fourier {'Ring' if ndim == 2 else 'Shell'} Correlation ({suffix})")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"FSC_{suffix}.png", bbox_inches="tight")
    show_figure(fig)


def show_ssnr_curve(fn, FSC, SSNR, SSNR_T, snrt, ndim):
    """Plot the SSNR curve with its threshold, resolution line, and asymptote.

    The figure shows the Spectral Signal-to-Noise Ratio (SSNR) and its
    frequency-dependent threshold on a semi-logarithmic scale, together
    with a horizontal dotted line at the asymptotic threshold value and a
    vertical dashed line at the estimated resolution crossing.  The figure
    is saved to ``SSNR_2D.png`` or ``SSNR_3D.png`` and then displayed.

    Parameters
    ----------
    fn : ndarray
        Spatial frequencies normalised by the Nyquist frequency.
    FSC : ndarray
        Fourier Shell/Ring Correlation curve (real part), included for
        potential future use but not plotted directly.
    SSNR : ndarray
        Spectral Signal-to-Noise Ratio curve, derived from FSC via
        ``SSNR = 2 * FSC / (1 - FSC)``.
    SSNR_T : ndarray
        Frequency-dependent SSNR threshold curve, derived from the FSC
        threshold ``T`` via ``SSNR_T = 2 * T / (1 - T)``.
    snrt : float
        SNR threshold value used to compute the asymptote and select the
        threshold name: ``0.2071`` → ``"half-bit"``, anything else →
        ``"one-bit"``.
    ndim : int
        Number of dimensions of the original data (2 or 3).

    Returns
    -------
    None
    """
    suffix = "2D" if ndim == 2 else "3D"
    eps = np.spacing(1)
    T_asymptote = snrt / (snrt + 1.0)
    SSNR_asymp = 2.0 * T_asymptote / max(1.0 - T_asymptote, eps)
    thr_name = "half-bit" if snrt == 0.2071 else "one-bit"

    # Resolution crossing: last index where SSNR > SSNR_T
    above = np.asarray(SSNR) > np.asarray(SSNR_T)
    if np.any(above):
        idx_res = int(np.where(above)[0][-1])
        fn_res = float(fn[idx_res])
    else:
        fn_res = None

    fig = plt.figure(figsize=(8, 6))
    plt.clf()
    ax = fig.add_subplot(111)

    ax.semilogy(fn, SSNR,   "-b", label="SSNR")
    ax.semilogy(fn, SSNR_T, "-r", label=f"SSNR threshold ({thr_name})")
    ax.axhline(SSNR_asymp, color="r", linestyle=":", alpha=0.6,
               label=f"Asymptote = {SSNR_asymp:.3f}")
    if fn_res is not None:
        ax.axvline(fn_res, color="k", linestyle="--", alpha=0.7,
                   label=f"Resolution ~ {fn_res:.3f} x Nyquist")

    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Spatial frequency / Nyquist")
    ax.set_ylabel("SSNR")
    ax.set_title(f"Spectral Signal-to-Noise Ratio ({suffix})")
    ax.grid(True, linestyle="--", alpha=0.5)

    # semilogy normally formats Y-axis ticks as "$10^{N}$" (mathtext).
    # On Python 3.14 + some matplotlib/pyparsing combos the mathtext parser
    # raises a ParseException at tight_layout time.  Replace the formatter
    # with a plain-text scientific-notation formatter to sidestep this.
    import matplotlib.ticker as _ticker
    ax.yaxis.set_major_formatter(
        _ticker.FuncFormatter(lambda val, _: f"{val:.2g}")
    )

    fig.tight_layout()
    fig.savefig(f"SSNR_{suffix}.png", bbox_inches="tight")
    show_figure(fig)


def show_random_fsc_curve(fn, fsc_obs, fsc_rand, fsc_corr, T, cutoff_fn, ndim):
    """Plot the phase-randomization FSC test results and save to disk.

    Left panel: ``FSC_obs``, ``FSC_rand``, ``FSC_corr``, the threshold ``T``,
    and a vertical dotted line at the randomisation cutoff frequency.
    Right panel: ``FSC_obs − FSC_rand`` (genuine signal above the cutoff),
    with a horizontal dashed line at zero.

    The figure is saved to ``RandomFSC.png`` and then displayed.

    Parameters
    ----------
    fn : ndarray
        Spatial frequencies normalised by the Nyquist frequency.
    fsc_obs : ndarray
        Observed (standard) FSC curve.
    fsc_rand : ndarray
        Phase-randomized FSC curve (noise floor).
    fsc_corr : ndarray
        Corrected FSC, defined as
        ``(FSC_obs - FSC_rand) / (1 - FSC_rand)`` above the randomization
        transition, ``FSC_obs`` at and below the cutoff, and ``np.nan``
        across the transition itself (plotted as a gap).  May be negative.
    T : ndarray
        Threshold curve.
    cutoff_fn : float
        Normalised cutoff frequency (``cutoff_shell / fnyquist``) at which
        phase randomisation begins.
    ndim : int
        Number of dimensions of the original data (2 or 3).  Currently used
        only for potential future labelling.

    Returns
    -------
    None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fn, fsc_obs,  "-b",  label="FSC_obs")
    ax1.plot(fn, fsc_rand, "-r",  label="FSC_rand")
    ax1.plot(fn, fsc_corr, "-g",  label="FSC_corr")
    ax1.plot(fn, T,        "--k", label="Threshold T")
    ax1.axvline(
        cutoff_fn,
        color="purple",
        linestyle=":",
        label=f"Cutoff fn = {cutoff_fn:.3f}",
    )
    ax1.set_xlim(0, 1)
    # FSC_corr is not clipped at zero: negative shells (FSC_rand > FSC_obs) are
    # diagnostic of model bias, so extend the axis rather than hiding them
    ymin = np.nanmin(np.asarray(fsc_corr, dtype=float), initial=-0.1)
    ax1.set_ylim(min(-0.1, ymin - 0.05), 1.1)
    ax1.set_xlabel("Spatial frequency / Nyquist")
    ax1.set_ylabel("FSC")
    ax1.set_title("Phase-Randomization FSC Test")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    bias = np.asarray(fsc_obs) - np.asarray(fsc_rand)
    ax2.plot(fn, bias, "-m", label="FSC_obs − FSC_rand")
    ax2.axhline(0.0, color="k", linestyle="--")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Spatial frequency / Nyquist")
    ax2.set_ylabel("FSC_obs − FSC_rand")
    ax2.set_title("Genuine signal: FSC_obs − FSC_rand")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig("RandomFSC.png", bbox_inches="tight")
    show_figure(fig)


def show_resolution_map(rmap, ndim, title, filename,
                        slice_idx=None, axis=0,
                        cmap="viridis_r", vmin=None, vmax=None):
    """Display a 2-D or 3-D local resolution map and save it to disk.

    For a 3-D map a single slice is extracted along *axis* and shown as a
    2-D image.  The slice index and axis are appended to the title.  For a
    2-D map the full array is shown.  A colorbar labelled
    ``"Local resolution (pixels)"`` is added, and the figure is saved to
    *filename* before being displayed.

    Parameters
    ----------
    rmap : ndarray
        Local resolution map.  May be 2-D or 3-D.
    ndim : int
        Number of dimensions of the original data (2 or 3).  Must match
        ``rmap.ndim``.
    title : str
        Base title string.  For 3-D data the slice information is appended
        automatically.
    filename : str
        Output filename (e.g. ``"LocalFSC_resmap.png"``).  Saved with
        ``bbox_inches='tight'``.
    slice_idx : int or None, optional
        Index of the slice to display along *axis*.  Defaults to the central
        slice (``rmap.shape[axis] // 2``) when ``None``.  Ignored for 2-D
        input.
    axis : int, optional
        Axis along which to extract the slice (3-D only).  Default ``0``.
    cmap : str, optional
        Matplotlib colormap name.  Default ``'viridis_r'``.
    vmin : float or None, optional
        Lower colour-scale limit.  ``None`` uses the data minimum.
    vmax : float or None, optional
        Upper colour-scale limit.  ``None`` uses the data maximum.

    Returns
    -------
    None
    """
    if ndim == 3:
        if slice_idx is None:
            slice_idx = rmap.shape[axis] // 2
        img = np.take(rmap, slice_idx, axis=axis)
        title = title + f"  (axis={axis}, slice={slice_idx})"
    else:
        img = rmap

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Local resolution (pixels)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    show_figure(fig)


def isnotebook():
    """
    Check if code is executed in the IPython notebook.
    This is important because jupyter notebook does not support iterative plots
    """
    ipy = sys.modules.get('IPython')
    if ipy is None:
        return False
    try:
        shell = ipy.get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (AttributeError, NameError):
        return False

def interativesession(func):
    """
    Decorator that ensures matplotlib interactive mode is active before calling a plot function.

    Parameters
    ----------
    func : callable
        Plot function to decorate.

    Returns
    -------
    callable
        Wrapped function that enables ``matplotlib.interactive(True)`` if
        it was not already active, then delegates to ``func``.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        flagmpl = matplotlib.is_interactive()
        if flagmpl == False:
            matplotlib.interactive(True)
        return func(*args, **kwargs)

    return new_func


class _NullContext:
    """A no-op context manager used as a drop-in for ``ipywidgets.Output``
    in terminal (non-notebook) environments."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def autoscale_y(ax, margin=0.1):
    """
    This function rescales the y-axis based on the data that is visible given the current xlim of the axis.

    Parameters
    ----------
    ax : object
        A matplotlib axes object
    margin : float
        The fraction of the total height of the y-data to pad the upper and lower ylims
    """

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        y_displayed = yd[((xd > lo) & (xd < hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed) - margin * h
        top = np.max(y_displayed) + margin * h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot:
            bot = new_bot
        if new_top > top:
            top = new_top

    ax.set_ylim(bot, top)


def _plotdelimiters(ax, limrow, limcol, airpixel=[]):
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
    airpixel : list of ints
        Position of pixel in the air/vacuum
    """
    ax.plot([limcol[0], limcol[-1]], [limrow[0], limrow[0]], "r-")
    ax.plot([limcol[0], limcol[-1]], [limrow[-1], limrow[-1]], "r-")
    ax.plot([limcol[0], limcol[0]], [limrow[0], limrow[-1]], "r-")
    ax.plot([limcol[-1], limcol[-1]], [limrow[0], limrow[-1]], "r-")
    if airpixel != []:
        ax.plot(airpixel[0], airpixel[1], "ob")
    return ax


def _createcanvashorizontal(
    recons, sinoorig, sinocurr, sinocomp, deltaslice, metric_error, **params
):
    """
    Create the unified 6-panel figure for horizontal alignment diagnostics.

    The figure is laid out as a 3 × 2 grid::

        Row 0 (tall):   reconstructed slice  |  initial sinogram (fixed)
        Row 1 (medium): shifts               |  synthetic sinogram
        Row 2 (medium): error metric         |  current sinogram

    Parameters
    ----------
    recons : ndarray
        Current reconstructed slice.
    sinoorig : ndarray
        Original (unaligned) sinogram — shown once, never updated.
    sinocurr : ndarray
        Current aligned sinogram.
    sinocomp : ndarray
        Synthetic sinogram computed from the reconstruction.
    deltaslice : ndarray
        Current horizontal shift estimates (already transposed by caller).
    metric_error : list of float
        Error metric history.
    **params
        Must contain ``'slicenum'`` (int), ``'sinohigh'`` (float),
        and ``'sinolow'`` (float).

    Returns
    -------
    fig_main : Figure
        Unified 6-panel diagnostic figure.
    arts : dict
        Named artist references for later in-place updates.
    axd : dict
        Named axes for later rescaling.
    """
    slicenum = params["slicenum"]
    cmax = params["sinohigh"]
    cmin = params["sinolow"]

    fig_main = plt.figure(figsize=(12, 12))
    gs = fig_main.add_gridspec(
        3, 2, height_ratios=[3, 2, 2], hspace=0.45, wspace=0.3
    )

    ax_recon     = fig_main.add_subplot(gs[0, 0])
    ax_initsino  = fig_main.add_subplot(gs[0, 1])
    ax_shifts    = fig_main.add_subplot(gs[1, 0])
    ax_synthsino = fig_main.add_subplot(gs[1, 1])
    ax_error     = fig_main.add_subplot(gs[2, 0])
    ax_currsino  = fig_main.add_subplot(gs[2, 1])

    # Reconstructed slice (updated each iteration)
    im_recon = ax_recon.imshow(recons, cmap="jet", aspect="auto")
    ax_recon.set_title("Reconstructed slice (slice {})".format(slicenum))
    ax_recon.set_xlabel("x [pixels]")
    ax_recon.set_ylabel("y [pixels]")

    # Initial sinogram (fixed — never updated after iter 0)
    im_initsino = ax_initsino.imshow(
        sinoorig, cmap="bone", vmin=cmin, vmax=cmax, aspect="auto"
    )
    ax_initsino.set_title("Initial sinogram (fixed)")
    ax_initsino.set_xlabel("Projection")
    ax_initsino.set_ylabel("x [pixels]")

    # Shifts (updated each iteration)
    im_shifts_lines = ax_shifts.plot(deltaslice)
    ax_shifts.axis("tight")
    ax_shifts.set_title("Object position (shifts)")
    ax_shifts.set_xlabel("Projection")
    ax_shifts.set_ylabel("Shift [pixels]")

    # Synthetic sinogram (updated each iteration)
    im_synthsino = ax_synthsino.imshow(
        sinocomp, cmap="bone", vmin=cmin, vmax=cmax, aspect="auto"
    )
    ax_synthsino.set_title("Synthetic sinogram")
    ax_synthsino.set_xlabel("Projection")
    ax_synthsino.set_ylabel("x [pixels]")

    # Error metric — grows one point per iteration
    (im_error,) = ax_error.plot(metric_error, "bo-")
    ax_error.axis("tight")
    ax_error.set_title("Error metric")
    ax_error.set_xlabel("Iteration")
    ax_error.set_ylabel("Error")

    # Current sinogram (updated each iteration)
    im_currsino = ax_currsino.imshow(
        sinocurr, cmap="bone", vmin=cmin, vmax=cmax, aspect="auto"
    )
    ax_currsino.set_title("Current sinogram")
    ax_currsino.set_xlabel("Projection")
    ax_currsino.set_ylabel("x [pixels]")

    fig_main.suptitle(
        "Horizontal alignment — Iter 0 | slice {}".format(slicenum), fontsize=13
    )

    arts = {
        "im_recon":        im_recon,
        "im_initsino":     im_initsino,
        "im_shifts_lines": im_shifts_lines,
        "im_synthsino":    im_synthsino,
        "im_error":        im_error,
        "im_currsino":     im_currsino,
    }
    axd = {
        "recon":     ax_recon,
        "initsino":  ax_initsino,
        "shifts":    ax_shifts,
        "synthsino": ax_synthsino,
        "error":     ax_error,
        "currsino":  ax_currsino,
    }

    return fig_main, arts, axd


def _createcanvasvertical(
    proj, lims, vertfluctinit, vertfluctcurr, deltastack, metric_error, **params
):
    """
    Create the static projection figure and unified 6-panel diagnostic figure
    for vertical alignment.

    The diagnostic figure uses :func:`~matplotlib.pyplot.subplot_mosaic` with
    the following layout::

        [ init2d  |  init2d ]   ← full-width: initial 2-D integral (fixed)
        [ curr2d  |  curr2d ]   ← full-width: current 2-D integral (updated)
        [ init1d  |  curr1d ]   ← initial 1-D profiles (fixed) | current (updated)
        [ shifts  |  error  ]   ← shifts (updated) | error metric (updated)

    Parameters
    ----------
    proj : ndarray
        First projection image (for the ROI overlay panel).
    lims : tuple of array_like
        ``(limrow, limcol)`` ROI limits.
    vertfluctinit : ndarray
        Initial vertical fluctuation signals (already transposed by caller,
        shape ``(n_rows_roi, n_projections)``).
    vertfluctcurr : ndarray
        Current vertical fluctuation signals (same shape, transposed).
    deltastack : ndarray
        Current shift estimates (already transposed by caller).
    metric_error : list of float
        Error metric history.
    **params
        Additional display parameters (forwarded for API consistency).

    Returns
    -------
    fig_proj : Figure
        Small static figure showing the projection with the ROI overlay.
        Displayed once at initialisation, never updated.
    fig_main : Figure
        Unified 6-panel diagnostic figure (updated each iteration).
    arts : dict
        Named artist references for later in-place updates.
    axd : dict
        Named axes (from ``subplot_mosaic``) for later rescaling.
    """
    limrow, limcol = lims

    # ------------------------------------------------------------------ #
    # fig_proj: projection + ROI overlay — shown once, never updated
    # ------------------------------------------------------------------ #
    fig_proj = plt.figure(figsize=(5, 4))
    ax_proj = fig_proj.add_subplot(111)
    im_proj = ax_proj.imshow(proj, cmap="bone")
    ax_proj.set_title("Projection with ROI")
    ax_proj.axis("image")
    _plotdelimiters(ax_proj, limrow, limcol)
    fig_proj.tight_layout()

    # ------------------------------------------------------------------ #
    # fig_main: unified 6-panel diagnostic figure
    # ------------------------------------------------------------------ #
    mosaic = [
        ["init2d", "init2d"],   # full width: initial 2-D integral (fixed)
        ["curr2d", "curr2d"],   # full width: current 2-D integral (updated)
        ["init1d", "curr1d"],   # initial 1-D profiles | current 1-D profiles
        ["shifts", "error"],    # shifts (updated) | error metric (updated)
    ]
    fig_main, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(12, 14),
        gridspec_kw={"height_ratios": [2, 2, 3, 3]},
    )

    # init2d: initial 2-D integral (fixed after iter 0)
    im_init2d = axd["init2d"].imshow(
        vertfluctinit, cmap="jet", interpolation="none", aspect="auto"
    )
    axd["init2d"].set_title("Initial Integral in x")
    axd["init2d"].set_xlabel("Projection")
    axd["init2d"].set_ylabel("y [pixels]")

    # curr2d: current 2-D integral (updated each iteration)
    im_curr2d = axd["curr2d"].imshow(
        vertfluctcurr, cmap="jet", interpolation="none", aspect="auto"
    )
    axd["curr2d"].set_title("Current Integral in x")
    axd["curr2d"].set_xlabel("Projection")
    axd["curr2d"].set_ylabel("y [pixels]")

    # init1d: initial 1-D profiles per projection (fixed after iter 0)
    im_init1d_lines = axd["init1d"].plot(vertfluctinit)
    avg_init = vertfluctinit.mean(axis=1)
    (im_init1d_avg,)  = axd["init1d"].plot(avg_init, "r",   linewidth=2.5)
    (im_init1d_avg2,) = axd["init1d"].plot(avg_init, "--w", linewidth=1.5)
    axd["init1d"].axis("tight")
    axd["init1d"].set_title("Initial Integral in x (1D)")
    axd["init1d"].set_xlabel("y [pixels]")
    axd["init1d"].set_ylabel("Intensity")

    # curr1d: current 1-D profiles per projection (updated each iteration)
    im_curr1d_lines = axd["curr1d"].plot(vertfluctcurr)
    avg_curr = vertfluctcurr.mean(axis=1)
    (im_curr1d_avg,)  = axd["curr1d"].plot(avg_curr, "r",   linewidth=2.5)
    (im_curr1d_avg2,) = axd["curr1d"].plot(avg_curr, "--w", linewidth=1.5)
    axd["curr1d"].axis("tight")
    axd["curr1d"].set_title("Current Integral in x (1D)")
    axd["curr1d"].set_xlabel("y [pixels]")
    axd["curr1d"].set_ylabel("Intensity")

    # shifts: object position per projection (updated each iteration)
    im_shifts_lines = axd["shifts"].plot(deltastack)
    axd["shifts"].axis("tight")
    axd["shifts"].set_title("Object position (shifts)")
    axd["shifts"].set_xlabel("Projection")
    axd["shifts"].set_ylabel("Shift [pixels]")

    # error: convergence history — grows one point per iteration
    (im_error,) = axd["error"].plot(metric_error, "bo-")
    axd["error"].axis("tight")
    axd["error"].set_title("Error metric")
    axd["error"].set_xlabel("Iteration")
    axd["error"].set_ylabel("Error")

    fig_main.suptitle("Vertical alignment — Iter 0", fontsize=13)
    fig_main.tight_layout()

    arts = {
        "im_proj":         im_proj,
        "im_init2d":       im_init2d,
        "im_curr2d":       im_curr2d,
        "im_init1d_lines": im_init1d_lines,
        "im_init1d_avg":   im_init1d_avg,
        "im_init1d_avg2":  im_init1d_avg2,
        "im_curr1d_lines": im_curr1d_lines,
        "im_curr1d_avg":   im_curr1d_avg,
        "im_curr1d_avg2":  im_curr1d_avg2,
        "im_shifts_lines": im_shifts_lines,
        "im_error":        im_error,
    }

    return fig_proj, fig_main, arts, axd


class RegisterPlot:
    """
    Manage live plot updates during tomographic projection alignment.

    Provides two high-level entry points:

    * :meth:`plotsvertical`  — vertical shift alignment diagnostics
    * :meth:`plotshorizontal` — horizontal (sinogram) alignment diagnostics

    Architecture
    ------------
    **Vertical alignment**
        * ``fig_proj``  — small static figure (projection + ROI), shown
          *once* at initialisation and never updated.
        * ``fig_main``  — unified 6-panel diagnostic figure, updated every
          iteration via its ``_out_main`` :class:`~ipywidgets.Output` widget.

    **Horizontal alignment**
        * ``fig_main``  — unified 6-panel diagnostic figure (single figure).

    Display strategy
    ----------------
    **Jupyter / ``%matplotlib widget``**
        Figures are rendered to PNG bytes via
        :meth:`~matplotlib.figure.Figure.savefig` and shown via IPython
        ``DisplayHandle`` objects obtained from
        ``IPython.display.display(..., display_id=True)``.  Subsequent
        updates call ``handle.update(new_png)`` — **no** ``clear_output``
        is ever called, so the figure display items are never accidentally
        removed.  The verbose text printed by the alignment loop is
        similarly updated in-place via a second ``DisplayHandle`` that
        holds an ``HTML`` block; the loop redirects ``stdout`` to a
        ``StringIO`` buffer and calls :meth:`_verbose_update` after each
        iteration.

    **Terminal**
        Figures are redrawn via ``canvas.draw_idle()`` + ``plt.pause()``.

    Parameters
    ----------
    **params
        Algorithm parameters forwarded to the canvas helpers.
        Must contain at least ``'slicenum'``, ``'sinohigh'``, and
        ``'sinolow'``.
    """

    def __init__(self, **params):
        self.params = params
        self.count = 0
        self.max_correction = None   # updated by plotsvertical each iteration
        self.stage_info = None       # (stage_num, n_stages, freqcutoff) — set by caller
        self._dh_verbose = None      # DisplayHandle for verbose text (notebook only)
        plt.close("all")

    # ------------------------------------------------------------------ #
    # PNG helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fig_to_png(fig):
        """Render *fig* to PNG bytes via the Agg renderer.

        Uses :meth:`~matplotlib.figure.Figure.savefig` which requires no
        figure manager — safe for ``%matplotlib widget`` (ipympl) where
        ``canvas.manager`` is ``None``.
        """
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        return buf.read()

    # ------------------------------------------------------------------ #
    # DisplayHandle helpers  (no clear_output — uses update() instead)
    # ------------------------------------------------------------------ #

    def _dh_init(self, fig, attr):
        """Display *fig* as a PNG and store the :class:`DisplayHandle` as
        ``self.<attr>``.

        The handle is used by :meth:`_dh_update` to replace the image
        in-place on subsequent calls — without any ``clear_output``.

        In terminal mode this is a no-op; the caller uses :meth:`_term_show`.
        """
        if not isnotebook():
            return
        png = display.Image(self._fig_to_png(fig))
        dh = display.display(png, display_id=True)
        setattr(self, attr, dh)

    def _dh_update(self, fig, attr):
        """Replace the PNG displayed by ``self.<attr>`` in-place.

        Uses :py:meth:`DisplayHandle.update` — no ``clear_output`` call,
        no new cell-output item is created.

        In terminal mode this is a no-op; the caller uses :meth:`_term_show`.
        """
        if not isnotebook():
            return
        dh = getattr(self, attr, None)
        png = display.Image(self._fig_to_png(fig))
        if dh is not None:
            dh.update(png)
        else:
            display.display(png)

    def _verbose_update(self, text):
        """Update the verbose-text area in-place (no ``clear_output``).

        *text* is rendered inside a ``<pre>`` block so newlines and
        indentation are preserved.  HTML special characters are escaped.

        ANSI escape sequences (tqdm colours, cursor moves) are stripped and
        carriage-return overwriting (``\\r``) is resolved so only the final
        content on each logical line is shown — this collapses the repeated
        tqdm progress-bar rewrites into a single clean line.

        Does nothing in terminal mode or when ``_dh_verbose`` is ``None``.
        """
        if not isnotebook():
            return
        dh = self._dh_verbose
        if dh is None:
            return

        # 1. Strip ANSI escape sequences (colours, cursor positioning, etc.)
        text = re.sub(r'\x1b\[[0-9;]*[mKGABCDEFHJSTfhilnprsu]', '', text)

        # 2. Resolve carriage-return overwriting: tqdm re-draws the same line
        #    by emitting \r; keep only the last segment after each \r so we
        #    see the final bar state instead of all intermediate rewrites.
        lines = []
        for line in text.split('\n'):
            parts = line.split('\r')
            lines.append(parts[-1])   # last overwrite wins
        text = '\n'.join(lines)

        # 3. Drop lines that are blank after the above processing
        text = '\n'.join(ln for ln in text.split('\n') if ln.strip())

        escaped = (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )
        dh.update(display.HTML(
            "<pre style='margin:0;white-space:pre-wrap;line-height:1.4'>"
            "{}</pre>".format(escaped)
        ))

    @staticmethod
    def _term_show(figs):
        """Redraw *figs* in a terminal GUI event loop."""
        for fig in figs:
            fig.canvas.draw_idle()
        plt.pause(0.001)

    # ------------------------------------------------------------------ #
    # Vertical alignment
    # ------------------------------------------------------------------ #

    @interativesession
    def plotsvertical(
        self, proj, lims, vertfluctinit, vertfluctcurr, deltastack, metric_error, count,
        max_correction=None,
    ):
        """Display or update the vertical-alignment diagnostic figures.

        **First call** — creates ``fig_proj`` (static, shown once) and
        ``fig_main`` (unified 6-panel, updated each iteration).

        **Subsequent calls** — delegates to :meth:`updatevertical` which
        updates artists in-place and refreshes only ``fig_main``.

        Parameters
        ----------
        proj : ndarray
            Current projection image (displayed once as ROI overlay).
        lims : tuple
            ``(limrow, limcol)`` ROI boundary indices.
        vertfluctinit : ndarray
            Initial vertical fluctuations (fixed reference).
        vertfluctcurr : ndarray
            Current vertical fluctuations (updated each iteration).
        deltastack : ndarray
            Current vertical shift estimates.
        metric_error : list of float
            Error metric history (grows by one element per iteration).
        count : int
            Current iteration number.
        max_correction : float or None
            Maximum absolute vertical shift change this iteration (pixels).
            Displayed in the figure suptitle when provided.
        """
        # Store data (transposed to match display convention)
        self.max_correction    = max_correction
        self.proj              = proj
        self.lims              = lims
        self.vertfluctinit     = vertfluctinit.T
        self.vertfluctinit_avg = self.vertfluctinit.mean(axis=1)
        self.vertfluctcurr     = vertfluctcurr.T
        self.vertfluctcurr_avg = self.vertfluctcurr.mean(axis=1)
        self.deltastack        = deltastack.T
        self.metric_error      = metric_error
        self.count             = count

        if not hasattr(self, "fig_proj"):
            # First call: create figures.
            # Wrap plt.figure() calls in a throwaway Output so any stray
            # auto-display from ipympl is captured and discarded here; we
            # then display our own controlled PNG Output containers below.
            if isnotebook():
                try:
                    from ipywidgets import Output as _Out
                    _cap = _Out()
                except ImportError:
                    _cap = _NullContext()
            else:
                _cap = _NullContext()

            with _cap:
                (self.fig_proj, self.fig_main,
                 self._v_arts, self._v_axd) = _createcanvasvertical(
                    self.proj, self.lims,
                    self.vertfluctinit, self.vertfluctcurr,
                    self.deltastack, self.metric_error,
                    **self.params
                )

            if isnotebook():
                # Display projection figure once (static).
                self._dh_init(self.fig_proj, "_dh_proj")
                # Display main diagnostic figure (updated each iteration
                # via _dh_update — no clear_output needed).
                self._dh_init(self.fig_main, "_dh_main")
                # Reserve a spot for the verbose text that the alignment
                # loop will update in-place via _verbose_update.
                self._dh_verbose = display.display(
                    display.HTML(""), display_id=True
                )
            else:
                self._term_show([self.fig_proj, self.fig_main])
                self._dh_verbose = None
        else:
            self.updatevertical()

    @interativesession
    def updatevertical(self):
        """Update the vertical-alignment diagnostic figure in-place.

        Modifies existing artist objects — no new figures or axes are
        created.  Only ``fig_main`` is refreshed; ``fig_proj`` is static.

        Panels updated
        --------------
        * **curr2d** — current 2-D integral image.
        * **curr1d** — current 1-D integral line plots and mean overlay.
        * **shifts** — shift curves for all projections.
        * **error**  — error-metric curve (grows one point per iteration).
        * ``fig_main.suptitle`` — iteration counter and latest error value.
        """
        arts = self._v_arts
        axd  = self._v_axd

        # curr2d: update 2-D image
        arts["im_curr2d"].set_data(self.vertfluctcurr)
        arts["im_curr2d"].autoscale()

        # curr1d: update individual profiles and mean overlay
        curr = self.vertfluctcurr
        for idx, line in enumerate(arts["im_curr1d_lines"]):
            line.set_ydata(curr[:, idx] if curr.ndim > 1 else curr)
        arts["im_curr1d_avg"].set_ydata(self.vertfluctcurr_avg)
        arts["im_curr1d_avg2"].set_ydata(self.vertfluctcurr_avg)
        axd["curr1d"].relim()
        # autoscale(enable=True) re-enables the autoscale flag (which savefig
        # disables internally via set_xlim/set_ylim) and immediately applies it.
        # tight=False (default) leaves room for the default margin padding.
        axd["curr1d"].autoscale(enable=True, axis="both")
        axd["curr1d"].margins(x=0.02, y=0.08)

        # shifts: update all shift curves
        delta = self.deltastack
        for idx, line in enumerate(arts["im_shifts_lines"]):
            line.set_ydata(delta[:, idx] if delta.ndim > 1 else delta)
        axd["shifts"].relim()
        axd["shifts"].autoscale(enable=True, axis="both")
        axd["shifts"].margins(x=0.02, y=0.08)

        # error: grow the convergence curve by one point
        n = len(self.metric_error)
        arts["im_error"].set_xdata(np.arange(n))
        arts["im_error"].set_ydata(self.metric_error)
        axd["error"].relim()
        axd["error"].autoscale(enable=True, axis="both")
        axd["error"].margins(x=0.02, y=0.08)

        # Update suptitle with current iteration, error, and max correction
        err_val = self.metric_error[-1] if self.metric_error else float("nan")
        corr_str = (
            " | Max Δy = {:.2f} px".format(self.max_correction)
            if self.max_correction is not None else ""
        )
        self.fig_main.suptitle(
            "Vertical alignment — Iter {} | E = {:.3e}{}".format(
                self.count, err_val, corr_str
            ),
            fontsize=13,
        )

        if isnotebook():
            self._dh_update(self.fig_main, "_dh_main")
        else:
            self._term_show([self.fig_main])

    # ------------------------------------------------------------------ #
    # Horizontal alignment
    # ------------------------------------------------------------------ #

    @interativesession
    def plotshorizontal(
        self, recons, sinoorig, sinocurr, sinocomp, deltaslice, metric_error, count
    ):
        """Display or update the horizontal-alignment diagnostic figure.

        **First call** — creates the unified 6-panel ``fig_main``.

        **Subsequent calls** — delegates to :meth:`updatehorizontal` which
        updates artists in-place and refreshes ``fig_main``.

        Parameters
        ----------
        recons : ndarray
            Current reconstructed slice.
        sinoorig : ndarray
            Original sinogram (fixed reference, never updated after iter 0).
        sinocurr : ndarray
            Current aligned sinogram.
        sinocomp : ndarray
            Synthetic sinogram computed from the reconstruction.
        deltaslice : ndarray
            Current horizontal shift estimates.
        metric_error : list of float
            Error metric history.
        count : int
            Current iteration number.
        """
        # Store data (transposed to match display convention)
        self.recons       = recons
        self.sinoorig     = sinoorig
        self.sinocurr     = sinocurr
        self.sinocomp     = sinocomp
        self.deltaslice   = deltaslice.T
        self.metric_error = metric_error
        self.count        = count

        if not hasattr(self, "fig_main"):
            # First call: create the unified figure.
            if isnotebook():
                try:
                    from ipywidgets import Output as _Out
                    _cap = _Out()
                except ImportError:
                    _cap = _NullContext()
            else:
                _cap = _NullContext()

            with _cap:
                (self.fig_main,
                 self._h_arts, self._h_axd) = _createcanvashorizontal(
                    self.recons, self.sinoorig, self.sinocurr, self.sinocomp,
                    self.deltaslice, self.metric_error,
                    **self.params
                )

            if isnotebook():
                # Display main figure (updated via _dh_update — no clear_output).
                self._dh_init(self.fig_main, "_dh_main")
                # Reserve a spot for verbose text.
                self._dh_verbose = display.display(
                    display.HTML(""), display_id=True
                )
            else:
                self._term_show([self.fig_main])
                self._dh_verbose = None
        else:
            self.updatehorizontal()

    @interativesession
    def updatehorizontal(self):
        """Update the horizontal-alignment diagnostic figure in-place.

        Modifies existing artist objects — no new figures or axes are created.

        Panels updated
        --------------
        * **recon**     — reconstructed slice (sharpens as alignment improves).
        * **synthsino** — synthetic sinogram.
        * **currsino**  — current sinogram.
        * **shifts**    — shift curves for all projections.
        * **error**     — error-metric curve (grows one point per iteration).
        * ``fig_main.suptitle`` — iteration counter and latest error value.
        """
        arts = self._h_arts
        axd  = self._h_axd

        # recon: update reconstructed slice
        arts["im_recon"].set_data(self.recons)
        arts["im_recon"].autoscale()
        axd["recon"].set_title(
            "Reconstructed slice — Iter {} (slice {})".format(
                self.count, self.params.get("slicenum", "?")
            )
        )

        # synthsino and currsino: update sinograms
        arts["im_synthsino"].set_data(self.sinocomp)
        arts["im_currsino"].set_data(self.sinocurr)

        # shifts: update all shift curves
        delta = self.deltaslice
        for idx, line in enumerate(arts["im_shifts_lines"]):
            line.set_ydata(delta[:, idx] if delta.ndim > 1 else delta)
        axd["shifts"].relim()
        axd["shifts"].autoscale(enable=True, axis="both")
        axd["shifts"].margins(x=0.02, y=0.08)

        # error: grow the convergence curve by one point
        n = len(self.metric_error)
        arts["im_error"].set_xdata(np.arange(n))
        arts["im_error"].set_ydata(self.metric_error)
        axd["error"].relim()
        axd["error"].autoscale(enable=True, axis="both")
        axd["error"].margins(x=0.02, y=0.08)

        # Update suptitle: include stage info when a schedule is in use
        err_val = self.metric_error[-1] if self.metric_error else float("nan")
        if self.stage_info is not None:
            stage_num, n_stages, fc = self.stage_info
            if n_stages > 1:
                stage_str = " | Stage {}/{} (fc={:.2f})".format(stage_num, n_stages, fc)
            else:
                stage_str = " (fc={:.2f})".format(fc)
        else:
            stage_str = ""
        self.fig_main.suptitle(
            "Horizontal alignment — Iter {}{} | E = {:.3e} | slice {}".format(
                self.count, stage_str, err_val, self.params.get("slicenum", "?")
            ),
            fontsize=13,
        )

        if isnotebook():
            self._dh_update(self.fig_main, "_dh_main")
        else:
            self._term_show([self.fig_main])


@interativesession
def iterative_show(
    stack_array,
    limrow=[],
    limcol=[],
    airpixel=[],
    onlyroi=False,
    colormap="bone",
    vmin=None,
    vmax=None,
):
    """
    Iterative plot of the images

    Parameters
    ----------
    stack_array : ndarray
        Array containing the stack of images to animate. The first index
        corresponds to the image number in the sequence of images.
    limrow : list of ints
        Limits of rows in the format [begining, end]
    limcol : list of ints
        Limits of cols in the format [begining, end]
    airpixel : list of ints
        Position of pixel in the air/vacuum
    onlyroi : bool
        If True, it displays only the ROI. If False, it displays the entire
        image.
    colormap : str, optional
        Colormap name. The default value is ``bone``
    vmin : float, None, optional
        Minimum gray-level. The default value is ``None``
    vmax : float, None, optional
        Maximum gray-level. The default value is ``None``

    """
    nproj, nr, nc = stack_array.shape
    if onlyroi:
        slarray0 = np.s_[limrow[0] : limrow[-1], limcol[0] : limcol[-1]]
        slarrayii = np.s_[limrow[0] : limrow[-1], limcol[0] : limcol[-1]]
    else:
        slarray0 = np.s_[:, :]
        slarrayii = np.s_[:, :]
        delimiters = True

    if limrow == [] or limrow == None:
        delimiters = False
    if limcol == [] or limcol == None:
        delimiters = False
    if vmin == "none":
        vmin = None
    if vmax == "none":
        vmax = None

    # display
    plt.close("all")
    plt.ion()
    fig = plt.figure(4)  # ,figsize=(14,6))
    ax1 = fig.add_subplot(111)
    im1 = ax1.imshow(stack_array[0][slarray0], cmap=colormap, vmin=vmin, vmax=vmax)
    if delimiters:
        ax1 = _plotdelimiters(ax1, limrow, limcol, airpixel)
    ax1.set_title("Projection: {}".format(1))
    fig.canvas.draw_idle()
    plt.pause(0.001)
    for ii in range(nproj):
        print("Projection: {}".format(ii + 1), end="\r")
        projection = stack_array[ii][slarrayii]
        im1.set_data(projection)
        ax1.set_title("Projection {}".format(ii + 1))
        if isnotebook():
            display.clear_output(wait=True)
            display.display(fig)
        else:
            fig.canvas.draw_idle()
        plt.pause(0.001)


def _animated_image(stack_array, *args):
    """
    Create an animation-ready figure using a text artist for the frame title.

    Parameters
    ----------
    stack_array : ndarray, shape (n, nr, nc)
        Stack of images to animate.
    *args
        args[0] : list of int, optional
            Row limits ``[row_start, row_end]``.
        args[1] : list of int, optional
            Column limits ``[col_start, col_end]``.
        If not provided, the full image dimensions are used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    updatefig : callable
        Frame-update function for :class:`matplotlib.animation.FuncAnimation`.
    nproj : int
        Total number of frames.
    """
    nproj, nr, nc = stack_array.shape
    if len(args) == 0:
        limrow = [0, nr]
        limcol = [0, nc]
    elif len(args) == 2:
        limrow = args[0]
        limcol = args[1]
    else:
        raise ValueError("This function accepts only two args")

    # display
    plt.close("all")
    # plt.ion()
    fig = plt.figure(4)  # ,figsize=(14,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        stack_array[0, limrow[0] : limrow[-1], limcol[0] : limcol[-1]],
        cmap="bone",
        animated=True,
    )
    # ~ title = ax.text(0.5,1.05,"",fontsize=20,bbox={'facecolor':'w','alpha':0.5,'pad':5},
    # ~ transform=ax.transAxes,ha='center')
    title = ax.text(0.5, 1.05, "", fontsize=20, transform=ax.transAxes, ha="center")
    # ~ plt.tight_layout()

    def updatefig(ii):
        global stack_array, limrow, limcol
        imgi = stack_array[ii, limrow[0] : limrow[-1], limcol[0] : limcol[-1]]
        im.set_data(imgi)
        title.set_text("Projection: {}".format(ii + 1))
        return im, title

    return fig, updatefig, nproj


def _animated_image2(stack_array, *args):
    """
    Create an animation-ready figure using the axes title for the frame label.

    Parameters
    ----------
    stack_array : ndarray, shape (n, nr, nc)
        Stack of images to animate.
    *args
        args[0] : list of int, optional
            Row limits ``[row_start, row_end]``.
        args[1] : list of int, optional
            Column limits ``[col_start, col_end]``.
        If not provided, the full image dimensions are used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    updatefig : callable
        Frame-update function for :class:`matplotlib.animation.FuncAnimation`.
    nproj : int
        Total number of frames.
    """
    nproj, nr, nc = stack_array.shape
    if len(args) == 0:
        limrow = [0, nr]
        limcol = [0, nc]
    elif len(args) == 2:
        limrow = args[0]
        limcol = args[1]
    else:
        raise ValueError("This function accepts only two args")

    # display
    plt.close("all")
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        stack_array[0, limrow[0] : limrow[-1], limcol[0] : limcol[-1]],
        cmap="bone",
        animated=True,
    )
    plt.tight_layout()
    arr1 = [None]

    def updatefig(ii):
        global stack_array, limrow, limcol
        ax.set_title("Projection: {}".format(ii + 1), fontsize=20)
        if arr1[0]:
            arr1[0].remove()
        arr1[0] = im.set_data(
            stack_array[ii, limrow[0] : limrow[-1], limcol[0] : limcol[-1]]
        )

    return fig, updatefig, nproj


def animated_image(stack_array, *args):
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
    fig, updatefig, nproj = _animated_image(stack_array, *args)
    ani = animation.FuncAnimation(
        fig, updatefig, frames=nproj, interval=50, blit=False, repeat=False
    )
    plt.show()


class ShowProjections:
    """
    Show projections and probe
    """

    def __init__(self):
        """
        Initializer of show_projections

        """
        self.idxp = 0
        plt.ion()
        print("Showing reconstructions for each angle")

    def __call__(self, obj, probe, idxp):
        return self.show_projections(obj, probe, idxp)

    @interativesession
    def show_projections(self, obj, probe, idxp):
        """
        Show the object and the probe
        Parameters
        ----------
        obj : ndarray
            Object to show
        probe : ndarray
            Probe to show
        idxp : int
            Projection number
        """
        if probe.ndim == 3:
            probe = probe[0]
        self.objamp = np.abs(obj)
        self.objph = np.angle(obj)
        self.probergb = hsv_to_rgb(self.probe2HSV(probe))
        self.idxp = idxp
        self.nr, self.nc = self.objph.shape
        plotgrid = (1, 3)
        plotsize = (18, 6)
        vabsmean = self.objamp.mean()
        perabsmean = 0.2 * vabsmean
        self.cmin = vabsmean - perabsmean
        self.cmax = vabsmean + perabsmean
        if idxp == 0:
            # display first image
            plt.close("all")
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
                num=1, nrows=plotgrid[0], ncols=plotgrid[1], figsize=plotsize
            )
            self.im1 = self.ax1.imshow(
                self.objamp,
                interpolation="none",
                cmap="gray",
                vmin=self.cmin,
                vmax=self.cmax,
            )
            self.ax1.set_title("Object magnitude - projection {}".format(self.idxp + 1))
            self.im2 = self.ax2.imshow(
                self.objph, interpolation="none", cmap="bone", vmin=-np.pi, vmax=np.pi
            )
            self.ax2.set_title("Object Phase - projection {}".format(self.idxp + 1))
            self.im3 = self.ax3.imshow(self.probergb, interpolation="none")
            self.ax3.set_title("Probe - projection {}".format(self.idxp + 1))
            self.ax3.axis("image")
            # ~ fig.colorbar(im1,ax=ax1)
            # ~ fig.colorbar(im2,ax=ax2)
            # ~ # Set the colormap and norm to correspond to the data for which
            # ~ # the colorbar will be used.
            # ~ norm = mpl.colors.Normalize(-np.pi,np.pi)
            # ~ cmap = mpl.cm.colors.hsv_to_rgb # TO BE FIXED
            # ~ fig.colorbar(im3,ax=ax3,cmap=mpl.cm.get_cmap('hsv'),norm=norm) # TO BE FIXED
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        else:
            self.update_show()

    @interativesession
    def update_show(self):
        """
        Update the canvas
        """
        self.im1.set_data(self.objamp)
        self.im1.set_cmap("gray")
        self.im1.set_clim((self.cmin, self.cmax))
        self.im1.set_interpolation(u"none")
        self.ax1.set_title("Object magnitude - projection {}".format(self.idxp + 1))
        self.im2.set_data(self.objph)
        self.im1.set_cmap("bone")
        self.im2.set_interpolation(u"none")
        self.ax2.set_title("Object Phase - projection {}".format(self.idxp + 1))
        self.im3.set_data(self.probergb)
        self.im3.set_interpolation(u"none")
        self.ax3.set_title("Probe (1st mode) - projection {}".format(self.idxp + 1))
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    @staticmethod
    def probe2HSV(probe):
        """
        Special tricks for the probe display in HSV
        """
        # Special tricks for the probe display
        H = np.angle(probe) / (2 * np.pi) + 0.5
        S = np.ones_like(H).astype(int)
        V = np.abs(probe) / np.max(np.abs(probe))
        return np.dstack((H, S, V))


@interativesession
def plot_checkangles(angles):
    """
    Plot the angles for each projections and the derivatives to check
    for anomalies

    Parameters
    ----------
    angles : array_like
        Array of angles
    """
    # plot the angles for verification
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(num=1, nrows=2, ncols=1)
    pltangles = ax1.plot(angles, "ro")
    ax1.set_xlabel("projection")
    ax1.set_ylabel("Theta angles")
    ax1.axis("tight")
    pltdiffangles = ax2.plot(np.diff(sorted(angles)), "ro-")
    ax2.set_xlabel("Sorted projections")
    ax2.set_ylabel("Angular spacing")
    ax2.axis("tight")
    plt.tight_layout()
    fig.canvas.draw_idle()


def show_linearphase(image, mask, *args):
    """
    Display a phase projection with an overlaid mask and a horizontal line cut.

    Parameters
    ----------
    image : ndarray, shape (nr, nc)
        Phase image to display.
    mask : ndarray, shape (nr, nc)
        Mask added to ``image`` for the 2-D panel.
    *args
        args[0] : int or str, optional
            Projection index used in the figure title.  Defaults to an
            empty string if not provided.
    """
    try:
        idxproj = args[0]
    except:
        idxproj = ""

    linecut = np.round(image.shape[0] / 2.0)

    fig, (ax1, ax2) = plt.subplots(num=3, nrows=2, ncols=1, figsize=(14, 10))
    im1 = ax1.imshow(image + mask, cmap="bone")
    ax1.set_title("Projection {}".format(idxproj))
    im2 = ax2.plot(image[linecut, :])
    ax2.plot([0, image.shape[1]], [0, 0])
    ax2.axis("tight")
    plt.draw()
    # ax2.cla()


def display_slice(recons, colormap="bone", vmin=None, vmax=None):
    """
    Display tomographic slice

    Parameters
    ----------
    recons : array_like
        Tomographic slice
    colormap : str, optional
        Colormap name. The default value is ``bone``
    vmin : float, None
        Minimum gray-level. The default value is ``None``
    vmax : float, None
        Maximum gray-level. The default value is ``None``
    """
    if vmin == "none":
        vmin = None
    if vmax == "none":
        vmax = None

    # plt.close("all")
    if isnotebook(): fig = plt.figure(figsize=(12,5))
    else: fig = plt.figure()
    plt.clf()
    ax1 = fig.add_subplot(111)
    ax1.imshow(recons, cmap=colormap, vmin=vmin, vmax=vmax)
    ax1.axis("image")
    ax1.set_title("Aligned tomographic slice")
    ax1.set_xlabel("x [pixels]")
    ax1.set_ylabel("y [pixels]")
    if isnotebook():
        display.display(fig)
        plt.close(fig)
    else:
        plt.show(block=False)
