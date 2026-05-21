#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import functools
import io as _io
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

    In Jupyter: calls display.display(fig) then closes if close=True, which
    prevents %matplotlib inline from rendering the figure a second time at
    cell end. Only pass close=False for figures that are updated across
    iterations or returned for later use.

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
        display.display(fig)
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
        fig = plt.figure(figsize=(10, 5))
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
    if isnotebook():
        display.display(fig)
        plt.close(fig)
    else:
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
                   label=f"Resolution ≈ {fn_res:.3f} × Nyquist")

    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Spatial frequency / Nyquist")
    ax.set_ylabel("SSNR")
    ax.set_title(f"Spectral Signal-to-Noise Ratio ({suffix})")
    ax.grid(True, linestyle="--", alpha=0.5)
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
        ``(FSC_obs - FSC_rand) / (1 - FSC_rand)``.
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
    ax1.set_ylim(-0.1, 1.1)
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
    Create the initial matplotlib canvas for horizontal alignment plots.

    Parameters
    ----------
    recons : ndarray
        Current reconstructed slice.
    sinoorig : ndarray
        Original (unaligned) sinogram.
    sinocurr : ndarray
        Current aligned sinogram.
    sinocomp : ndarray
        Synthetic sinogram computed from the reconstruction.
    deltaslice : ndarray
        Current horizontal shift estimates.
    metric_error : list of float
        Error metric history.
    **params
        Must contain ``'slicenum'`` (int), ``'sinohigh'`` (float),
        and ``'sinolow'`` (float).

    Returns
    -------
    fig_array : list of Figure
        Figures ``[fig1, fig2, fig3]``.
    im_array : list
        Image/line artist objects for later updates.
    ax_array : list of Axes
        Axes objects for later updates.
    """
    slicenum = params["slicenum"]
    cmax = params["sinohigh"]
    cmin = params["sinolow"]

    # Display one reconstructed slice
    if isnotebook(): fig1 = plt.figure(num=1,figsize=(12,5))
    else: fig1 = plt.figure(num=1)
    plt.clf()
    ax11 = fig1.add_subplot(111)
    im11 = ax11.imshow(recons, cmap="jet")
    ax11.axis("image")
    ax11.set_title("Initial slice number: {}".format(slicenum))
    ax11.set_xlabel("x [pixels]")
    ax11.set_ylabel("y [pixels]")
    fig1.tight_layout()

    # Display initial, current and synthetic sinograms
    fig2 = plt.figure(num=2, figsize=(6, 10))
    plt.clf()
    ax21 = fig2.add_subplot(311)
    im21 = ax21.imshow(sinoorig, cmap="bone", vmin=cmin, vmax=cmax)
    ax21.axis("tight")
    ax21.set_title("Initial sinogram")
    ax21.set_xlabel("Projection")
    ax21.set_ylabel("x [pixels]")
    ax22 = fig2.add_subplot(312)
    im22 = ax22.imshow(sinocurr, cmap="bone", vmin=cmin, vmax=cmax)
    ax22.axis("tight")
    ax22.set_title("Current sinogram")
    ax22.set_xlabel("Projection")
    ax22.set_ylabel("x [pixels]")
    ax23 = fig2.add_subplot(313)
    im23 = ax23.imshow(sinocomp, cmap="bone", vmin=cmin, vmax=cmax)
    ax23.axis("tight")
    ax23.set_title("Synthetic sinogram")
    ax23.set_xlabel("Projection")
    ax23.set_ylabel("x [pixels]")
    fig2.tight_layout()

    # Display deltaslice and metric_error
    fig3 = plt.figure(num=3)
    plt.clf()
    ax31 = fig3.add_subplot(211)
    im31 = ax31.plot(deltaslice)
    ax31.axis("tight")
    ax31.set_title("Object position")
    ax32 = fig3.add_subplot(212)
    (im32,) = ax32.plot(metric_error, "bo-")
    ax32.axis("tight")
    ax32.set_title("Error metric — iter 0")
    fig3.tight_layout()

    fig_array = [fig1, fig2, fig3]
    im_array = [im11, im21, im22, im23, im31, im32]
    ax_array = [ax11, ax21, ax22, ax23, ax31, ax32]

    return (fig_array, im_array, ax_array)


def _createcanvasvertical(
    proj, lims, vertfluctinit, vertfluctcurr, deltastack, metric_error, **params
):
    """
    Create the initial matplotlib canvas for vertical alignment plots.

    Parameters
    ----------
    proj : ndarray
        First projection image (for the ROI overlay panel).
    lims : tuple of array_like
        ``(limrow, limcol)`` ROI limits.
    vertfluctinit : ndarray, shape (n, n_rows_roi)
        Initial vertical fluctuation signals.
    vertfluctcurr : ndarray, shape (n, n_rows_roi)
        Current vertical fluctuation signals.
    deltastack : ndarray, shape (2, n)
        Current shift estimates.
    metric_error : list of float
        Error metric history.
    **params
        Additional display parameters (unused directly; forwarded for
        consistency with ``plotsvertical``).

    Returns
    -------
    fig_array : list of Figure
        Figures ``[fig1, fig2, fig3, fig4]``.
    im_array : list
        Image/line artist objects for later updates.
    ax_array : list of Axes
        Axes objects for later updates.
    """
    limrow, limcol = lims

    # figures display
    nr, nc = vertfluctinit.shape  # for the image display
    if nc > nr:
        figsize = (np.round(6 * nc / nr), 6)
    else:
        figsize = (6, np.round(6 * nr / nc))

    # display one projection with limits
    fig1 = plt.figure(num=1)
    plt.clf()
    ax11 = fig1.add_subplot(111)
    im11 = ax11.imshow(proj, cmap="bone")
    ax11.set_title("Projection")
    ax11.axis("image")
    ax11 = _plotdelimiters(ax11, limrow, limcol)
    fig1.tight_layout()

    # display vertical fluctuations as 2D images
    fig2 = plt.figure(num=2, figsize=figsize)
    plt.clf()
    ax21 = fig2.add_subplot(211)
    im21 = ax21.imshow(vertfluctinit, cmap="jet", interpolation="none")
    ax21.axis("tight")
    ax21.set_title("Initial Integral in x")
    ax21.set_xlabel("Projection")
    ax21.set_ylabel("y [pixels]")
    ax22 = fig2.add_subplot(212)
    im22 = ax22.imshow(vertfluctcurr, cmap="jet", interpolation="none")
    ax22.axis("tight")
    ax22.set_title("Current Integral in x")
    ax22.set_xlabel("Projection")
    ax22.set_ylabel("y [pixels]")
    fig2.tight_layout()

    # display vertical fluctuations as plots
    fig3 = plt.figure(num=3)
    plt.clf()
    ax31 = fig3.add_subplot(211)
    im31 = ax31.plot(vertfluctinit)
    (im31a,) = ax31.plot(vertfluctinit.mean(axis=1), "r", linewidth=2.5)
    (im31b,) = ax31.plot(vertfluctinit.mean(axis=1), "--w", linewidth=1.5)
    ax31.axis("tight")
    ax31.set_title("Initial Integral in x")
    ax31.set_xlabel("Vertical coordinates [pixels]")
    ax31.set_ylabel("y [pixels]")
    ax32 = fig3.add_subplot(212)
    im32 = ax32.plot(vertfluctcurr)
    (im32a,) = ax32.plot(vertfluctcurr.mean(axis=1), "r", linewidth=2.5)
    (im32b,) = ax32.plot(vertfluctcurr.mean(axis=1), "--w", linewidth=1.5)
    ax32.axis("tight")
    ax32.set_title("Current Integral in x — iter 0")
    ax32.set_xlabel("Vertical coordinates [pixels]")
    ax32.set_ylabel("y [pixels]")
    fig3.tight_layout()

    # shifts and error metric
    fig4 = plt.figure(num=4)
    plt.clf()
    ax41 = fig4.add_subplot(211)
    im41 = ax41.plot(deltastack)
    ax41.axis("tight")
    ax41.set_title("Object position")
    ax42 = fig4.add_subplot(212)
    (im42,) = ax42.plot(metric_error, "bo-")
    ax42.axis("tight")
    ax42.set_title("Error metric — iter 0")
    fig4.tight_layout()

    fig_array = [fig1, fig2, fig3, fig4]
    im_array = [im11, im21, im22, im31, im31a, im31b, im32, im32a, im32b, im41, im42]
    ax_array = [ax11, ax21, ax22, ax31, ax32, ax41, ax42]

    return (fig_array, im_array, ax_array)


class RegisterPlot:
    """
    Manage live plot updates during tomographic projection alignment.

    Provides two high-level methods — :meth:`plotsvertical` and
    :meth:`plotshorizontal` — that create and update the alignment
    diagnostic figures in both Jupyter (``%matplotlib widget``) and
    terminal environments.

    Display strategy
    ----------------
    **Jupyter / %matplotlib widget**
        Figures are rendered to PNG via :meth:`~matplotlib.figure.Figure.savefig`
        (which requires no figure manager) and shown inside
        :class:`ipywidgets.Output` containers.  On the first call the
        containers are created and embedded in the cell output; on every
        subsequent call :meth:`ipywidgets.Output.clear_output` replaces
        the PNG in-place without spawning a new cell output.  This avoids
        all issues with ``canvas.draw()``, ``canvas.manager``, and
        ipympl's ``_shown`` flag.

    **Terminal**
        Figures are updated via ``canvas.draw_idle()`` + ``plt.pause()``.

    Parameters
    ----------
    **params
        Algorithm parameters forwarded to the underlying canvas helpers.
        Must include at least ``'slicenum'``, ``'sinohigh'``, and
        ``'sinolow'``.
    """

    def __init__(self, **params):
        self.params = params
        self.count = 0
        plt.close("all")

    # ------------------------------------------------------------------ #
    # Internal display helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fig_to_png(fig):
        """Render *fig* to a PNG byte-string using the Agg renderer.

        Works without a figure manager, so it is safe for all matplotlib
        backends including ``%matplotlib widget`` (ipympl).
        """
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        return buf.read()

    def _nb_init(self, figs):
        """Create one ``ipywidgets.Output`` container per figure, display
        all containers in the current cell, then populate each with a PNG
        of its figure.

        Containers are stored in ``self._nb_outs`` so that ``_nb_update``
        can refresh them in-place on later iterations.
        """
        try:
            from ipywidgets import Output
        except ImportError:
            # ipywidgets not available — plain display fallback
            for fig in figs:
                display.display(display.Image(self._fig_to_png(fig)))
            self._nb_outs = None
            return

        self._nb_outs = [Output() for _ in figs]
        # Embed all containers now so they appear in document order
        for out in self._nb_outs:
            display.display(out)
        # Fill each container with its figure's initial render
        for fig, out in zip(figs, self._nb_outs):
            with out:
                display.display(display.Image(self._fig_to_png(fig)))

    def _nb_update(self, figs):
        """Replace each Output container's content with a fresh PNG render.

        ``clear_output(wait=True)`` swaps the old image for the new one
        in-place — no new cell output is created.
        """
        if not hasattr(self, "_nb_outs") or self._nb_outs is None:
            # Fallback: plain display
            for fig in figs:
                display.display(display.Image(self._fig_to_png(fig)))
            return
        for fig, out in zip(figs, self._nb_outs):
            out.clear_output(wait=True)
            with out:
                display.display(display.Image(self._fig_to_png(fig)))

    @staticmethod
    def _term_show(figs):
        """Redraw figures in a terminal GUI event loop."""
        for fig in figs:
            fig.canvas.draw_idle()
        plt.pause(0.001)

    def _show(self, figs, init=False):
        """Backend-agnostic show/update dispatcher.

        Parameters
        ----------
        figs : tuple of Figure
        init : bool
            True on the very first call (creates Output containers in
            notebook mode).
        """
        if isnotebook():
            if init:
                self._nb_init(figs)
            else:
                self._nb_update(figs)
        else:
            self._term_show(figs)

    # ------------------------------------------------------------------ #
    # Vertical alignment
    # ------------------------------------------------------------------ #

    @interativesession
    def plotsvertical(
        self, proj, lims, vertfluctinit, vertfluctcurr, deltastack, metric_error, count
    ):
        """Display or update the four vertical-alignment diagnostic figures.

        On the first call the figures are created via
        :func:`_createcanvasvertical` and displayed.  On every subsequent
        call the existing artists are updated in-place by
        :meth:`updatevertical`.

        Parameters
        ----------
        proj : ndarray
            Current projection image.
        lims : tuple
            ``(limrow, limcol)`` ROI limits.
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
        """
        self.proj = proj
        self.lims = lims
        self.vertfluctinit     = vertfluctinit.T
        self.vertfluctinit_avg = self.vertfluctinit.mean(axis=1)
        self.vertfluctcurr     = vertfluctcurr.T
        self.vertfluctcurr_avg = self.vertfluctcurr.mean(axis=1)
        self.deltastack        = deltastack.T
        self.metric_error      = metric_error
        self.count             = count

        if not hasattr(self, "fig1"):
            # First call — create all four figures.
            # In notebook mode the figure creation is wrapped in a throwaway
            # Output so any stray auto-display from plt.figure() is captured
            # and discarded; we then display our own PNG Output containers.
            if isnotebook():
                try:
                    from ipywidgets import Output as _Out
                    _cap = _Out()
                except ImportError:
                    _cap = _NullContext()
            else:
                _cap = _NullContext()

            with _cap:
                fig_array, im_array, ax_array = _createcanvasvertical(
                    self.proj, self.lims,
                    self.vertfluctinit, self.vertfluctcurr,
                    self.deltastack, self.metric_error,
                    **self.params
                )

            # Store figure references
            self.fig1, self.fig2, self.fig3, self.fig4 = fig_array

            # Store artist references
            (self.im11, self.im21, self.im22,
             self.im31, self.im31a, self.im31b,
             self.im32, self.im32a, self.im32b,
             self.im41, self.im42) = im_array

            # Store axes references
            (self.ax11, self.ax21, self.ax22,
             self.ax31, self.ax32,
             self.ax41, self.ax42) = ax_array

            self._show((self.fig1, self.fig2, self.fig3, self.fig4), init=True)
        else:
            self.updatevertical()

    @interativesession
    def updatevertical(self):
        """Update the four vertical-alignment figures in-place.

        Modifies existing artist objects (no new figures or axes created)
        and re-renders via PNG for notebook or ``draw_idle`` for terminal.

        Evolution tracking
        ------------------
        * **Fig 2** – current vertical-fluctuation image; initial image fixed.
        * **Fig 3** – current integral line plot; initial panel fixed.
        * **Fig 4** – shift curves updated; error-metric curve *grows* one
          point per iteration so the full convergence history is visible.
        """
        # Fig 2: current vertical-fluctuation image
        self.im22.set_data(self.vertfluctcurr)
        self.im22.autoscale()
        self.ax22.set_title("Current Integral in x — iter {}".format(self.count))

        # Fig 3: current integral lines + averages
        curr = self.vertfluctcurr
        for idx, line in enumerate(self.im32):
            line.set_ydata(curr[:, idx] if curr.ndim > 1 else curr)
        self.im32a.set_ydata(self.vertfluctcurr_avg)
        self.im32b.set_ydata(self.vertfluctcurr_avg)
        self.ax32.relim()
        self.ax32.autoscale_view()
        self.ax32.set_title("Current Integral in x — iter {}".format(self.count))

        # Fig 4: shift curves + growing error-metric
        delta = self.deltastack
        for idx, line in enumerate(self.im41):
            line.set_ydata(delta[:, idx] if delta.ndim > 1 else delta)
        self.ax41.relim()
        self.ax41.autoscale_view()
        n = len(self.metric_error)
        self.im42.set_xdata(range(n))
        self.im42.set_ydata(self.metric_error)
        self.ax42.relim()
        self.ax42.autoscale_view()
        self.ax42.set_title("Error metric — iter {}".format(self.count))

        self._show((self.fig1, self.fig2, self.fig3, self.fig4))

    # ------------------------------------------------------------------ #
    # Horizontal alignment
    # ------------------------------------------------------------------ #

    @interativesession
    def plotshorizontal(
        self, recons, sinoorig, sinocurr, sinocomp, deltaslice, metric_error, count
    ):
        """Display or update the three horizontal-alignment diagnostic figures.

        On the first call the figures are created via
        :func:`_createcanvashorizontal` and displayed.  On every subsequent
        call the existing artists are updated in-place by
        :meth:`updatehorizontal`.

        Parameters
        ----------
        recons : ndarray
            Current reconstructed slice.
        sinoorig : ndarray
            Original sinogram (fixed reference).
        sinocurr : ndarray
            Current aligned sinogram.
        sinocomp : ndarray
            Synthetic sinogram from reconstruction.
        deltaslice : ndarray
            Current horizontal shift estimates.
        metric_error : list of float
            Error metric history.
        count : int
            Current iteration number.
        """
        self.recons       = recons
        self.sinoorig     = sinoorig
        self.sinocurr     = sinocurr
        self.sinocomp     = sinocomp
        self.deltaslice   = deltaslice.T
        self.metric_error = metric_error
        self.count        = count

        if not hasattr(self, "fig1"):
            if isnotebook():
                try:
                    from ipywidgets import Output as _Out
                    _cap = _Out()
                except ImportError:
                    _cap = _NullContext()
            else:
                _cap = _NullContext()

            with _cap:
                fig_array, im_array, ax_array = _createcanvashorizontal(
                    self.recons, self.sinoorig, self.sinocurr, self.sinocomp,
                    self.deltaslice, self.metric_error,
                    **self.params
                )

            self.fig1, self.fig2, self.fig3 = fig_array

            (self.im11, self.im21, self.im22,
             self.im23, self.im31, self.im32) = im_array

            (self.ax11, self.ax21, self.ax22,
             self.ax23, self.ax31, self.ax32) = ax_array

            self._show((self.fig1, self.fig2, self.fig3), init=True)
        else:
            self.updatehorizontal()

    @interativesession
    def updatehorizontal(self):
        """Update the three horizontal-alignment figures in-place.

        Modifies existing artist objects and re-renders via PNG (notebook)
        or ``draw_idle`` (terminal).

        Evolution tracking
        ------------------
        * **Fig 1** – reconstructed slice sharpens as alignment improves.
        * **Fig 2** – current and synthetic sinograms updated; original fixed.
        * **Fig 3** – shift curves updated; error-metric curve *grows* one
          point per iteration.
        """
        # Fig 1: reconstructed slice
        self.im11.set_data(self.recons)
        self.im11.autoscale()
        self.ax11.set_title("Reconstructed slice — iteration {}".format(self.count))

        # Fig 2: current and synthetic sinograms (original stays fixed)
        self.im22.set_data(self.sinocurr)
        self.im23.set_data(self.sinocomp)

        # Fig 3: shift curves + growing error-metric
        delta = self.deltaslice
        for idx, line in enumerate(self.im31):
            line.set_ydata(delta[:, idx] if delta.ndim > 1 else delta)
        self.ax31.relim()
        self.ax31.autoscale_view()
        n = len(self.metric_error)
        self.im32.set_xdata(range(n))
        self.im32.set_ydata(self.metric_error)
        self.ax32.relim()
        self.ax32.autoscale_view()
        self.ax32.set_title("Error metric — iter {}".format(self.count))

        self._show((self.fig1, self.fig2, self.fig3))


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
