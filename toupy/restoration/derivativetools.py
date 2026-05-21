#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard packages
import os
import warnings

# third party packages
from matplotlib.widgets import Button
from ..utils.plot_utils import plt
import numpy as np
from scipy.fft import fftfreq, fft, ifft
from ..utils import tqdm

# local packages
from ..registration.shift import ShiftFunc
from ..utils.plot_utils import _plotdelimiters
from ..utils import isnotebook

__all__ = [
    "calculate_derivatives",
    "calculate_derivatives_fft",   # deprecated alias — use calculate_derivatives
    "chooseregiontoderivatives",
    "derivatives",
    "derivatives_fft",             # deprecated alias — use derivatives
    "derivatives_sino",
    "gradient_axis",
]


def gradient_axis(x, axis=-1):
    """
    Compute the forward-difference gradient along one axis, preserving shape.

    Unlike :func:`numpy.gradient`, this function keeps all dimensions
    unchanged and sets the last slice along the chosen axis to zero.

    Parameters
    ----------
    x : ndarray, shape (..., M, N)
        Input 2-D (or higher-dimensional) array.
    axis : int, optional
        Axis along which to compute the difference.  ``-1`` (default)
        computes the difference along columns; ``0`` computes it along rows.

    Returns
    -------
    ndarray
        Array of the same shape as ``x`` containing the forward finite
        differences along ``axis``.
    """
    # Single output array: out[i] = x[i+1] - x[i], last slice = 0.
    # Avoids allocating two full temporaries of the same shape.
    out = np.empty_like(x)
    if axis != 0:
        out[:, :-1] = x[:, 1:] - x[:, :-1]
        out[:, -1] = 0
    else:
        out[:-1, :] = x[1:, :] - x[:-1, :]
        out[-1, :] = 0
    return out


class _DerivROIPicker:
    """
    Interactive GUI for selecting the derivative ROI.

    Click two corners of the desired rectangle on the first projection:

    1. **Top-left** corner  → sets the left X margin (``deltax``) and the
       top row limit (``roiy[0]``).
    2. **Bottom-right** corner → sets the bottom row limit (``roiy[-1]``);
       the right X margin mirrors ``deltax`` symmetrically.

    Then click **Confirm** to accept, or **Reset** to start over.

    The ROI is:

    - ``roix = range(deltax, width - deltax)``
    - ``roiy = range(y_top, y_bottom)``

    Terminal / Jupyter two-cell workflow
    ------------------------------------
    Used internally by :func:`chooseregiontoderivatives`.  In Jupyter access
    ``picker.roix`` and ``picker.roiy`` in the **next** cell after clicking
    Confirm.
    """

    _STEPS = [
        "Step 1/2: click the TOP-LEFT corner of the derivative region",
        "Step 2/2: click the BOTTOM-RIGHT corner of the derivative region",
    ]

    def __init__(self, stack_array, deltax_init, limsy_init):
        nr, nc = stack_array[0].shape
        self._nc   = nc
        self._step = 0
        self._tl   = None   # top-left (x, y) click
        self._done = False

        # public results — pre-populated with the params defaults
        self.roix = range(deltax_init, nc - deltax_init)
        self.roiy = range(limsy_init[0], limsy_init[1])

        # ---- figure ----
        plt.close("all")
        fig, ax = plt.subplots(figsize=(9, 7))
        self.fig = fig
        self.ax  = ax

        ax.imshow(stack_array[0], cmap="bone", aspect="auto")
        # draw the initial ROI
        self._rect_lines = []
        self._draw_rect(deltax_init, limsy_init[0], nc - deltax_init, limsy_init[1])
        ax.axis("tight")
        ax.set_title(
            "First projection — current ROI shown in red\n"
            "Image: {} rows × {} cols".format(nr, nc),
            fontsize=9,
        )

        self._status_txt = fig.text(
            0.5, 0.975, self._STEPS[0],
            ha="center", va="top", fontsize=10,
            color="steelblue", fontweight="bold",
        )
        self._info_txt = fig.text(
            0.5, 0.935, "Current: deltax={}, y=[{}, {}]".format(
                deltax_init, limsy_init[0], limsy_init[1]
            ),
            ha="center", va="top", fontsize=9, color="0.35",
        )

        ax_reset   = fig.add_axes([0.25, 0.01, 0.2,  0.055])
        ax_confirm = fig.add_axes([0.55, 0.01, 0.2,  0.055])
        self._btn_reset   = Button(ax_reset,   "Reset")
        self._btn_confirm = Button(ax_confirm, "Confirm")
        self._btn_confirm.ax.set_facecolor("0.85")
        self._btn_reset.on_clicked(self._on_reset)
        self._btn_confirm.on_clicked(self._on_confirm)

        self._cid = fig.canvas.mpl_connect("button_press_event", self._on_click)
        print(self._STEPS[0], flush=True)

    # ------------------------------------------------------------------ helpers

    def _clear_rect(self):
        for ln in self._rect_lines:
            try: ln.remove()
            except Exception: pass
        self._rect_lines = []

    def _draw_rect(self, x0, y0, x1, y1):
        self._clear_rect()
        kw = dict(color="r", linewidth=1.5, linestyle="-")
        for xs, ys in [([x0, x1], [y0, y0]), ([x0, x1], [y1, y1]),
                       ([x0, x0], [y0, y1]), ([x1, x1], [y0, y1])]:
            ln, = self.ax.plot(xs, ys, **kw)
            self._rect_lines.append(ln)

    def _set_info(self, msg):
        self._info_txt.set_text(msg)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ events

    def _on_click(self, event):
        if event.button != 1 or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._done or self._step >= 2:
            return

        x, y = event.xdata, event.ydata

        if self._step == 0:
            self._tl = (x, y)
            deltax = max(0, int(round(x)))
            dot, = self.ax.plot(x, y, "bs", markersize=7)
            self._rect_lines.append(dot)
            self.fig.canvas.draw_idle()
            self._step = 1
            self._status_txt.set_text(self._STEPS[1])
            self._set_info(
                "Top-left: ({:.0f}, {:.0f})  →  deltax={}".format(x, y, deltax)
            )
            print("  Top-left: ({:.0f}, {:.0f})  deltax={}".format(x, y, deltax),
                  flush=True)
            print(self._STEPS[1], flush=True)

        elif self._step == 1:
            x0, y0 = self._tl
            x1, y1 = x, y
            # enforce ordering
            if y1 < y0: y0, y1 = y1, y0
            deltax  = max(0, int(round(min(x0, x1))))
            y_top   = int(round(y0))
            y_bot   = int(round(y1))
            self.roix = range(deltax, self._nc - deltax)
            self.roiy = range(y_top, y_bot)
            self._draw_rect(deltax, y_top, self._nc - deltax, y_bot)
            self._step = 2
            self._status_txt.set_text(
                "All set — click  Confirm  to accept  |  Reset  to redo"
            )
            self._btn_confirm.ax.set_facecolor("limegreen")
            info = "deltax={}  y=[{}, {}]  → roix len={}, roiy len={}".format(
                deltax, y_top, y_bot, len(self.roix), len(self.roiy)
            )
            self._set_info(info)
            print("  " + info, flush=True)
            print("All set — click  Confirm  to accept, or  Reset  to redo.",
                  flush=True)

    def _on_reset(self, event):
        self._step = 0
        self._tl   = None
        self.roix  = None
        self.roiy  = None
        self._btn_confirm.ax.set_facecolor("0.85")
        self._status_txt.set_text(self._STEPS[0])
        self._set_info("")
        print("Reset — " + self._STEPS[0], flush=True)

    def _on_confirm(self, event):
        if self._step < 2:
            print("Not ready yet — complete both clicks first.", flush=True)
            return
        self._done = True
        try: self.fig.canvas.mpl_disconnect(self._cid)
        except Exception: pass
        print(
            "Confirmed — roix=[{}, {}]  roiy=[{}, {}]".format(
                self.roix[0], self.roix[-1], self.roiy[0], self.roiy[-1]
            ),
            flush=True,
        )
        plt.close(self.fig)


def chooseregiontoderivatives(stack_array, **params):
    """
    Interactively choose the region of interest for derivative computation.

    Opens a GUI figure showing the first projection with the current ROI
    (from ``params``) overlaid in red.  Click two corners to redefine the
    rectangle, then click **Confirm**.

    Parameters
    ----------
    stack_array : ndarray, shape (n, nr, nc)
        Stack of projection images.
    **params
        Must contain:

        deltax : int
            Initial horizontal margin (pixels) from the left/right edges.
        limsy : tuple of int
            Initial ``(row_start, row_end)`` vertical limits.

    Returns
    -------
    Terminal mode
        ``(roix, roiy)`` — both are :class:`range` objects, ready to use.
    Jupyter mode  (two-cell workflow)
        ``picker`` (:class:`_DerivROIPicker`) — access ``picker.roix`` and
        ``picker.roiy`` in the **next** cell after clicking Confirm.
    """
    deltax_init = params["deltax"]
    limsy_init  = params["limsy"]

    picker = _DerivROIPicker(stack_array, deltax_init, limsy_init)

    if isnotebook():
        try:
            import matplotlib as _mpl
            _interactive = "inline" not in _mpl.get_backend().lower()
        except Exception:
            _interactive = False
        if _interactive:
            plt.show(block=False)
            picker.fig.canvas.draw()
            print(
                "\nJupyter two-cell workflow:\n"
                "  Interact with the figure, then in the NEXT cell:\n"
                "      roix, roiy = picker.roix, picker.roiy",
                flush=True,
            )
        else:
            from IPython import display as _ipy_display
            _ipy_display.display(picker.fig)
        return picker
    else:
        plt.show(block=True)
        return picker.roix, picker.roiy


def derivatives(input_array, shift_method="fourier", symmetric=True, n_cpus=-1):
    """
    Calculate the horizontal derivative of a 2-D image.

    Parameters
    ----------
    input_array : array_like
        Input 2-D image.
    shift_method : str, optional
        Shift / differentiation method.

        ``"fourier"`` (default)
            Pure FFT symmetric-difference filter.  Applied directly via
            :func:`scipy.fft.fft` — no ``ShiftFunc`` overhead.  Supports
            the ``symmetric`` and ``n_cpus`` parameters.

        ``"spline"``, ``"linear"``
            Sub-pixel shift via :class:`~toupy.registration.shift.ShiftFunc`
            (always symmetric ±0.5 px).  ``symmetric`` and ``n_cpus`` are
            ignored for these methods.

    symmetric : bool, optional
        Only used when ``shift_method="fourier"``.  If ``True`` (default),
        applies a symmetric ±½-pixel difference (filter
        ``2i·sin(π f)``).  If ``False``, applies a forward 1-pixel
        difference (filter ``exp(2πi f) − 1``).
    n_cpus : int, optional
        Number of threads passed to :func:`scipy.fft.fft`.
        Only used when ``shift_method="fourier"``.
        ``-1`` (default) uses all available cores.

    Returns
    -------
    diffimg : ndarray
        Derivative image (same shape as ``input_array``).
    """
    if shift_method == "fourier":
        if n_cpus < 0:
            n_cpus = os.cpu_count() or 1
        freqs = fftfreq(input_array.shape[1])
        if symmetric:
            rshift, lshift = 0.5, 0.5
        else:
            rshift, lshift = 1.0, 0.0
        kernel = (
            np.exp( 1j * 2.0 * np.pi * freqs * rshift)
            - np.exp(-1j * 2.0 * np.pi * freqs * lshift)
        )
        return ifft(kernel * fft(input_array, workers=n_cpus),
                    workers=n_cpus).real

    # Non-fourier methods use ShiftFunc (C extensions, GIL released)
    S = ShiftFunc(shiftmeth=shift_method)
    return S(input_array, [0, 0.5]) - S(input_array, [0, -0.5])


def derivatives_fft(input_img, symmetric=True, n_cpus=-1):
    """
    .. deprecated::
        Use :func:`derivatives` with ``shift_method="fourier"`` instead.
    """
    warnings.warn(
        "derivatives_fft() is deprecated; use derivatives(shift_method='fourier') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return derivatives(input_img, shift_method="fourier",
                       symmetric=symmetric, n_cpus=n_cpus)


def calculate_derivatives(stack_array, roiy, roix,
                           shift_method="fourier", symmetric=True, n_cpus=-1):
    """
    Compute projection derivatives over a stack of images.

    Parameters
    ----------
    stack_array : array_like, shape (nprojs, nr, nc)
        Input stack of projection images.
    roiy, roix : range or tuple
        Row and column limits of the ROI.
    shift_method : str, optional
        Passed to :func:`derivatives`.  Default ``"fourier"``.
    symmetric : bool, optional
        Passed to :func:`derivatives` (fourier path only).
        Default ``True``.
    n_cpus : int, optional
        Number of CPU cores / threads.

        * ``"fourier"`` — passed as ``workers`` to :func:`scipy.fft.fft`,
          which parallelises across **all** rows of all projections in a
          single call (no Python loop).
        * Other methods — ignored; each C extension manages its own
          threading internally.

        ``-1`` (default) uses all available cores.

    Returns
    -------
    aligned_diff : ndarray, shape (nprojs, roi_nr, roi_nc)
        Stack of derivative images.
    """
    roi_stack = stack_array[:, roiy[0]:roiy[-1], roix[0]:roix[-1]]

    if shift_method == "fourier":
        # Vectorise over the entire (nprojs, roi_rows, nc) array in one
        # scipy.fft call — no Python-level loop needed.
        if n_cpus < 0:
            n_cpus = os.cpu_count() or 1
        nc_roi = roi_stack.shape[2]
        freqs = fftfreq(nc_roi)
        if symmetric:
            rshift, lshift = 0.5, 0.5
        else:
            rshift, lshift = 1.0, 0.0
        kernel = (
            np.exp( 1j * 2.0 * np.pi * freqs * rshift)
            - np.exp(-1j * 2.0 * np.pi * freqs * lshift)
        )                                              # shape (nc_roi,)
        fft_all = fft(roi_stack, workers=n_cpus)       # axis=-1 by default
        fft_all *= kernel                              # broadcast over leading axes
        return ifft(fft_all, workers=n_cpus).real.astype(roi_stack.dtype)

    # Non-fourier: sequential loop; C extensions handle their own threading
    aligned_diff = np.empty_like(roi_stack)
    for ii in tqdm(range(stack_array.shape[0]), desc="Computing derivatives"):
        aligned_diff[ii] = derivatives(roi_stack[ii], shift_method)
    return aligned_diff


def calculate_derivatives_fft(stack_array, roiy, roix, n_cpus=-1):
    """
    .. deprecated::
        Use :func:`calculate_derivatives` with ``shift_method="fourier"`` instead.
    """
    warnings.warn(
        "calculate_derivatives_fft() is deprecated; "
        "use calculate_derivatives(shift_method='fourier') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calculate_derivatives(stack_array, roiy, roix,
                                 shift_method="fourier", n_cpus=n_cpus)


def derivatives_sino(input_sino, shift_method="fourier"):
    """
    Calculate the derivative of the sinogram along the radial direction.

    Parameters
    ----------
    input_sino : array_like
        Input sinogram.
    shift_method : str, optional
        Passed to :func:`derivatives`.  Default ``"fourier"``.

    Returns
    -------
    diffsino : array_like
        Derivative of the sinogram along the radial direction.
    """
    rollsino = np.rollaxis(input_sino, 1)  # (nc, nprojs) → derivative along rows
    rolldiff = derivatives(rollsino, shift_method)
    return np.rollaxis(rolldiff, 1)
