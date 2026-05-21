#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard packages
import contextlib
import heapq
import os
import sys
from collections import deque

# third party packages
from joblib import Parallel, delayed, parallel_backend
from matplotlib.widgets import Button
from ..utils.plot_utils import plt
import multiprocessing
import numpy as np
from scipy.fft import dctn, idctn
from ..utils import tqdm

# optional: SNAPHU (pip install snaphu)
try:
    import snaphu as _snaphu
    _SNAPHU_AVAILABLE = True
except ImportError:
    _SNAPHU_AVAILABLE = False

# local packages
from ..utils.plot_utils import _plotdelimiters
from ..utils import isnotebook

__all__ = [
    "wraptopi",
    "wrap",
    "distance",
    "get_charge",
    "phaseresidues",
    "phaseresiduesStack",
    "phaseresiduesStack_parallel",
    "chooseregiontounwrap",
    "unwrap_phase_2d",
    "unwrapping_phase"
]


def wraptopi(phase, endpoint=True):
    """
    Wrap a scalar value or an entire array

    Parameters
    ----------
    phase : float or array_like
        The value or signal to wrapped.
    endpoint : bool, optional
        If ``endpoint=False``, the scalar value or array is wrapped
        to [-pi, pi), whereas if ``endpoint=True``, it is wrapped to (-pi, pi].
        The default value is ``endpoint=True``

    Returns
    -------
    float or array
        Wrapped value or array

    Examples
    --------
    >>> import numpy as np
    >>> wraptopi(np.linspace(-np.pi,np.pi,7),endpoint=True)
    array([ 3.14159265, -2.0943951 , -1.04719755, -0.        ,  1.04719755,
        2.0943951 ,  3.14159265])
    >>> wraptopi(np.linspace(-np.pi,np.pi,7),endpoint=False)
    array([-3.14159265, -2.0943951 , -1.04719755,  0.        ,  1.04719755,
        2.0943951 , -3.14159265])
    """
    if not endpoint:  # case [-pi, pi)
        return (phase + np.pi) % (2 * np.pi) - np.pi
    else:  # case (-pi, pi]
        return ((-phase + np.pi) % (2.0 * np.pi) - np.pi) * -1.0


def wrap(phase):
    """
    Wrap a scalar value or an entire array to [-0.5, 0.5).

    Parameters
    ----------
    phase : float or array_like
        The value or signal to wrapped.

    Returns
    -------
    float or array
        Wrapped value or array

    Notes
    -----
    Created by Sebastian Theilenberg, PyMRR, which is available at
    Github repository: https://github.com/theilen/PyMRR.git
    """
    if hasattr(phase, "__len__"):
        phase = phase.copy()
        phase[phase > 0.5] -= 1.0
        phase[phase <= -0.5] += 1.0
    else:
        if phase > 0.5:
            phase -= 1.0
        elif phase <= -0.5:
            phase += 1.0
    return phase


def distance(pixel1, pixel2):
    """
    Return the Euclidean distance between two pixels.

    Parameters
    ----------
    pixel1 : array_like
        Coordinates of the first pixel, e.g. ``(row, col)``.
    pixel2 : array_like
        Coordinates of the second pixel, e.g. ``(row, col)``.

    Returns
    -------
    float
        Euclidean distance between ``pixel1`` and ``pixel2``.

    Examples
    --------
    >>> distance(np.arange(1,10),np.arange(2,11))
    3.0
    """
    if (not isinstance(pixel1, np.ndarray)) and (not isinstance(pixel2, np.ndarray)):
        pixel1 = np.asarray(pixel1)
        pixel2 = np.asarray(pixel2)
    return np.sqrt(np.sum((pixel1 - pixel2) ** 2))


def _get_charge(residues):
    """
    Auxiliary function to get the residues charges

    Parameters
    ----------
    residues : ndarray
        A 2-dimensional array containing the with residues

    Returns
    -------
    posres : array_like
        Positions of the residues with positive charge
    negres : array_like
        Positions of the residues with negative charge
    """
    posres = np.where(np.round(residues) == 1)
    respos = len(posres[0])
    negres = np.where(np.round(residues) == -1)
    resneg = len(negres[0])

    nres = respos + resneg

    return posres, negres, nres


def get_charge(residues):
    """
    Get the residues charges

    Parameters
    ----------
    residues : ndarray
        A 2-dimensional array containing the with residues

    Returns
    -------
    posres : array_like
        Positions of the residues with positive charge
    negres : array_like
        Positions of the residues with negative charge
    """
    posres, negres, nres = _get_charge(residues)

    print("Found {:>3.0f} residues".format(nres), end="")

    return posres, negres


def phaseresidues(phimage):
    """
    Calculates the phase residues [1]_ for a given wrapped phase image.

    Parameters
    ----------
    phimage : ndarray
        A 2-dimensional array containing the phase-contrast images with gray-level
        in radians

    Returns
    -------
    residues : ndarray
        A 2-dimensional array containing the map of residues (valued +1 or -1)

    Note
    -----
    Note that by convention the positions of the phase residues are
    marked on the top left corner of the 2 by 2 regions as shown below:

    .. graphviz::

        graph g {
            node [shape=plaintext];
            active -- right [label="  res4   "];
            right  -- belowright [label=" res3 "];
            below  -- belowright [label=" res2 "];
            below  -- active [label=" res1 "];
            { rank=same; active right }
            { rank=same; belowright below }
        }

    Inspired by PhaseResidues.m created by B.S. Spottiswoode on 07/10/2004
    and by find_residues.m created by Manuel Guizar - Sept 27, 2011

    References
    ----------
    .. [1] R. M. Goldstein, H. A. Zebker and C. L. Werner,
      Radio Science 23, 713-720 (1988).
    """
    # Stack all four loop increments into a single (4, nr-2, nc-2) array so
    # that wraptopi is called once instead of four times, reducing Python
    # overhead and allowing NumPy to process the full batch in one pass.
    diffs = np.stack([
        phimage[2:,   1:-1] - phimage[1:-1, 1:-1],
        phimage[2:,   2:]   - phimage[2:,   1:-1],
        phimage[1:-1, 2:]   - phimage[2:,   2:],
        phimage[1:-1, 1:-1] - phimage[1:-1, 2:],
    ])                                              # (4, nr-2, nc-2)
    residues = np.sum(wraptopi(diffs), axis=0) / (2.0 * np.pi)

    respos, resneg, nres = _get_charge(residues)
    residues_charge = dict(pos=respos, neg=resneg)

    return residues, residues_charge, nres


def phaseresiduesStack(stack_array, threshold=5000):
    """
    Calculate the map of residues on the stack.

    Parameters
    ----------
    stack_array : ndarray
        A 3-dimensional array containing the stack of projections
        from which to calculate the phase residues.
    threshold : int, optional
        Maximum number of acceptable phase residues per projection.
        Projections with more residues than ``threshold`` are flagged
        as problematic.  Default ``5000``.

    Returns
    -------
    resmap : ndarray
        2-D phase residue accumulation map (sum of absolute residue maps
        across all projections).
    posres : tuple of ndarray
        Indices ``(yres, xres)`` of pixels where ``resmap >= 1``.
    nres : int
        Total number of residues found in the last processed projection.
    """
    resmap = 0
    wrong = []
    nproj = stack_array.shape[0]
    for ii in tqdm(range(nproj), desc="Searching phase residues"):
        residues, residues_charge, nres = phaseresidues(stack_array[ii])
        if np.any(np.isnan(residues)):
            raise ValueError(f"NaN found in projection {ii+1}")
        if nres > threshold:
            wrong.append(ii)
        resmap += np.abs(residues)
    print(". Done")
    posres = np.where(resmap >= 1.0)
    if wrong:
        print("The following projections are problematic: {}".format(wrong))
    return resmap, posres, nres


def phaseresiduesStack_parallel(stack_array, threshold=1000, ncores=2):
    """
    Calculate the map of residues on the stack using parallel processing.

    Parameters
    ----------
    stack_array : ndarray
        A 3-dimensional array containing the stack of projections
        from which to calculate the phase residues.
    threshold : int, optional
        Maximum number of acceptable phase residues per projection.
        Projections with more residues than ``threshold`` are flagged
        as problematic.  Default ``1000``.
    ncores : int, optional
        Number of CPU cores for parallel computation via joblib.
        Default ``2``.

    Returns
    -------
    resmap : ndarray
        2-D phase residue accumulation map.
    posres : tuple of ndarray
        Indices ``(yres, xres)`` of pixels where ``resmap >= 1``.
    nres : tuple of int
        Tuple of per-projection residue counts.
    """
    with parallel_backend("loky", inner_max_num_threads=2):
        residues, residues_charge, nres = zip(
            *Parallel(n_jobs=ncores)(
                delayed(phaseresidues)(ii) for ii in tqdm(stack_array)
            )
        )
    print("Done")
    print("Creating the map of residues")
    # Convert tuple of (nproj) residue arrays → (nproj, nr-2, nc-2) and
    # sum absolute values in one vectorised call instead of a Python loop.
    resmap = np.sum(np.abs(np.array(residues)), axis=0)
    del residues
    del residues_charge
    posres = np.where(resmap >= 1.0)
    wrong = np.where(np.array(nres) > threshold)[0]
    if len(wrong) > 0:
        print("The following projections are problematic: \n {}".format(wrong))
    # return residues, residues_charge, nres
    return resmap, posres, nres


class _RegionPicker:
    """
    Interactive GUI for selecting the unwrap region and an air pixel.

    Three sequential clicks on the image define the region:

    1. Top-left corner of the rectangular unwrap region.
    2. Bottom-right corner  (blue rectangle drawn + residue count printed).
    3. A pixel in air/vacuum (must fall inside the rectangle).

    Then click **Confirm** to accept, or **Reset** to start over.

    Terminal usage
    --------------
    This class is not meant to be used directly; use :func:`chooseregiontounwrap`.

    Jupyter two-cell workflow
    -------------------------
    The picker is returned immediately; access results in the **next** cell::

        # Cell 1
        picker = chooseregiontounwrap(stack_array, ...)
        # Cell 2 — run AFTER clicking Confirm
        rx, ry, airpix = picker.rx, picker.ry, picker.airpix
    """

    _STEPS = [
        "Step 1/3: click the TOP-LEFT corner of the unwrap region",
        "Step 2/3: click the BOTTOM-RIGHT corner of the unwrap region",
        "Step 3/3: click a pixel in AIR / vacuum (inside the rectangle)",
    ]

    def __init__(self, stack_array, resmap, xres, yres):
        self._stack   = stack_array
        self._resmap  = resmap
        self._xres    = xres
        self._yres    = yres

        # public results
        self.rx     = None
        self.ry     = None
        self.airpix = None
        self._done  = False

        # internal state
        self._step       = 0   # 0=TL, 1=BR, 2=air, 3=ready
        self._corners    = []  # (x, y) world-coord tuples
        self._rect_lines = []  # rectangle artist handles
        self._air_dot    = None

        # ---- figure ----
        fig, ax = plt.subplots(figsize=(9, 7))
        self.fig = fig
        self.ax  = ax

        ax.imshow(stack_array[0], cmap="bone", aspect="auto")
        ax.plot(xres, yres, "or", markersize=3, alpha=0.6, label="residues")
        ax.axis("tight")
        ax.set_title(
            "First projection — red dots = phase residues\n"
            "Image size: {} rows × {} cols".format(
                stack_array[0].shape[0], stack_array[0].shape[1]
            ),
            fontsize=9,
        )

        # step-instruction banner (centred over full figure)
        self._status_txt = fig.text(
            0.5, 0.975, self._STEPS[0],
            ha="center", va="top", fontsize=10,
            color="steelblue", fontweight="bold",
        )
        # secondary info line (residue count, coordinates, warnings)
        self._info_txt = fig.text(
            0.5, 0.935, "",
            ha="center", va="top", fontsize=9, color="0.35",
        )

        # buttons
        ax_reset   = fig.add_axes([0.25, 0.01, 0.2,  0.055])
        ax_confirm = fig.add_axes([0.55, 0.01, 0.2,  0.055])
        self._btn_reset   = Button(ax_reset,   "Reset")
        self._btn_confirm = Button(ax_confirm, "Confirm")
        self._btn_confirm.ax.set_facecolor("0.85")   # greyed out until ready
        self._btn_reset.on_clicked(self._on_reset)
        self._btn_confirm.on_clicked(self._on_confirm)

        self._cid = fig.canvas.mpl_connect("button_press_event", self._on_click)
        print(self._STEPS[0], flush=True)

    # ------------------------------------------------------------------ helpers

    def _clear_overlays(self):
        for ln in self._rect_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._rect_lines = []
        if self._air_dot is not None:
            try:
                self._air_dot.remove()
            except Exception:
                pass
            self._air_dot = None

    def _draw_rect(self, x0, y0, x1, y1):
        """Draw blue rectangle from top-left (x0,y0) to bottom-right (x1,y1)."""
        self._clear_overlays()
        kw = dict(color="b", linewidth=1.5, linestyle="-")
        for xs, ys in [
            ([x0, x1], [y0, y0]),
            ([x0, x1], [y1, y1]),
            ([x0, x0], [y0, y1]),
            ([x1, x1], [y0, y1]),
        ]:
            ln, = self.ax.plot(xs, ys, **kw)
            self._rect_lines.append(ln)

    def _set_info(self, msg):
        self._info_txt.set_text(msg)
        self.fig.canvas.draw_idle()

    def _set_status(self, msg):
        self._status_txt.set_text(msg)

    # ------------------------------------------------------------------ events

    def _on_click(self, event):
        if event.button != 1 or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._done or self._step >= 3:
            return

        x, y = event.xdata, event.ydata

        if self._step == 0:
            # ---- step 1: top-left corner ----
            self._corners = [(x, y)]
            self._clear_overlays()
            dot, = self.ax.plot(x, y, "bs", markersize=7)
            self._rect_lines.append(dot)
            self._step = 1
            self._set_status(self._STEPS[1])
            self._set_info("Top-left: ({:.0f}, {:.0f})".format(x, y))
            print("  Top-left corner: ({:.0f}, {:.0f})".format(x, y), flush=True)
            print(self._STEPS[1], flush=True)

        elif self._step == 1:
            # ---- step 2: bottom-right corner — enforce ordering ----
            x0, y0 = self._corners[0]
            x0, x1 = (x0, x) if x0 <= x else (x, x0)
            y0, y1 = (y0, y) if y0 <= y else (y, y0)
            self._corners = [(x0, y0), (x1, y1)]
            self._draw_rect(x0, y0, x1, y1)
            rx = range(int(round(x0)), int(round(x1)) + 1)
            ry = range(int(round(y0)), int(round(y1)) + 1)
            nres = int(np.round(
                self._resmap[ry[0]:ry[-1], rx[0]:rx[-1]].sum()
            ))
            self._step = 2
            self._set_status(self._STEPS[2])
            info = (
                "Region: x=[{}, {}]  y=[{}, {}]   {} residues inside"
            ).format(rx[0], rx[-1], ry[0], ry[-1], nres)
            self._set_info(info)
            print("  " + info, flush=True)
            print(self._STEPS[2], flush=True)

        elif self._step == 2:
            # ---- step 3: air pixel (must be inside the rectangle) ----
            (x0, y0), (x1, y1) = self._corners
            rx = range(int(round(x0)), int(round(x1)) + 1)
            ry = range(int(round(y0)), int(round(y1)) + 1)
            if not (rx[0] <= x <= rx[-1] and ry[0] <= y <= ry[-1]):
                msg = "Air pixel must be inside the blue rectangle — try again."
                self._set_info(msg)
                print("  WARNING: " + msg, flush=True)
                return
            self._air_dot, = self.ax.plot(x, y, "ob", markersize=8)
            self.rx     = rx
            self.ry     = ry
            self.airpix = (int(round(x)), int(round(y)))
            self._step  = 3   # waiting for Confirm
            self._set_status(
                "All set — click  Confirm  to accept  |  Reset  to start over"
            )
            self._btn_confirm.ax.set_facecolor("limegreen")
            self._set_info("Air pixel: ({}, {})".format(*self.airpix))
            print("  Air pixel: ({}, {})".format(*self.airpix), flush=True)
            print(
                "All set — click  Confirm  to accept, or  Reset  to redo.",
                flush=True,
            )

    def _on_reset(self, event):
        self._clear_overlays()
        self._corners = []
        self._step    = 0
        self.rx       = None
        self.ry       = None
        self.airpix   = None
        self._btn_confirm.ax.set_facecolor("0.85")
        self._set_status(self._STEPS[0])
        self._set_info("")
        print("Reset — " + self._STEPS[0], flush=True)

    def _on_confirm(self, event):
        if self._step < 3:
            print(
                "Not ready yet — complete all 3 clicks first.", flush=True
            )
            return
        self._done = True
        try:
            self.fig.canvas.mpl_disconnect(self._cid)
        except Exception:
            pass
        print(
            "Confirmed — rx=[{}, {}]  ry=[{}, {}]  airpix={}".format(
                self.rx[0], self.rx[-1],
                self.ry[0], self.ry[-1],
                self.airpix,
            ),
            flush=True,
        )
        plt.close(self.fig)


def chooseregiontounwrap(stack_array, threshold=5000, parallel=False, ncores=1):
    """
    Choose the region to be unwrapped interactively.

    A GUI figure opens showing the first projection overlaid with
    phase-residue locations (red dots).  Click three points in order:

    1. Top-left corner of the rectangular unwrap region.
    2. Bottom-right corner of the unwrap region.
    3. A pixel in air / vacuum (must be inside the rectangle).

    Then click **Confirm** to accept, or **Reset** to start over.

    Parameters
    ----------
    stack_array : ndarray
        A 3-dimensional array containing the stack of projections
        to be unwrapped.
    threshold : int, optional
        Threshold for the number of acceptable phase residues. (Default = 5000)
    parallel : bool, optional
        If ``True``, multiprocessing is used for residue computation.
        (Default = ``False``)
    ncores : int, optional
        Number of cores for parallel computation. (Default = 1)

    Returns
    -------
    Terminal mode
        ``(rx, ry, airpix)`` — ranges and air-pixel tuple, ready to use.
    Jupyter mode  (two-cell workflow)
        ``picker`` — :class:`_RegionPicker` instance.  Access
        ``picker.rx``, ``picker.ry``, ``picker.airpix`` in the **next**
        cell after clicking Confirm.
    """
    # ---- residue computation ----
    print("Checking for phase residues …", flush=True)
    if ncores == 1:
        parallel = False
    if parallel:
        if ncores == -1:
            try:
                ncores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
            except Exception:
                ncores = multiprocessing.cpu_count()
        if ncores == 1:
            print("{} core used: parallel calculations are not possible".format(ncores))
        resmap, posres, nres = phaseresiduesStack_parallel(
            stack_array, threshold, ncores
        )
    else:
        resmap, posres, nres = phaseresiduesStack(stack_array, threshold)
    yres, xres = posres
    print("→ residue map done.", flush=True)

    # warn about projections with too many residues
    wrong = np.where(np.array(nres) > threshold)[0]
    if len(wrong) > 0:
        print(
            "The following projections exceed the residue threshold: {}".format(wrong)
        )

    # ---- interactive picker ----
    plt.close("all")
    picker = _RegionPicker(stack_array, resmap, xres, yres)

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
                "  Interact with the figure above (3 clicks + Confirm),\n"
                "  then run in the NEXT cell:\n"
                "      rx, ry, airpix = picker.rx, picker.ry, picker.airpix",
                flush=True,
            )
        else:
            import warnings
            from IPython import display as _ipy_display
            _ipy_display.display(picker.fig)
            warnings.warn(
                "Interactive region picking requires a non-inline backend.\n"
                "Add  %matplotlib widget  as the first notebook cell\n"
                "and run  pip install ipympl  once, then restart the kernel.",
                UserWarning,
                stacklevel=2,
            )
        return picker
    else:
        plt.show(block=True)   # blocks until Confirm closes the figure
        return picker.rx, picker.ry, picker.airpix


# ---------------------------------------------------------------------------
# Internal reliability-map helper
# ---------------------------------------------------------------------------

def _reliability_map(phase):
    """
    Compute a per-pixel reliability map for Herraez's reliability-guided
    phase unwrapping algorithm.

    Reliability = 1 / D where D = sqrt(H^2 + V^2 + D1^2 + D2^2).
    H, V, D1, D2 are second-order phase differences computed via wraptopi.
    Border pixels receive reliability 0.

    Parameters
    ----------
    phase : ndarray
        2-D wrapped phase array (radians).

    Returns
    -------
    rel : ndarray
        Float64 reliability array, same shape as ``phase``.
    """
    nr, nc = phase.shape
    rel = np.zeros((nr, nc), dtype=np.float64)

    # Second differences (interior pixels only).
    # Stack all 8 first-order differences into a single (8, nr-2, nc-2)
    # array so wraptopi is called once instead of eight times.
    inner = phase[1:-1, 1:-1]
    all_diffs = np.stack([
        phase[1:-1, 2:]  - inner,   # H forward
        inner - phase[1:-1, :-2],   # H backward
        phase[2:,  1:-1] - inner,   # V forward
        inner - phase[:-2,  1:-1],  # V backward
        phase[2:,  2:]   - inner,   # D1 forward
        inner - phase[:-2,  :-2],   # D1 backward
        phase[2:,  :-2]  - inner,   # D2 forward
        inner - phase[:-2,  2:],    # D2 backward
    ])                              # (8, nr-2, nc-2)
    w = wraptopi(all_diffs)         # single call
    H  = w[0] - w[1]
    V  = w[2] - w[3]
    D1 = w[4] - w[5]
    D2 = w[6] - w[7]

    D = np.sqrt(H ** 2 + V ** 2 + D1 ** 2 + D2 ** 2)
    # Avoid division by zero; zero D gives maximum reliability (flat region)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel[1:-1, 1:-1] = np.where(D == 0.0, np.finfo(np.float64).max, 1.0 / D)

    return rel


# ---------------------------------------------------------------------------
# Algorithm 1: Herraez reliability-guided BFS
# ---------------------------------------------------------------------------

def _unwrap_herraez(phase):
    """
    Reliability-guided phase unwrapping (Herraez et al., 2002).

    Uses a max-heap (implemented as a min-heap with negated priorities) to
    process pixel edges in decreasing order of edge reliability, where edge
    reliability = min(reliability[p1], reliability[p2]).

    Compared with a naïve heap implementation this version reduces per-
    element overhead by:

    * Encoding each heap entry as a compact 3-tuple
      ``(-edge_rel, flat_dst_idx, flat_src_idx)`` instead of a 6-tuple
      with a monotone counter, saving the ``next(ctr)`` call and one
      int per push.
    * Inlining the neighbor-push logic to avoid a Python function call on
      every visited pixel (the hot path).
    * Caching local references to ``heapq.heappush`` / ``heapq.heappop``
      to cut attribute-lookup time in the inner loop.
    * Using ``divmod(flat, nc)`` to recover ``(row, col)`` in a single
      C-level operation instead of two Python integer divisions.

    Parameters
    ----------
    phase : ndarray
        2-D wrapped phase array (radians).

    Returns
    -------
    unwrapped : ndarray
        Float64 unwrapped phase array, same shape as ``phase``.
    """
    nr, nc = phase.shape
    rel = _reliability_map(phase)

    unwrapped = phase.astype(np.float64).copy()
    visited = np.zeros((nr, nc), dtype=bool)

    # Seed from pixel (0, 0)
    visited[0, 0] = True

    # Cache hot-path references to avoid per-call attribute lookup
    _push = heapq.heappush
    _pop  = heapq.heappop
    heap  = []

    # Seed neighbours
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r2, c2 = dr, dc
        if 0 <= r2 < nr and 0 <= c2 < nc:
            er = rel[r2, c2] if rel[r2, c2] < rel[0, 0] else rel[0, 0]
            _push(heap, (-er, r2 * nc + c2, 0))

    while heap:
        neg_rel, flat2, flat1 = _pop(heap)
        r, c = divmod(flat2, nc)
        if visited[r, c]:
            continue
        visited[r, c] = True
        src_r, src_c = divmod(flat1, nc)
        d = wraptopi(phase[r, c] - unwrapped[src_r, src_c])
        unwrapped[r, c] = unwrapped[src_r, src_c] + d
        # Inline neighbour push
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < nr and 0 <= c2 < nc and not visited[r2, c2]:
                er = rel[r2, c2] if rel[r2, c2] < rel[r, c] else rel[r, c]
                _push(heap, (-er, r2 * nc + c2, flat2))

    return unwrapped


# ---------------------------------------------------------------------------
# Algorithm 2: Goldstein branch-cut + BFS flood fill
# ---------------------------------------------------------------------------

def _unwrap_goldstein(phase):
    """
    Goldstein branch-cut phase unwrapping (Goldstein et al., 1988).

    Locates phase residues, pairs them with a greedy nearest-neighbour
    strategy, draws L-shaped branch cuts, then flood-fills from the image
    centre while respecting the branch cuts.

    Parameters
    ----------
    phase : ndarray
        2-D wrapped phase array (radians).

    Returns
    -------
    unwrapped : ndarray
        Float64 unwrapped phase array, same shape as ``phase``.
        Pixels unreachable due to branch cuts retain their wrapped value.
    """
    nr, nc = phase.shape
    residues, residues_charge, nres = phaseresidues(phase)

    if nres == 0:
        return _unwrap_flynn(phase)

    # Convert residue indices to image coordinates (+1 offset)
    pos_rows = residues_charge["pos"][0] + 1
    pos_cols = residues_charge["pos"][1] + 1
    neg_rows = residues_charge["neg"][0] + 1
    neg_cols = residues_charge["neg"][1] + 1

    branch_cuts = np.zeros((nr, nc), dtype=bool)

    pos_used = [False] * len(pos_rows)
    neg_used = [False] * len(neg_rows)

    # Greedy nearest-neighbour pairing of positive to negative residues
    for pi in range(len(pos_rows)):
        r1, c1 = pos_rows[pi], pos_cols[pi]
        best_dist = np.inf
        best_ni = -1
        for ni in range(len(neg_rows)):
            if neg_used[ni]:
                continue
            d = distance((r1, c1), (neg_rows[ni], neg_cols[ni]))
            if d < best_dist:
                best_dist = d
                best_ni = ni
        if best_ni >= 0:
            pos_used[pi] = True
            neg_used[best_ni] = True
            r2, c2 = neg_rows[best_ni], neg_cols[best_ni]
            # L-shaped branch cut: horizontal segment then vertical segment
            c_lo, c_hi = min(c1, c2), max(c1, c2)
            branch_cuts[r1, c_lo:c_hi + 1] = True
            r_lo, r_hi = min(r1, r2), max(r1, r2)
            branch_cuts[r_lo:r_hi + 1, c2] = True

    # Unmatched positives: vertical cut to top border
    for pi in range(len(pos_rows)):
        if not pos_used[pi]:
            r, c = pos_rows[pi], pos_cols[pi]
            branch_cuts[:r + 1, c] = True

    # Unmatched negatives: vertical cut to bottom border
    for ni in range(len(neg_rows)):
        if not neg_used[ni]:
            r, c = neg_rows[ni], neg_cols[ni]
            branch_cuts[r:, c] = True

    # BFS flood fill from centre
    unwrapped = phase.astype(np.float64).copy()
    visited = np.zeros((nr, nc), dtype=bool)

    start_r, start_c = nr // 2, nc // 2
    # If start pixel is on a branch cut, search nearby for a free pixel
    if branch_cuts[start_r, start_c]:
        found = False
        for dr in range(nr):
            for dc in range(nc):
                for sr, sc in [(start_r + dr, start_c + dc),
                               (start_r - dr, start_c - dc),
                               (start_r + dr, start_c - dc),
                               (start_r - dr, start_c + dc)]:
                    if 0 <= sr < nr and 0 <= sc < nc and not branch_cuts[sr, sc]:
                        start_r, start_c = sr, sc
                        found = True
                        break
                if found:
                    break
            if found:
                break

    queue = deque()
    queue.append((start_r, start_c))
    visited[start_r, start_c] = True
    # seed value is the wrapped phase itself (already set in unwrapped)

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < nr and 0 <= c2 < nc and not visited[r2, c2] and not branch_cuts[r2, c2]:
                visited[r2, c2] = True
                d = wraptopi(phase[r2, c2] - unwrapped[r, c])
                unwrapped[r2, c2] = unwrapped[r, c] + d
                queue.append((r2, c2))

    return unwrapped


# ---------------------------------------------------------------------------
# Algorithm 3: Flynn row-integration + median row correction
# ---------------------------------------------------------------------------


def _unwrap_flynn(phase):
    """
    Flynn's integration-based phase unwrapping (Flynn, 1997).

    Fast O(N) approach:
    1. Integrate wrapped horizontal differences row by row.
    2. Fix row-to-row jumps using a median correction.

    Parameters
    ----------
    phase : ndarray
        2-D wrapped phase array (radians).

    Returns
    -------
    unwrapped : ndarray
        Float64 unwrapped phase array, same shape as ``phase``.
    """
    nr, nc = phase.shape
    two_pi = 2.0 * np.pi

    unwrapped = np.empty((nr, nc), dtype=np.float64)

    # Step 1: integrate wrapped horizontal differences row by row
    dh = wraptopi(np.diff(phase, axis=1))
    unwrapped[:, 0] = phase[:, 0]
    unwrapped[:, 1:] = phase[:, :1] + np.cumsum(dh, axis=1)

    # Step 2: fix row-to-row jumps using median correction
    for r in range(1, nr):
        dv = unwrapped[r] - unwrapped[r - 1]
        correction = int(np.round(np.median(dv) / two_pi))
        if correction != 0:
            unwrapped[r:] -= two_pi * correction

    return unwrapped


# ---------------------------------------------------------------------------
# Algorithm 4: DCT-based Weighted Least Squares (Ghiglia & Romero, 1994)
# ---------------------------------------------------------------------------

def _unwrap_wls(phase):
    """
    DCT-based unweighted least-squares phase unwrapping (Ghiglia & Romero, 1994).

    Solves the 2-D discrete Poisson equation  ∇²φ = ρ  where ρ is the
    divergence of the wrapped phase gradients.  The system is solved via a
    DCT-II transform (Neumann boundary conditions) in O(N log N) time.

    This gives a globally optimal solution under a flat (uniform) weight
    model, making it substantially more robust than the greedy Herraez or
    Goldstein methods for images that are smooth but wrapped many times.

    Parameters
    ----------
    phase : ndarray
        2-D wrapped phase array (radians).

    Returns
    -------
    unwrapped : ndarray
        Float64 unwrapped phase array, same shape as ``phase``.

    Reference
    ---------
    D. C. Ghiglia and L. A. Romero, "Robust two-dimensional weighted and
    unweighted phase unwrapping that uses fast transforms and iterative
    methods", J. Opt. Soc. Am. A 11(1), 107-117 (1994).
    """
    nr, nc = phase.shape

    # Wrapped forward differences (phase gradients)
    dy = wraptopi(np.diff(phase, axis=0))   # shape (nr-1, nc)
    dx = wraptopi(np.diff(phase, axis=1))   # shape (nr, nc-1)

    # Discrete divergence with Neumann (zero-flux) boundary conditions.
    # Interior: rho[i,j] = dx[i,j] - dx[i,j-1] + dy[i,j] - dy[i-1,j]
    rho = np.zeros((nr, nc), dtype=np.float64)
    rho[:, 1:-1] += dx[:, 1:] - dx[:, :-1]   # interior x
    rho[:, 0]    += dx[:, 0]                   # left boundary
    rho[:, -1]   -= dx[:, -1]                  # right boundary
    rho[1:-1, :] += dy[1:, :] - dy[:-1, :]    # interior y
    rho[0, :]    += dy[0, :]                   # top boundary
    rho[-1, :]   -= dy[-1, :]                  # bottom boundary

    # Residue correction: phase residues make the gradient field non-
    # conservative; the Poisson solver would otherwise spread that
    # inconsistency as a smooth large-scale gradient across the whole image.
    # We detect residue locations (2×2 loops where the wrapped path integral
    # ≠ 0) and zero-out their divergence contribution before solving, which
    # localises the residue error rather than distributing it globally.
    res_map, _, _ = phaseresidues(phase)   # values ±1 at residue corners
    # phaseresidues uses a (nr-2)×(nc-2) interior grid with +1 offset
    res_mask = np.zeros((nr, nc), dtype=bool)
    res_mask[1:-1, 1:-1] = np.abs(np.round(res_map)) >= 1
    # Dilate one pixel so the correction covers the full 2×2 residue cell
    from scipy.ndimage import binary_dilation
    res_mask = binary_dilation(res_mask)
    rho[res_mask] = 0.0

    # Eigenvalues of the discrete Laplacian in DCT-II / Neumann basis:
    #   mu[k, l] = 2*(cos(pi*k/nr) - 1) + 2*(cos(pi*l/nc) - 1)
    k = np.arange(nr, dtype=np.float64)
    l = np.arange(nc, dtype=np.float64)
    mu = (
        2.0 * (np.cos(np.pi * k / nr) - 1.0)[:, None]
        + 2.0 * (np.cos(np.pi * l / nc) - 1.0)[None, :]
    )
    mu[0, 0] = 1.0   # avoid division by zero; DC set to 0 below

    # Solve in DCT domain
    rho_hat = dctn(rho, type=2, norm="ortho")
    phi_hat = rho_hat / mu
    phi_hat[0, 0] = 0.0   # DC is a free gauge (arbitrary global offset)
    phi = idctn(phi_hat, type=2, norm="ortho")

    return phi


# ---------------------------------------------------------------------------
# Algorithm 5: SNAPHU (Chen & Zebker, 2001) — optional dependency
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_fd1():
    """
    Context manager that silences C-level stdout (file descriptor 1).

    ``contextlib.redirect_stdout`` only redirects Python's ``sys.stdout``
    object.  SNAPHU's C library writes directly to fd 1, so the only
    reliable way to suppress it is to temporarily replace fd 1 with
    /dev/null at the OS level via ``os.dup2``.
    """
    sys.stdout.flush()
    saved_fd = os.dup(1)                          # save real fd 1
    devnull_fd = os.open(os.devnull, os.O_WRONLY) # open /dev/null
    try:
        os.dup2(devnull_fd, 1)                    # redirect fd 1 → /dev/null
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)                      # restore real fd 1
        os.close(saved_fd)
        os.close(devnull_fd)


def _unwrap_snaphu(phase, verbose=False):
    """
    SNAPHU phase unwrapping (Chen & Zebker, 2001).

    Uses the Statistical-cost, Network-flow Algorithm for Phase Unwrapping
    (SNAPHU) via the ``snaphu-py`` package.  The per-pixel reliability map
    is normalised to [0, 1] and used as a coherence estimate so that SNAPHU's
    statistical cost model is guided by local phase quality.

    Requires the optional dependency::

        pip install snaphu

    Parameters
    ----------
    phase : ndarray
        2-D wrapped phase array (radians).
    verbose : bool, optional
        If ``False`` (default) SNAPHU's extensive C-level output is
        suppressed and a single progress line is printed instead.
        Set to ``True`` to see the full SNAPHU log (useful for debugging).

    Returns
    -------
    unwrapped : ndarray
        Float64 unwrapped phase array, same shape as ``phase``.

    Raises
    ------
    ImportError
        If ``snaphu-py`` is not installed.

    Reference
    ---------
    C. W. Chen and H. A. Zebker, "Two-dimensional phase unwrapping with use
    of statistical models for cost functions in nonlinear optimization",
    J. Opt. Soc. Am. A 18(2), 338-351 (2001).
    """
    if not _SNAPHU_AVAILABLE:
        raise ImportError(
            "snaphu-py is required for method='snaphu'.\n"
            "Install it with:  pip install snaphu"
        )

    phase = np.asarray(phase, dtype=np.float64)

    # Convert wrapped phase → complex interferogram  e^{i φ}
    igram = np.exp(1j * phase).astype(np.complex64)

    # Use reliability map as a coherence proxy (normalised to [0, 1])
    rel = _reliability_map(phase)
    rel_range = rel.max() - rel.min()
    if rel_range == 0.0:
        corr = np.ones(phase.shape, dtype=np.float32)
    else:
        corr = ((rel - rel.min()) / rel_range).astype(np.float32)

    # Run SNAPHU, optionally suppressing its C-level output
    print("Running SNAPHU ...", end=" ", flush=True)
    if verbose:
        print()   # newline before SNAPHU's own output
        unw, _ = _snaphu.unwrap(igram, corr, nlooks=1)
    else:
        with _suppress_fd1():
            unw, _ = _snaphu.unwrap(igram, corr, nlooks=1)
        print("done.", flush=True)

    return np.asarray(unw, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def unwrap_phase_2d(phase, method="herraez", verbose=False):
    """
    Unwrap a 2-D phase array using one of three internal algorithms.

    Parameters
    ----------
    phase : ndarray
        2-D wrapped phase array (radians).
    method : str, optional
        Algorithm to use.  One of:

        ``"herraez"`` (default)
            Reliability-guided BFS.  Computes a per-pixel reliability from
            second-order phase differences and processes pixel edges in
            decreasing reliability order via a priority queue.  Generally
            the most robust choice for noisy data.

            Reference: M. A. Herraez, D. R. Burton, M. J. Lalor, and
            M. A. Gdeisat, "Fast two-dimensional phase-unwrapping algorithm
            based on sorting by reliability", Applied Optics 41(35), 2002.

        ``"goldstein"``
            Branch-cut algorithm.  Locates phase residues, pairs them with a
            greedy nearest-neighbour strategy, draws L-shaped branch cuts,
            then unwraps via BFS flood fill from the image centre.  Falls
            back to Flynn's method when no residues are found.

            Reference: R. M. Goldstein, H. A. Zebker, and C. L. Werner,
            "Satellite radar interferometry: Two-dimensional phase
            unwrapping", Radio Science 23(4), 1988.

        ``"flynn"``
            Row-integration + median row correction.  Integrates wrapped
            horizontal differences row by row, then corrects accumulated
            row-to-row offsets using a median estimator.  O(N) complexity,
            fastest but least robust to noise.

            Reference: T. J. Flynn, "Two-dimensional phase unwrapping with
            minimum weighted discontinuity", Journal of the Optical Society
            of America A 14(10), 1997.

        ``"wls"``
            DCT-based unweighted least-squares (Ghiglia & Romero, 1994).
            Solves the 2-D Poisson equation whose right-hand side is the
            divergence of the wrapped phase gradients, using a DCT-II
            transform (Neumann boundary conditions).  Globally optimal under
            a flat weight model; O(N log N) complexity.  More robust than
            the greedy methods for densely wrapped, smooth phase fields.

            Reference: D. C. Ghiglia and L. A. Romero, "Robust two-dimensional
            weighted and unweighted phase unwrapping that uses fast transforms
            and iterative methods", J. Opt. Soc. Am. A 11(1), 107-117 (1994).

        ``"snaphu"``
            Statistical-cost network-flow algorithm (Chen & Zebker, 2001).
            Regarded as the gold standard for 2-D phase unwrapping.  The
            per-pixel reliability map is used as a coherence estimate.
            Requires the optional package ``snaphu-py``
            (``pip install snaphu``).

            Reference: C. W. Chen and H. A. Zebker, "Two-dimensional phase
            unwrapping with use of statistical models for cost functions in
            nonlinear optimization", J. Opt. Soc. Am. A 18(2), 338-351 (2001).

    verbose : bool, optional
        Only used when ``method='snaphu'``.  If ``False`` (default) SNAPHU's
        extensive C-level log is suppressed and replaced by a single line.
        Set to ``True`` to see the full SNAPHU output.

    Returns
    -------
    unwrapped : ndarray
        Float64 unwrapped phase array, same shape as ``phase``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the recognised algorithm names.
    ImportError
        If ``method='snaphu'`` and ``snaphu-py`` is not installed.
    """
    method = method.lower()
    if method == "herraez":
        return _unwrap_herraez(phase)
    elif method == "goldstein":
        return _unwrap_goldstein(phase)
    elif method == "flynn":
        return _unwrap_flynn(phase)
    elif method == "wls":
        return _unwrap_wls(phase)
    elif method == "snaphu":
        return _unwrap_snaphu(phase, verbose=verbose)
    else:
        raise ValueError(
            f"Unknown unwrapping method '{method}'. "
            "Choose one of: 'herraez', 'goldstein', 'flynn', 'wls', 'snaphu'."
        )


# ---------------------------------------------------------------------------
# Internal single-image unwrapping helper
# ---------------------------------------------------------------------------

def _unwrapping_phase(img2unwrap, rx=[], ry=[], airpix=[], method="herraez", verbose=False):
    """
    Unwrap the phases of a projection

    Parameters
    ----------
    img2unwrap : ndarray
        A 2-dimensional array containing the image to be unwrapped
    rx, ry : tuple or list of ints
        Limits of the are to be unwrapped in x and y
    airpix : tuple or list of ints
        Position of pixel in the air/vacuum area
    method : str, optional
        Phase unwrapping algorithm. See ``unwrap_phase_2d`` for details.
        Default is ``"herraez"``.
    verbose : bool, optional
        Passed to ``unwrap_phase_2d`` (only relevant for ``method='snaphu'``).
        Default is ``False``.

    Returns
    -------
    unwrapped : ndarray
        Unwrapped image (new array — the input ``img2unwrap`` is never modified).
    """
    if rx == [] and ry == []:
        # unwrap_phase_2d always returns a new array, so the original is safe
        unwrapped = unwrap_phase_2d(img2unwrap, method=method, verbose=verbose)
        unwrapped -= -2 * np.pi * np.round(unwrapped / (2 * np.pi))
    else:
        # Take an explicit copy of the ROI so the original wrapped data is
        # never overwritten — this guarantees that calling with a different
        # method later always starts from the same original wrapped phase.
        img_wrap_sel = img2unwrap[ry[0] : ry[-1], rx[0] : rx[-1]].copy()

        # Unwrap the selected region
        img_unwrap_sel = unwrap_phase_2d(img_wrap_sel, method=method, verbose=verbose)

        # Air-pixel correction: read from the unwrapped result
        # (airpix is (col, row) == (x, y); chooseregiontounwrap guarantees
        #  it lies inside [ry, rx])
        air_val = img_unwrap_sel[airpix[1] - ry[0], airpix[0] - rx[0]]
        air_offset = 2 * np.pi * np.round(air_val / (2 * np.pi))

        # Build the output image: start from a copy of the original so that
        # pixels outside the ROI keep their original values.
        unwrapped = img2unwrap.copy()
        unwrapped[ry[0] : ry[-1], rx[0] : rx[-1]] = img_unwrap_sel - air_offset

    return unwrapped


# ---------------------------------------------------------------------------
# Internal parallel unwrapping helper
# ---------------------------------------------------------------------------

def _unwrapping_phase_parallel(stack2unwrap, rx=[], ry=[], airpix=[], ncores=1, method="herraez", verbose=False):
    """
    Unwrap the phases of a stack of projections in parallel.

    Parameters
    ----------
    stack2unwrap : ndarray, shape (n, nr, nc)
        Stack of 2-D wrapped phase images to be unwrapped.
    rx : list or range of int, optional
        Column index range of the ROI.  Default ``[]`` (full width).
    ry : list or range of int, optional
        Row index range of the ROI.  Default ``[]`` (full height).
    airpix : tuple or list of int, optional
        Position ``(col, row)`` of a pixel in the air/vacuum region used
        to set the absolute phase offset.  Default ``[]``.
    ncores : int, optional
        Number of CPU cores for parallel computation via joblib.
        Pass ``-1`` to use all available cores.  Default ``1``.
    method : str, optional
        Phase unwrapping algorithm.  See :func:`unwrap_phase_2d` for
        details.  Default ``"herraez"``.
    verbose : bool, optional
        Passed to :func:`unwrap_phase_2d`; only relevant for
        ``method='snaphu'``.  Default ``False``.

    Returns
    -------
    stack_out : ndarray, shape (n, nr, nc)
        Stack of unwrapped phase images.
    """
    if ncores == -1:
        try:
            ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except:
            ncpus = multiprocessing.cpu_count()
    else:
        ncpus = ncores
    print(f"Parallel calculations using {ncpus} cpus")
    # Unwrap each projection in the ROI (copy so the original is never touched)
    roi_wrapped = stack2unwrap[:, ry[0] : ry[-1], rx[0] : rx[-1]].copy()
    with parallel_backend("loky", inner_max_num_threads=2):
        results = Parallel(n_jobs=ncpus)(
            delayed(unwrap_phase_2d)(img, method=method, verbose=verbose)
            for img in tqdm(roi_wrapped)
        )

    # Air-pixel indices within the ROI sub-array
    air_row = airpix[1] - ry[0]   # airpix is (col, row)
    air_col = airpix[0] - rx[0]

    # Build the output from a copy of the original (preserves pixels outside
    # the ROI and guarantees the caller's stack_phasecorr is never modified)
    stack_out = stack2unwrap.copy()
    print("Correcting for air values")
    for ii in range(stack2unwrap.shape[0]):
        # Read air correction from the UNWRAPPED result (same logic as the
        # sequential path) — the wrapped value would always round to 0
        air_val = results[ii][air_row, air_col]
        airphase = np.round(air_val / (2 * np.pi))
        stack_out[ii, ry[0] : ry[-1], rx[0] : rx[-1]] = (
            results[ii] - 2 * np.pi * airphase
        )
    return stack_out


# ---------------------------------------------------------------------------
# Public stack unwrapping function
# ---------------------------------------------------------------------------

def unwrapping_phase(stack_phasecorr, rx, ry, airpix, **params):
    """
    Unwrap the phase of the projections in a stack.

    Parameters
    ----------
    stack_phasecorr : ndarray
        A 3-dimensional array containing the stack of projections to be unwrapped
    rx, ry : tuple or list of ints
        Limits of the are to be unwrapped in x and y
    airpix : tuple or list of ints
        Position of pixel in the air/vacuum area
    **params
        Dictionary of additional parameters.

        vmin : float or None
            Minimum value for the gray level at each display.
        vmax : float or None
            Maximum value for the gray level at each display.
        unwrap_method : str, optional
            Phase unwrapping algorithm to use.  One of ``"herraez"``
            (default), ``"goldstein"``, ``"flynn"``, ``"wls"``, or
            ``"snaphu"``.  See :func:`unwrap_phase_2d` for details.
            ``"snaphu"`` requires the optional package ``snaphu-py``
            (``pip install snaphu``).
        parallel : bool, optional
            If ``True``, use parallel processing.  Default ``True``.
        n_cpus : int, optional
            Number of CPU cores for parallel computation.  Pass ``-1``
            to use all available cores.  Default ``-1``.

    Returns
    -------
    stack_unwrap : ndarray
        A 3-dimensional array containing the stack of unwrapped projections

    Notes
    -----
    Five algorithms are available.  The default is the reliability-guided
    algorithm by Herraez et al. [#herraez]_.  For the best robustness use
    ``"wls"`` (no extra dependency) or ``"snaphu"`` (requires
    ``pip install snaphu``).

    References
    ----------
    .. [#herraez] M. A. Herraez et al., "Fast two-dimensional phase-unwrapping
      algorithm based on sorting by reliability following a noncontinuous path",
      Applied Optics 41(35), 7437 (2002).
    .. [#ghiglia] D. C. Ghiglia and L. A. Romero, "Robust two-dimensional
      weighted and unweighted phase unwrapping that uses fast transforms and
      iterative methods", J. Opt. Soc. Am. A 11(1), 107-117 (1994).
    .. [#chen] C. W. Chen and H. A. Zebker, "Two-dimensional phase unwrapping
      with use of statistical models for cost functions in nonlinear
      optimization", J. Opt. Soc. Am. A 18(2), 338-351 (2001).
    """
    params.setdefault("parallel", True)
    params.setdefault("unwrap_method", "herraez")
    params.setdefault("snaphu_verbose", False)
    verbose = params["snaphu_verbose"]
    if params["parallel"]:
        if params["n_cpus"] == -1:
            try:
                ncores = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
            except:
                ncores = multiprocessing.cpu_count()
        else:
            ncores = params["n_cpus"]
    else:
        ncores = 1
    stack_unwrap = np.empty_like(stack_phasecorr)
    # test on first projection
    print("Testing unwrapping on the first projection")
    img0_unwrap = _unwrapping_phase(
        stack_phasecorr[0], rx, ry, airpix,
        method=params["unwrap_method"], verbose=verbose,
    )
    # displaying
    plt.close("all")
    plt.ion()
    fig = plt.figure(7)
    ax1 = fig.add_subplot(111)
    im1 = ax1.imshow(
        stack_phasecorr[0], cmap="bone", vmin=params["vmin"], vmax=params["vmax"]
    )
    # update images with boudaries
    ax1 = _plotdelimiters(ax1, ry, rx, airpix)
    ax1.axis("tight")
    if isnotebook():
        from IPython import display
        display.display(fig)
        plt.close(fig)
        display.clear_output(wait=True)
    else:
        plt.show(block=False)
    while True:
        a = input("Do you want to edit the color scale?([y]/n)").lower()
        if str(a) == "" or str(a) == "y":
            while True:
                color_vmin = eval(input("Minimum color scale value: "))
                if isinstance(color_vmin, int) or isinstance(color_vmin, float):
                    break
                else:
                    print("Wrong typing. Try it again.")
            while True:
                color_vmax = eval(input("Maximum color scale value: "))
                if isinstance(color_vmax, int) or isinstance(color_vmax, float):
                    break
                else:
                    print("Wrong typing. Try it again.")
            params["vmin"] = color_vmin
            params["vmax"] = color_vmax
            print("Using vmin={} and vmax={}".format(params["vmin"], params["vmax"]))
            # displaying the update images
            im1.set_data(stack_phasecorr[0])
            im1.set_clim(params["vmin"], params["vmax"])
            ax1 = _plotdelimiters(ax1, ry, rx, airpix)
            ax1.axis("tight")
            plt.show(block=False)
        else:
            print(
                "Color scale was not changed. Using vmin={} and vmax={}".format(
                    params["vmin"], params["vmax"]
                )
            )
            break
    if not params["parallel"] or ncores == 1:
        # main loop for the unwrapping
        nprojs = stack_phasecorr.shape[0]
        for ii in tqdm(range(nprojs), desc="Unwrapping projections"):
            img_unwrap = _unwrapping_phase(
                stack_phasecorr[ii], rx, ry, airpix,
                method=params["unwrap_method"], verbose=verbose,
            )
            stack_unwrap[ii] = img_unwrap  # update the stack
    else:
        stack_unwrap = _unwrapping_phase_parallel(
            stack_phasecorr, rx, ry, airpix,
            ncores=ncores, method=params["unwrap_method"], verbose=verbose,
        )
        # ~ stack_unwrap_sel = _unwrapping_phase_parallel(
        # ~ stack2unwrap[:,ry[0] : ry[-1], rx[0] : rx[-1]]
        # ~ )
        # ~ for ii in range(stack2unwrap.shape[0]):
        # ~ stack2unwrap[ii,ry[0] : ry[-1], rx[0] : rx[-1]] = stack_unwrap_sel[ii]
        # ~ stack2unwrap[ii,ry[0] : ry[-1], rx[0] : rx[-1]] = (
        # ~ stack2unwrap_sel[ii]
        # ~ - 2 * np.pi * np.round(stack2unwrap[:,airpix[1], airpix[0]] / (2 * np.pi))
        # ~ )

    return stack_unwrap