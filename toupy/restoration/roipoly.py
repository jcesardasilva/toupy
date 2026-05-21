# -*- coding: utf-8 -*-

"""
Draw polygon regions of interest (ROIs) in matplotlib images,
similar to Matlab's roipoly function.

Rewritten to use :class:`matplotlib.widgets.PolygonSelector` so that
the implementation is backend-agnostic and works with the ipympl
(``%matplotlib widget``) backend in Jupyter notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from matplotlib.widgets import PolygonSelector


class roipoly:
    """
    Draw a polygonal region of interest (ROI) interactively.

    Wraps :class:`matplotlib.widgets.PolygonSelector`.  Left-click to
    add vertices.  Click the first vertex again (or press **Enter** in
    matplotlib ≥ 3.7) to close the polygon.  The mask figure closes
    automatically once the polygon is finalised.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure on which to draw.  Defaults to the current figure.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw.  Defaults to the current axes.
    roicolor : str, optional
        Colour of the polygon line.  Default ``'b'`` (blue).

    Attributes
    ----------
    allxpoints : list of float
        X-coordinates of the polygon vertices.
    allypoints : list of float
        Y-coordinates of the polygon vertices.

    Notes
    -----
    After the polygon is finalised the mask figure is closed and the
    vertices are available via :attr:`allxpoints` / :attr:`allypoints`.
    Call :meth:`getMask` to rasterise the polygon onto any image grid,
    and :meth:`displayROI` to overlay the outline on another axes.
    """

    def __init__(self, fig=None, ax=None, roicolor="b"):
        if ax is None:
            ax = plt.gca()
        if fig is None:
            fig = ax.figure

        self.fig = fig
        self.ax = ax
        self.roicolor = roicolor
        self.allxpoints = []
        self.allypoints = []

        # PolygonSelector API: 'props' (≥ 3.5) vs 'lineprops' (< 3.5)
        _props = dict(color=roicolor, linewidth=1.5, alpha=0.8)
        try:
            self._selector = PolygonSelector(ax, self._on_select,
                                             props=_props)
        except TypeError:                          # matplotlib < 3.5
            self._selector = PolygonSelector(ax, self._on_select,
                                             lineprops=_props)

        ax.set_title(
            "Left-click: add vertex   |   Click first vertex (or Enter): finish"
        )
        plt.show(block=False)

    # ------------------------------------------------------------------
    def _on_select(self, verts):
        """Called by PolygonSelector when the polygon is finalised."""
        xs, ys = zip(*verts)
        self.allxpoints = list(xs)
        self.allypoints = list(ys)
        # Disconnect so further clicks don't modify the vertices
        self._selector.disconnect_events()
        # Close the mask figure to return focus to the main GUI
        self.fig.canvas.draw_idle()
        plt.close(self.fig)

    # ------------------------------------------------------------------
    def getMask(self, currentImage):
        """
        Compute a boolean mask for the drawn polygon.

        Parameters
        ----------
        currentImage : array_like, shape (ny, nx)
            Image whose pixel grid is used to rasterise the polygon.

        Returns
        -------
        ndarray of bool, shape (ny, nx)
            ``True`` for pixels inside the polygon, ``False`` outside.
            Returns an all-False array if fewer than 3 vertices were
            collected (i.e. the polygon was not completed).
        """
        ny, nx = np.shape(currentImage)
        if len(self.allxpoints) < 3:
            return np.zeros((ny, nx), dtype=bool)
        poly_verts = list(zip(self.allxpoints, self.allypoints))
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.ravel(), y.ravel())).T
        path = mplPath.Path(poly_verts)
        return path.contains_points(points).reshape(ny, nx)

    # ------------------------------------------------------------------
    def displayROI(self, ax=None, **linekwargs):
        """
        Overlay the closed polygon outline on an axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes.  Defaults to ``plt.gca()`` (the current axes
            at call time, usually the main GUI axes after the mask
            figure has been closed).
        **linekwargs
            Extra keyword arguments forwarded to
            :class:`matplotlib.lines.Line2D`.
        """
        if len(self.allxpoints) < 2:
            return
        if ax is None:
            ax = plt.gca()
        xs = self.allxpoints + [self.allxpoints[0]]
        ys = self.allypoints + [self.allypoints[0]]
        line = plt.Line2D(xs, ys, color=self.roicolor, **linekwargs)
        ax.add_line(line)
        ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    def displayMean(self, currentImage, ax=None, **textkwargs):
        """
        Annotate with the mean ± std of the ROI pixels.

        Parameters
        ----------
        currentImage : array_like, shape (ny, nx)
            Source image.
        ax : matplotlib.axes.Axes, optional
            Target axes.  Defaults to ``plt.gca()``.
        **textkwargs
            Extra keyword arguments forwarded to
            :func:`matplotlib.pyplot.text`.
        """
        if ax is None:
            ax = plt.gca()
        mask = self.getMask(currentImage)
        vals = np.extract(mask, currentImage)
        if vals.size == 0:
            return
        string = "{:.3f} ± {:.3f}".format(vals.mean(), vals.std())
        ax.text(
            self.allxpoints[0], self.allypoints[0],
            string,
            color=self.roicolor,
            bbox=dict(facecolor="w", alpha=0.6),
            **textkwargs,
        )
        ax.figure.canvas.draw_idle()
