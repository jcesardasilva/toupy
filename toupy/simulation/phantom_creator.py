#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2010  Alex Opie  <lx_op@orcon.net.nz>
##
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or (at
## your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

"""
Module to create the Shepp-Logan phantom for simulation
Forked from https://jenda.hrach.eu/f2/cat-py/phantom.py
"""

import numpy as np

__all__ = ["phantom"]


def phantom(N=256, phantom_type="Modified Shepp-Logan", ellipses=None):
    """
    Create a Shepp-Logan [#shepp-logan]_ or modified Shepp-Logan
    phantom [#toft]_ . A phantom is a known object (either real or
    purely mathematical) that is used for testing image reconstruction
    algorithms. The Shepp-Logan phantom  is a popular mathematical model
    of a cranial slice, made up of a set of ellipses. This allows
    rigorous testing of computed tomography (CT) algorithms as it can be
    analytically transformed with the radon transform.

    Parameters
    ----------
    N : int
        The edge length of the square image to be produced

    phantom_type : str, optional
        The type of phantom to produce. Either ``Modified Shepp-Logan``
        or ``Shepp-Logan``. The default value is ``Modified Shepp-Logan``.
        This is overriden if ``ellipses`` is also specified.

    ellipses : array like
        Custom set of ellipses to use.

    Note
    ----
    To use ellipses, these should be in the
    form ``[[I, a, b, x0, y0, phi], [I, a, b, x0, y0, phi], ...]``
    where each row defines an ellipse and:

    * ``I`` : Additive intensity of the ellipse.
    * ``a`` : Length of the major axis.
    * ``b`` : Length of the minor axis.
    * ``x0`` : Horizontal offset of the centre of the ellipse.
    * ``y0`` : Vertical offset of the centre of the ellipse.
    * ``phi`` : Counterclockwise rotation of the ellipse in degrees,
      measured as the angle between the horizontal axis and the
      ellipse major axis.

    The image bouding box in the algorithm is ``[-1, -1], [1, 1]``, so
    the values of ``a``, ``b``, ``x0`` and ``y0`` should all be specified
    with respect to this box.

    Returns
    -------
    P : ndarray
        A 2-dimensional array containing th Shepp-Logan phantom image.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> P = phantom()
    >>> # P = phantom(256, 'Modified Shepp-Logan', None)
    >>> plt.imshow(P)

    References
    ----------

    .. [#shepp-logan] Shepp, L. A., Logan, B. F., "Reconstructing Interior Head Tissue
      from X-Ray Transmission", IEEE Transactions on Nuclear Science,
      Feb. 1974, p. 232

    .. [#toft] Toft, P., "The Radon Transform - Theory and Implementation",
      Ph.D. thesis, Department of Mathematical Modelling, Technical
      University of Denmark, June 1996
    """
    if ellipses is None:
        ellipses = _select_phantom(phantom_type)
    elif np.size(ellipses, 1) != 6:
        raise AssertionError("Wrong number of columns in user phantom")

    # Initiate the image with zeros
    p = np.zeros((N, N))

    # Create the pixel grid
    ygrid, xgrid = np.mgrid[-1 : 1 : (1j * N), -1 : 1 : (1j * N)]

    for ellip in ellipses:
        I = ellip[0]
        a2 = ellip[1] ** 2
        b2 = ellip[2] ** 2
        x0 = ellip[3]
        y0 = ellip[4]
        phi = ellip[5] * np.pi / 180  # Rotation angle in radians

        # Create the offset x and y values for the grid
        x = xgrid - x0
        y = ygrid - y0

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # Find the pixels within the ellipse
        locs = (
            ((x * cos_p + y * sin_p) ** 2) / a2 + ((y * cos_p - x * sin_p) ** 2) / b2
        ) <= 1

        # Add the ellipse intensity to those pixels
        p[locs] += I

    return p


def _select_phantom(name):
    """
    Wrapper to select the phantom type
    """
    if name.lower() == "shepp-logan":
        e = _shepp_logan()
    elif name.lower() == "modified shepp-logan":
        e = _mod_shepp_logan()
    else:
        raise ValueError("Unknown phantom type: %s" % name)
    return e


def _shepp_logan():
    """
    Standard head phantom, taken from Shepp-Logan
    """
    return [
        [2, 0.69, 0.92, 0, 0, 0],
        [-0.98, 0.6624, 0.8740, 0, -0.0184, 0],
        [-0.02, 0.1100, 0.3100, 0.22, 0, -18],
        [-0.02, 0.1600, 0.4100, -0.22, 0, 18],
        [0.01, 0.2100, 0.2500, 0, 0.35, 0],
        [0.01, 0.0460, 0.0460, 0, 0.1, 0],
        [0.02, 0.0460, 0.0460, 0, -0.1, 0],
        [0.01, 0.0460, 0.0230, -0.08, -0.605, 0],
        [0.01, 0.0230, 0.0230, 0, -0.606, 0],
        [0.01, 0.0230, 0.0460, 0.06, -0.605, 0],
    ]


def _mod_shepp_logan():
    """
    Modified version of Shepp-Logan hean phantom
    ajusted to improve contrast.
    """
    return [
        [1, 0.69, 0.92, 0, 0, 0],
        [-0.80, 0.6624, 0.8740, 0, -0.0184, 0],
        [-0.20, 0.1100, 0.3100, 0.22, 0, -18],
        [-0.20, 0.1600, 0.4100, -0.22, 0, 18],
        [0.10, 0.2100, 0.2500, 0, 0.35, 0],
        [0.10, 0.0460, 0.0460, 0, 0.1, 0],
        [0.10, 0.0460, 0.0460, 0, -0.1, 0],
        [0.10, 0.0460, 0.0230, -0.08, -0.605, 0],
        [0.10, 0.0230, 0.0230, 0, -0.606, 0],
        [0.10, 0.0230, 0.0460, 0.06, -0.605, 0],
    ]
