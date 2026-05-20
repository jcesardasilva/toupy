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

__all__ = ["phantom", "phantom3d"]


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

    Notes
    -----
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


def phantom3d(N=128, n_v=None, phantom_type="Modified Shepp-Logan", ellipsoids=None):
    """
    Create a 3-D Shepp-Logan or modified Shepp-Logan phantom.

    Extends :func:`phantom` to three dimensions by adding a z semi-axis ``c``
    and a z centre offset ``z0`` to each ellipsoid.  The coordinate system
    is normalised to ``[-1, 1]`` along every axis, so all ellipsoid parameters
    should be given relative to that box.

    Parameters
    ----------
    N : int, optional
        Number of voxels along the transaxial (x and y) directions.
        Default is ``128``.
    n_v : int or None, optional
        Number of voxels along the axial (z) direction.  If ``None`` (default),
        ``n_v = N`` is used, producing a cube.
    phantom_type : str, optional
        Built-in phantom variant.  Either ``'Shepp-Logan'`` or
        ``'Modified Shepp-Logan'`` (default).  Ignored when *ellipsoids*
        is supplied.
    ellipsoids : array-like or None, optional
        Custom ellipsoid table.  Each row must have **eight** elements::

            [I, a, b, c, x0, y0, z0, phi]

        where

        * ``I``   — additive intensity.
        * ``a``   — semi-axis length along x (after rotation by ``phi``).
        * ``b``   — semi-axis length along y (after rotation by ``phi``).
        * ``c``   — semi-axis length along z.
        * ``x0``  — x centre offset in ``[-1, 1]``.
        * ``y0``  — y centre offset in ``[-1, 1]``.
        * ``z0``  — z centre offset in ``[-1, 1]``.
        * ``phi`` — counter-clockwise rotation of the ellipsoid in the xy-plane,
          in degrees.

    Returns
    -------
    p : ndarray, shape (n_v, N, N)
        3-D phantom volume.  Axis order is ``(z, y, x)``.

    Examples
    --------
    >>> from toupy.simulation import phantom3d
    >>> vol = phantom3d(N=64)            # cube, 64^3
    >>> vol = phantom3d(N=64, n_v=48)   # 64 x 64 transaxial, 48 axial slices
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(vol[vol.shape[0]//2], cmap='bone')  # central axial slice

    Notes
    -----
    The 3-D Shepp-Logan ellipsoid parameters are taken from the extension of
    the original Shepp-Logan table commonly used in cone-beam CT simulation
    literature.  The z semi-axes are chosen to give a realistic head-shaped
    volume at the typical aspect ratio used in medical imaging.

    See Also
    --------
    phantom : 2-D Shepp-Logan phantom.

    References
    ----------
    .. [#shepp-logan] Shepp, L. A., Logan, B. F., "Reconstructing Interior Head
       Tissue from X-Ray Transmission", IEEE Transactions on Nuclear Science,
       Feb. 1974, p. 232.
    .. [#toft] Toft, P., "The Radon Transform - Theory and Implementation",
       Ph.D. thesis, Department of Mathematical Modelling, Technical University
       of Denmark, June 1996.
    """
    if n_v is None:
        n_v = N

    if ellipsoids is None:
        ellipsoids = _select_phantom3d(phantom_type)
    else:
        ellipsoids = np.asarray(ellipsoids)
        if ellipsoids.ndim != 2 or ellipsoids.shape[1] != 8:
            raise AssertionError(
                "Custom ellipsoids must have shape (n_ellipsoids, 8); "
                "each row is [I, a, b, c, x0, y0, z0, phi]."
            )

    # Initialise volume
    p = np.zeros((n_v, N, N), dtype=np.float64)

    # Normalised coordinate grids — shape (n_v, N, N)
    zgrid, ygrid, xgrid = np.mgrid[
        -1 : 1 : (1j * n_v),
        -1 : 1 : (1j * N),
        -1 : 1 : (1j * N),
    ]

    for ellip in ellipsoids:
        I   = ellip[0]
        a2  = ellip[1] ** 2
        b2  = ellip[2] ** 2
        c2  = ellip[3] ** 2
        x0  = ellip[4]
        y0  = ellip[5]
        z0  = ellip[6]
        phi = ellip[7] * np.pi / 180  # degrees → radians

        # Translate
        x = xgrid - x0
        y = ygrid - y0
        z = zgrid - z0

        # Rotate in xy-plane by phi
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        xr = x * cos_p + y * sin_p
        yr = y * cos_p - x * sin_p

        # Ellipsoid membership test
        locs = (xr ** 2) / a2 + (yr ** 2) / b2 + (z ** 2) / c2 <= 1
        p[locs] += I

    return p


def _select_phantom3d(name):
    """
    Select a built-in 3-D phantom ellipsoid table by name.

    Parameters
    ----------
    name : str
        Phantom type.  Either ``'Shepp-Logan'`` or
        ``'Modified Shepp-Logan'`` (case-insensitive).

    Returns
    -------
    list of list
        List of ellipsoid parameter rows, each of the form
        ``[I, a, b, c, x0, y0, z0, phi]``.

    Raises
    ------
    ValueError
        If ``name`` does not match a known phantom type.
    """
    if name.lower() == "shepp-logan":
        return _shepp_logan_3d()
    elif name.lower() == "modified shepp-logan":
        return _mod_shepp_logan_3d()
    else:
        raise ValueError("Unknown 3-D phantom type: %s" % name)


def _shepp_logan_3d():
    """
    Standard 3-D Shepp-Logan head phantom ellipsoid parameters.

    Returns
    -------
    list of list
        Ellipsoid parameters ``[I, a, b, c, x0, y0, z0, phi]``.
    """
    #       I        a       b       c       x0      y0       z0     phi
    return [
        [ 2.00,  0.6900, 0.9200, 0.9000,  0.000,  0.0000,  0.000,  0.0],
        [-0.98,  0.6624, 0.8740, 0.8800,  0.000, -0.0184,  0.000,  0.0],
        [-0.02,  0.1100, 0.3100, 0.2200,  0.220,  0.0000, -0.250,-18.0],
        [-0.02,  0.1600, 0.4100, 0.2800, -0.220,  0.0000, -0.250, 18.0],
        [ 0.01,  0.2100, 0.2500, 0.4100,  0.000,  0.3500, -0.250,  0.0],
        [ 0.01,  0.0460, 0.0460, 0.0500,  0.000,  0.1000, -0.250,  0.0],
        [ 0.02,  0.0460, 0.0460, 0.0500,  0.000, -0.1000, -0.250,  0.0],
        [ 0.01,  0.0460, 0.0230, 0.0500, -0.080, -0.6050, -0.250,  0.0],
        [ 0.01,  0.0230, 0.0230, 0.0200,  0.000, -0.6060, -0.250,  0.0],
        [ 0.01,  0.0230, 0.0460, 0.0350,  0.060, -0.6050, -0.250,  0.0],
    ]


def _mod_shepp_logan_3d():
    """
    Modified 3-D Shepp-Logan head phantom with improved contrast.

    Returns
    -------
    list of list
        Ellipsoid parameters ``[I, a, b, c, x0, y0, z0, phi]``.
    """
    #       I        a       b       c       x0      y0       z0     phi
    return [
        [ 1.00,  0.6900, 0.9200, 0.9000,  0.000,  0.0000,  0.000,  0.0],
        [-0.80,  0.6624, 0.8740, 0.8800,  0.000, -0.0184,  0.000,  0.0],
        [-0.20,  0.1100, 0.3100, 0.2200,  0.220,  0.0000, -0.250,-18.0],
        [-0.20,  0.1600, 0.4100, 0.2800, -0.220,  0.0000, -0.250, 18.0],
        [ 0.10,  0.2100, 0.2500, 0.4100,  0.000,  0.3500, -0.250,  0.0],
        [ 0.10,  0.0460, 0.0460, 0.0500,  0.000,  0.1000, -0.250,  0.0],
        [ 0.10,  0.0460, 0.0460, 0.0500,  0.000, -0.1000, -0.250,  0.0],
        [ 0.10,  0.0460, 0.0230, 0.0500, -0.080, -0.6050, -0.250,  0.0],
        [ 0.10,  0.0230, 0.0230, 0.0200,  0.000, -0.6060, -0.250,  0.0],
        [ 0.10,  0.0230, 0.0460, 0.0350,  0.060, -0.6050, -0.250,  0.0],
    ]


def _select_phantom(name):
    """
    Select a built-in phantom ellipse set by name.

    Parameters
    ----------
    name : str
        Phantom type.  Either ``'Shepp-Logan'`` or
        ``'Modified Shepp-Logan'`` (case-insensitive).

    Returns
    -------
    list of list
        List of ellipse parameter rows, each of the form
        ``[I, a, b, x0, y0, phi]``.

    Raises
    ------
    ValueError
        If ``name`` does not match a known phantom type.
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
    Standard Shepp-Logan head phantom ellipse parameters.

    Returns
    -------
    list of list
        Ellipse parameters ``[I, a, b, x0, y0, phi]`` for the original
        Shepp-Logan phantom (10 ellipses).
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
    Modified Shepp-Logan head phantom ellipse parameters with improved contrast.

    Returns
    -------
    list of list
        Ellipse parameters ``[I, a, b, x0, y0, phi]`` for the modified
        Shepp-Logan phantom (10 ellipses), adjusted to improve contrast.
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
