#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cone-beam acquisition geometry descriptor.

``ConeBeamGeometry`` is a plain dataclass that encodes every metric
parameter of a circular-orbit cone-beam CT setup.  It is intentionally
kept free of reconstruction imports so that the registration sub-package
can import it without pulling in NumPy FFT or CuPy.

Typical usage::

    from toupy.tomo.geometry import ConeBeamGeometry
    import numpy as np

    geom = ConeBeamGeometry(
        SOD=500.0, SDD=1000.0,
        det_pixel_size=0.1,
        n_u=1024, n_v=512,
        angles=np.linspace(0, 360, 720, endpoint=False),
    )
    geom.validate()
    u = geom.u_coords()   # shape (1024,)
    v = geom.v_coords()   # shape (512,)
"""

# standard library
from __future__ import annotations
from dataclasses import dataclass, field

# third-party
import numpy as np
from numpy import ndarray

__all__ = ["ConeBeamGeometry"]


@dataclass
class ConeBeamGeometry:
    """
    Circular-orbit cone-beam acquisition geometry.

    All distances are expected in consistent physical units (e.g. mm or µm).
    Angles are in degrees.

    Parameters
    ----------
    SOD : float
        Source-to-object distance (source to rotation axis), in physical
        length units.  Must satisfy ``SOD < SDD``.
    SDD : float
        Source-to-detector distance, in the same units as ``SOD``.
    det_pixel_size : float
        Physical pitch of one detector pixel (square pixels assumed), in the
        same units as ``SOD``/``SDD``.  Must be strictly positive.
    n_u : int
        Number of detector columns (horizontal / transaxial direction).
        Must be a positive integer.
    n_v : int
        Number of detector rows (vertical / axial direction).
        Must be a positive integer.
    angles : ndarray, shape (n_angles,)
        Projection angles in degrees.  Arbitrary spacing is allowed;
        the FDK weighting step uses them only to iterate over views.

    Attributes
    ----------
    magnification : float
        Geometric magnification ``SDD / SOD``.  A voxel at the rotation
        axis appears enlarged by this factor on the detector.
    effective_pixel_size : float
        Physical size of one *object-space* pixel at the rotation axis,
        ``det_pixel_size / magnification``.  Use this value to set the
        reconstruction voxel size.

    Examples
    --------
    >>> import numpy as np
    >>> from toupy.tomo.geometry import ConeBeamGeometry
    >>> geom = ConeBeamGeometry(
    ...     SOD=500.0, SDD=1000.0, det_pixel_size=0.055,
    ...     n_u=2048, n_v=2048,
    ...     angles=np.linspace(0, 360, 1800, endpoint=False),
    ... )
    >>> geom.validate()
    >>> print(geom.magnification)          # 2.0
    >>> print(geom.effective_pixel_size)   # 0.0275
    """

    SOD: float
    SDD: float
    det_pixel_size: float
    n_u: int
    n_v: int
    angles: ndarray = field(default_factory=lambda: np.array([]))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def magnification(self) -> float:
        """
        Geometric magnification factor ``SDD / SOD``.

        Returns
        -------
        float
            The ratio of source-to-detector distance to source-to-object
            distance.  Always > 1 for a valid geometry (``SDD > SOD``).

        Raises
        ------
        ZeroDivisionError
            If ``SOD`` is zero (caught earlier by :meth:`validate`).
        """
        raise NotImplementedError

    @property
    def effective_pixel_size(self) -> float:
        """
        Object-space pixel size at the rotation axis.

        Defined as ``det_pixel_size / magnification``, i.e. the physical
        extent of one detector pixel back-projected to the object plane.
        Use this as the reconstruction voxel pitch.

        Returns
        -------
        float
            Effective voxel pitch in the same units as ``det_pixel_size``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def u_coords(self) -> ndarray:
        """
        Centred horizontal (transaxial) detector coordinate array.

        Returns a 1-D array of length ``n_u`` containing the physical
        *detector-plane* u-coordinates of each column centre, measured
        from the beam axis.  The coordinate of column ``i`` is::

            u_i = (i - (n_u - 1) / 2) * det_pixel_size

        Returns
        -------
        u : ndarray, shape (n_u,)
            Detector u-coordinates in physical units, centred on zero.
        """
        raise NotImplementedError

    def v_coords(self) -> ndarray:
        """
        Centred vertical (axial) detector coordinate array.

        Analogous to :meth:`u_coords` but along the vertical detector
        axis.  The coordinate of row ``j`` is::

            v_j = (j - (n_v - 1) / 2) * det_pixel_size

        Returns
        -------
        v : ndarray, shape (n_v,)
            Detector v-coordinates in physical units, centred on zero.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Check self-consistency of all geometry fields.

        Should be called once after construction, before passing the
        geometry object to any reconstruction or registration function.

        Raises
        ------
        ValueError
            With a descriptive message for each invalid condition:

            * ``SOD <= 0`` — source-to-object distance must be positive.
            * ``SDD <= 0`` — source-to-detector distance must be positive.
            * ``SOD >= SDD`` — the detector must lie beyond the object;
              equivalently, ``magnification > 1`` must hold.
            * ``det_pixel_size <= 0`` — pixel pitch must be positive.
            * ``n_u <= 0`` or ``n_v <= 0`` — detector dimensions must be
              positive integers.
            * ``len(angles) == 0`` — at least one projection angle is
              required.
        """
        raise NotImplementedError
