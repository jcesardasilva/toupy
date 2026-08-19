#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pluggable frame-reader layer for reconstructed projections.

This module decouples Toupy's core loading logic from the concrete file
formats it reads.  Each supported format is described by a small object that

* declares which filename extensions it handles (``extensions``), and
* knows how to turn one file into a normalised :class:`ReconFrame`.

Readers register themselves in a process-wide registry keyed by extension.
Callers ask the registry for the reader that matches a path
(:func:`get_reader`) instead of hard-coding an ``if fileext == ".ptyr"``
chain.  Adding support for a new facility's format then means writing one
:class:`FrameReader` and registering it -- no changes to the core loaders.

The three formats Toupy has always supported are provided here as thin
adapters around the existing functions in :mod:`toupy.io.filesrw`, so this
layer changes *how* readers are selected without changing *what* they read.

Notes
-----
All three built-in formats are read one projection at a time: each file
yields a single 2D object.  ``.ptyr`` (PtyPy) and ``.cxi`` (PyNX) objects
are complex and come with a probe; ``.edf`` objects are real and carry no
probe, but the EDF header additionally records how many projections the
whole scan contains (``nvue``), surfaced as
:attr:`ReconFrame.num_projections`.
"""

# Standard library imports
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

# third party packages
import numpy as np

# local packages
from .filesrw import read_cxi, read_edf, read_ptyr

__all__ = [
    "ReconFrame",
    "FrameReader",
    "register_reader",
    "get_reader",
    "available_extensions",
]


@dataclass
class ReconFrame:
    """
    Normalised result of reading one reconstructed projection.

    Attributes
    ----------
    obj : ndarray
        The reconstructed object for a single projection (2D).  Complex for
        ptyr/cxi, real for edf.
    pixelsize : ndarray or float
        Pixel size(s) in metres.  Per-projection readers return a
        ``(vertical, horizontal)`` array; the edf reader returns a scalar.
    energy : float
        Photon energy of the reconstruction.
    probe : ndarray or None, optional
        Reconstructed probe, when the format provides one.  ``None`` for
        formats that do not (edf).  Default ``None``.
    num_projections : int or None, optional
        Total number of projections in the scan, when the file records it in
        its header (edf ``nvue``).  ``None`` for formats that do not carry it
        (ptyr, cxi).  Default ``None``.
    """

    obj: np.ndarray
    pixelsize: object
    energy: float
    probe: Optional[np.ndarray] = None
    num_projections: Optional[int] = None


@runtime_checkable
class FrameReader(Protocol):
    """
    Structural interface a reconstruction-file reader must satisfy.

    A reader is any object that declares the extensions it handles and is
    callable on a path, returning a :class:`ReconFrame`.  Implementations do
    not need to subclass anything -- matching this shape is enough.
    """

    #: Filename extensions handled by this reader (each including the dot).
    extensions: Tuple[str, ...]

    def __call__(
        self, pathfilename: str, correct_orientation: bool = True
    ) -> ReconFrame:
        ...


_REGISTRY: Dict[str, "FrameReader"] = {}


def _normalise_ext(ext: str) -> str:
    """Lower-case ``ext`` and ensure it starts with a dot."""
    ext = ext.lower()
    return ext if ext.startswith(".") else "." + ext


def register_reader(reader: "FrameReader") -> "FrameReader":
    """
    Register ``reader`` for every extension it declares.

    Parameters
    ----------
    reader : FrameReader
        Reader instance exposing an ``extensions`` attribute and callable as
        ``reader(pathfilename, correct_orientation=True) -> ReconFrame``.

    Returns
    -------
    FrameReader
        The same ``reader``, so this can be used as a decorator.

    Raises
    ------
    ValueError
        If ``reader`` declares no extensions.
    """
    extensions = getattr(reader, "extensions", None)
    if not extensions:
        raise ValueError(
            "Reader {!r} declares no extensions to register".format(reader)
        )
    for ext in extensions:
        _REGISTRY[_normalise_ext(ext)] = reader
    return reader


def get_reader(path_or_ext: str) -> "FrameReader":
    """
    Return the reader registered for a path or extension.

    Parameters
    ----------
    path_or_ext : str
        A file path (``"proj_0000.ptyr"``) or a bare extension (``".ptyr"``
        or ``"ptyr"``).

    Returns
    -------
    FrameReader
        The reader registered for the extension.

    Raises
    ------
    IOError
        If no reader is registered for the extension.
    """
    ext = os.path.splitext(path_or_ext)[1] or path_or_ext
    ext = _normalise_ext(ext)
    try:
        return _REGISTRY[ext]
    except KeyError:
        supported = ", ".join(available_extensions())
        raise IOError(
            "No reader registered for '{}'. Supported extensions: {}.".format(
                ext, supported
            )
        )


def available_extensions() -> Tuple[str, ...]:
    """Return the sorted tuple of registered extensions."""
    return tuple(sorted(_REGISTRY))


class _PtyrReader:
    """Adapter around :func:`toupy.io.filesrw.read_ptyr` (PtyPy)."""

    extensions = (".ptyr",)

    def __call__(
        self, pathfilename: str, correct_orientation: bool = True
    ) -> ReconFrame:
        obj, probe, pixelsize, energy = read_ptyr(pathfilename, correct_orientation)
        return ReconFrame(
            obj=obj, pixelsize=pixelsize, energy=energy, probe=probe
        )


class _CxiReader:
    """Adapter around :func:`toupy.io.filesrw.read_cxi` (PyNX)."""

    extensions = (".cxi",)

    def __call__(
        self, pathfilename: str, correct_orientation: bool = True
    ) -> ReconFrame:
        obj, probe, pixelsize, energy = read_cxi(pathfilename, correct_orientation)
        return ReconFrame(
            obj=obj, pixelsize=pixelsize, energy=energy, probe=probe
        )


class _EdfReader:
    """
    Adapter around :func:`toupy.io.filesrw.read_edf`.

    Each EDF file holds one projection and carries no probe, so
    ``correct_orientation`` does not apply; the scan's total projection count
    from the header (``nvue``) is surfaced as
    :attr:`ReconFrame.num_projections`.
    """

    extensions = (".edf",)

    def __call__(
        self, pathfilename: str, correct_orientation: bool = True
    ) -> ReconFrame:
        projection, pixelsize, energy, nvue = read_edf(pathfilename)
        return ReconFrame(
            obj=projection,
            pixelsize=pixelsize,
            energy=energy,
            probe=None,
            num_projections=nvue,
        )


register_reader(_PtyrReader())
register_reader(_CxiReader())
register_reader(_EdfReader())
