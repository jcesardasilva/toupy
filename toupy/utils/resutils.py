#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resource-availability utilities.

Portable helpers to estimate the peak memory an array operation needs and
compare it against the RAM currently available, so that a computation can
warn (or, opt-in, refuse to start) before it exhausts the machine's memory.

These replace the old, ESRF-specific ``@checkhostname`` guard: rather than
inferring "is this a big enough machine?" from the hostname, they measure the
actual resource that matters.

Notes
-----
Reading available memory relies on :mod:`psutil`, which is an *optional*
dependency.  If it is not installed, the checks degrade gracefully: they
skip the comparison and return without warning (or, in ``strict`` mode,
raise :class:`RuntimeError` so the caller knows the guarantee could not be
provided).  Install it with ``pip install toupy[resource]`` or
``pip install psutil``.
"""

# standard library imports
import warnings

# third party package imports
import numpy as np

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - depends on the environment
    psutil = None
    _HAS_PSUTIL = False

__all__ = [
    "psutil_available",
    "available_memory",
    "estimate_peak_bytes",
    "humanize_bytes",
    "check_memory_requirement",
]


def psutil_available():
    """
    Report whether :mod:`psutil` is importable.

    Returns
    -------
    bool
        ``True`` if :mod:`psutil` was imported successfully, ``False``
        otherwise.  When ``False`` the memory checks cannot read the amount
        of available RAM and degrade gracefully.
    """
    return _HAS_PSUTIL


def available_memory():
    """
    Return the amount of readily-available RAM in bytes.

    Uses ``psutil.virtual_memory().available``, i.e. the memory that can be
    given to processes without the system starting to swap.

    Returns
    -------
    int or None
        Available memory in bytes, or ``None`` if :mod:`psutil` is not
        installed.
    """
    if not _HAS_PSUTIL:
        return None
    return psutil.virtual_memory().available


def humanize_bytes(nbytes):
    """
    Format a byte count as a human-readable string.

    Parameters
    ----------
    nbytes : int or float
        Number of bytes.

    Returns
    -------
    str
        A string such as ``'1.5 GiB'`` using binary (1024-based) units.
    """
    nbytes = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024.0 or unit == "TiB":
            return "{:.2f} {}".format(nbytes, unit)
        nbytes /= 1024.0


def estimate_peak_bytes(shape, dtype=np.float64, safety_factor=3.0):
    """
    Estimate the peak memory an array operation needs, in bytes.

    The estimate is the footprint of one full array of the given ``shape``
    and ``dtype``, multiplied by ``safety_factor`` to account for the
    temporary copies most numerical routines allocate.  FSC and
    tomographic reconstruction, for instance, hold several full-volume
    arrays simultaneously (the two input volumes, FFTs, masks, ...), so a
    factor of around ``3`` is a sensible default.

    Parameters
    ----------
    shape : tuple of int, or sequence of such tuples
        Shape of the array. A sequence of shapes may be given, in which case
        their individual footprints are summed (useful when several arrays of
        different sizes coexist); a single ``dtype`` is applied to all of them.
    dtype : data-type, optional
        The array data-type.  Default is ``numpy.float64``.
    safety_factor : float, optional
        Multiplier accounting for temporary copies held simultaneously.
        Must be ``>= 1``.  Default is ``3.0``.

    Returns
    -------
    float
        Estimated peak memory requirement in bytes.
    """
    if safety_factor < 1:
        raise ValueError("safety_factor must be >= 1")

    itemsize = np.dtype(dtype).itemsize

    # allow a single shape tuple, or a sequence of shape tuples
    shapes = shape
    if len(shape) > 0 and np.isscalar(shape[0]):
        shapes = [shape]

    total_elems = 0
    for shp in shapes:
        nelem = 1
        for dim in shp:
            nelem *= int(dim)
        total_elems += nelem

    return float(total_elems) * itemsize * float(safety_factor)


def check_memory_requirement(
    shape,
    dtype=np.float64,
    safety_factor=3.0,
    strict=False,
    operation="this operation",
    available=None,
):
    """
    Check whether an array operation is likely to fit in available RAM.

    Estimates the peak memory needed for an operation on an array of the
    given ``shape`` and ``dtype`` (see :func:`estimate_peak_bytes`) and
    compares it against the memory currently available.  By default an
    insufficient-memory situation only emits a warning, so a legitimate run
    is never blocked; pass ``strict=True`` to raise instead.

    This is portable and contains no hostname logic; it is the resource-aware
    replacement for the former ``@checkhostname`` guard.

    Parameters
    ----------
    shape : tuple of int, or sequence of such tuples
        Shape(s) of the array(s) the operation works on.
    dtype : data-type, optional
        The array data-type.  Default is ``numpy.float64``.
    safety_factor : float, optional
        Multiplier accounting for temporary copies.  Default is ``3.0``.
    strict : bool, optional
        If ``True``, raise :class:`MemoryError` when the estimated
        requirement exceeds the available memory (and :class:`RuntimeError`
        when the amount of available memory cannot be determined because
        :mod:`psutil` is missing).  If ``False`` (default), only a
        :class:`warnings.warn` message is emitted and the function returns
        normally.
    operation : str, optional
        Human-readable name of the operation, used in the messages.
    available : int, optional
        Override for the amount of available memory in bytes.  Mainly useful
        for testing; if ``None`` (default) it is read via
        :func:`available_memory`.

    Returns
    -------
    bool
        ``True`` if the operation is expected to fit (or if the check could
        not be performed and ``strict`` is ``False``), ``False`` if the
        estimate exceeds the available memory and ``strict`` is ``False``.

    Raises
    ------
    MemoryError
        If ``strict`` is ``True`` and the estimate exceeds available memory.
    RuntimeError
        If ``strict`` is ``True`` and available memory cannot be determined.

    Examples
    --------
    >>> # a 1000**3 float32 volume, non-fatal warning if it will not fit
    >>> check_memory_requirement((1000, 1000, 1000), np.float32)  # doctest: +SKIP
    """
    required = estimate_peak_bytes(shape, dtype, safety_factor)

    if available is None:
        available = available_memory()

    if available is None:
        # psutil not installed: cannot compare, degrade gracefully
        msg = (
            "psutil is not installed; skipping the available-memory check "
            "for {} (estimated peak requirement {}). Install 'toupy[resource]' "
            "to enable it.".format(operation, humanize_bytes(required))
        )
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return True

    if required > available:
        msg = (
            "{} may need about {} of RAM (shape {}, dtype {}, safety factor "
            "{:g}), but only {} is currently available. The process may run "
            "out of memory or start swapping.".format(
                operation,
                humanize_bytes(required),
                shape,
                np.dtype(dtype).name,
                safety_factor,
                humanize_bytes(available),
            )
        )
        if strict:
            raise MemoryError(msg)
        warnings.warn(msg, ResourceWarning, stacklevel=2)
        return False

    return True
