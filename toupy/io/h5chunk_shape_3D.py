#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
from functools import reduce
import math
import operator

__all__ = ["binlist", "numVals", "perturbShape", "chunk_shape_3D"]

"""
"""


def binlist(n, width=0):
    """
    Return list of bits that represent a non-negative integer.

    Parameters
    ----------
    n : int
        non-negative integer
    width : int
        number of bits in returned zero-filled list (default 0)
    """
    return list(map(int, list(bin(n)[2:].zfill(width))))


def numVals(shape):
    """
    Return number of values in chunk of specified shape, given by a list of dimension lengths.

    Parameters
    ----------
    shape : sequence of ints
        list of variable dimension sizes
    """
    if len(shape) == 0:
        return 1
    return reduce(operator.mul, shape)


def perturbShape(shape, onbits):
    """
    Return shape perturbed by adding 1 to elements corresponding to 1 bits in onbits

    Parameters
    ----------
    shape : sequence of ints
        list of variable dimension sizes
    onbits : int
        non-negative integer less than 2**len(shape)
    """
    return list(map(sum, zip(shape, binlist(onbits, len(shape)))))


def chunk_shape_3D(varShape, valSize=4, chunkSize=4096):
    """
    Return a 'good shape' for a 3D variable, assuming balanced 1D/(n-1)D access [#site]_

    Parameters
    ----------
    varShape : sequence of ints
        length 3 list of variable dimension sizes
    chunkSize : int, optional
        maximum chunksize desired, in bytes (default 4096)
    valSize : int, optional
        size of each data value, in bytes (default 4)

    Returns
    -------
    tuple
        Returns integer chunk lengths of a chunk shape that provides
        balanced access of 1D subsets and 2D subsets of a netCDF or HDF5
        variable var with shape (T, X, Y), where the 1D subsets are of the
        form var[:,x,y] and the 2D slices are of the form var[t,:,:],
        typically 1D time series and 2D spatial slices.

    Notes
    -----
    'Good shape' for chunks means that the number of chunks accessed to read either
    kind of 1D or 2D subset is approximately equal, and the size of
    each chunk (uncompressed) is no more than chunkSize, which is
    often a disk block size. Code fetched from [#code1]_ and [#code2]_.

    References
    ----------
    .. [#site] https://www.unidata.ucar.edu/blogs/developer/en/entry/chunking_data_choosing_shapes
    .. [#code1] https://www.unidata.ucar.edu/blog_content/data/2013/chunk_shape_3D.py
    .. [#code2] https://github.com/HDFGroup/datacontainer/blob/master/lib/chunking.py
    """

    rank = 3
    chunkVals = chunkSize / float(valSize)  # ideal number of values in a chunk
    numChunks = (
        varShape[0] * varShape[1] * varShape[2] / chunkVals
    )  # ideal number of chunks
    axisChunks = numChunks ** 0.25  # ideal number of chunks along each 2D axis
    cFloor = []  # will be first estimate of good chunk shape
    # cFloor  = [varShape[0] // axisChunks**2, varShape[1] // axisChunks, varShape[2] // axisChunks]
    # except that each chunk shape dimension must be at least 1
    # chunkDim = max(1.0, varShape[0] // axisChunks**2)
    if varShape[0] / axisChunks ** 2 < 1.0:
        chunkDim = 1.0
        axisChunks = axisChunks / math.sqrt(varShape[0] / axisChunks ** 2)
    else:
        chunkDim = varShape[0] // axisChunks ** 2
    cFloor.append(chunkDim)
    prod = 1.0  # factor to increase other dims if some must be increased to 1.0
    for ii in range(1, rank):
        if varShape[ii] / axisChunks < 1.0:
            prod *= axisChunks / varShape[ii]
    for ii in range(1, rank):
        if varShape[ii] / axisChunks < 1.0:
            chunkDim = 1.0
        else:
            chunkDim = (prod * varShape[ii]) // axisChunks
        cFloor.append(chunkDim)

    # cFloor is typically too small, (numVals(cFloor) < chunkSize)
    # Adding 1 to each shape dim results in chunks that are too large,
    # (numVals(cCeil) > chunkSize).  Want to just add 1 to some of the
    # axes to get as close as possible to chunkSize without exceeding
    # it.  Here we use brute force, compute numVals(cCand) for all
    # 2**rank candidates and return the one closest to chunkSize
    # without exceeding it.
    bestChunkSize = 0
    cBest = cFloor
    for ii in range(8):
        # cCand = map(sum,zip(cFloor, binlist(i, rank)))
        cCand = perturbShape(cFloor, ii)
        thisChunkSize = valSize * numVals(cCand)
        if bestChunkSize < thisChunkSize <= chunkSize:
            bestChunkSize = thisChunkSize
            cBest = list(cCand)  # make a copy of best candidate so far
    return tuple(map(int, cBest))
