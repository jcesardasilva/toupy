#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import functools
import math
import re
import shutil
import socket
import warnings

__all__ = ["switch", "deprecated", "checkhostname", "progbar"]


class switch(object):
    """
    This class provides the functionality of switch or case in other
    languages than python
    Python does not have switch
    """

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite """
        if self.fall or not args:
            return True
        elif self.value in args:
            self.fall = True
            return True
        else:
            return False


def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def checkhostname(func):
    """
    Check if running in OAR, if not, exit.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        hostname = socket.gethostname()  # os.environ['HOST']
        # hostname.find('rnice')==0:
        if re.search("hpc", hostname) or re.search("hib", hostname):
            print("You are working on the OAR machine: {}".format(hostname))
        elif re.search("rnice", hostname):  # os.system('oarprint host')==0:
            print("You are working on the RNICE machine: {}".format(hostname))
            raise SystemExit("You must use OAR machines, not RNICE")
        elif re.search("gpu", hostname) or re.search("gpid16a", hostname):
            print("You are working on the GPU: {}".format(hostname))
        else:
            print(
                "You running in machine {}, which is not an ESRF machine".format(
                    hostname
                )
            )
            a = input(
                "Possibly running in the wrong machine. Do you have enough memory? (y/[n])"
            ).lower()
            if str(a) == "" or str(a) == "n":
                raise SystemExit("You must use more powerfull machines")
            if str(a) == "y":
                print("Ok, you assume all the risks!!!!")
        return func(*args, **kwargs)

    return new_func


def close_allopenfiles(obj_test):
    """
    Browse through all objects, check if there is files open of type of
    obj_test and close them.
    
    Example
    -------
    >> import h5py
    >> obj_test = h5py.File
    >> close_allopenfiles(obj_test) # will close all open HDF5 file
    """
    import gc

    for obj in gc.get_objects():  # Browse through ALL objects
        if isinstance(obj, obj_test):  # Just HDF5 files
            try:
                obj.close()
            except:
                pass  # Was already closed


def progbar(curr, total, textstr=""):
    termwidth, termheight = shutil.get_terminal_size()
    full_progbar = int(math.ceil(termwidth / 2))
    #~ full_progbar = termwidth - len(textstr) - 2 # some margin
    frac = curr / total
    filled_progbar = round(frac * full_progbar)
    textbar = "#" * filled_progbar + "-" * (full_progbar - filled_progbar)
    textperc = "[{:>7.2%}]".format(frac)
    print("\r", textbar, textperc, textstr, end="")
