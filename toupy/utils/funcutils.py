#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard libraries imports
import functools
import math
import os
import re
import sys
import shutil
import socket
import urllib
import urllib.request
import warnings

# local libraries imports
from .plot_utils import isnotebook

__all__ = ["switch", "deprecated", "checkhostname", "progress_bar", "tqdm", "downloadURL", "downloadURLfile"]


from tqdm import tqdm


class switch(object):
    """
    Provide switch/case functionality for Python.

    Mimics the ``switch`` statement found in C, Java, and similar
    languages.  Intended to be used in a ``for`` loop with a single
    iteration, where each ``case`` is an ``if`` guarded by a call to
    :meth:`match`.

    Parameters
    ----------
    value : object
        The value to compare against in each case.

    Examples
    --------
    >>> for case in switch(x):
    ...     if case(1):
    ...         print("one")
    ...         break
    ...     if case(2, 3):
    ...         print("two or three")
    ...         break
    ...     if case():  # default
    ...         print("other")
    """

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop."""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """
        Indicate whether to enter a case suite.

        Parameters
        ----------
        *args : object
            Values to match against.  If no arguments are given (default
            case), always returns ``True``.

        Returns
        -------
        bool
            ``True`` if the switch value matches any of ``args``, or if
            fall-through is active, or if no args are given.
        """
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
        if isnotebook():
            print("You are running in a Jupyter Notebook enviroment")
        elif re.search("hpc", hostname) or re.search("hib", hostname):
            print("You are working on the OAR machine: {}".format(hostname))
        elif re.search("rnice", hostname):  # os.system('oarprint host')==0:
            print("You are working on the RNICE machine: {}".format(hostname))
            raise SystemExit("You must use OAR machines, not RNICE")
        elif re.search("gpu", hostname) or re.search("gpid16a", hostname):
            print("You are working on the GPU: {}".format(hostname))
        else:
            print("You running in the machine {}".format(hostname))
            a = input("Do you have enough memory? (y/[n])").lower()
            if str(a) == "" or str(a) == "n":
                raise SystemExit("You must use more powerfull machines")
            if str(a) == "y":
                print("Ok, you assume all the risks!!!!")
        return func(*args, **kwargs)

    return new_func


def close_allopenfiles(obj_test):
    """
    Close all open objects of a given type found in the garbage collector.

    Parameters
    ----------
    obj_test : type
        Type to search for among all live Python objects.

    Examples
    --------
    >>> import h5py
    >>> close_allopenfiles(h5py.File)  # closes all open HDF5 files
    """
    import gc

    for obj in gc.get_objects():  # Browse through ALL objects
        if isinstance(obj, obj_test):  # Just HDF5 files
            try:
                obj.close()
            except:
                pass  # Was already closed


@deprecated
def progbar(curr, total, textstr=""):
    """
    Create a progress bar for for-loops.

    .. deprecated::
        Use ``tqdm.auto.tqdm`` instead of ``progbar``.

    Parameters
    ----------
    curr : int
        Current value to shown in the progress bar
    total : int
        Maximum size of the progress bar.
    textstr : str
        String to be shown at the right side of the progress bar
    """
    termwidth, termheight = shutil.get_terminal_size()
    full_progbar = int(math.ceil(termwidth / 2))
    # ~ full_progbar = termwidth - len(textstr) - 2 # some margin
    frac = curr / total
    filled_progbar = round(frac * full_progbar)
    textbar = "#" * filled_progbar + "-" * (full_progbar - filled_progbar)
    textperc = "[{:>7.2%}]".format(frac)
    print("\r", textbar, textperc, textstr, end="")

def progress_bar(count, block_size, total_size):
    """
    Display a download progress bar in the terminal.

    Intended as a ``reporthook`` callback for
    :func:`urllib.request.urlretrieve`.

    Parameters
    ----------
    count : int
        Number of blocks transferred so far.
    block_size : int
        Size of each block in bytes.
    total_size : int
        Total file size in bytes.  If ``<= 0`` the function returns
        immediately without printing.
    """
    if total_size <= 0:
        return
        
    # Calcul du pourcentage
    percent = int(count * block_size * 100 / total_size)
    # On s'assure de ne pas dépasser 100% à cause du dernier bloc
    percent = min(100, percent)
    
    # Création visuelle de la barre [##########          ]
    bar_length = 40
    filled_length = int(bar_length * percent / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # \r permet de revenir au début de la ligne sans sauter à la suivante
    sys.stdout.write(f'\rIndicateur : |{bar}| {percent}% complet')
    sys.stdout.flush()
  
def downloadURL(url,fname):
    """
    Download file from a URL.
    
    Parameters
    ----------
    url : str
        URL address
    fname : str
        Filename as to be stored
    """
    try:
        print(f"Downloading {fname} from {url}. Please be patient!")
        # reporthook est l'argument magique pour la progression
        urllib.request.urlretrieve(url, fname, reporthook=progress_bar)
        print("\n\nDone")
    except Exception as e:
        print(f"\nErreur: {e}")

@deprecated
def downloadURLfile(url, filename):
    """
    Download and save file from a URL.
    
    Parameters
    ----------
    url : str
        URL address
    fname : str
        Filename as to be stored
    """
    u = urllib.request.urlopen(url)

    with open(filename, 'wb') as f:
        meta = u.info()
        meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])
        print("Downloading: {0} Bytes: {1}".format(url, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)

            status = "{0:16}".format(file_size_dl)
            if file_size:
                status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
            status += chr(13)
            print(status, end="")
        print()

    return filename
    
