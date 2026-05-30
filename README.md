<img src="resources/toupy_logo.png" alt="toupy" width="200"> 

TOUPY - Tomographic Utilities for Python
========================================
DOI: [![DOI](https://zenodo.org/badge/196718112.svg)](https://zenodo.org/badge/latestdoi/196718112)

Introduction
------------

The name **Toupy** stands for Tomographic Utilities for Python and it is a wordplay with the French 
word *toupie* (pronounced *too-pee*) for spinning top, the toy designed to spin rapidly on the ground, the motion of 
which causes it to remain precisely balanced on its tip due to its rotational inertia. 
You can find the information in the wikipedia page in [English](https://en.wikipedia.org/wiki/Top) and in [French](https://fr.wikipedia.org/wiki/Toupie_(jouet)).

Documentation
-------------

The documentation can be found in [ReadTheDocs](https://toupy.readthedocs.io/en/latest/).

Installation
------------

Install the latest release from PyPI:

    pip install toupy

For a local editable install (recommended for development), clone or download the repository and run:

    pip install -e .

from the folder containing `setup.py`. To also install optional Jupyter notebook dependencies
(interactive plotting with `%matplotlib widget`):

    pip install -e ".[notebook]"

To upgrade an existing installation:

    pip install -U toupy

Dependencies
------------

Toupy requires Python >= 3.8 and the following packages:

* numpy >= 1.20
* scipy >= 1.7
* matplotlib >= 3.3
* PyFFTW >= 0.12
* fabio >= 0.11
* h5py >= 3.1
* scikit-image >= 0.18
* tqdm >= 4.61
* ipython >= 7.16
* ipywidgets >= 7.6

Optional (for Jupyter notebook interactive plotting):

* ipympl >= 0.9  (`pip install ipympl`)
* jupyterlab >= 3.0

Optional (for GPU acceleration of the phase-retrieval and reconstruction methods):

* cupy  (e.g. `pip install cupy-cuda12x` — matched to your CUDA version)

Get started
-----------

Get started quickly with the examples in the [templates](https://github.com/jcesardasilva/toupy/tree/master/templates) directory
and the [tutorial notebooks](https://github.com/jcesardasilva/toupy/tree/master/tutorial).

Phase retrieval, ring correction and TV reconstruction
------------------------------------------------------

Methods for propagation-based / holotomographic nanotomography.  All compute
functions accept ``cuda=True`` for optional GPU acceleration via CuPy, falling
back to CPU automatically when CuPy is not installed:

* **Phase retrieval** (`toupy.restoration`)
  * `tie_hom` — single-distance TIE-Hom (Paganin) retrieval for homogeneous objects.
  * `ctf_retrieve` — CTF retrieval; single distance *or* multi-distance
    (holotomographic) inversion via `delta_beta` and a list of distances.
  * `iterative_phase_retrieval` — nonlinear retrieval using the exact Fresnel
    forward model (single or multi-distance), with Tikhonov / total-variation
    regularisation. Accurate in the multi-fringe / higher-δ/β regime where
    TIE-Hom blurs.
  * `suggest_holo_distances` — rule-based, gap-free multi-distance series for
    holotomography (CTF-zero interleaving).
* **Ring artifact correction** (`toupy.restoration`)
  * `remove_rings_wavelet_fft`, `remove_rings_titarenko`, `remove_rings_stack`.
* **Total-variation reconstruction** (`toupy.tomo`)
  * `tv_reconstruction`, `chambolle_pock_tv` — TV-regularised FBP (Chambolle–Pock).

Runnable examples in [`tutorial/`](tutorial/):

* `example_phase_retrieval_tv_ring.py` — phase retrieval + ring correction + TV pipeline.
* `example_iterative_phase_retrieval.py` — TIE-Hom vs nonlinear iterative (single & multi-distance).
* `example_iterative_phase_tomography.py` — full phase-contrast tomography pipeline + CTF coverage diagnostic.

See [`CHANGELOG.md`](CHANGELOG.md) for details.

Call for Contributions
----------------------

Toupy welcomes help and improvements from a wide range of different backgrounds.
For example, work on the documentation is well appreciated.
Please open an issue or pull request on [GitHub](https://github.com/jcesardasilva/toupy).
