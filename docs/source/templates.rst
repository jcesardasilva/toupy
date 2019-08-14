*********
Templates
*********

The templates here are routines to analyze the X-ray tomographic data.
It is exemplified by the case of data acquired at ID16A beamline of
ESRF, but it can easily adapted to datas from any other beamline.

They consists basically into an step of load the projections data, the
main step to be adapted for different beamlines, the implementation of
processing step and, finally the saving step. For ID16A data, you should
not need to adapt the code.

Tomographic Reconstruction
==========================
The tomographic reconstruction of high resolution data is divided in
several steps:

* Processing of the projections, for example, the phase-retrieval for
  phase contrast imaging, the fluorescence fitting for XRF-tomographic
  dataset and other.
* Restoration of the projections in case of phase wrapping or linear
  phase ramp for phase-contrast imaging, and the normalization of the XRF
  datasets.
* Alignment of the stack of projections.
* Finally, the tomographic reconstruction.

The templates here will then guide you through the steps above. After
each step, the files are saved and can be used for the next step.
All the files, except Tiff conversion, are saved in HDF5 format.
For the analysis, it is important to have the latest version of Python
packages. For people working with data from ID16A and with access to
beamline computing resources, this can be obtained by using the ID16A
Python environment, which is activated by typing `py3venv_on` on the Linux prompt.

The python scripts can be run from shell, from ipython or from Jupyter
notebook. This dependes on the user`s preference. For illustration
purposes only, the description below supposes you will launch the scripts
from shell.

Loading of the projections
--------------------------
Edit `load_projections.py`  with proper parameters and run

.. code-block:: shell

  python load_projections.py

The instructions of what to do appear on the screen.
It loads either .tif or .edf files.
The next step consists of the vertical registration of the projections.

Linear Phase ramp removal
-------------------------
Edit `remove_phase_ramp.py` with proper parameters and run

.. code-block:: shell

  python remove_phase_ramp.py

This open a GUI interface with buttons to allow to proceed with the phase ramp removal.

Phase unwrapping
----------------
Edit `phase_unwrapping.py`  with proper parameters and run

.. code-block:: shell

  python phase_unwrapping.py

The instructions of what to do appear on the screen.

Vertical alignment
------------------
Edit `vertical_alignment.py`  with proper parameters and run

.. code-block:: shell

  python vertical_alignment.py

The instructions of what to do appear on the screen.

Derivatives of the Projection
-----------------------------
Edit `projections_derivatives.py`  with proper parameters and run

.. code-block:: shell

  python projections_derivatives.py

The instructions of what to do appear on the screen.

Sinogram inspection
-------------------
Edit `sinogram_inspection.py`  with proper parameters and run

.. code-block:: shell

  python sinogram_inspection.py

The instructions of what to do appear on the screen.

Horizontal alignment
--------------------
Edit `horizontal_alignment.py`  with proper parameters and run

.. code-block:: shell

  python horizontal_alignment.py

The instructions of what to do appear on the screen.

Tomographic reconstruction
--------------------------
Edit `tomographic_reconstruction.py`  with proper parameters and run

.. code-block:: shell

  python tomographic_reconstruction.py

The instructions of what to do appear on the screen.

Tiff 8 or 16 bits conversion
-----------------------------
This step is only necessary for people who want to have the tomographic
slices as tiff rather than as HDF5.

Edit `tiff_conversion.py`  with proper parameters and run

.. code-block:: shell

  python tiff_conversion.py

The instructions of what to do appear on the screen.
