#!/usr/bin/env python
"""
The script for building/installing packages
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fid:
    long_description = fid.read()

setuptools.setup(
    name="toupy",
    version="0.4.0",
    author="Julio Cesar da Silva",
    author_email="julio-cesar.da-silva@neel.cnrs.fr",
    description="Tomographic Utilities for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jcesardasilva/toupy",
    project_urls={"Bug Tracker": "https://github.com/jcesardasilva/toupy/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Operating System :: Unix",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    scripts=[
        "bin/file_comp",
        "bin/missing_recons",
        "bin/plot_projections",
        "bin/create_toupy_templates",
    ],
    install_requires=[
        "fabio>=0.11.0",
        "h5py>=3.1.0",
        "ipython>=7.16.1",
        "ipywidgets>=7.6.0",
        "matplotlib>=3.3.4",
        "numpy>=1.20.0",
        "PyFFTW>=0.12.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "tqdm>=4.61.2",
    ],
    extras_require={
        "resource": [
            "psutil>=5.8.0",
        ],
        "notebook": [
            "ipympl>=0.9.0",
            "jupyterlab>=3.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
        # Optional PyTorch backend for the two-pass multislice and FBaP GPU
        # paths.  Not required: toupy falls back to the pure NumPy backend when
        # torch is absent.  Install with:  pip install toupy[torch]
        "torch": [
            "torch>=1.10.0",
        ],
    },
)
