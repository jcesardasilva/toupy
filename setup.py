#!/usr/bin/env python
"""
The script for building/installing packages
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fid:
    long_description = fid.read()

long_description_toupy = "**Toupy** - Tomographic Utilites for Python"

setuptools.setup(
    name="toupy",
    version="0.2.2",
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
    package_dir={"toupy": "toupy"},
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    scripts=[
        "bin/file_comp",
        "bin/missing_recons",
        "bin/plot_projections",
        "bin/create_toupy_templates",
    ],
    install_requires=[
        "fabio>=2022.12.1",
        "h5py>=3.8.0",
        "ipython>=8.10.0",
        "joblib>=1.2.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.2",
        "numexpr>=2.8.4",
        "scipy>=1.10.0",
        "scikit-image>=0.18.3",
        "silx>=1.1.2",
        "tqdm>=4.64.1",
    ],
)
