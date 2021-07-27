#!/usr/bin/env python
"""
The script for building/installing packages
"""

import setuptools

with open("./README.md", "r", encoding="utf-8") as fid:
    long_description = fid.read()

long_description_toupy = "**Toupy** - Tomographic Utilites for Python"

setuptools.setup(
    name="toupy",
    version="0.2.0",
    author="Julio Cesar da Silva",
    author_email="julio-cesar.da-silva@neel.cnrs.fr",
    # console_scripts=["bin/create_toupy_templates"],
    license="LICENCE",
    description="Tomographic Utilities for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jcesardasilva/toupy",
    project_urls={"Bug Tracker": "https://github.com/jcesardasilva/toupy/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Operating System :: Unix",
    ],
    package_dir={"toupy": "toupy"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
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
        "joblib>=1.0.1",
        "matplotlib>=3.3.4",
        "numpy>=1.16.5",
        "numexpr>=2.6.9",
        "scipy>=1.5.4",
        "PyFFTW>=0.11.1",
        "pyopencl>=2021.1.1",
        "scikit_image>=0.17.2",
        "silx>=0.9.0",
        "tqdm>=4.61.2",
    ],
)
