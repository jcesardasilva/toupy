#!/usr/bin/env python
"""
The script for building/installing packages
"""

import setuptools

def readreadme(filename):
    with open(filename, "r") as fid:
        long_description = fid.read()
    return long_description

long_description_toupy = "**Toupy** - Tomographic Utilites for Python"
packages_list = setuptools.find_packages()

if __name__=='__main__':
    packages_list = setuptools.find_packages()
    long_description_toupy = readreadme("./README.md")
    if packages_list is None:
        sys.exit("Failed to fetch packages")

setuptools.setup(
    name="toupy",
    version="0.1.2",
    author="Julio Cesar da Silva",
    author_email="jdasilva@esrf.fr",
    package_dir={"toupy": "toupy"},
    packages=packages_list,
    scripts=[
        "bin/file_comp",
        "bin/missing_recons",
        "bin/create_toupy_templates",
        "bin/plot_projections",
    ],
    license="LICENCE",
    description="Tomographic Utilities for Python",
    #long_description=long_description_toupy,
    long_description=readreadme("./README.md"),
    #long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Operating System :: Unix",
    ],
    install_requires=[
        "fabio>=0.9.0",
        "h5py>=2.9.0",
        "matplotlib>=3.1.1",
        "libtiff>=0.4.2",
        "numpy>=1.16.4",
        "numexpr>=2.6.9",
        "scipy>=1.3.0",
        "PyFFTW>=0.11.1",
        "scikit-image>=0.15.0",
        "silx>=0.9.0",
        "sphinx>=2.1.2",
    ],
)
