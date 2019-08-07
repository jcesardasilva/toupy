#!/usr/bin/env python
"""
The script for building/installing packages
"""

import setuptools

def readreadme(filename):
    with open(filename, "r") as fid:
        long_description = fid.read()
    return long_description

setuptools.setup(
    name="toupy",
    version="0.1.0",
    author="Julio Cesar da Silva",
    author_email="jdasilva@esrf.fr",
    package_dir={"toupy": "toupy"},
    packages=setuptools.find_packages(),
    scripts=["bin/file_comp", "bin/missing_recons"],
    license="LICENCE",
    description="Tomographic Utilities for Python",
    long_description=readreadme("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Operating System :: Unix",
    ],
)
