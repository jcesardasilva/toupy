#!/usr/bin/env python
"""
The script for building/installing packages
"""

import setuptools

with open("README.md", "r") as fid:
    long_description = fid.read()

setuptools.setup(
    name="toupy",
    version="0.1.0",
    author="Julio Cesar da Silva",
    author_email="jdasilva@esrf.fr",
    package_dir={"toupy": "toupy"},
    packages=setuptools.find_packages(),
    scripts=["bin/file_comp", "bin/missing_recons"],
    license="LICENCE",
    description="Tomography utils in Python for high-resolution imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Operating System :: Unix",
    ],
    install_requires=[
        "fabio >= 0.9.0",
        "h5py >= 2.9.0",
        "matplotlib >= 3.1.1",
        "libtiff >=0.4.2",
        "numpy >= 1.16.4",
        "scipy >= 1.3.0",
        "PyFFTW >= 0.11.1",
        "roipoly >= 0.5.0",
        "scikit-image >= 0.15.0",
        "silx >= 0.9.0",
    ],
)

"""
packages=['toupy',
                'toupy.io',
                'toupy.registration',
                'toupy.resolution',
                'toupy.tomo',
                'toupy.unwrapping',
                'toupy.utils'],
"""
