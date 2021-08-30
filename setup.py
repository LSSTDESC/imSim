from __future__ import print_function

import sys,os,glob,re
import platform
import ctypes
import ctypes.util
import types
import subprocess
import re
import tempfile
import urllib.request as urllib2
import tarfile
import shutil
import setuptools
from setuptools import setup, find_packages

print("Using setuptools version",setuptools.__version__)
print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

def all_files_from(dir, ext=''):
    files = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(ext) and not filename.startswith( ('.', 'SCons') ):
                files.append(os.path.join(root, filename))
    return files

run_dep = ['numpy', 'galsim']

with open('README.md') as file:
    long_description = file.read()

packages = find_packages()
print('packages = ',packages)

def all_files_from(dir, ext=''):
    """Quick function to get all files from directory and all subdirectories
    """
    files = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(ext) and not filename.startswith('.'):
                files.append(os.path.join(root, filename))
    return files

shared_data = all_files_from('data')

# Read in the version from imsim/_version.py
# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file=os.path.join('imsim','_version.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    imsim_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print('ImSim version is %s'%(imsim_version))

dist = setup(name="ImSim",
    version=imsim_version,
    author="ImSim Developers (point of contact: Chris Walter)",
    author_email="chris.walter@duke.edu",
    description="Image Simulation tools for LSST DESC",
    long_description=long_description,
    license = "BSD License",
    url="https://github.com/LSSTDESC/imSim",
    download_url="https://github.com/LSSTDESC/imSim/releases/tag/v%s.zip"%imsim_version,
    packages=packages,
    package_data={'imsim': shared_data},
    install_requires=run_dep,
    )
