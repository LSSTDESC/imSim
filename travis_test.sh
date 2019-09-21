#!/bin/bash
source scl_source enable devtoolset-6
source loadLSST.bash
setup -t current lsst_sims
pip install nose
pip install coveralls
pip install pylint
git clone https://github.com/lsst/obs_lsst.git
cd obs_lsst
setup -r . -j

# Build the obs_lsst package, but avoid the time consuming and lengthy
# output from the 'tests' target.  The version module is built by the
# 'tests', so build that explicitly since it is imported by the
# package itself.
scons lib python shebang examples doc policy python/lsst/obs/lsst/version.py

cd ..

# Clone sims_GalSimInterface and set up master to suppress
# lsst.afw.geom deprecation warnings.
git clone https://github.com/lsst/sims_GalSimInterface.git
cd sims_GalSimInterface
setup -r . -j
cd ..

eups declare imsim -r ${TRAVIS_BUILD_DIR} -t current
setup imsim
cd ${TRAVIS_BUILD_DIR}
scons
nosetests -s --with-coverage --cover-package=desc.imsim
