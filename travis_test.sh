#!/bin/bash
source scl_source enable devtoolset-8
source loadLSST.bash
setup -t sims_w_2019_42 lsst_sims
setup -t DC2production throughputs
setup -t DC2production sims_skybrightness_data
pip install nose
pip install coveralls
pip install pylint
git clone https://github.com/lsst/obs_lsst.git
cd obs_lsst
git checkout dc2/run2.1
setup -r . -j

# Build the obs_lsst package, but avoid the time consuming and lengthy
# output from the 'tests' target.  The version module is built by the
# 'tests', so build that explicitly since it is imported by the
# package itself.
scons lib python shebang examples doc policy python/lsst/obs/lsst/version.py

cd ..

eups declare imsim -r ${TRAVIS_BUILD_DIR} -t current
setup imsim
cd ${TRAVIS_BUILD_DIR}
scons
nosetests -s --with-coverage --cover-package=desc.imsim
