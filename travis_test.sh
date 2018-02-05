#!/bin/bash
source scl_source enable devtoolset-6
source loadLSST.bash
setup lsst_sims
pip install nose
pip install coveralls
pip install pylint
eups declare imsim -r ${TRAVIS_BUILD_DIR} -t current
setup imsim
cd ${TRAVIS_BUILD_DIR}
scons
nosetests -s --with-coverage --cover-package=desc.imsim
