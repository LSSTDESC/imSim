#!/bin/bash
source scl_source enable devtoolset-6
source loadLSST.bash
setup lsst_sims
pip install nose
pip install coveralls
pip install pylint
git clone https://github.com/lsst/obs_lsst.git
cd obs_lsst
setup -r . -j
scons lib python shebang examples doc policy
cd ..
eups declare imsim -r ${TRAVIS_BUILD_DIR} -t current
setup imsim
cd ${TRAVIS_BUILD_DIR}
scons
nosetests -s --with-coverage --cover-package=desc.imsim
