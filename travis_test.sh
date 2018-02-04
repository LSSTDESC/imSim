#!/bin/bash
source scl_source enable devtoolset-6
source loadLSST.bash
setup lsst_sims
pip install nose
pip install coveralls
pip install pylint
eups declare imsim -r /home/travis/imSim -t current
setup imsim
cd /home/travis/imSim
scons
nosetests -s --with-coverage --cover-package=desc.imsim
