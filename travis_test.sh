#!/bin/bash
source scl_source enable devtoolset-6
source /lsst/stack/loadLSST.bash
setup lsst_sims
pip install nose
pip install coveralls
pip install pylint
eups declare imsim -r /home/imSim -t current
setup imsim
cd /home/imSim
scons
nosetests -s --with-coverage --cover-package=desc.imsim
pylint --py3k `find . -name \*.py -print`
