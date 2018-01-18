#!/bin/bash
source /opt/lsst/software/stack/loadLSST.bash
setup lsst_sims
pip install nose
pip install coveralls
pip install pylint
eups declare imsim -r /home/imSim -t current
setup imsim
cd /home/imSim
scons
nosetests -s --with-coverage --cover-package=desc.imsim
