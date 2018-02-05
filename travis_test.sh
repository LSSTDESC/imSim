#!/bin/bash
package_dir=$( cd $(dirname $BASH_SOURCE)/..; pwd -P )
source scl_source enable devtoolset-6
source loadLSST.bash
setup lsst_sims
pip install nose
pip install coveralls
pip install pylint
eups declare imsim -r ${package_dir} -t current
setup imsim
cd ${package_dir}
scons
nosetests -s --with-coverage --cover-package=desc.imsim
