# To set up this package using eups, source this script after setting
# up the LSST Stack in your bash environment.  You may also need to
# set up some Stack dependencies explicitly, e.g.,
#
# $ setup lsst_sims
#
# as well as add non-eups-managed dependencies to your PYTHONPATH,
# PATH, etc..

inst_dir=$( cd $(dirname $BASH_SOURCE)/..; pwd -P )
eups declare imSim -r ${inst_dir} -t ${USER}
setup imSim -t ${USER}
