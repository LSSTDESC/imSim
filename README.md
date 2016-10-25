[![Build Status](https://travis-ci.org/DarkEnergyScienceCollaboration/imSim.svg?branch=issue%2F21%2Fcomputing_infrastructure)](https://travis-ci.org/DarkEnergyScienceCollaboration/imSim)
[![Coverage Status](https://coveralls.io/repos/github/DarkEnergyScienceCollaboration/imSim/badge.svg?branch=master)](https://coveralls.io/github/DarkEnergyScienceCollaboration/imSim?branch=master)

# imSim
GalSim based LSST simulation package

## Set up
If you have the `lsst_sims` distribution set up, including the
`sims_catalogs_measures` and `sims_catalogs_generation` repos, then
enter the following from this repo directory:
```
$ eups declare -r . imsim -t current
$ setup imsim
```

## Usage
```
$ imsim.py --help
usage: imsim.py [-h] [-v] [-f FILE] [-n NUMROWS] [--outdir OUTDIR]

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -f FILE, --file FILE  Specify the instance file
  -n NUMROWS, --numrows NUMROWS
                        read the first numrows of the file.
  --outdir OUTDIR       output directory for eimage file
```
