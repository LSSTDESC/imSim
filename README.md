[![Build Status](https://travis-ci.org/DarkEnergyScienceCollaboration/imSim.svg?branch=issue%2F21%2Fcomputing_infrastructure)](https://travis-ci.org/DarkEnergyScienceCollaboration/imSim)
[![Coverage Status](https://coveralls.io/repos/github/DarkEnergyScienceCollaboration/imSim/badge.svg?branch=master)](https://coveralls.io/github/DarkEnergyScienceCollaboration/imSim?branch=master)

# imSim
GalSim based LSST simulation package

## Set up
Set up the `lsst_sims` distribution, then enter the following from
this repo directory:
```
$ eups declare -r . imsim -t current
$ setup imsim
$ scons
```
Once you have `eups declared` a package like this, you only need to
execute the `setup imsim` command, and that command can be issued from
any directory.  The `scons` build step needs only to be re-run if a
new command line executable is added to the `bin.src` folder.

## Usage
The executables in the `bin` folder should be in your path and so
should be runnable directly from the command line:
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

Similarly, by doing `setup imsim`, the `PYTHONPATH` environment
variable has been modified so that the `desc.imsim` module is available:
```
>>> import desc.imsim
>>>
```
