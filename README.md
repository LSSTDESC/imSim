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
usage: imsim.py [-h] [-n NUMROWS] [--outdir OUTDIR] [--sensor SENSOR]
                [--config_file CONFIG_FILE]
                [--log_level {DEBUG,INFO,WARN,ERROR,CRITICAL}]
                file

positional arguments:
  file                  The instance catalog

optional arguments:
  -h, --help            show this help message and exit
  -n NUMROWS, --numrows NUMROWS
                        Read the first numrows of the file.
  --outdir OUTDIR       Output directory for eimage file
  --sensor SENSOR       Sensor to simulate, e.g., "R:2,2 S:1,1". If None, then
                        simulate all sensors with sources on them
  --config_file CONFIG_FILE
                        Config file. If None, the default config will be used.
  --log_level {DEBUG,INFO,WARN,ERROR,CRITICAL}
                        Logging level. Default: "INFO"
```

Similarly, by doing `setup imsim`, the `PYTHONPATH` environment
variable has been modified so that the `desc.imsim` module is available:
```
>>> import desc.imsim
>>>
```
