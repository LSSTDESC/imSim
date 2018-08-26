[![Build Status](https://travis-ci.org/LSSTDESC/imSim.svg?branch=master)](https://travis-ci.org/LSSTDESC/imSim)
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/imSim/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/imSim?branch=master)

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
usage: imsim.py [-h] [-n NUMROWS] [--outdir OUTDIR] [--sensors SENSORS]
                [--config_file CONFIG_FILE]
                [--log_level {DEBUG,INFO,WARN,ERROR,CRITICAL}]
                [--psf {DoubleGaussian,Kolmogorov,Atmospheric}]
                [--disable_sensor_model] [--file_id FILE_ID]
                [--create_centroid_file] [--seed SEED] [--processes PROCESSES]
                instcat

positional arguments:
  instcat               The instance catalog

optional arguments:
  -h, --help            show this help message and exit
  -n NUMROWS, --numrows NUMROWS
                        Read the first numrows of the file.
  --outdir OUTDIR       Output directory for eimage file
  --sensors SENSORS     Sensors to simulate, e.g., "R:2,2 S:1,1^R:2,2 S:1,0".
                        If None, then simulate all sensors with sources on
                        them
  --config_file CONFIG_FILE
                        Config file. If None, the default config will be used.
  --log_level {DEBUG,INFO,WARN,ERROR,CRITICAL}
                        Logging level. Default: INFO
  --psf {DoubleGaussian,Kolmogorov,Atmospheric}
                        PSF model to use. Default: Kolmogorov
  --disable_sensor_model
                        disable sensor effects
  --file_id FILE_ID     ID string to use for checkpoint and centroid
                        filenames.
  --create_centroid_file
                        Write centroid file(s).
  --seed SEED           integer used to seed random number generator
  --processes PROCESSES
                        number of processes to use in multiprocessing mode
```
