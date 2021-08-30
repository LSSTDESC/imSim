[![Build Status](https://travis-ci.org/LSSTDESC/imSim.svg?branch=main)](https://travis-ci.org/LSSTDESC/imSim)
[![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/imSim/badge.svg?branch=main)](https://coveralls.io/github/LSSTDESC/imSim?branch=main)

# imSim
imSim is a software package that simulates the LSST telescope and survey.
It produces simulated images from the 3.25 Gigapixel camera which are suitable
to be processed through the LSST Data Management pipeline. imSim takes as an
input a catalog of astronomical sources along with information about how the light
is distorted on the way to Earth including lensing and extinction information.  The
images which are produced include the systematic effects of the atmosphere,
optics and sensors on the observed PSF.

imSim calls the [GalSim](https://github.com/GalSim-developers/GalSim "GalSim GitHub Page") library for astronomical object rendering and is designed to be used inside of the
LSST Data Management and LSST Simulation Group software environment.  It is not a stand alone program.  It requires a working LSST software stack to build and run.

Communication with the imSim development team should take place through the
[imSim GitHub repository](https://github.com/LSSTDESC/imSim) which is part of the
DESC GitHub organization.  Other questions can be directed to Chris Walter at Duke University.

## imSim models

Details on the models implemented by imSim including their validations can be found in the [imSim Feature Matrix](https://github.com/LSSTDESC/imSim/wiki/Feature-Matrix).

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
                [--psf_file PSF_FILE] [--image_path IMAGE_PATH]
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
  --file_id FILE_ID     ID string to use for checkpoint filenames.
  --create_centroid_file
                        Write centroid file(s).
  --seed SEED           integer used to seed random number generator
  --processes PROCESSES
                        number of processes to use in multiprocessing mode
  --psf_file PSF_FILE   Pickle file containing for the persisted PSF. If the
                        file exists, the psf will be loaded from that file,
                        ignoring the --psf option; if not, a PSF will be
                        created and saved to that filename.
  --image_path IMAGE_PATH
                        search path for FITS postage stamp images.This will be
                        prepended to any existing IMSIM_IMAGE_PATH environment
                        variable, for which $CWD is included by default.
```

## Release Status
imSim is currently still considered beta code. A list of tagged versions and pre-releases of this package can be found [here](https://github.com/LSSTDESC/imSim/releases), with the main branch including the most recent (non-tagged) development.

## Licensing and ownership
This software was developed within the LSST DESC using LSST DESC resources, and
so meets the criteria given in, and is bound by, the LSST DESC Publication Policy
for being a “DESC product”.  imSim is licensed with a BSD 3-Clause "New" or "Revised" License as detailed in the included LICENSE file.
