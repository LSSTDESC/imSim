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

Details on the models implemented by imSim including their validations can be found in the [imSim Feature Matrix](https://lsstdesc.github.io/imSim/features.html).

## Set up

Please visit the [docs about installation](https://lsstdesc.github.io/imSim/install.html).

## Usage

Basic usage:

Create a configuration `config.yaml`, then run

```sh
galsim config.yaml
```

For more details, visit the [docs about usage](https://lsstdesc.github.io/imSim/usage.html).

## Release Status
imSim is currently still considered beta code. A list of tagged versions and pre-releases of this package can be found [here](https://github.com/LSSTDESC/imSim/releases), with the main branch including the most recent (non-tagged) development.

## Licensing and ownership
This software was developed within the LSST DESC using LSST DESC resources, and
so meets the criteria given in, and is bound by, the LSST DESC Publication Policy
for being a “DESC product”.  imSim is licensed with a BSD 3-Clause "New" or "Revised" License as detailed in the included LICENSE file.
