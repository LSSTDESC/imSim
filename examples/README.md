To use imSim you should first follow the instruction in the README.md in the main directory of this repository to make sure it is setup.  Then, you can see a list of possible commands by issuing:

imsim.py --help

There you will see a set of commands-line flags you can use to control the program including the choice of PSF, how the sensors are simulated etc. You can also change more technical settings in a config file. The default config file is in data/default_imsim_configs.  You can edit this file or, better, make a new file which you can then use with the --config-file command line option.

This directory contains a simple example instance catalog (the input to imSim) you can use to test the program called example_instance_catalog.txt

Here are some examples of how you would use it.

_Simple run with all options at their defaults:_

imsim.py example_instance_catalog.txt

_Use a parametric Kolmogorov PSF_

imsim.py --psf Kolmogorov example_instance_catalog.txt

_Use an atmospheric PSF and disable the sensor model_

imsim.py --psf Atmospheric  --disable_sensor_model example_instance_catalog.txt

_Use an atmospheric PSF, disable the sensor model, and restrict the simulation to only one sensor_

imsim.py --psf Atmospheric  --disable_sensor_model --sensor 'R:2,2 S:1,1' example_instance_catalog.txt

_Use an atmospheric PSF, disable the sensor model, restrict the simulation to only one sensor, and write a centroid file that contains information on every source that was drawn_

imsim.py --psf Atmospheric  --disable_sensor_model --sensor 'R:2,2 S:1,1' --create_centroid_file example_instance_catalog.txt

Note by default imSim will not overwrite existing fits files.  So, to re-run you should either remove the previous files or, if you would like to change the behavior, change the value of the 'overwrite' parameter in the config file.
