
# Register some configuration files with default desired imSim behavior with convenient names.
# These live in the imsim/config directory, which gets installed with imsim.
# Users can just use the given aliases rather than specifying the full location (which will
# typically be somewhere in the depths of a python site-packages directory).

import os
import galsim
from .meta_data import config_dir

# The RegisterTemplate feature was added in GalSim 2.4.
# This is currently the development branch (main).
# TODO: Remove this once we can set galsim>=2.4 as a dependency.
if galsim.version >= '2.4':

    galsim.config.RegisterTemplate('imsim-config', os.path.join(config_dir, 'imsim-config.yaml'))
