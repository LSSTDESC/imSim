
# Register some configuration files with default desired imSim behavior with
# convenient names. These live in the imsim/config directory, which gets
# installed with imsim. Users can just use the given aliases rather than
# specifying the full location (which will typically be somewhere in the depths
# of a python site-packages directory).

import os
import galsim
from .meta_data import config_dir

galsim.config.RegisterTemplate('imsim-config', os.path.join(config_dir, 'imsim-config.yaml'))
galsim.config.RegisterTemplate('imsim-instcat', os.path.join(config_dir, 'imsim-instcat.yaml'))
galsim.config.RegisterTemplate('imsim-skycat', os.path.join(config_dir, 'imsim-skycat.yaml'))
