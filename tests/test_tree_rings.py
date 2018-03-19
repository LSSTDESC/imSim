"""
Rudimentary test for tree ring code
Craig Lage 19-Mar-18
"""

from __future__ import absolute_import, print_function
import os
import numpy as np
import lsst.utils as lsstUtils
from lsst.sims.GalSimInterface import make_galsim_detector
from lsst.sims.GalSimInterface import LSSTCameraWrapper
from lsst.sims.GalSimInterface import GalSimInterpreter
import desc.imsim

sensors = ['R:2,2 S:1,1', 'R:3,4 S:2,2']

camera_wrapper = LSSTCameraWrapper()

desc.imsim.read_config()
instcat_file = os.path.join(lsstUtils.getPackageDir('imsim'),
                               'tests', 'tiny_instcat.txt')

stuff = desc.imsim.parsePhoSimInstanceFile(instcat_file)

obs_md = stuff.obs_metadata
phot_params = stuff.phot_params

detector_list = []
for sensor in sensors:
    detector_list.append(make_galsim_detector(camera_wrapper, sensor, phot_params, obs_md))

gs_interpreter = GalSimInterpreter(detectors=detector_list)

desc.imsim.add_treering_info(gs_interpreter)

r_value = 5280.0

for detector in gs_interpreter.detectors:
    print("Detector = ", detector.name)
    print("Detector center = ",detector.tree_rings.center)
    print("At a value of %f, Detector radial function = "%r_value, detector.tree_rings.func(r_value))


