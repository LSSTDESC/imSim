"""
Time execution of galsim.SiliconSensor.accumulate.
"""
import os
import sys
import time
import argparse
import numpy as np
import galsim
import desc.imsim

parser = argparse.ArgumentParser(description="Time SiliconSensor.accumulate")
parser.add_argument('--seed', type=int, default=1001, help='random number seed')
parser.add_argument('--nphot', type=int, default=100, help='# photons')
parser.add_argument('--nxy_min', type=int, default=10,
                    help='minimum number of pixels along x or y axis')
parser.add_argument('--nxy_max', type=int, default=5000,
                    help='maximum number of pixels along x or y axis')
parser.add_argument('--npts', type=int, default=16, help='# points in nxy')
parser.add_argument('--bundles_per_pix', type=int, default=1,
                    help='# photon bundles per pixel')
parser.add_argument('--nrecalc_factor', type=float, default=1000,
                    help='nrecalc = nrecalc_factor*<# image pixels>')
args = parser.parse_args()

desc.imsim.read_config()
instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                            'tiny_instcat.txt')
commands, objects = desc.imsim.parsePhoSimInstanceFile(instcat_file)
obs_md = desc.imsim.phosim_obs_metadata(commands)

pixel_scale = 0.2

skymodel = desc.imsim.ESOSkyModel(obs_md, seed=args.seed, addNoise=False,
                                  addBackground=True, fast_background=False,
                                  bundles_per_pix=args.bundles_per_pix)
waves = skymodel.waves
angles = skymodel.angles

for nxy in np.logspace(np.log10(args.nxy_min),
                       np.log10(args.nxy_max), args.npts):
    image = galsim.Image(int(nxy), int(nxy), scale=pixel_scale)
    nrecalc = int(args.nrecalc_factor*np.prod(image.array.shape))
    sensor = galsim.SiliconSensor(rng=skymodel.randomNumbers, nrecalc=nrecalc)
    photon_array = skymodel.get_photon_array(image, args.nphot)
    waves.applyTo(photon_array)
    angles.applyTo(photon_array)

    t0 = time.clock()
    sensor.accumulate(photon_array, image)
    print(int(nxy)**2, time.clock() - t0)
    sys.stdout.flush()
