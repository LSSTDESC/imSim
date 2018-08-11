#!/usr/bin/env python
"""
Script to create imSim flats.  This includes electorstatic sensor effects
such as treerings and brighter-fatter.
"""
import argparse
import galsim
import lsst.afw.cameraGeom as cameraGeom
from lsst.sims.GalSimInterface import LSSTCameraWrapper, make_galsim_detector
import desc.imsim

parser = argparse.ArgumentParser("imSim flat production script.")
parser.add_argument('instcat', type=str, help="instance catalog file")
parser.add_argument('--sensors', type=str, default=None,
                    help="sensors to simulate")
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--counts_total', type=int, default=8e4,
                    help="total # electrons/pixel in flat")
parser.add_argument('--counts_per_iter', type=int, default=4e3,
                    help="# electrons/pixel per iteration")
parser.add_argument('--log_level', type=str,
                    choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'],
                    default='INFO', help='Logging level. Default: INFO')

args = parser.parse_args()

obs_md, phot_params, _ = desc.imsim.parsePhoSimInstanceFile(args.instcat,
                                                            numRows=30)
sensor_list = args.sensors.split('^') if args.sensors is not None \
    else args.sensors
rng = galsim.UniformDeviate(args.seed)
niter = int(args.counts_total/args.counter_per_iter + 0.5)
counts_per_iter = args.counts_total/niter
logger = desc.imsim.get_logger(args.log_level, name='make_flats')
config = desc.imsim.read_config()

visit = obs_md.OpsimMetaData['obshistID']

camera_wrapper = LSSTCameraWrapper()

for det in camera_wrapper.camera:
    det_name = det.getName()
    logger.info("processing %s", det_name)
    if (det.getType() != cameraGeom.SCIENCE or
            (args.sensors is not None and det_name not in sensor_list)):
        continue
    gs_det = make_galsim_detector(camera_wrapper, det_name, phot_params, obs_md)
    desc.imsim.add_treering_info([gs_det])
    my_flat = desc.imsim.make_flat(gs_det, counts_per_iter, niter, rng)
    ccd_id = "R{}_S{}".format(det_name[2:5:2], det_name[8:11:2])
    my_flat.write('flat_{}_{}_{}.fits'.format(visit, ccd_id, obs_md.bandpass))
