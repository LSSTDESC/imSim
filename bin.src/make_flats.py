#!/usr/bin/env python
"""
Script to create imSim flats.  This includes electrostatic sensor effects
such as treerings and brighter-fatter.
"""
import argparse
import galsim
import lsst.afw.cameraGeom as cameraGeom
from lsst.sims.GalSimInterface import LSSTCameraWrapper, make_galsim_detector
import desc.imsim

parser = argparse.ArgumentParser(description="imSim flat production script.")
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
parser.add_argument('--wcs_file', type=str, default=None,
                    help='FITS file to use for the WCS (as galsim.FitsWCS)')

args = parser.parse_args()

config = desc.imsim.read_config()

obs_md, phot_params, _ = desc.imsim.parsePhoSimInstanceFile(args.instcat,
                                                            numRows=30)
sensor_list = args.sensors.split('^') if args.sensors is not None \
    else args.sensors
rng = galsim.UniformDeviate(args.seed)
niter = int(args.counts_total/args.counts_per_iter + 0.5)
counts_per_iter = args.counts_total/niter
logger = desc.imsim.get_logger(args.log_level, name='make_flats')

visit = obs_md.OpsimMetaData['obshistID']

camera_wrapper = LSSTCameraWrapper()

wcs = galsim.FitsWCS(args.wcs_file) if args.wcs_file is not None else None

for det in camera_wrapper.camera:
    det_name = det.getName()
    if (det.getType() != cameraGeom.SCIENCE or
            (args.sensors is not None and det_name not in sensor_list)):
        continue
    logger.info("processing %s", det_name)
    gs_det = make_galsim_detector(camera_wrapper, det_name, phot_params, obs_md)
    desc.imsim.add_treering_info([gs_det])
    my_flat = desc.imsim.make_flat(gs_det, counts_per_iter, niter, rng,
                                   logger=logger, wcs=wcs)
    ccd_id = "R{}_S{}".format(det_name[2:5:2], det_name[8:11:2])
    my_flat.write('flat_{}_{}_{}.fits'.format(visit, ccd_id, obs_md.bandpass))
