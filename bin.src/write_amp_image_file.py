#!/usr/bin/env python
"""
Process an eimage file through a simulated electronics readout chain
and write out a FITS file conforming to the format of CCS-produced
outputs.
"""
from __future__ import absolute_import, print_function
import os
import argparse
import desc.imsim

parser = argparse.ArgumentParser()
parser.add_argument("eimage_file", help="eimage file to process")
parser.add_argument("--log_level", type=str,
                    choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'],
                    default='INFO', help='Logging level. Default: "INFO"')
parser.add_argument("--seg_file", default=None, type=str,
                    help="text file describing the layout of the focalplane")
parser.add_argument("--opsim_db", default=None, type=str,
                    help="OpSim db file as alternative source of pointing info")
args = parser.parse_args()

logger = desc.imsim.get_logger(args.log_level)

image_source = desc.imsim.ImageSource.create_from_eimage(args.eimage_file,
                                                         seg_file=args.seg_file,
                                                         opsim_db=args.opsim_db,
                                                         logger=logger)
outfile = os.path.basename(args.eimage_file).replace('lsst_e', 'lsst_a')
image_source.write_fits_file(outfile)
