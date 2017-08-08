#!/usr/bin/env python
"""
Process an eimage file through a simulated electronics readout chain
and write out a FITS file conforming to the format of CCS-produced
outputs.
"""
from __future__ import absolute_import, print_function
import os
import argparse
import astropy.io.fits as fits
import desc.imsim

parser = argparse.ArgumentParser()
parser.add_argument("eimage_file", help="eimage file to process")
parser.add_argument("--seg_file", default=None, type=str,
                    help="text file describing the layout of the focalplane")
args = parser.parse_args()

image_source = desc.imsim.make_ImageSource(args.eimage_file,
                                           seg_file=args.seg_file)
outfile = os.path.basename(args.eimage_file).replace('lsst_e', 'lsst_a')
image_source.write_fits_file(outfile)

