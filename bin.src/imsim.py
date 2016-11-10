#!/usr/bin/env python
"""
This is the imSim program, used to drive GalSim to simulate the LSST.  Written
for the DESC collaboration and LSST project.  This version of the program can
read phoSim instance files as is. It leverages the LSST Sims GalSim interface
code found in sims_GalSimInterface.
"""
from __future__ import absolute_import, print_function
import os
import argparse
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.coordUtils import chipNameFromRaDec
import desc.imsim


def main():
    """
    Drive GalSim to simulate the LSST.
    """
    # Setup a parser to take command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="The instance catalog")
    parser.add_argument('-n', '--numrows', default=None, type=int,
                        help="Read the first numrows of the file.")
    parser.add_argument('--outdir', type=str, default='fits',
                        help='Output directory for eimage file')
    parser.add_argument('--sensor', type=str, default=None,
                        help='Sensor to simulate, e.g., "R:2,2 S:1,1".' +
                        'If None, then simulate all sensors with sources on them')
    parser.add_argument('--config_file', type=str, default=None,
                        help="Config file. If None, the default config will be used.")
    parser.add_argument('--log_level', type=str,
                        choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Logging level. Default: "INFO"')
    arguments = parser.parse_args()

    config = desc.imsim.read_config(arguments.config_file)

    logger = desc.imsim.get_logger(arguments.log_level)

    # Get the number of rows to read from the instance file.  Use
    # default if not specified.
    numRows = arguments.numrows
    if numRows is not None:
        logger.info("Reading %i rows from the instance catalog %s.",
                    numRows, arguments.file)
    else:
        logger.info("Reading all rows from the instance catalog %s.",
                    arguments.file)

    # The PhoSim instance file contains both pointing commands and
    # objects.  The parser will split them and return a both phosim
    # command dictionary and a dataframe of objects.
    commands, phosim_objects = \
        desc.imsim.parsePhoSimInstanceFile(arguments.file, numRows)

    phosim_objects = \
        desc.imsim.validate_phosim_object_list(phosim_objects).accepted

    # Build the ObservationMetaData with values taken from the
    # PhoSim commands at the top of the instance file.
    obs_md = desc.imsim.phosim_obs_metadata(commands)

    camera = LsstSimMapper().camera

    # Sub-divide the source dataframe into stars and galaxies.
    if arguments.sensor is not None:
        # Trim the input catalog to a single chip.
        phosim_objects['chipName'] = \
            chipNameFromRaDec(phosim_objects['raICRS'].values,
                              phosim_objects['decICRS'].values,
                              camera=camera, obs_metadata=obs_md,
                              epoch=2000.0)

        starDataBase = \
            phosim_objects.query("galSimType=='pointSource' and chipName=='%s'"
                                 % arguments.sensor)
        galaxyDataBase = \
            phosim_objects.query("galSimType=='sersic' and chipName=='%s'"
                                 % arguments.sensor)
    else:
        starDataBase = \
            phosim_objects.query("galSimType=='pointSource'")
        galaxyDataBase = \
            phosim_objects.query("galSimType=='sersic'")

    # Simulate the objects in the Pandas Dataframes.

    # First simulate stars
    phoSimStarCatalog = desc.imsim.ImSimStars(starDataBase, obs_md)
    phoSimStarCatalog.photParams = desc.imsim.photometricParameters(commands)
    phoSimStarCatalog.camera = camera
    phoSimStarCatalog.get_fitsFiles()

    # Now galaxies
    phoSimGalaxyCatalog = desc.imsim.ImSimGalaxies(galaxyDataBase, obs_md)
    phoSimGalaxyCatalog.copyGalSimInterpreter(phoSimStarCatalog)
    phoSimGalaxyCatalog.get_fitsFiles()

    # Write out the fits files
    outdir = arguments.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    prefix = config['eimage_prefix']
    phoSimGalaxyCatalog.write_images(nameRoot=os.path.join(outdir, prefix) +
                                     str(commands['obshistid']))

if __name__ == "__main__":
    main()
