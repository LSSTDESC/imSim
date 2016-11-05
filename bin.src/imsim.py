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

from lsst.sims.photUtils import LSSTdefaults
from lsst.sims.utils import ObservationMetaData
from lsst.sims.coordUtils import chipNameFromRaDec

from lsst.sims.GalSimInterface import GalSimStars
from lsst.sims.GalSimInterface import GalSimGalaxies

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
                        help='Sensor to simulate, e.g., "R:2,2 S:1,1". If None, then simulate all sensors with sources on them')
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

    # Now build the ObservationMetaData with values taken from the PhoSim
    # commands at the top of the instance file.

    # Access the relevant pointing commands from the dictionary.
    visitID = commands['obshistid']
    mjd = commands['mjd']
    rightAscension = commands['rightascension']
    declination = commands['declination']
    rotSkyPosition = commands['rotskypos']
    bandpass = commands['bandpass']
    # @todo: The seeing from the instance catalog is the seeing at
    # 500nm at zenith.  Do we need to do a band-specific calculation?
    seeing = commands['seeing']

    # We need to set the M5 limiting magnitudes etc.
    defaults = LSSTdefaults()

    camera = LsstSimMapper().camera
    obs = ObservationMetaData(pointingRA=rightAscension,
                              pointingDec=declination,
                              mjd=mjd, rotSkyPos=rotSkyPosition,
                              bandpassName=bandpass,
                              m5=defaults.m5(bandpass),
                              seeing=seeing)

    # Set the OpsimMetaData attribute with the obshistID info.
    obs.OpsimMetaData = {'obshistID': visitID}

    # Now further sub-divide the source dataframe into stars and galaxies.
    if arguments.sensor is not None:
        # Trim the input catalog to a single chip.
        raICRS = phosim_objects['raICRS'].values
        decICRS = phosim_objects['decICRS'].values
        phosim_objects['chipName'] = chipNameFromRaDec(raICRS, decICRS,
                                                       camera=camera,
                                                       obs_metadata=obs,
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

    # The GalSim[Stars,Galaxies] classes are subclassed via
    # desc.imsim.imSim_class_factory, with the new subclass
    # constructor taking a pandas DataFrame as a parameter instead of
    # a CatalogDBObject.  In that constructor, the PSF and CCD noise
    # model are set.  For simulation the LSST this should be the same
    # between all types of objects.

    phot_params = desc.imsim.photometricParameters(commands)

    # First simulate stars
    ImSimStars = desc.imsim.imSim_class_factory(GalSimStars)
    phoSimStarCatalog = ImSimStars(starDataBase, obs)
    phoSimStarCatalog.photParams = phot_params
    phoSimStarCatalog.camera = camera
    phoSimStarCatalog.get_fitsFiles()

    # Now galaxies
    ImSimGalaxies = desc.imsim.imSim_class_factory(GalSimGalaxies)
    phoSimGalaxyCatalog = ImSimGalaxies(galaxyDataBase, obs)
    phoSimGalaxyCatalog.copyGalSimInterpreter(phoSimStarCatalog)
    phoSimGalaxyCatalog.get_fitsFiles()

    # Write out the fits files
    outdir = arguments.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    prefix = config['eimage_prefix']
    phoSimGalaxyCatalog.write_images(nameRoot=os.path.join(outdir, prefix) +
                                     str(visitID))

if __name__ == "__main__":
    main()
