#!/usr/bin/env python
"""
This is the imSim program, used to drive GalSim to simulate the LSST.  Written
for the DESC collaboration and LSST project.  This version of the program can
read phoSim instance files as is. It leverages the LSST Sims GalSim interface
code found in sims_GalSimInterface.
"""
from __future__ import absolute_import, print_function
import os
import sys
import logging
import argparse

from lsst.obs.lsstSim import LsstSimMapper

from lsst.sims.photUtils import LSSTdefaults
from lsst.sims.utils import ObservationMetaData
from lsst.sims.coordUtils import chipNameFromRaDec

from lsst.sims.GalSimInterface.galSimCatalogs import GalSimBase
from lsst.sims.GalSimInterface import GalSimStars
from lsst.sims.GalSimInterface import GalSimGalaxies

import desc.imsim
from desc.imsim.monkeyPatchedGalSimBase import \
    phoSimCalculateGalSimSeds, phoSimInitializer, get_phoSimInstanceCatalog


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
    parser.add_argument('--log_level', type=str,
                        choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Logging level. Default: "INFO"')
    arguments = parser.parse_args()

    # Setup logging output.
    logging.basicConfig(format="%(message)s", level=logging.INFO,
                        stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(eval('logging.' + arguments.log_level))

    # Monkey Patch the GalSimBase class and replace the routines that
    # gets the objects and builds the catalog with my version that gets the
    # information from a phoSim instance catalog file.  This should be temporary
    # and is a hack.
    GalSimBase.__init__ = \
        lambda *args, **kwds: phoSimInitializer(*args, logger=logger, **kwds)
    GalSimBase._calculateGalSimSeds = phoSimCalculateGalSimSeds
    GalSimBase.get_fitsFiles = get_phoSimInstanceCatalog

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
    commandDictionary, phoSimObjectList = \
        desc.imsim.parsePhoSimInstanceFile(arguments.file, numRows)[:2]

    # Now build the ObservationMetaData with values taken from the PhoSim
    # commands at the top of the instance file.

    # Access the relevant pointing commands from the dictionary.
    visitIDString = str(int(commandDictionary['obshistid'][0]))
    mjd = commandDictionary['mjd'][0]
    rightAscension = commandDictionary['rightascension'][0]
    declination = commandDictionary['declination'][0]
    rotSkyPosition = commandDictionary['rotskypos'][0]
    bandpass = 'ugrizy'[int(commandDictionary['filter'][0])]
    seeing = commandDictionary['seeing'][0]

    # We need to set the M5 limiting magnitudes etc.
    defaults = LSSTdefaults()

    camera = LsstSimMapper().camera
    obs = ObservationMetaData(pointingRA=rightAscension,
                              pointingDec=declination,
                              mjd=mjd, rotSkyPos=rotSkyPosition,
                              bandpassName=[bandpass],
                              m5=[defaults.m5(bandpass)],
                              seeing=[seeing])

    # Trim the input catalog to a single chip.
    raICRS = phoSimObjectList['ra'].values
    decICRS = phoSimObjectList['dec'].values
    phoSimObjectList['chipName'] = chipNameFromRaDec(raICRS, decICRS,
                                                     camera=camera,
                                                     obs_metadata=obs,
                                                     epoch=2000.0)
    # Now further sub-divide the source dataframe into stars and galaxies.
    if arguments.sensor is not None:
        starDataBase = \
            phoSimObjectList.query("galSimType == 'pointSource' and magNorm < 50 and chipName=='%s'" % arguments.sensor)
        galaxyDataBase = \
            phoSimObjectList.query("galSimType == 'sersic' and magNorm < 50 and chipName=='%s'" % arguments.sensor)
    else:
        starDataBase = \
            phoSimObjectList.query("galSimType == 'pointSource' and magNorm < 50")
        galaxyDataBase = \
            phoSimObjectList.query("galSimType == 'sersic' and magNorm < 50")

    # Simulate the objects in the Pandas Dataframes. I monkey patched the
    # abstract base class GalSimBase to take my dataFrame instead of using the
    # internal databases. This is a hack until we rethink the class design.
    #
    # Additionally, I have set the PSF and noise etc globally for all of the
    # derived classes in the __init__.  For simulation the LSST this should be
    # the same between all types of objects.

    # First simulate stars
    phoSimStarCatalog = GalSimStars(starDataBase, obs)
    phoSimStarCatalog.camera = camera
    phoSimStarCatalog.get_fitsFiles()

    # Now galaxies
    phoSimGalaxyCatalog = GalSimGalaxies(galaxyDataBase, obs)
    phoSimGalaxyCatalog.copyGalSimInterpreter(phoSimStarCatalog)
    phoSimGalaxyCatalog.get_fitsFiles()

    # Write out the fits files
    outdir = arguments.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    phoSimGalaxyCatalog.write_images(nameRoot=os.path.join(outdir, 'e-image_') +
                                     visitIDString)

if __name__ == "__main__":
    main()
