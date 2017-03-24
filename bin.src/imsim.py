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
from lsst.sims.GalSimInterface import SNRdocumentPSF
try:
    from lsst.sims.GalSimInterface import Kolmogorov_and_Gaussian_PSF
except:
    # in case we are running with an old version of lsst_sims
    pass
from desc.imsim.skyModel import ESOSkyModel
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
    parser.add_argument('--psf', type=str, default='DoubleGaussian',
                        choices=['DoubleGaussian', 'Kolmogorov'],
                        help="PSF model to use; either the double Gaussian "
                        "from LSE=40 (equation 30), or the Kolmogorov convolved "
                        "with a Gaussian proposed by David Kirkby at the "
                        "23 March 2017 SSims telecon")
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
                              parallax=phosim_objects['parallax'].values,
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

    # Add noise and sky background
    # The simple code using the default lsst-GalSim interface would be:
    #
    #    PhoSimStarCatalog.noise_and_background = ExampleCCDNoise(addNoise=True,
    #                                                             addBackground=True)
    #
    # But, we need a more realistic sky model and we need to pass more than
    # this basic info to use Peter Y's ESO sky model.
    # We must pass obs_metadata, chip information etc...
    phoSimStarCatalog.noise_and_background = ESOSkyModel(obs_metadata, addNoise=True,
                                                         addBackground=True)

    # Add a PSF.
    if arguments.psf.lower() == "doublegaussian":
        # This one is taken from equation 30 of
        # www.astro.washington.edu/users/ivezic/Astr511/LSST_SNRdoc.pdf .
        #
        # Set seeing from self.obs_metadata.
        phoSimStarCatalog.PSF = \
            SNRdocumentPSF(obs_md.OpsimMetaData['FWHMgeom'])
    elif arguments.psf.lower() == "kolmogorov":
        # This PSF was presented by David Kirkby at the 23 March 2017
        # Survey Simulations Working Group telecon
        #
        # https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=LSSTDESC&title=SSim+2017-03-23

        phoSimStarCatalog.PSF = \
            Kolmogorov_and_Gaussian_PSF(airmass=airmass,
                                        rawSeeing=obs_md.OpsimMetaData['rawSeeing'],
                                        band=obs_md.bandpass)
    else:
        raise RuntimeError("Do not know what to do with psf model: "
                           "%s" % arguments.psf)

    phoSimStarCatalog.camera = camera
    phoSimStarCatalog.get_fitsFiles()

    # Now galaxies
    phoSimGalaxyCatalog = desc.imsim.ImSimGalaxies(galaxyDataBase, obs_md)
    phoSimGalaxyCatalog.copyGalSimInterpreter(phoSimStarCatalog)
    phoSimGalaxyCatalog.PSF = phoSimStarCatalog.PSF
    phoSimGalaxyCatalog.noise_and_background = phoSimStarCatalog.noise_and_background
    phoSimGalaxyCatalog.get_fitsFiles()

    # Write out the fits files
    outdir = arguments.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    prefix = config['persistence']['eimage_prefix']
    phoSimGalaxyCatalog.write_images(nameRoot=os.path.join(outdir, prefix) +
                                     str(commands['obshistid']))

if __name__ == "__main__":
    main()
