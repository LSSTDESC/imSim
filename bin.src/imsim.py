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
import numpy as np
from lsst.sims.coordUtils import chipNameFromRaDecLSST
from lsst.sims.catUtils.mixins import PhoSimAstrometryBase
from lsst.sims.GalSimInterface import SNRdocumentPSF
from lsst.sims.GalSimInterface import LSSTCameraWrapper
from lsst.sims.GalSimInterface import Kolmogorov_and_Gaussian_PSF
from desc.imsim.skyModel import ESOSkyModel
import desc.imsim

_POINT_SOURCE = 1
_SERSIC_2D = 2

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
    parser.add_argument('--psf', type=str, default='Kolmogorov',
                        choices=['DoubleGaussian', 'Kolmogorov'],
                        help="PSF model to use; either the double Gaussian "
                        "from LSE=40 (equation 30), or the Kolmogorov convolved "
                        "with a Gaussian proposed by David Kirkby at the "
                        "23 March 2017 SSims telecon")
    parser.add_argument('--checkpoint_file', type=str, default=None,
                        help='Checkpoint file name.')
    parser.add_argument('--nobj_checkpoint', type=int, default=1000,
                        help='# objects to process between checkpoints')
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
    commands = \
        desc.imsim.parsePhoSimInstanceFile(arguments.file, numRows)

    # Build the ObservationMetaData with values taken from the
    # PhoSim commands at the top of the instance file.
    obs_md = desc.imsim.phosim_obs_metadata(commands)

    camera = desc.imsim.get_obs_lsstSim_camera()

    num_objects = 0
    ct_rows = 0
    with open(arguments.file, 'r') as input_file:
        for line in input_file:
            ct_rows += 1
            params = line.strip().split()
            if params[0] == 'object':
                num_objects += 1
            if numRows is not None and ct_rows>=numRows:
                break

    # RA, Dec in the coordinate system expected by PhoSim
    ra_phosim = np.zeros(num_objects, dtype=float)
    dec_phosim = np.zeros(num_objects, dtype=float)

    sed_name = [None]*num_objects
    mag_norm = np.zeros(num_objects, dtype=float)
    gamma1 = np.zeros(num_objects, dtype=float)
    gamma2 = np.zeros(num_objects, dtype=float)
    kappa = np.zeros(num_objects, dtype=float)

    internal_av = np.zeros(num_objects, dtype=float)
    internal_rv = np.zeros(num_objects, dtype=float)
    galactic_av = np.zeros(num_objects, dtype=float)
    galactic_rv = np.zeros(num_objects, dtype=float)
    semi_major_arcsec = np.zeros(num_objects, dtype=float)
    semi_minor_arcsec = np.zeros(num_objects, dtype=float)
    position_angle_degrees = np.zeros(num_objects, dtype=float)
    sersic_index = np.zeros(num_objects, dtype=float)
    redshift = np.zeros(num_objects, dtype=float)


    unique_id = np.zeros(num_objects, dtype=int)
    object_type = np.zeros(num_objects, dtype=int)

    i_obj = 0
    with open(arguments.file, 'r') as input_file:
        for line in input_file:
            params = line.strip().split()
            if params[0] != 'object':
                continue
            if numRows is not None and i_obj>=num_objects:
                break
            unique_id[i_obj] = int(params[1])
            ra_phosim[i_obj] = float(params[2])
            dec_phosim[i_obj] = float(params[3])
            mag_norm[i_obj] = float(params[4])
            sed_name[i_obj] = params[5]
            redshift[i_obj] = float(params[6])
            gamma1[i_obj] = float(params[7])
            gamma2[i_obj] = float(params[8])
            kappa[i_obj] = float(params[9])
            if params[12].lower() == 'point':
                object_type[i_obj] = _POINT_SOURCE
                i_gal_dust_model = 14
                if params[13].lower() != 'none':
                    i_gal_dust_model = 16
                    internal_av[i_obj] = float(params[14])
                    internal_rv[i_obj] =float(params[15])
                if params[i_gal_dust_model].lower() != 'none':
                    galactic_av[i_obj] = float(params[i_gal_dust_model+1])
                    galactic_rv[i_obj] = float(params[i_gal_dust_model+2])
            elif params[12].lower() == 'sersic2d':
                object_type[i_obj] = _SERSIC_2D
                semi_major_arcsec[i_obj] = float(params[13])
                semi_minor_arcsec[i_obj] = float(params[14])
                position_angle_degrees[i_obj] = float(params[15])
                sersic_index[i_obj] = float(params[16])
                i_gal_dust_model = 18
                if params[17].lower() != 'none':
                    i_gal_dust_model = 19
                    internal_av[i_obj] = float(params[17])
                    internal_rv[i_obj] = float(params[18])
                if params[i_gal_dust_model].lower() != 'none':
                    galactic_av[i_obj] = float(params[i_gal_dust_model+1])
                    galactic_rv[i_obj] =float(params[i_gal_dust_model+2])

            else:
                raise RuntimeError("Do not know how to handle "
                                   "object type: %s" % params[12])

    ra_obs, dec_obs = PhoSimAstrometryBase.icrsFromPhoSim(ra_phosim, dec_phosim,
                                                          obs_md)

    chip_name = chipNameFromRaDecLSST(ra_obs, dec_obs, obs_metadata=obs_md)

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
    # Restrict to a single sensor if requested.
    if arguments.sensor is not None:
        phoSimStarCatalog.allowed_chips = [arguments.sensor]
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
    phoSimStarCatalog.noise_and_background \
        = ESOSkyModel(obs_md, addNoise=True, addBackground=True)

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

        # equation 3 of Krisciunas and Schaefer 1991
        airmass = 1.0/np.sqrt(1.0-0.96*(np.sin(0.5*np.pi-obs_md.OpsimMetaData['altitude']))**2)

        phoSimStarCatalog.PSF = \
            Kolmogorov_and_Gaussian_PSF(airmass=airmass,
                                        rawSeeing=obs_md.OpsimMetaData['rawSeeing'],
                                        band=obs_md.bandpass)
    else:
        raise RuntimeError("Do not know what to do with psf model: "
                           "%s" % arguments.psf)

    phoSimStarCatalog.camera = camera
    phoSimStarCatalog.camera_wrapper = LSSTCameraWrapper()
    phoSimStarCatalog.get_fitsFiles(arguments.checkpoint_file,
                                    arguments.nobj_checkpoint)

    # Now galaxies
    phoSimGalaxyCatalog = desc.imsim.ImSimGalaxies(galaxyDataBase, obs_md)
    phoSimGalaxyCatalog.copyGalSimInterpreter(phoSimStarCatalog)
    phoSimGalaxyCatalog.PSF = phoSimStarCatalog.PSF
    phoSimGalaxyCatalog.noise_and_background = phoSimStarCatalog.noise_and_background
    phoSimGalaxyCatalog.get_fitsFiles(arguments.checkpoint_file,
                                      arguments.nobj_checkpoint)

    # Add cosmic rays to the eimages.
    phoSimGalaxyCatalog.add_cosmic_rays()

    # Write out the fits files
    outdir = arguments.outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    prefix = config['persistence']['eimage_prefix']
    phoSimGalaxyCatalog.write_images(nameRoot=os.path.join(outdir, prefix) +
                                     str(commands['obshistid']))

if __name__ == "__main__":
    main()
