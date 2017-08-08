'''
Classes to represent realistic sky models.
Note that this extends the default classes located in
sims_GalSimInterface/python/lsst/sims/GalSimInterface/galSimNoiseAndBackground.py
'''

from __future__ import absolute_import, division

import numpy as np
import astropy.units as u

import lsst.sims.skybrightness as skybrightness

import galsim
from lsst.sims.GalSimInterface.galSimNoiseAndBackground import NoiseAndBackgroundBase

from .imSim import get_config

__all__ = ['skyCountsPerSec', 'ESOSkyModel', 'get_skyModel_params']


# Code snippet from D. Kirkby.  Note the use of astropy units.
def skyCountsPerSec(surface_brightness=21, filter_band='r',
                    effective_area=32.4*u.m**2, pixel_size=0.2*u.arcsec):
    pars = get_skyModel_params()
    # Lookup the zero point corresponding to 24 mag/arcsec**2
    s0 = pars[filter_band] * u.electron / u.s / u.m ** 2

    # Calculate the rate in detected electrons / second
    dB = (surface_brightness - pars['B0']) * u.mag(1 / u.arcsec ** 2)
    return s0 * dB.to(1 / u.arcsec ** 2) * pixel_size ** 2 * effective_area


# Here we are defining our own class derived from NoiseAndBackgroundBase for
# use instead of ExampleCCDNoise
class ESOSkyModel(NoiseAndBackgroundBase):
    """
    This class wraps the GalSim class CCDNoise.  This derived class returns
    a sky model based on the ESO model as implemented in
    """

    def __init__(self, obs_metadata, seed=None, addNoise=True, addBackground=True):
        """
        @param [in] addNoise is a boolean telling the wrapper whether or not
        to add noise to the image

        @param [in] addBackground is a boolean telling the wrapper whether
        or not to add the skybackground to the image

        @param [in] seed is an (optional) int that will seed the
        random number generator used by the noise model. Defaults to None,
        which causes GalSim to generate the seed from the system.
        """

        self.obs_metadata = obs_metadata

        self.addNoise = addNoise
        self.addBackground = addBackground

        if seed is None:
            self.randomNumbers = galsim.UniformDeviate()
        else:
            self.randomNumbers = galsim.UniformDeviate(seed)

    def addNoiseAndBackground(self, image, bandpass=None, m5=None,
                              FWHMeff=None,
                              photParams=None):
        """
        This method actually adds the sky background and noise to an image.

        Note: default parameters are defined in

        sims_photUtils/python/lsst/sims/photUtils/photometricDefaults.py

        @param [in] image is the GalSim image object to which the background
        and noise are being added.

        @param [in] bandpass is a CatSim bandpass object (not a GalSim bandpass
        object) characterizing the filter through which the image is being taken.

        @param [in] FWHMeff is the FWHMeff in arcseconds

        @param [in] photParams is an instantiation of the
        PhotometricParameters class that carries details about the
        photometric response of the telescope.  Defaults to None.

        @param [out] the input image with the background and noise model added to it.
        """

        # calculate the sky background to be added to each pixel
        skyModel = skybrightness.SkyModel(mags=True)
        ra = np.array([self.obs_metadata.pointingRA])
        dec = np.array([self.obs_metadata.pointingDec])
        mjd = self.obs_metadata.mjd.TAI
        skyModel.setRaDecMjd(ra, dec, mjd, degrees=True)

        bandPassName = self.obs_metadata.bandpass
        skyMagnitude = skyModel.returnMags()[bandPassName]

        # Since we are only producing one eimage, account for cases
        # where nsnap > 1 with an effective exposure time for the
        # visit as a whole.  TODO: Undo this change when we write
        # separate images per exposure.
        exposureTime = photParams.nexp*photParams.exptime*u.s

        skyCounts = skyCountsPerSec(surface_brightness=skyMagnitude,
                                    filter_band=bandPassName)*exposureTime

        # print "Magnitude:", skyMagnitude
        # print "Brightness:", skyMagnitude, skyCounts

        image = image.copy()

        if self.addBackground:
            image += skyCounts

            # if we are adding the skyCounts to the image,there is no need # to pass
            # a skyLevel parameter to the noise model.  skyLevel is # just used to
            # calculate the level of Poisson noise.  If the # sky background is
            # included in the image, the Poisson noise # will be calculated from the
            # actual image brightness.
            skyLevel = 0.0

        else:
            skyLevel = skyCounts*photParams.gain

        if self.addNoise:
            noiseModel = self.getNoiseModel(skyLevel=skyLevel, photParams=photParams)
            image.addNoise(noiseModel)

        return image

    def getNoiseModel(self, skyLevel=0.0, photParams=None):

        """
        This method returns the noise model implemented for this wrapper
        class.

        This is currently the same as implemented in ExampleCCDNoise.  This
        routine can both Poisson fluctuate the background and add read noise.
        We turn off the read noise by adjusting the parameters in the photParams.
        """

        return galsim.CCDNoise(self.randomNumbers, sky_level=skyLevel,
                               gain=photParams.gain, read_noise=photParams.readnoise)


def get_skyModel_params():
    """
    Get the zero points and reference magnitude for the sky model.

    Returns
    -------
    dict : the skyModel zero points and reference magnitudes.
    """
    config = get_config()
    return config['skyModel_params']
