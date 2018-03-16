'''
Classes to represent realistic sky models.
Note that this extends the default classes located in
sims_GalSimInterface/python/lsst/sims/GalSimInterface/galSimNoiseAndBackground.py
'''

from __future__ import absolute_import, division

import numpy as np
import astropy.units as u

import galsim

from lsst.obs.lsstSim import LsstSimMapper
import lsst.sims.skybrightness as skybrightness
import lsst.sims.coordUtils
from lsst.sims.GalSimInterface.galSimNoiseAndBackground import NoiseAndBackgroundBase
from lsst.sims.photUtils import BandpassDict
from lsst.sims.photUtils import Sed

from .imSim import get_config

__all__ = ['SkyCountsPerSec', 'ESOSkyModel', 'get_skyModel_params']


class SkyCountsPerSec(object):
    """
    This is a class that is used to calculate the number of sky counts per
    second.
    """
    def __init__(self, skyModel, photParams, bandpassdic):
        """

        @param [in] skyModel is an instantation of the skybrightness.SkyModel
        class that carries information about the sky for the current conditions.

        @photParams [in] is an instantiation of the
        PhotometricParameters class that carries details about the
        photometric response of the telescope.

        @bandpassdic [in] is an instantation of the Bandpassdict class that
        holds the bandpasses.
        """

        self.skyModel = skyModel
        self.photParams = photParams
        self.bandpassdic = bandpassdic

    def __call__(self, filter_name='u', magNorm=None):
        """
        This method calls the SkyCountsPerSec object and calculates the sky
        counts.

        @param [in] filter_name is a string that indicates the name of the filter
        for which to make the calculation.

        @param [in] magNorm is an option to calculate the sky counts for a given
        magnitude.  When calculating the counts from just the information in skyModel
        this should be set as MagNorm=None.
        """

        bandpass = self.bandpassdic[filter_name]
        wave, spec = self.skyModel.returnWaveSpec()
        skymodel_Sed = Sed(wavelen=wave, flambda=spec[0, :])
        if magNorm:
            skymodel_fluxNorm = skymodel_Sed.calcFluxNorm(magNorm, bandpass)
            skymodel_Sed.multiplyFluxNorm(skymodel_fluxNorm)
        sky_counts = skymodel_Sed.calcADU(bandpass=bandpass, photParams=self.photParams)
        expTime = self.photParams.nexp * self.photParams.exptime * u.s
        sky_counts_persec = sky_counts * 0.2**2 / expTime

        return sky_counts_persec


def get_chip_names_centers():
    """Get the names and pixel centers for each sensor on the LSST camera.

    Returns
    -------
    name_list: list
        List specifying raft and labels for each sensor on the camera. For example,
        one sensor may be labelled: 'R:4,2 S:1,0'.
    center_x: np.array length name_list
        x-coordinates of pixel centers for each chip
    center_y: np.array length name_list
        y-coordinates of pixel centers for each chip
    """

    name_list = []
    x_pix_list = []
    y_pix_list = []
    n_chips = 0

    lsst_camera = LsstSimMapper().camera

    # Get chip names
    for chip in lsst_camera:
        chip_name = chip.getName()
        n_chips += 1

        corner_list = lsst.sims.coordUtils.getCornerPixels(chip_name, lsst_camera)

        for corner in corner_list:
            x_pix_list.append(corner[0])
            y_pix_list.append(corner[1])
            name_list.append(chip_name)

    # Get chip centers
    center_x = np.empty(n_chips, dtype=float)
    center_y = np.empty(n_chips, dtype=float)

    for ix_ct in range(n_chips):
        ix = ix_ct*4
        chip_name = name_list[ix]

        xx = 0.25*(x_pix_list[ix] + x_pix_list[ix+1] +
                   x_pix_list[ix+2] + x_pix_list[ix+3])

        yy = 0.25*(y_pix_list[ix] + y_pix_list[ix+1] +
                   y_pix_list[ix+2] + y_pix_list[ix+3])

        center_x[ix_ct] = xx
        center_y[ix_ct] = yy

    return(name_list, center_x, center_y)


def get_chip_center(chip_name, camera):
    """
    Get center of the chip in focal plane pixel coordinates

    Parameters
    ----------
    chip_name: str
        The name of the chip, e.g., "R:2,2 S:1,1".
    camera: lsst.afw.cameraGeom.camera.Camera
        The camera object, e.g., LsstSimCameraMapper().camera.

    Returns
    -------
    (float, float): focal plane pixel coordinates of chip center.
    """
    corner_list = lsst.sims.coordUtils.getCornerPixels(chip_name, camera)

    x_pix_list = []
    y_pix_list = []

    for corner in corner_list:
        x_pix_list.append(corner[0])
        y_pix_list.append(corner[1])

    center_x = 0.25*(x_pix_list[0] + x_pix_list[1] +
                     x_pix_list[2] + x_pix_list[3])

    center_y = 0.25*(y_pix_list[0] + y_pix_list[1] +
                     y_pix_list[2] + y_pix_list[3])

    return center_x, center_y


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
                              FWHMeff=None, photParams=None, chipName=None):
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

        @param [in] chipName is the name of the sensor being considered,
        e.g., "R:2,2 S:1,1".

        @param [out] the input image with the background and noise model added to it.
        """
        camera = LsstSimMapper().camera
        center_x, center_y = get_chip_center(chipName, camera)

        # calculate the sky background to be added to each pixel
        skyModel = skybrightness.SkyModel(mags=False)
        ra, dec = lsst.sims.coordUtils.raDecFromPixelCoords(
            xPix=center_x, yPix=center_y, chipName=chipName, camera=camera,
            obs_metadata=self.obs_metadata, epoch=2000.0,
            includeDistortion=True)
        mjd = self.obs_metadata.mjd.TAI
        skyModel.setRaDecMjd(ra, dec, mjd, degrees=True)

        bandPassName = self.obs_metadata.bandpass
        bandPassdic = BandpassDict.loadTotalBandpassesFromFiles(['u', 'g', 'r', 'i', 'z', 'y'])

        # Since we are only producing one eimage, account for cases
        # where nsnap > 1 with an effective exposure time for the
        # visit as a whole.  TODO: Undo this change when we write
        # separate images per exposure.
        exposureTime = photParams.nexp * photParams.exptime

        skycounts_persec = SkyCountsPerSec(skyModel, photParams, bandPassdic)
        skyCounts = skycounts_persec(bandPassName) * exposureTime * u.s

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
