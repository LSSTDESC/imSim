"""
This file defines the model classes that wrap noise models from
galsim into the CatSim interface
"""

from builtins import object
import numpy
import galsim
from lsst.sims.photUtils import calcSkyCountsPerPixelForM5, PhotometricParameters, \
                                LSSTdefaults

__all__ = ["ExampleCCDNoise"]

class NoiseAndBackgroundBase(object):
    """
    This is the base class for all wrappers of sky background
    and noise in the GalSim interface.  Daughters of this class
    are meant to be included as the noise_and_background
    class member variable of GalSim InstanceCatalog classes.
    To implement a new noise model, users should write a new class
    that inherits from this one.  That new class should only define
    a method getNoiseModel() that takes as arguments skyLevel, and an
    instantiation of the PhotometricParameters class defined in sims_photUtils
    (this will carry gain and readnoise information).  See the docstring
    for getNoiseModel() for further details.
    """

    def __init__(self, seed=None, addNoise=True, addBackground=True):
        """
        @param [in] addNoise is a boolean telling the wrapper whether or not
        to add noise to the image

        @param [in] addBackground is a boolean telling the wrapper whether
        or not to add the skybackground to the image

        @param [in] seed is an (optional) int that will seed the
        random number generator used by the noise model. Defaults to None,
        which causes GalSim to generate the seed from the system.
        """

        self.addNoise = addNoise
        self.addBackground = addBackground

        if seed is None:
            self.randomNumbers = galsim.UniformDeviate()
        else:
            self.randomNumbers = galsim.UniformDeviate(seed)


    def getNoiseModel(self, skyLevel=0.0, photParams=None):
        """
        This method returns the noise model implemented for this wrapper
        class.

        @param [in] skyLevel is the number of electrons per pixel due
        to the sky background.  However, this value should only be non-zero
        if the sky background has been subtracted from the image.  The
        purpose of this parameter is to provide an extra background value
        when calculating the level of Poisson noise in each pixel.  If the
        sky background is already present in the image, then the noise model
        will just set the noise level based on the intensity in each pixel
        and there is no need to add an additional skyLevel.  If the sky
        background is still included in the image, set skyLevel equal to zero.

        @param [in] photParams is an instantiation of the
        PhotometricParameters class that carries details about the
        photometric response of the telescope.  Defaults to None.

        @param [out] returns an instantiation of a GalSim noise class, as
        specified by the particular wrapper class to which this method belongs.
        """

        raise NotImplementedError("There is no noise model for NoiseAndBackgroundBase")


    def addNoiseAndBackground(self, image, bandpass=None, m5=None,
                              FWHMeff=None,
                              photParams=None,
                              detector=None):
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

        @param [in] detector is the GalSimDetector corresponding to the image.
        Defaults to None.

        @param [out] the input image with the background and noise model added to it.

        """


        #calculate the sky background to be added to each pixel
        skyCounts = calcSkyCountsPerPixelForM5(m5, bandpass, FWHMeff=FWHMeff, photParams=photParams)

        image = image.copy()

        if self.addBackground:
            image += skyCounts
            skyLevel = 0.0 #if we are adding the skyCounts to the image,there is no need
                           #to pass a skyLevel parameter to the noise model.  skyLevel is
                           #just used to calculate the level of Poisson noise.  If the
                           #sky background is included in the image, the Poisson noise
                           #will be calculated from the actual image brightness.
        else:
            skyLevel = skyCounts*photParams.gain

        if self.addNoise:
            noiseModel = self.getNoiseModel(skyLevel=skyLevel, photParams=photParams)
            image.addNoise(noiseModel)

        return image


class ExampleCCDNoise(NoiseAndBackgroundBase):
    """
    This class wraps the GalSim class CCDNoise.  It is meant to be assigned as
    the self.noise_and_background member variable in a GalSim InstanceCatalog.
    To instantiatea different noise model, write a class like this one that
    defines a method getNoiseModel() which accepts as its arguments skyLevel,
    readNoise, and gain and returns an instantiation of a GalSim noise model
    """

    def getNoiseModel(self, skyLevel=0.0, photParams=None):

        """
        This method returns the noise model implemented for this wrapper
        class.

        @param [in] skyLevel is the number of electrons per pixel due
        to the sky background.  However, this value should only be non-zero
        if the sky background has been subtracted from the image.  The
        purpose of this parameter is to provide an extra background value
        when calculating the level of Poisson noise in each pixel.  If the
        sky background is already present in the image, then the noise model
        will just set the noise level based on the intensity in each pixel
        and there is no need to add an additional skyLevel.  If the sky
        background is still included in the image, set skyLevel equal to zero.

        @param [in] photParams is an instantiation of the
        PhotometricParameters class that carries details about the
        photometric response of the telescope.  Defaults to None.

        @param [out] returns an instantiation of the GalSim CCDNoise class
        """

        return galsim.CCDNoise(self.randomNumbers, sky_level=skyLevel,
                               gain=photParams.gain, read_noise=photParams.readnoise)


