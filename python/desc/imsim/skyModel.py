'''
Classes to represent realistic sky models.
Note that this extends the default classes located in
sims_GalSimInterface/python/lsst/sims/GalSimInterface/galSimNoiseAndBackground.py
'''

from __future__ import absolute_import, division
from collections import namedtuple
import numpy as np
import astropy.units as u
import galsim
from lsst.sims.photUtils import BandpassDict
import lsst.sims.skybrightness as skybrightness
from lsst.sims.GalSimInterface.galSimNoiseAndBackground import NoiseAndBackgroundBase
from lsst.sims.photUtils import Sed
from .imSim import get_config, get_logger

__all__ = ['make_sky_model', 'SkyCountsPerSec', 'ESOSkyModel',
           'ESOSiliconSkyModel', 'FastSiliconSkyModel']

def make_sky_model(obs_metadata, photParams, seed=None, bandpassDict=None,
                   addNoise=True, addBackground=True, bundles_per_pix=20,
                   apply_sensor_model=False, logger=None, fast_silicon=True):
    "Function to provide ESOSkyModel object."
    if apply_sensor_model:
        if fast_silicon:
            return FastSiliconSkyModel(obs_metadata, photParams,
                                       seed=seed,
                                       bandpassDict=bandpassDict,
                                       logger=logger)
        else:
            return ESOSiliconSkyModel(obs_metadata, photParams, seed=seed,
                                      bandpassDict=bandpassDict,
                                      bundles_per_pix=bundles_per_pix,
                                      logger=logger)
    return ESOSkyModel(obs_metadata, photParams, seed=seed,
                       bandpassDict=bandpassDict, addNoise=addNoise,
                       addBackground=addBackground, logger=logger)


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


# Here we are defining our own class derived from NoiseAndBackgroundBase for
# use instead of ExampleCCDNoise
class ESOSkyModel(NoiseAndBackgroundBase):
    """
    This class wraps the GalSim class CCDNoise.  This derived class returns
    a sky model based on the ESO model as implemented in
    """
    def __init__(self, obs_metadata, photParams, seed=None, bandpassDict=None,
                 addNoise=True, addBackground=True, logger=None):
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
        self.photParams = photParams

        if bandpassDict is None:
            self.bandpassDict = BandpassDict.loadBandpassesFromFiles()[0]

        # Computing the skybrightness.SkyModel object is expensive, so
        # do it only once in the constructor.
        self.skyModel = skybrightness.SkyModel(mags=False)
        ra = np.array([self.obs_metadata.pointingRA])
        dec = np.array([self.obs_metadata.pointingDec])
        mjd = self.obs_metadata.mjd.TAI
        self.skyModel.setRaDecMjd(ra, dec, mjd, degrees=True)

        self.addNoise = addNoise
        self.addBackground = addBackground
        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_logger('INFO')

        if seed is None:
            self.randomNumbers = galsim.UniformDeviate()
        else:
            self.randomNumbers = galsim.UniformDeviate(seed)

    def addNoiseAndBackground(self, image, bandpass='u', m5=None,
                              FWHMeff=None,
                              photParams=None, chipName=None):
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

        skyCounts = self.sky_counts()

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
            skyLevel = skyCounts * photParams.gain

        if self.addNoise:
            noiseModel = self.getNoiseModel(
                skyLevel=skyLevel, photParams=photParams)
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

        return galsim.CCDNoise(
            self.randomNumbers,
            sky_level=skyLevel,
            gain=photParams.gain,
            read_noise=photParams.readnoise)

    def sky_counts(self):
        """
        Returns
        -------
        float: sky background counts per pixel.
        """
        bandPassName = self.obs_metadata.bandpass

        # Since we are only producing one eimage, account for cases
        # where nsnap > 1 with an effective exposure time for the
        # visit as a whole.  TODO: Undo this change when we write
        # separate images per exposure.
        exposureTime = self.photParams.nexp*self.photParams.exptime
        skycounts_persec = SkyCountsPerSec(self.skyModel, self.photParams, self.bandpassDict)
        return float(skycounts_persec(bandPassName)*exposureTime*u.s)

class FastSiliconSkyModel(ESOSkyModel):
    """
    This version produces a sky background image by scaling the counts
    in each pixel by the areas of distorted pixel geometries in the
    galsim.Silicon model to account for electrostatic effects such as
    tree rings.
    """
    def __init__(self, obs_metadata, photParams, seed=None,
                 bandpassDict=None, logger=None):
        super(FastSiliconSkyModel, self).__init__(obs_metadata, photParams,
                                                  seed=seed,
                                                  bandpassDict=bandpassDict,
                                                  logger=logger)
    def addNoiseAndBackground(self, image, bandpass=None, m5=None,
                              FWHMeff=None, photParams=None, detector=None,
                              chipName=None):
        """
        This implementation is based on GalSim/devel/lsst/treering_skybg2.py.
        """
        if detector is None:
            raise RuntimeError("A GalSimDetector object must be provided.")

        # Create an image with the detector wcs onto which to draw the
        # sky background.
        nrow, ncol = image.array.shape
        nrecalc = 1e300 # disable pixel boundary updating.
        sensor = galsim.SiliconSensor(rng=self.randomNumbers,
                                      nrecalc=nrecalc,
                                      treering_func=detector.tree_rings.func,
                                      treering_center=detector.tree_rings.center)
        # Loop over amplifiers to save memory when storing the 36
        # pixel vertices per pixel.
        nx, ny = 2, 8
        dx = ncol//nx
        dy = nrow//ny

        for i in range(nx):
            xmin = i*dx + 1
            xmax = (i + 1)*dx
            for j in range(ny):
                self.logger.info("FastSiliconSkyModel: processing amp %d" %
                                 (i*ny + j + 1))
                ymin = j*dy + 1
                ymax = (j + 1)*dy

                # Create a temporary image with the detector wcs to
                # contain the single amp data.
                temp_image = galsim.ImageF(ncol, nrow, wcs=detector.wcs)

                # Include a 2-pixel buffer around the perimeter of the
                # amp region to account for charge redistribution
                # across pixel boundaries into and out of neighboring
                # segments.
                buf = 2
                bounds = (galsim.BoundsI(xmin-buf, xmax+buf, ymin-buf, ymax+buf)
                          & temp_image.bounds)
                temp_amp = temp_image[bounds]

                # Compute the pixel areas as distorted by tree rings, etc..
                sensor_area = sensor.calculate_pixel_areas(temp_amp)

                # Apply distortion from wcs.
                temp_amp.wcs.makeSkyImage(temp_amp, sky_level=1.)

                # Since sky_level was 1, the temp_amp array at this
                # point contains the pixel areas.
                mean_pixel_area = temp_amp.array.mean()

                # Scale by the sky counts.
                temp_amp *= self.sky_counts()/mean_pixel_area

                # Include pixel area scaling from electrostatic distortions.
                temp_amp *= sensor_area

                # Add Poisson noise.
                noise = galsim.PoissonNoise(self.randomNumbers)
                temp_amp.addNoise(noise)

                # Actual bounds of the amplifier segment to be filled.
                amp_bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)

                # Add the single amp image to the final full CCD image.
                image[amp_bounds] += temp_image[amp_bounds]
        return image


class ESOSiliconSkyModel(ESOSkyModel):

    """
    This is a subclass of ESOSkyModel and applies the galsim.Silicon
    sensor model to the sky background photons derived from the ESO
    sky brightness model.
    """
    def __init__(self, obs_metadata, photParams, seed=None, bandpassDict=None,
                 bundles_per_pix=20, logger=None):
        """
        Parameters
        ----------
        obs_metadata: lsst.sims.utils.ObservationMetaData
            Visit-specific data such as pointing direction,
            observation time, seeing, bandpass info, etc..  This info
            is extracted from the phosim instance catalog headers.
        photParams: lsst.sims.photUtils.PhotometricParameters
            Visit-specific photometric parameters, such as exposure time, gain,
            bandpass, etc..
        seed: int, optional
            Seed value passed to the random number generator used by
            the noise model. Defaults to None, which causes GalSim to
            generate the seed from the system.
        bandpassDict: lsst.sims.photUtils.BandpassDict, optional
            Bandpass dictionary used by the sims code.  If None (default),
            the BandpassDict.loadBandpassesFromFiles function is called
            which reads in the standard LSST bandpasses.
        bundles_per_pix: int, optional [20]
            Number of photon bundles per pixel for use with SiliconSensor.
            If bundles_per_pix <= 1, then use unbundled photons.
        """
        super(ESOSiliconSkyModel, self).__init__(obs_metadata, photParams, seed=seed,
                                                 bandpassDict=bandpassDict,
                                                 addNoise=addNoise,
                                                 addBackground=addBackground,
                                                 logger=logger)
        self.bundles_per_pix = bundles_per_pix

        # Wavelength and angle samplers need only be constructed at
        # most once per SED and bandpass, so build them lazily as
        # properties.
        self._waves = None
        self._angles = None

    @property
    def waves(self):
        if self._waves is None:
            sed = galsim.SED(galsim.LookupTable(x=self.skyModel.wave,
                                                f=self.skyModel.spec[0,:]),
                             wave_type='nm', flux_type='flambda')
            bandPassName = self.obs_metadata.bandpass
            bandpass = self.bandpassDict[bandPassName]
            index = np.where(bandpass.sb != 0)
            gs_bandpass \
                = galsim.Bandpass(galsim.LookupTable(x=bandpass.wavelen[index],
                                                     f=bandpass.sb[index]),
                                  wave_type='nm')
            self._waves = galsim.WavelengthSampler(sed=sed,
                                                   bandpass=gs_bandpass,
                                                   rng=self.randomNumbers)
        return self._waves

    @property
    def angles(self):
        if self._angles is None:
            fratio = 1.234  # From https://www.lsst.org/scientists/keynumbers
            obscuration = 0.606  # (8.4**2 - 6.68**2)**0.5 / 8.4
            self._angles \
                = galsim.FRatioAngles(fratio, obscuration, self.randomNumbers)
        return self._angles

    def addNoiseAndBackground(self, image, bandpass=None, m5=None,
                              FWHMeff=None,
                              photParams=None, detector=None, chipName=None):
        """
        This method adds the sky background and noise to an image.

        Parameters
        ----------
        image: galsim.Image
            The GalSim image object to which the background and noise
            are being added.
        bandpass: lsst.sims.photUtils.Bandpass [None]
            This is here just for interface compatibility with the
            base class.
        m5: float [None]
            Not used in this function.
        FWHMeff: float [None]
            Not used in this function.
        photParams: lsst.sims.photUtils.PhotometricParameters, optional
            Object that carries details about the photometric response
            of the telescope.  Default: None
        detector: GalSimDetector, optional
            This is used to pass the tree ring model to the sensor model.
            If None, an interface-compatible detector object is created
            with a null tree ring model.

        Returns
        -------
        galsim.Image: The image with the sky background and noise added.
        """
        if detector is None:
            # Make a dummy detector object with default tree ring
            # properties.
            DummyDetector = namedtuple('DummyDetector', ['tree_rings'])
            TreeRingInfo = namedtuple('TreeRingInfo', ['center', 'func'])
            detector = DummyDetector(TreeRingInfo(galsim.PositionD(0, 0), None))

        return self.process_photons(image.copy(), self.sky_counts(), detector)

    def process_photons(self, image, skyCounts, detector, chunk_size=int(5e6)):
        tree_rings = detector.tree_rings

        # Add photons by amplifier since a full 4k x 4k sensor uses too much
        # memory to represent all of the pixel vertices.

        # imSim images use the Camera Coordinate System where the
        # parallel transfer direction is along the x-axis:
        nx, ny = 2, 8
        nrow, ncol = image.array.shape
        dx = ncol//nx   # number of pixels in x for an amp
        dy = nrow//ny   # number of pixels in y
        # Disable the updating of the pixel boundaries by
        # setting nrecalc to 1e300
        nrecalc = 1e300
        sensor = galsim.SiliconSensor(rng=self.randomNumbers,
                                      nrecalc=nrecalc,
                                      treering_center=tree_rings.center,
                                      treering_func=tree_rings.func)
        for i in range(nx):
            # galsim boundaries start at 1 and include pixels at both ends.
            xmin = i*dx + 1
            xmax = (i + 1)*dx
            for j in range(ny):
                self.logger.info("ESOSiliconSkyModel: processing amp %d" %
                                 (i*ny + j + 1))
                ymin = j*dy + 1
                ymax = (j + 1)*dy
                # Actual bounds of the amplifier region.
                amp_bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)

                # Create a temporary image to contain the single amp data
                # with a 1-pixel buffer around the perimeter.
                temp_image = galsim.ImageF(ncol, nrow)
                bounds = (galsim.BoundsI(xmin-1, xmax+1, ymin-1, ymax+1)
                          & temp_image.bounds)
                temp_amp = temp_image[bounds]
                nphotons = self.get_nphotons(temp_amp, skyCounts)
                chunks = [chunk_size]*(nphotons//chunk_size)
                if nphotons % chunk_size > 0:
                    chunks.append(nphotons % chunk_size)
                for ichunk, nphot in enumerate(chunks):
                    if self.bundles_per_pix > 1:
                        photon_array \
                            = self.get_bundled_photon_array(temp_amp, nphot,
                                                            skyCounts)
                    else:
                        photon_array = self.get_photon_array(temp_amp, nphot)
                        self.logger.info("chunk %d of %d", ichunk + 1,
                                         len(chunks))
                    self.waves.applyTo(photon_array)
                    self.angles.applyTo(photon_array)

                    # Accumulate the photons on the temporary amp image.
                    sensor.accumulate(photon_array, temp_amp, resume=(ichunk>0))
                # Add the temp_amp image to the final image, excluding
                # the 1-pixel buffer.
                image[amp_bounds] += temp_image[amp_bounds]
        return image

    def get_bundled_photon_array(self, image, nphotons, skyCounts):
        """
        Use bundled photons for speed.
        """
        # skyCounts is the expectation value of the counts in each
        # pixel.  We want these values to vary according to Poisson
        # statistics.  One way to do that would be to make
        #     nphotons = npix * skyCounts
        # and give them all random positions within the image.  Then
        # each pixel would have a Poisson variation in its count of
        # how many photons fall in it.  However, we would like to
        # speed up the calculation (and reduce the memory demands) by
        # dropping in photon bundles that effectively do many photons
        # at once.  Unfortunately, if we still just did random
        # positions, this would increase the variance in each pixel,
        # so the values would have the wrong statistics.  Instead, we
        # fix the number of bundles per pixel and put all the variance
        # in the flux.
        #
        # To get the right expected flux and variance in each pixel, we need:
        #
        #     nbundles_per_pixel * mean(flux_per_bundle) = skyCounts
        #     nbundles_per_pixel * var(flux_per_bundle) = skyCounts
        #
        # I.e., we can get the right statistics if
        #
        #     mean(flux_per_bundle) = var(flux_per_bundle).
        #

        # A convenient way to do that is to have the fluxes of the
        # bundles be generated from a Poisson distribution.

        # Make a PhotonArray to hold the sky photons
        npix = np.prod(image.array.shape)
        nbundles_per_pix = self._nbundles_per_pix(skyCounts)
        flux_per_bundle = np.float(skyCounts) / nbundles_per_pix
        photon_array = galsim.PhotonArray(nphotons)

        # Generate the x,y values.
        xx, yy = np.meshgrid(np.arange(image.xmin, image.xmax+1),
                             np.arange(image.ymin, image.ymax+1))
        xx = xx.ravel()
        yy = yy.ravel()
        assert len(xx) == npix
        assert len(yy) == npix
        xx = np.repeat(xx, nbundles_per_pix)
        yy = np.repeat(yy, nbundles_per_pix)
        # If the photon_array is smaller than xx and yy,
        # randomly select the corresponding number of xy values.
        if photon_array.size() < len(xx):
            index = (np.random.permutation(np.arange(len(xx)))[:photon_array.size()],)
            xx = xx[index]
            yy = yy[index]
        assert len(xx) == photon_array.size()
        assert len(yy) == photon_array.size()
        galsim.random.permute(self.randomNumbers, xx, yy)  # Randomly reshuffle in place

        # The above values are pixel centers.  Add a random offset within each pixel.
        self.randomNumbers.generate(photon_array.x)  # Random values from 0..1
        photon_array.x -= 0.5
        self.randomNumbers.generate(photon_array.y)
        photon_array.y -= 0.5
        photon_array.x += xx
        photon_array.y += yy

        # Set the flux of the photons
        flux_pd = galsim.PoissonDeviate(self.randomNumbers, mean=flux_per_bundle)
        flux_pd.generate(photon_array.flux)

        return photon_array

    def get_photon_array(self, image, nphotons):
        # Simpler method that has all the pixels with flux=1.
        # Might be too slow, in which case consider switching to the above code.
        photon_array = galsim.PhotonArray(int(nphotons))

        # Generate the x,y values.
        self.randomNumbers.generate(photon_array.x) # 0..1 so far
        photon_array.x *= (image.xmax - image.xmin + 1)
        photon_array.x += image.xmin - 0.5  # Now from xmin-0.5 .. xmax+0.5
        self.randomNumbers.generate(photon_array.y)
        photon_array.y *= (image.ymax - image.ymin + 1)
        photon_array.y += image.ymin - 0.5

        # Flux in this case is simple.  All flux = 1.
        photon_array.flux = 1

        return photon_array

    def get_nphotons(self, image, skyCounts):
        npix = np.prod(image.array.shape)
        if self.bundles_per_pix <= 1:
            # No bundling, so return the total number of sky bg
            # photons drawn from a Poisson distribution.
            return int(galsim.PoissonDeviate(self.randomNumbers,
                                             mean=npix*skyCounts)())
        # Compute the effective number of photons for bundled photons.
        return int(npix*self._nbundles_per_pix(skyCounts))

    def _nbundles_per_pix(self, skyCounts):
        return min(int(skyCounts), self.bundles_per_pix)


def get_skyModel_params():
    """
    Get the zero points and reference magnitude for the sky model.

    Returns
    -------
    dict : the skyModel zero points and reference magnitudes.
    """
    config = get_config()
    return config['skyModel_params']
