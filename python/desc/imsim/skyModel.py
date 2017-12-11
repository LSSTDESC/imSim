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


class ESOSkyModel(NoiseAndBackgroundBase):
    """
    This class wraps the GalSim class CCDNoise.  This derived class returns
    a sky model based on the ESO model as implemented in
    """

    def __init__(self, obs_metadata, seed=None, bandpassDict=None,
                 addNoise=True, addBackground=True, fast_background=False):
        """
        Parameters
        ----------
        obs_metadata: lsst.sims.utils.ObservationMetaData
            Visit-specific data such as pointing direction,
            observation time, seeing, bandpass info, etc..  This info
            is extracted from the phosim instance catalog headers.
        seed: int, optional
            Seed value passed to the random number generator used by
            the noise model. Defaults to None, which causes GalSim to
            generate the seed from the system.
        bandpassDict: lsst.sims.photUtils.BandpassDict, optional
            Bandpass dictionary used by the sims code.  If None (default),
            the BandpassDict.loadBandpassesFromFiles function is called
            which reads in the standard LSST bandpasses.
        addNoise: bool, optional [True]
            Flag to add noise from the NoiseAndBackgroundBase noise model.
            TODO: Determine what this does.  If it is read noise, it should
            be turned off, since the electronic readout code will add that.
        addBackground: bool, optional [True]
            Add sky background.
        fast_background: bool, optional [False]
            If True, just add the expected sky background counts to
            each pixel.  Otherwise, process the background photons
            through the sensor model.
        """
        self.obs_metadata = obs_metadata
        if bandpassDict is None:
            self.bandpassDict = BandpassDict.loadBandpassesFromFiles()[0]

        self.addNoise = addNoise
        self.addBackground = addBackground

        self.fast_background = fast_background

        if seed is None:
            self.randomNumbers = galsim.UniformDeviate()
        else:
            self.randomNumbers = galsim.UniformDeviate(seed)

        # Computing the skybrightness.SkyModel object is expensive, so
        # do it only once in the constructor.
        self.skyModel = skybrightness.SkyModel(mags=True)

    def addNoiseAndBackground(self, image, bandpass=None, m5=None,
                              FWHMeff=None,
                              photParams=None,
                              detector=None,
                              bundle_photons=True):
        """This method adds the sky background and noise to an image.

        Parameters
        ----------
        image: galsim.Image
            The GalSim image object to which the background and noise
            are being added.
        bandpass: lsst.sims.photUtils.Bandpass, optional
            Bandpass object characterizing the filter through which
            the image is being taken. This seems to be ignored in this code.
        FWHMeff: float, optional
            Effective FWHM (of the psf???).  Not used in this code.
        photParams: lsst.sims.photUtils.PhotometricParameters, optional
            Object that carries details about the photometric response
            of the telescope.  Default: None
        detector: GalSimDetector, optional
            This is used to pass the tree ring model to the sensor model.
            If None, an interface-compatible detector object is created
            with a null tree ring model.
        bundle_photons: bool, optional
            Flag to use photon bundling in sensor model handling of sky
            photons. Default: True

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

        # calculate the sky background to be added to each pixel
        ra = np.array([self.obs_metadata.pointingRA])
        dec = np.array([self.obs_metadata.pointingDec])
        mjd = self.obs_metadata.mjd.TAI
        self.skyModel.setRaDecMjd(ra, dec, mjd, degrees=True)

        bandPassName = self.obs_metadata.bandpass
        skyMagnitude = self.skyModel.returnMags()[bandPassName]

        # Since we are only producing one eimage, account for cases
        # where nsnap > 1 with an effective exposure time for the
        # visit as a whole.  TODO: Undo this change when we write
        # separate images per exposure.
        exposureTime = photParams.nexp*photParams.exptime*u.s

        # TODO: figure out why skyCountsPerSec returns a np.array
        skyCounts \
            = (skyCountsPerSec(surface_brightness=skyMagnitude,
                               filter_band=bandPassName)*exposureTime).value[0]

        # print "Magnitude:", skyMagnitude
        # print "Brightness:", skyMagnitude, skyCounts

        image = image.copy()

        if self.addBackground:
            if self.fast_background:
                image += skyCounts
            else:   # The below stuff does this more carefully via Craig's sensor code.
                if bundle_photons:
                    # Use bundled photons to speed things up at the expense of some accuracy.
                    photon_array = self.get_bundled_photon_array(image, skyCounts)
                else:
                    # Unbundled version where each photon has flux=1
                    photon_array = self.get_photon_array(image, skyCounts)
                image = self.process_photon_array(image, detector, photon_array)

            # If we are adding the skyCounts to the image, there is no
            # need to pass a skyLevel parameter to the noise model.
            # skyLevel is just used to calculate the level of Poisson
            # noise.  If the sky background is included in the image,
            # the Poisson noise will be calculated from the actual
            # image brightness.
            skyLevel = 0
        else:
            skyLevel = skyCounts*photParams.gain

        if self.addNoise:
            noiseModel = self.getNoiseModel(skyLevel=skyLevel, photParams=photParams)
            image.addNoise(noiseModel)

        return image

    def get_bundled_photon_array(self, image, skyCounts):
        # Saving this option in case we need it for speed, but for now, we'll do the
        # simple thing of having all the photons with flux=1 and get the Poisson variation
        # naturally by randomizing the position across the whole image.

        # skyCounts is the expectation value of the counts in each pixel.
        # We want these values to vary according to Poisson statistics.
        # One way to do that would be to make nphotons = npix * skyCounts and give them all
        # random positions within the image.  Then each pixel would have a Poisson variation
        # in its count of how many photons fall in it.
        # However, we would like to speed up the calculation (and reduce the memory demands)
        # by dropping in photon bundles that effectively do many photons at once.
        # Unfortunately, if we still just did random positions, this would increase the
        # variance in each pixel, so the values would have the wrong statistics.  Instead,
        # we fix the number of bundles per pixel and put all the variance in the flux.
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
        # A convenient way to do that is to have the fluxes of the bundles be generated from
        # a Poisson distribution.

        # Make a PhotonArray to hold the sky photons
        npix = np.prod(image.array.shape)
        nbundles_per_pix = 20  # Somewhat arbitrary.  Larger is more accurate, but slower.
        if skyCounts < nbundles_per_pix:  # Don't put < 1 real photon per "bundle"
            nbundles_per_pix = int(skyCounts)
        flux_per_bundle = skyCounts / nbundles_per_pix
        nbundles = npix * nbundles_per_pix
        photon_array = galsim.PhotonArray(int(nbundles))

        # Generate the x,y values.
        xx, yy = np.meshgrid(np.arange(image.xmin, image.xmax+1),
                             np.arange(image.ymin, image.ymax+1))
        xx = xx.ravel()
        yy = yy.ravel()
        assert len(xx) == npix
        assert len(yy) == npix
        xx = np.repeat(xx, nbundles_per_pix)
        yy = np.repeat(yy, nbundles_per_pix)
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

    def get_photon_array(self, image, skyCounts):
        # Simpler method that has all the pixels with flux=1.
        # Might be too slow, in which case consider switching to the above code.
        npix = np.prod(image.array.shape)
        nphotons = npix * skyCounts
        # Technically, this should be a Poisson variate with this expectation value.
        nphotons = galsim.PoissonDeviate(self.randomNumbers, mean=nphotons)()
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

    def process_photon_array(self, image, detector, photon_array):
        sed = galsim.SED(galsim.LookupTable(x=self.skyModel.wave, f=self.skyModel.spec[0,:]),
                         wave_type='nm', flux_type='flambda')

        bandPassName = self.obs_metadata.bandpass
        bandpass = self.bandpassDict[bandPassName]
        index = np.where(bandpass.sb != 0)
        gs_bandpass = galsim.Bandpass(galsim.LookupTable(x=bandpass.wavelen[index], f=bandpass.sb[index]),
                                      wave_type='nm')
        waves = galsim.WavelengthSampler(sed=sed, bandpass=gs_bandpass, rng=self.randomNumbers)
        waves.applyTo(photon_array)

        # Set the angles
        fratio = 1.234  # From https://www.lsst.org/scientists/keynumbers
        obscuration = 0.606  # (8.4**2 - 6.68**2)**0.5 / 8.4
        angles = galsim.FRatioAngles(fratio, obscuration, self.randomNumbers)
        angles.applyTo(photon_array)

        # The main slow bit in the Silicon sensor is when recalculating the pixel boundaries.
        # We do this after every nrecalc electrons.  It's only really important to do when
        # the flux per pixel is around 1000.  So in this case, that's npix * 1000 electrons.
        nrecalc = np.prod(image.array.shape) * 1000
        sensor = galsim.SiliconSensor(rng=self.randomNumbers, nrecalc=nrecalc,
                                      treering_center=detector.tree_rings.center,
                                      treering_func=detector.tree_rings.func)

        # Accumulate the photons on the image.
        sensor.accumulate(photon_array, image)
        return image

    def add_sky_bg_without_sensor_effects(self, image, skyCounts):
        image += skyCounts
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
