'''
Classes to represent realistic sky models.
Note that this extends the default classes located in
sims_GalSimInterface/python/lsst/sims/GalSimInterface/galSimNoiseAndBackground.py
'''
import numpy as np
import galsim
import lsst.sims.coordUtils
from lsst.sims.photUtils import BandpassDict, Sed
import lsst.sims.skybrightness as skybrightness
from .sims_GalSimInterface.galSimNoiseAndBackground import NoiseAndBackgroundBase
from .imSim import get_config, get_logger, get_obs_lsstSim_camera

__all__ = ['make_sky_model', 'get_chip_center', 'sky_counts_per_sec',
           'ESOSkyModel', 'ESOSiliconSkyModel', 'FastSiliconSkyModel']


def make_sky_model(obs_metadata, photParams, seed=None, bandpassDict=None,
                   addNoise=True, addBackground=True, apply_sensor_model=False,
                   logger=None, fast_silicon=True, disable_sky_model=False):
    "Function to provide ESOSkyModel object."
    if disable_sky_model:
        return NullSkyModel()
    if apply_sensor_model:
        if fast_silicon:
            return FastSiliconSkyModel(obs_metadata, photParams,
                                       seed=seed,
                                       bandpassDict=bandpassDict,
                                       logger=logger)
        else:
            return ESOSiliconSkyModel(obs_metadata, photParams, seed=seed,
                                      bandpassDict=bandpassDict,
                                      logger=logger)
    return ESOSkyModel(obs_metadata, photParams, seed=seed,
                       bandpassDict=bandpassDict, addNoise=addNoise,
                       addBackground=addBackground, logger=logger)


def sky_counts_per_sec(skyModel, photParams, bandpass, magNorm=None):
    """
    Compute the sky background counts per pixel per second.  Note that
    the gain in photParams is applied to the return value such that
    "counts" are in units of ADU.

    Parameters
    ----------
    skyModel: lsst.sims.skybrightness.SkyModel
        Model of the sky for the current epoch.
    photParams: lsst.sims.photUtils.PhotometricParameters
        Object containing parameters of the photometric response of the
        telescope, including pixel scale, gain, effective area, exposure
        time, number of exposures, etc.
    bandpass: lsst.sims.photUtils.Bandpass
        Instrumental throughput for a particular passband.
    magNorm: float [None]
        If not None, then renormalize the sky SED to have a monochromatic
        magnitude of magNorm at 500nm.  Otherwise, use the default
        skyModel normalization.

    Returns
    -------
    ADUs per second per pixel
    """
    wave, spec = skyModel.returnWaveSpec()
    sed = Sed(wavelen=wave, flambda=spec[0, :])
    if magNorm is not None:
        flux_norm = sed.calcFluxNorm(magNorm, bandpass)
        sed.multiplyFluxNorm(flux_norm)
    countrate_per_arcsec = sed.calcADU(bandpass=bandpass, photParams=photParams)
    exptime = photParams.nexp*photParams.exptime
    return countrate_per_arcsec*photParams.platescale**2/exptime


def get_chip_center(chip_name, camera):
    """
    Get center of the chip in focal plane pixel coordinates

    Parameters
    ----------
    chip_name: str
        The name of the chip, e.g., "R:2,2 S:1,1".
    camera: lsst.afw.cameraGeom.camera.Camera
        The camera object, e.g., LsstSimMapper().camera.

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


class NullSkyModel(NoiseAndBackgroundBase):
    """
    Class to disable the sky model entirely. Can be used to produce
    FITS file for object injection or for testing individual object
    rendering.
    """
    def __init__(self):
        pass

    def addNoiseAndBackground(self, image, **kwds):
        """
        Return the image as-is, i.e., without adding any sky background
        photons.
        """
        return image

    def sky_counts(self, chipName):
        """Return zero sky background counts."""
        return 0

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

    def addNoiseAndBackground(self, image, bandpass=None, m5=None,
                              FWHMeff=None, photParams=None, detector=None):
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

        @param [in] detector is the sensor being considered.

        @param [out] the input image with the background and noise model added to it.
        """
        skyCounts = self.sky_counts(detector.name)

        if self.addBackground:
            image += skyCounts
            # If we are adding the skyCounts to the image, we
            # should set skyLevel=0 in the call to the noise model.
            # skyLevel is just used to calculate the level of Poisson
            # noise.  If the sky background is included in the image,
            # the Poisson noise will be calculated from the actual
            # image brightness.
            skyLevel = 0.0
        else:
            skyLevel = skyCounts*photParams.gain

        if self.addNoise:
            noiseModel \
                = self.getNoiseModel(skyLevel=skyLevel, photParams=photParams)
            image.addNoise(noiseModel)

        return image

    def getNoiseModel(self, skyLevel=0.0, photParams=None):
        """
        This method returns the noise model implemented for this wrapper
        class.

        This is currently the same as implemented in ExampleCCDNoise.
        This routine can both Poisson fluctuate the background and add
        read noise.  We turn off the read noise by adjusting the
        parameters in the photParams.
        """

        return galsim.CCDNoise(self.randomNumbers, sky_level=skyLevel,
                               gain=photParams.gain,
                               read_noise=photParams.readnoise)

    def sky_counts(self, chipName):
        """
        Parameters
        ----------
        chipName: str
           The name of the sensor at which the sky background will be
           evaluated.

        Returns
        -------
        float: sky background counts per pixel.
        """
        camera = get_obs_lsstSim_camera()
        center_x, center_y = get_chip_center(chipName, camera)
        ra, dec = lsst.sims.coordUtils.raDecFromPixelCoords(
            xPix=center_x, yPix=center_y, chipName=chipName,
            camera=camera, obs_metadata=self.obs_metadata, epoch=2000.0,
            includeDistortion=True)
        mjd = self.obs_metadata.mjd.TAI
        self.skyModel.setRaDecMjd(ra, dec, mjd, degrees=True)

        bandpass = self.bandpassDict[self.obs_metadata.bandpass]

        # Since we are only producing one eimage, account for cases
        # where nsnap > 1 with an effective exposure time for the
        # visit as a whole.  TODO: Undo this change when we write
        # separate images per exposure.
        exposureTime = self.photParams.nexp*self.photParams.exptime
        return exposureTime*sky_counts_per_sec(self.skyModel, self.photParams,
                                               bandpass)


class FastSiliconSkyModel(ESOSkyModel):
    """
    This version produces an image by scaling the counts in each pixel
    by the areas of distorted pixel geometries in the galsim.Silicon
    model to account for electrostatic effects such as tree rings. It
    is currently used for making the sky background or flats which
    require a large number of electrons in every pixel.
    """
    def __init__(self, obs_metadata, photParams, seed=None,
                 bandpassDict=None, logger=None):
        super(FastSiliconSkyModel, self).__init__(obs_metadata, photParams,
                                                  seed=seed,
                                                  bandpassDict=bandpassDict,
                                                  logger=logger)

    def addNoiseAndBackground(self, image, bandpass=None, m5=None,
                              FWHMeff=None, photParams=None, detector=None):
        """
        Add the sky level counts to the image, rescale by the distorted
        pixel areas to account for tree rings, etc., then add Poisson noise.
        This implementation is based on GalSim/devel/lsst/treering_skybg2.py.
        """
        if detector is None:
            raise RuntimeError("A GalSimDetector object must be provided.")

        # Create a SiliconSensor object to handle the calculations
        # of the quantities related to the pixel boundary distortions
        # from electrostatic effects such as tree rings, etc..
        # Use the transpose=True option since "eimages" of LSST sensors
        # follow the Camera Coordinate System convention where the
        # parallel transfer direction is along the x-axis.
        config = get_config()
        nrecalc = 1e300  # disable pixel boundary updating.
        sensor = galsim.SiliconSensor(rng=self.randomNumbers,
                                      nrecalc=nrecalc,
                                      strength=config['ccd']['bf_strength'],
                                      treering_func=detector.tree_rings.func,
                                      treering_center=detector.tree_rings.center,
                                      transpose=True)

        # Loop over 1/2 amplifiers to save memory when storing the 36
        # pixel vertices per pixel. The 36 vertices arise from the 8
        # vertices per side + 4 corners (8*(4 sides) + 4 = 36).  This
        # corresponds to 72 floats per pixel (x and y coordinates) for
        # representing the pixel distortions in memory.
        nrow, ncol = image.array.shape
        nx, ny = 4, 8
        dx = ncol//nx
        dy = nrow//ny

        for i in range(nx):
            xmin = i*dx + 1
            xmax = (i + 1)*dx
            for j in range(ny):
                self.logger.debug("FastSiliconSkyModel: processing amp region %d" %
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
                temp_amp *= self.sky_counts(detector.name)/mean_pixel_area

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
                 addNoise=True, addBackground=True, logger=None):
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
        """
        super(ESOSiliconSkyModel, self).__init__(obs_metadata, photParams,
                                                 seed=seed,
                                                 bandpassDict=bandpassDict,
                                                 addNoise=addNoise,
                                                 addBackground=addBackground,
                                                 logger=logger)

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
                              FWHMeff=None, photParams=None, detector=None):
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
        detector: GalSimDetector
            This is used to pass the tree ring model to the sensor model.

        Returns
        -------
        galsim.Image: The image with the sky background and noise added.
        """
        if detector is None:
            raise RuntimeError("A detector must be specified.")

        return self.process_photons(image, self.sky_counts(detector.name),
                                    detector)

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

        config = get_config()
        # Disable the updating of the pixel boundaries by
        # setting nrecalc to 1e300
        nrecalc = 1e300
        sensor = galsim.SiliconSensor(rng=self.randomNumbers,
                                      nrecalc=nrecalc,
                                      strength=config['ccd']['bf_strength'],
                                      treering_center=tree_rings.center,
                                      treering_func=tree_rings.func,
                                      transpose=True)
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
                    photon_array = self.get_photon_array(temp_amp, nphot)
                    self.logger.info("chunk %d of %d", ichunk + 1, len(chunks))

                    self.waves.applyTo(photon_array)
                    self.angles.applyTo(photon_array)

                    # Accumulate the photons on the temporary amp image.
                    sensor.accumulate(photon_array, temp_amp, resume=(ichunk>0))
                # Add the temp_amp image to the final image, excluding
                # the 1-pixel buffer.
                image[amp_bounds] += temp_image[amp_bounds]
        return image

    def get_photon_array(self, image, nphotons):
        """
        Generate an array of photons randomly distributed over the
        surface of the sensor.
        """
        photon_array = galsim.PhotonArray(int(nphotons))

        # Generate the x,y values.
        self.randomNumbers.generate(photon_array.x) # 0..1 so far
        photon_array.x *= (image.xmax - image.xmin + 1)
        photon_array.x += image.xmin - 0.5  # Now from xmin-0.5 .. xmax+0.5
        self.randomNumbers.generate(photon_array.y)
        photon_array.y *= (image.ymax - image.ymin + 1)
        photon_array.y += image.ymin - 0.5
        photon_array.flux = 1

        return photon_array

    def get_nphotons(self, image, skyCounts):
        npix = np.prod(image.array.shape)
        # Return the total number of sky bg photons drawn from a
        # Poisson distribution.
        return int(galsim.PoissonDeviate(self.randomNumbers,
                                         mean=npix*skyCounts)())
