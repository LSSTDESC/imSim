"""
This file defines the following classes:

GalSimInterpreter -- a class which takes objects passed by a GalSim
Instance Catalog (see galSimCatalogs.py) and uses GalSim to write them
to FITS images.
"""
import math
import os
import pickle
import tempfile
import gzip
import numpy as np
import astropy
import galsim
from lsst.sims.utils import  observedFromPupilCoords
from . import make_galsim_detector, SNRdocumentPSF, \
    Kolmogorov_and_Gaussian_PSF, LsstObservatory

__all__ = ["make_gs_interpreter", "GalSimInterpreter", "ObjectFlags"]

def make_gs_interpreter(obs_md, detector, bandpassDict, noiseWrapper,
                        epoch=None, seed=None, apply_sensor_model=False,
                        bf_strength=1):
    """Function to make a GalSimInterpreter object."""
    disable_sensor_model = not apply_sensor_model
    return GalSimInterpreter(obs_md, detector,
                             bandpassDict=bandpassDict,
                             noiseWrapper=noiseWrapper,
                             epoch=epoch, seed=seed,
                             bf_strength=bf_strength,
                             disable_sensor_model=disable_sensor_model)

class GalSimInterpreter:
    """
    This is the class which actually takes the objects contained in
    the GalSim InstanceCatalog and converts them into FITS images.
    """
    _observatory = LsstObservatory()

    def __init__(self, obs_metadata, detector,
                 bandpassDict, noiseWrapper=None,
                 epoch=None, seed=None, bf_strength=1,
                 disable_sensor_model=False):

        """
        @param [in] obs_metadata is an instantiation of the
        ObservationMetaData class which carries data about this
        particular observation (telescope site and pointing
        information)

        @param [in] The GalSimDetector objects for which
        we are drawing a FITS image

        @param [in] bandpassDict is a BandpassDict containing all of
        the bandpasses for which we are generating images

        @param [in] noiseWrapper is an instantiation of a
        NoiseAndBackgroundBase class which tells the interpreter how
        to add sky noise to its images.

        @param [in] seed is an integer that will use to seed the
        random number generator used when drawing images (if None,
        GalSim will automatically create a random number generator
        seeded with the system clock)
        """

        self.obs_metadata = obs_metadata
        self.detector = detector
        self.bandpassDict = bandpassDict
        self.noiseWrapper = noiseWrapper
        self.epoch = epoch
        if seed is not None:
            self._rng = galsim.UniformDeviate(seed)
        else:
            self._rng = None

        self.PSF = None

        # This will contain the galsim Image
        self.detectorImage = None

        self.checkpoint_file = None
        self.drawn_objects = set()
        self.nobj_checkpoint = 1000
        self.centroid_base_name = None
        # This dict will contain the file handles for each centroid
        # file where sources are found.
        self.centroid_handles = {}
        # This is a list of the centroid objects which will be written
        # to the file.
        self.centroid_list = []

        # Initialization from GalSimSiliconInterpreter
        self.disable_sensor_model = disable_sensor_model

        self.gs_bandpass_dict = {}
        for bandpassName in bandpassDict:
            bandpass = bandpassDict[bandpassName]
            index = np.where(bandpass.sb != 0)
            bp_lut = galsim.LookupTable(x=bandpass.wavelen[index],
                                        f=bandpass.sb[index])
            self.gs_bandpass_dict[bandpassName] \
                = galsim.Bandpass(bp_lut, wave_type='nm')

        self.sky_bg_per_pixel = None

        # Create a PSF that's fast to evaluate for the postage stamp
        # size calculation for extended objects in .getStampBounds.
        FWHMgeom = obs_metadata.OpsimMetaData['FWHMgeom']
        self._double_gaussian_psf = SNRdocumentPSF(FWHMgeom)

        # Save the parameters needed to create a Kolmogorov PSF for a
        # custom value of gsparams.folding_threshold.  That PSF will
        # to be used in the .getStampBounds function for bright stars.
        altRad = np.radians(obs_metadata.OpsimMetaData['altitude'])
        self._airmass = 1.0/np.sqrt(1.0-0.96*(np.sin(0.5*np.pi-altRad))**2)
        self._rawSeeing = obs_metadata.OpsimMetaData['rawSeeing']
        self._band = obs_metadata.bandpass

        # Save the default folding threshold for determining when to recompute
        # the PSF for bright point sources.
        self._ft_default = galsim.GSParams().folding_threshold

        # Save these, which are needed for DCR
        self.local_hour_angle \
            = self.getHourAngle(self.obs_metadata.mjd.TAI,
                                self.obs_metadata.pointingRA)*galsim.degrees
        self.obs_latitude \
            = self.observatory.getLatitude().asDegrees()*galsim.degrees

        # Make a trivial SED to use for faint things.
        blue_limit = np.min([bp.blue_limit for bp
                             in self.gs_bandpass_dict.values()])
        red_limit = np.max([bp.red_limit for bp
                            in self.gs_bandpass_dict.values()])
        constant_func = galsim.LookupTable([blue_limit, red_limit],
                                           [1, 1], interpolant='linear')
        self.trivial_sed = galsim.SED(constant_func, wave_type='nm',
                                      flux_type='fphotons')

        # Create a SiliconSensor object.
        self.sensor = galsim.SiliconSensor(
            strength=bf_strength,
            treering_center=detector.tree_rings.center,
            treering_func=detector.tree_rings.func,
            transpose=True)


    def setPSF(self, PSF=None):
        """
        Set the PSF wrapper for this GalSimInterpreter

        @param [in] PSF is an instantiation of a class which inherits
        from PSFbase and defines _getPSF()
        """
        self.PSF = PSF

    def getFileName(self):
        """
        Return the partial name of the FITS file to be written
        """
        return (self.detector.fileName + '_'
                + self.obs_metadata.bandpass + '.fits')

    def blankImage(self):
        """
        Draw a blank image associated with a specific detector.  The
        image will have the correct size for the given detector.
        """
        return galsim.Image(self.detector.xMaxPix - self.detector.xMinPix + 1,
                            self.detector.yMaxPix - self.detector.yMinPix + 1,
                            wcs=self.detector.wcs)

    def drawObject(self, gsObject, max_flux_simple=0, sensor_limit=0,
                   fft_sb_thresh=None):
        """
        Draw an astronomical object on all of the relevant FITS files.

        @param [in] gsObject is an instantiation of the GalSimCelestialObject
        class carrying all of the information for the object whose image
        is to be drawn

        @param [in] max_flux_simple is the maximum flux at which
        various simplifying approximations are used.  These include
        using a flat SED and possibly omitting the realistic sensor
        effects. (default = 0, which means always use the full SED)

        @param [in] sensor_limit is the limiting value of the existing
        flux in the postage stamp image, above which the use of a
        SiliconSensor model is forced.  For faint things, if there is
        not already flux at this level, then a simple sensor model
        will be used instead.  (default = 0, which means the
        SiliconSensor is always used, even for the faint things)

        @param [in] fft_sb_thresh is a surface brightness
        (photons/pixel) where we will switch from photon shooting to
        drawing with fft if any pixel is above this.  Should be at
        least the saturation level, if not higher. (default = None,
        which means never switch to fft.)

        @param [out] outputString is a string denoting which detectors
        the astronomical object illumines, suitable for output in the
        GalSim InstanceCatalog
        """
        object_flags = ObjectFlags()

        outputString = None
        centeredObj = self.createCenteredObject(gsObject)

        # Make sure this object is marked as "drawn" since we only
        # care that this method has been called for this object.
        self.drawn_objects.add(gsObject.uniqueId)

        # Compute the realized object flux (as drawn from the
        # corresponding Poisson distribution) for the observed band
        # and return right away if zero in order to save compute.
        bandpassName = self.obs_metadata.bandpass
        flux = gsObject.flux(bandpassName)
        realized_flux = galsim.PoissonDeviate(self._rng, mean=flux)()
        if flux == 0:
            object_flags.set_flag('skipped')
            self._store_zero_flux_centroid_info(self.detector, flux, gsObject,
                                                object_flags.value)
            return outputString

        self._addNoiseAndBackground()

        # Create a surface operation to sample incident angles and a
        # galsim.SED object for sampling the wavelengths of the
        # incident photons.
        fratio = 1.234  # From https://www.lsst.org/scientists/keynumbers
        obscuration = 0.606  # (8.4**2 - 6.68**2)**0.5 / 8.4
        angles = galsim.FRatioAngles(fratio, obscuration, self._rng)

        faint = realized_flux < max_flux_simple

        if faint:
            # For faint things, use a very simple SED, since we don't
            # really care about getting the exact right distribution
            # of wavelengths here.  (Impacts DCR and electron
            # conversion depth in silicon)
            gs_sed = self.trivial_sed
            object_flags.set_flag('simple_sed')
        else:
            sed_lut = galsim.LookupTable(x=gsObject.sed.wavelen,
                                         f=gsObject.sed.flambda)
            gs_sed = galsim.SED(sed_lut, wave_type='nm', flux_type='flambda',
                                redshift=0.)

        ra_obs, dec_obs = observedFromPupilCoords(gsObject.xPupilRadians,
                                                  gsObject.yPupilRadians,
                                                  obs_metadata=self.obs_metadata)
        obj_coord = galsim.CelestialCoord(ra_obs*galsim.degrees,
                                          dec_obs*galsim.degrees)

        gs_bandpass = self.gs_bandpass_dict[bandpassName]
        waves = galsim.WavelengthSampler(sed=gs_sed, bandpass=gs_bandpass,
                                         rng=self._rng)
        dcr = galsim.PhotonDCR(
            base_wavelength=gs_bandpass.effective_wavelength,
            HA=self.local_hour_angle,
            latitude=self.obs_latitude,
            obj_coord=obj_coord)

        # Set the object flux to the value realized from the
        # Poisson distribution.
        obj = centeredObj.withFlux(realized_flux)

        use_fft = False
        if (realized_flux > 1.e6 and fft_sb_thresh is not None
            and realized_flux > fft_sb_thresh):
            # Note: Don't bother with this check unless the total
            # flux is > thresh.  Otherwise, there is no chance
            # that the flux in 1 pixel is > thresh.  Also, the
            # cross-over point for time to where the fft becomes
            # faster is emprically around 1.e6 photons, so also
            # don't bother unless the flux is more than this.
            obj, use_fft = self.maybeSwitchPSF(obj, fft_sb_thresh)

        if use_fft:
            object_flags.set_flag('fft_rendered')
            object_flags.set_flag('no_silicon')

        detector = self.detector

        xPix, yPix = detector.camera_wrapper\
           .pixelCoordsFromPupilCoords(gsObject.xPupilRadians,
                                       gsObject.yPupilRadians,
                                       chipName=detector.name,
                                       obs_metadata=self.obs_metadata)

        # Desired position to draw the object.
        image_pos = galsim.PositionD(xPix, yPix)

        # Find a postage stamp region to draw onto.  Use (sky
        # noise)/3. as the nominal minimum surface brightness
        # for rendering an extended object.
        keep_sb_level = np.sqrt(self.sky_bg_per_pixel)/3.
        full_bounds = self.getStampBounds(gsObject, realized_flux, image_pos,
                                          keep_sb_level, 3*keep_sb_level)

        # Ensure the bounds of the postage stamp lie within the image.
        bounds = full_bounds & self.detectorImage.bounds

        if not bounds.isDefined():
            return outputString

        # Offset is relative to the "true" center of the postage stamp.
        offset = image_pos - bounds.true_center

        image = self.detectorImage[bounds]

        if faint:
            # For faint things, only use the silicon sensor if
            # there is already some significant flux on the
            # image near the object.  Brighter-fatter doesn't
            # start having any measurable effect until at
            # least around 1000 e-/pixel. So a limit of 200 is
            # conservative by a factor of 5.  Do the
            # calculation relative to the median, since a
            # perfectly flat sky level will not have any B/F
            # effect.  (But noise fluctuations due to the sky
            # will be properly included here if the sky is
            # drawn first.)
            if (np.max(image.array) > np.median(image.array)
                + sensor_limit):
                sensor = self.sensor
            else:
                sensor = None
                object_flags.set_flag('no_silicon')
        else:
            sensor = self.sensor

        if self.disable_sensor_model:
            sensor = None
            object_flags.set_flag('no_silicon')

        if sensor:
            # Ensure the rng used by the sensor object is set
            # to the desired state.
            self.sensor.rng.reset(self._rng)
            surface_ops = [waves, dcr, angles]
        else:
            # Don't need angles if not doing silicon sensor.
            surface_ops = [waves, dcr]

        if use_fft:
            # When drawing with FFTs, large offsets can be a
            # problem, since they can blow up the required FFT
            # size.  We'll guard for that below with a try
            # block, but we can minimize how often this
            # happens by making sure the offset is close to
            # 0,0.
            if abs(offset.x) > 2 or abs(offset.y) > 2:
                # Make a larger image that has the object near
                # the center.
                fft_image = galsim.Image(full_bounds, dtype=image.dtype,
                                         wcs=image.wcs)
                fft_image[bounds] = image
                fft_offset = image_pos - full_bounds.true_center
            else:
                fft_image = image.copy()
                fft_offset = offset

            try:
                obj.drawImage(method='fft',
                              offset=fft_offset,
                              image=fft_image,
                              gain=detector.photParams.gain)
            except galsim.errors.GalSimFFTSizeError:
                use_fft = False
                object_flags.unset_flag('fft_rendered')
                if sensor is not None:
                    object_flags.unset_flag('no_silicon')
            else:
                # Some pixels can end up negative from FFT
                # numerics.  Just set them to 0.
                fft_image.array[fft_image.array < 0] = 0.
                fft_image.addNoise(galsim.PoissonNoise(rng=self._rng))
                # In case we had to make a bigger image, just
                # copy the part we need.
                image += fft_image[bounds]
        if not use_fft:
            obj.drawImage(method='phot',
                          offset=offset,
                          rng=self._rng,
                          maxN=int(1e6),
                          image=image,
                          sensor=sensor,
                          surface_ops=surface_ops,
                          add_to_image=True,
                          poisson_flux=False,
                          gain=detector.photParams.gain)

        # If we are writing centroid files,store the entry.
        if self.centroid_base_name is not None:
            centroid_tuple = (detector.fileName, bandpassName,
                              gsObject.uniqueId,
                              flux, realized_flux, xPix, yPix,
                              object_flags.value,
                              gsObject.galSimType)
            self.centroid_list.append(centroid_tuple)

        # Because rendering FitsImage object types can take a long
        # time for bright objects (>1e4 photons takes longer than ~30s
        # on cori-haswell), force a checkpoint after each object is
        # drawn.
        force_checkpoint = ((gsObject.galSimType == 'FitsImage')
                            and (realized_flux > 1e4))
        self.write_checkpoint(force=force_checkpoint)
        return outputString

    @staticmethod
    def maybeSwitchPSF(obj, fft_sb_thresh, pixel_scale=0.2):
        """
        Check if the maximum surface brightness of the object is high
        enough that we should switch to using an fft method rather
        than photon shooting.

        When we do this, we also switch the PSF model to something
        slightly simpler with roughly the same wings, but not as
        complicated in the central core.  Thus, this should only be
        done when the core is going to be saturated anyway, so we only
        really care about the wings of the PSF.

        Note: This function assumes that obj at this point is a
              convolution with the PSF at the end, and that it has had
              its flux set to a new value with `withFlux()`.  If this
              is not the case, an AttributeError will be raised.

        Parameters
        ----------
        gsObject: GalSimCelestialObject
            This contains the information needed to construct a
            galsim.GSObject convolved with the desired PSF.
        obj: galsim.GSObject
            The current GSObject to draw, which might need to be modified
            if we decide to switch to fft drawing.
        fft_sb_thresh: float
            The surface brightness (photons/pixel) where we will switch from
            photon shooting to drawing with fft if any pixel is above this.
            Should be at least the saturation level, if not higher.
        pixel_scale: float [0.2]
            The CCD pixel scale in arcsec.

        Returns
        -------
        galsim.GSObj, bool: obj = the object to actually use
                            use_fft = whether to use fft drawing

        """
        if not fft_sb_thresh:
            return obj, False

        # obj.original should be a Convolution with the PSF at the
        # end.  Extract it.
        geom_psf = obj.original.obj_list[-1]
        all_but_psf = obj.original.obj_list[:-1]
        try:
            screen_list = geom_psf.screen_list
        except AttributeError:
            # If it's not a galsim.PhaseScreenPSF, just use whatever it is.
            fft_psf = [geom_psf]
        else:
            # If geom_psf is a PhaseScreenPSF, then make a simpler one
            # the just convolves a Kolmogorov profile with an
            # OpticalPSF.
            opt_screens = [s for s in geom_psf.screen_list
                           if isinstance(s, galsim.OpticalScreen)]
            if len(opt_screens) >= 1:
                # Should never be more than 1, but it there weirdly
                # is, just use the first.
                opt_screen = opt_screens[0]
                optical_psf = galsim.OpticalPSF(
                        lam=geom_psf.lam,
                        diam=opt_screen.diam,
                        aberrations=opt_screen.aberrations,
                        annular_zernike=opt_screen.annular_zernike,
                        obscuration=opt_screen.obscuration,
                        gsparams=geom_psf.gsparams)
                fft_psf = [optical_psf]
            else:
                fft_psf = []
            r0_500 = screen_list.r0_500_effective
            atm_psf = galsim.Kolmogorov(lam=geom_psf.lam, r0_500=r0_500,
                                        gsparams=geom_psf.gsparams)
            fft_psf.append(atm_psf)

        fft_obj = galsim.Convolve(all_but_psf + fft_psf).withFlux(obj.flux)

        # Now this object should have a much better estimate of the
        # real maximum surface brightness than the original geom_psf
        # did.  However, the max_sb feature gives an over-estimate,
        # whereas to be conservative, we would rather an
        # under-estimate.  For this kind of profile, dividing by 2
        # does a good job of giving us an underestimate of the max
        # surface brightness.  Also note that `max_sb` is in
        # photons/arcsec^2, so multiply by pixel_scale**2 to get
        # photons/pixel, which we compare to fft_sb_thresh.
        if fft_obj.max_sb/2. * pixel_scale**2 > fft_sb_thresh:
            return fft_obj, True

        return obj, False

    def getStampBounds(self, gsObject, flux, image_pos, keep_sb_level,
                       large_object_sb_level, Nmax=1400, pixel_scale=0.2):
        """
        Get the postage stamp bounds for drawing an object within the stamp
        to include the specified minimum surface brightness.  Use the
        folding_threshold criterion for point source objects.  For
        extended objects, use the getGoodPhotImageSize function, where
        if the initial stamp is too large (> Nmax**2 ~ 1GB of RSS
        memory for a 72 vertex/pixel sensor model), use the relaxed
        surface brightness level for large objects.

        Parameters
        ----------
        gsObject: GalSimCelestialObject
            This contains the information needed to construct a
            galsim.GSObject convolved with the desired PSF.
        flux: float
            The flux of the object in e-.
        keep_sb_level: float
            The minimum surface brightness (photons/pixel) out to which to
            extend the postage stamp, e.g., a value of
            sqrt(sky_bg_per_pixel)/3 would be 1/3 the Poisson noise
            per pixel from the sky background.
        large_object_sb_level: float
            Surface brightness level to use for large/bright objects that
            would otherwise yield stamps with more than Nmax**2 pixels.
        Nmax: int [1400]
            The largest stamp size to consider at the nominal keep_sb_level.
            1400**2*72*8/1024**3 = 1GB.
        pixel_scale: float [0.2]
            The CCD pixel scale in arcsec.

        Returns
        -------
        galsim.BoundsI: The postage stamp bounds.

        """
        if flux < 10:
            # For really faint things, don't try too hard.  Just use 32x32.
            image_size = 32
        elif gsObject.galSimType.lower() == "pointsource":
            # For bright stars, set the folding threshold for the
            # stamp size calculation.  Use a
            # Kolmogorov_and_Gaussian_PSF since it is faster to
            # evaluate than an AtmosphericPSF.
            folding_threshold = self.sky_bg_per_pixel/flux
            if folding_threshold >= self._ft_default:
                gsparams = None
            else:
                gsparams = galsim.GSParams(folding_threshold=folding_threshold)
            psf = Kolmogorov_and_Gaussian_PSF(airmass=self._airmass,
                                              rawSeeing=self._rawSeeing,
                                              band=self._band,
                                              gsparams=gsparams)
            obj = self.drawPointSource(gsObject, psf=psf)
            image_size = obj.getGoodImageSize(pixel_scale)
        else:
            # For extended objects, recreate the object to draw, but
            # convolved with the faster DoubleGaussian PSF.
            obj = self.createCenteredObject(gsObject,
                                            psf=self._double_gaussian_psf)
            obj = obj.withFlux(flux)

            # Start with GalSim's estimate of a good box size.
            image_size = obj.getGoodImageSize(pixel_scale)

            # For bright things, defined as having an average of at
            # least 10 photons per pixel on average, try to be careful
            # about not truncating the surface brightness at the edge
            # of the box.
            if flux > 10 * image_size**2:
                image_size = self._getGoodPhotImageSize(gsObject, flux,
                                                        keep_sb_level,
                                                        pixel_scale=pixel_scale)

            # If the above size comes out really huge, scale back to
            # what you get for a somewhat brighter surface brightness
            # limit.
            if image_size > Nmax:
                image_size = self._getGoodPhotImageSize(gsObject, flux,
                                                        large_object_sb_level,
                                                        pixel_scale=pixel_scale)
                image_size = max(image_size, Nmax)

        # Create the bounds object centered on the desired location.
        xmin = int(math.floor(image_pos.x) - image_size/2)
        xmax = int(math.ceil(image_pos.x) + image_size/2)
        ymin = int(math.floor(image_pos.y) - image_size/2)
        ymax = int(math.ceil(image_pos.y) + image_size/2)

        return galsim.BoundsI(xmin, xmax, ymin, ymax)

    def _getGoodPhotImageSize(self, gsObject, flux, keep_sb_level,
                              pixel_scale=0.2):
        point_source = self.drawPointSource(gsObject, self._double_gaussian_psf)
        point_source = point_source.withFlux(flux)
        ps_size = getGoodPhotImageSize(point_source, keep_sb_level,
                                       pixel_scale=pixel_scale)
        unconvolved_obj = self._createCenteredObject(gsObject, psf=None)
        unconvolved_obj = unconvolved_obj.withFlux(flux)
        obj_size = getGoodPhotImageSize(unconvolved_obj, keep_sb_level,
                                        pixel_scale=pixel_scale)
        return int(np.sqrt(ps_size**2 + obj_size**2))

    def _store_zero_flux_centroid_info(self, detector, flux, gsObject,
                                       obj_flags_value):
        if self.centroid_base_name is None:
            return
        realized_flux = 0
        xPix, yPix = detector.camera_wrapper.pixelCoordsFromPupilCoords(
            gsObject.xPupilRadians, gsObject.yPupilRadians,
            detector.name, self.obs_metadata)
        centroid_tuple = (detector.fileName, self.obs_metadata.bandpass,
                          gsObject.uniqueId,
                          flux, realized_flux, xPix, yPix,
                          obj_flags_value, gsObject.galSimType)
        self.centroid_list.append(centroid_tuple)

    def _addNoiseAndBackground(self):
        """
        Add sky background and noise to the detector image.
        """
        if self.detectorImage is not None:
            return
        self.detectorImage = self.blankImage()
        bandpassName = self.obs_metadata.bandpass
        if self.noiseWrapper is not None:
            # Add sky background and noise to the image
            self.detectorImage = self.noiseWrapper.addNoiseAndBackground(
                self.detectorImage, bandpass=self.bandpassDict[bandpassName],
                m5=self.obs_metadata.m5[bandpassName],
                FWHMeff=self.obs_metadata.seeing[bandpassName],
                photParams=self.detector.photParams, detector=self.detector)

        self.write_checkpoint(force=True, object_list=set())

    def drawPointSource(self, gsObject, psf=None):
        """
        Draw an image of a point source.

        @param [in] gsObject is an instantiation of the
        GalSimCelestialObject class carrying information about the
        object whose image is to be drawn

        @param [in] psf PSF to use for the convolution.  If None, then
        use self.PSF.
        """
        if psf is None:
            psf = self.PSF
        return self._drawPointSource(gsObject, psf=psf)

    def _drawPointSource(self, gsObject, psf=None):
        if psf is None:
            raise RuntimeError("Cannot draw a point source in GalSim "
                               "without a PSF")
        return psf.applyPSF(xPupil=gsObject.xPupilArcsec,
                            yPupil=gsObject.yPupilArcsec)

    def drawSersic(self, gsObject, psf=None):
        """
        Draw the image of a Sersic profile.

        @param [in] gsObject is an instantiation of the
        GalSimCelestialObject class carrying information about the
        object whose image is to be drawn

        @param [in] psf PSF to use for the convolution.  If None, then
        use self.PSF.
        """
        if psf is None:
            psf = self.PSF
        return self._drawSersic(gsObject, psf=psf)

    def _drawSersic(self, gsObject, psf=None):
        # create a Sersic profile
        centeredObj = galsim.Sersic(
            n=float(gsObject.sindex),
            half_light_radius=float(gsObject.halfLightRadiusArcsec))

        # Turn the Sersic profile into an ellipse
        centeredObj = centeredObj.shear(
            q=gsObject.minorAxisRadians/gsObject.majorAxisRadians,
            beta=(0.5*np.pi+gsObject.positionAngleRadians)*galsim.radians)

        # Apply weak lensing distortion.
        centeredObj = centeredObj.lens(gsObject.g1, gsObject.g2, gsObject.mu)

        # Apply the PSF.
        if psf is not None:
            centeredObj = psf.applyPSF(xPupil=gsObject.xPupilArcsec,
                                       yPupil=gsObject.yPupilArcsec,
                                       obj=centeredObj)

        return centeredObj

    def drawRandomWalk(self, gsObject, psf=None):
        """
        Draw the image of a RandomWalk light profile. In orider to allow for
        reproducibility, the specific realisation of the random walk is seeded
        by the object unique identifier, if provided.

        @param [in] gsObject is an instantiation of the
        GalSimCelestialObject class carrying information about the
        object whose image is to be drawn

        @param [in] psf PSF to use for the convolution.  If None, then
        use self.PSF.
        """
        if psf is None:
            psf = self.PSF
        return self._drawRandomWalk(gsObject, psf=psf)

    def _drawRandomWalk(self, gsObject, psf=None):
        # Seeds the random walk with the object id if available
        if gsObject.uniqueId is None:
            rng = None
        else:
            rng = galsim.BaseDeviate(int(gsObject.uniqueId))

        # Create the RandomWalk profile
        centeredObj = galsim.RandomKnots(
            npoints=int(gsObject.npoints),
            half_light_radius=float(gsObject.halfLightRadiusArcsec),
            rng=rng)

        # Apply intrinsic ellipticity to the profile
        centeredObj = centeredObj.shear(
            q=gsObject.minorAxisRadians/gsObject.majorAxisRadians,
            beta=(0.5*np.pi+gsObject.positionAngleRadians)*galsim.radians)

        # Apply weak lensing distortion.
        centeredObj = centeredObj.lens(gsObject.g1, gsObject.g2, gsObject.mu)

        # Apply the PSF.
        if psf is not None:
            centeredObj = psf.applyPSF(xPupil=gsObject.xPupilArcsec,
                                       yPupil=gsObject.yPupilArcsec,
                                       obj=centeredObj)

        return centeredObj

    def drawFitsImage(self, gsObject, psf=None):
        """
        Draw the image of a FitsImage light profile.

        @param [in] gsObject is an instantiation of the
        GalSimCelestialObject class carrying information about the
        object whose image is to be drawn

        @param [in] psf PSF to use for the convolution.  If None, then
        use self.PSF.
        """
        if psf is None:
            psf = self.PSF
        return self._drawFitsImage(gsObject, psf=psf)

    def _drawFitsImage(self, gsObject, psf=None):
        # Create the galsim.InterpolatedImage profile from the FITS image.
        centeredObj = galsim.InterpolatedImage(gsObject.fits_image_file,
                                               scale=gsObject.pixel_scale)
        if gsObject.rotation_angle != 0:
            centeredObj = centeredObj.rotate(
                gsObject.rotation_angle*galsim.degrees)

        # Apply weak lensing distortion.
        centeredObj = centeredObj.lens(gsObject.g1, gsObject.g2, gsObject.mu)

        # Apply the PSF
        if psf is not None:
            centeredObj = psf.applyPSF(xPupil=gsObject.xPupilArcsec,
                                       yPupil=gsObject.yPupilArcsec,
                                       obj=centeredObj)

        return centeredObj

    def createCenteredObject(self, gsObject, psf=None):
        """.
        Create a centered GalSim Object (i.e. if we were just to draw
        this object as an image, the object would be centered on the
        frame)

        @param [in] gsObject is an instantiation of the
        GalSimCelestialObject class carrying information about the
        object whose image is to be drawn

        Note: parameters that obviously only apply to Sersic profiles
        will be ignored in the case of point sources
        """
        if psf is None:
            psf = self.PSF
        return self._createCenteredObject(gsObject, psf=psf)

    def _createCenteredObject(self, gsObject, psf=None):
        if gsObject.galSimType == 'sersic':
            centeredObj = self._drawSersic(gsObject, psf=psf)

        elif gsObject.galSimType == 'pointSource':
            centeredObj = self._drawPointSource(gsObject, psf=psf)

        elif gsObject.galSimType == 'RandomWalk':
            centeredObj = self._drawRandomWalk(gsObject, psf=psf)

        elif gsObject.galSimType == 'FitsImage':
            centeredObj = self._drawFitsImage(gsObject, psf=psf)

        else:
            raise RuntimeError("Apologies: the GalSimInterpreter "
                               "does not yet have a method to draw " +
                               gsObject.galSimType + " objects")

        return centeredObj

    def writeImages(self, nameRoot=None):
        """
        Write the FITS files to disk.

        @param [in] nameRoot is a string that will be prepended to the
        names of the output FITS files.  The files will be named like

        @param [out] namesWritten is a list of the names of the FITS
        files written

        nameRoot_detectorName_bandpassName.fits

        myImages_R_0_0_S_1_1_y.fits is an example of an image for an
        LSST-like camera with nameRoot = 'myImages'
        """
        namesWritten = []
        name = self.getFileName()
        if nameRoot is not None:
            fileName = nameRoot + '_' + name
        else:
            fileName = name
        self.detectorImage.write(file_name=fileName)
        namesWritten.append(fileName)

        return namesWritten

    def open_centroid_file(self, centroid_name):
        """
        Open a centroid file.  This file will have one line per-object
        and the it will be labeled with the objectID and then followed
        by the average X Y position of the photons from the
        object. Either the true photon position or the average of the
        pixelated electrons collected on a finite sensor can be
        chosen.
        """

        visitID = self.obs_metadata.OpsimMetaData['obshistID']
        file_name = self.centroid_base_name + str(visitID) + \
                    '_' + centroid_name + '.txt.gz'

        # Open the centroid file for this sensor with the gzip module to write
        # the centroid files in gzipped format.  Note the 'wt' which writes in
        # text mode which you must explicitly specify with gzip.
        self.centroid_handles[centroid_name] = gzip.open(file_name, 'wt')
        self.centroid_handles[centroid_name].write(
            '{:15} {:>15} {:>15} {:>10} {:>10} {:>11} {:>15}\n'.
            format('SourceID', 'Flux', 'Realized flux',
                   'xPix', 'yPix', 'flags', 'GalSimType'))

    def _writeObjectToCentroidFile(self, detector_name, bandpass_name, uniqueId,
                                   flux, realized_flux, xPix, yPix,
                                   obj_flags_value, object_type):
        """
        Write the flux and the the object position on the sensor for
        this object into a centroid file.  First check if a centroid
        file exists for this detector and, if it doesn't create it.

        @param [in] detector_name is the name of the sensor the
        gsObject falls on.

        @param [in] bandpass_name is the name of the filter used in
        this exposure.

        @param [in] uniqueId is the unique ID of the gsObject.

        @param [in] flux is the calculated flux for the gsObject in
        the given bandpass.

        @param [in] realized_flux is the Poisson realization of the object flux.

        @param [in] xPix x-pixel coordinate of object.

        @param [in] yPix y-pixel coordinate of object.

        @param [in] obj_flags_value is the bit flags for the object
              handling composed as an integer.

        @param [in] object_type is the gsObject.galSimType
        """
        centroid_name = detector_name + '_' + bandpass_name

        # If we haven't seen this sensor before open a centroid file for it.
        if centroid_name not in self.centroid_handles:
            self.open_centroid_file(centroid_name)

        # Write the object to the file
        self.centroid_handles[centroid_name].write(
            '{:<15} {:15.5f} {:15.5f} {:10.2f} {:10.2f} {:11d} {:>15}\n'.
            format(uniqueId, flux, realized_flux, xPix, yPix,
                   obj_flags_value, object_type))

    def write_centroid_files(self):
        """
        Write the centroid data structure out to the files.

        This function loops over the entries in the centroid list and
        then sends them each to be writen to a file. The
        _writeObjectToCentroidFile will decide how to put them in files.

        After writing the files are closed.
        """
        # Loop over entries
        for centroid_tuple in self.centroid_list:
            self._writeObjectToCentroidFile(*centroid_tuple)

        # Now close the centroid files.
        for name in self.centroid_handles:
            self.centroid_handles[name].close()

    def write_checkpoint(self, force=False, object_list=None):
        """
        Write a pickle file of detector images packaged with the
        objects that have been drawn. By default, write the checkpoint
        every self.nobj_checkpoint objects.
        """
        if self.checkpoint_file is None:
            return
        if force or len(self.drawn_objects) % self.nobj_checkpoint == 0:
            # The galsim.Image in self.detectorImage cannot be
            # pickled because it contains references to unpickleable
            # afw objects, so just save the array data and rebuild
            # the galsim.Image from scratch, given the detector name.
            images = {self.getFileName(): self.detectorImage}
            drawn_objects = self.drawn_objects if object_list is None \
                            else object_list
            image_state = dict(images=images,
                               rng=self._rng,
                               drawn_objects=drawn_objects,
                               centroid_objects=self.centroid_list)
            with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                             dir='.') as tmp:
                pickle.dump(image_state, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
                os.chmod(tmp.name, 0o660)
            os.rename(tmp.name, self.checkpoint_file)

    def restore_checkpoint(self, camera_wrapper, phot_params, obs_metadata,
                           epoch=2000.0):
        """
        Restore self.detectorImage, self._rng, and self.drawn_objects states
        from the checkpoint file.

        Parameters
        ----------
        camera_wrapper: lsst.sims.GalSimInterface.GalSimCameraWrapper
            An object representing the camera being simulated

        phot_params: lsst.sims.photUtils.PhotometricParameters
            An object containing the physical parameters representing
            the photometric properties of the system

        obs_metadata: lsst.sims.utils.ObservationMetaData
            Characterizing the pointing of the telescope

        epoch: float
            Representing the Julian epoch against which RA, Dec are
            reckoned (default = 2000)
        """
        if (self.checkpoint_file is None
            or not os.path.isfile(self.checkpoint_file)):
            return
        with open(self.checkpoint_file, 'rb') as input_:
            image_state = pickle.load(input_)
            images = image_state['images']
            # Loop over images dict for backwards compatibility, but
            # there should only be exactly one entry, so raise an
            # exception if not.
            if len(images) != 1:
                raise RuntimeError(f'{len(images)} images found '
                                   f'in {self.checkpoint_file}.')
            for key in images:
                # Unmangle the detector name.
                detname = "R:{},{} S:{},{}".format(*tuple(key[1:3] + key[5:7]))
                # Create the galsim.Image from scratch as a blank image and
                # set the pixel data from the persisted image data array.
                detector = make_galsim_detector(camera_wrapper, detname,
                                                phot_params, obs_metadata,
                                                epoch=epoch)
                self.detectorImage = self.blankImage()
                self.detectorImage += image_state['images'][key]
            self._rng = image_state['rng']
            self.drawn_objects = image_state['drawn_objects']
            self.centroid_list = image_state['centroid_objects']

    def getHourAngle(self, mjd, ra):
        """
        Compute the local hour angle of an object for the specified
        MJD and RA.

        Parameters
        ----------
        mjd: float
            Modified Julian Date of the observation.
        ra: float
            Right Ascension (in degrees) of the object.

        Returns
        -------
        float: hour angle in degrees
        """
        time = astropy.time.Time(mjd, format='mjd',
                                 location=self.observatory.getLocation())
        # Get the local apparent sidereal time.
        last = time.sidereal_time('apparent').degree
        ha = last - ra
        return ha

    @property
    def observatory(self):
        """Return the observatory object."""
        return self._observatory


def getGoodPhotImageSize(obj, keep_sb_level, pixel_scale=0.2):
    """
    Get a postage stamp size (appropriate for photon-shooting) given a
    minimum surface brightness in photons/pixel out to which to
    extend the stamp region.

    Parameters
    ----------
    obj: galsim.GSObject
        The GalSim object for which we will call .drawImage.
    keep_sb_level: float
        The minimum surface brightness (photons/pixel) out to which to
        extend the postage stamp, e.g., a value of
        sqrt(sky_bg_per_pixel)/3 would be 1/3 the Poisson noise
        per pixel from the sky background.
    pixel_scale: float [0.2]
        The CCD pixel scale in arcsec.

    Returns
    -------
    int: The length N of the desired NxN postage stamp.

    Notes
    -----
    Use of this function should be avoided with PSF implementations that
    are costly to evaluate.  A roughly equivalent DoubleGaussian
    could be used as a proxy.

    This function was originally written by Mike Jarvis.
    """
    # The factor by which to adjust N in each step.
    factor = 1.1

    # Start with the normal image size from GalSim
    N = obj.getGoodImageSize(pixel_scale)
    #print('N = ',N)

    if isinstance(obj, galsim.RandomKnots):
        # If the galaxy is a RandomWalk, extract the underlying
        # profile for this calculation rather than using the knotty
        # version, which will pose problems for the xValue function.
        obj = obj._profile

    # This can be too small for bright stars, so increase it in steps
    # until the edges are all below the requested sb level.  (Don't go
    # bigger than 4096)
    Nmax = 4096
    while N < Nmax:
        # Check the edges and corners of the current square
        h = N / 2 * pixel_scale
        xvalues = [obj.xValue(h, 0), obj.xValue(-h, 0),
                   obj.xValue(0, h), obj.xValue(0, -h),
                   obj.xValue(h, h), obj.xValue(h, -h),
                   obj.xValue(-h, h), obj.xValue(-h, -h)]
        maxval = np.max(xvalues)
        #print(N, maxval)
        if maxval < keep_sb_level:
            break
        N *= factor

    N = min(N, Nmax)

    # This can be quite huge for Devauc profiles, but we don't
    # actually have much surface brightness way out in the wings.  So
    # cut it back some.  (Don't go below 64 though.)
    while N >= 64 * factor:
        # Check the edges and corners of a square smaller by a factor of N.
        h = N / (2 * factor) * pixel_scale
        xvalues = [obj.xValue(h, 0), obj.xValue(-h, 0),
                   obj.xValue(0, h), obj.xValue(0, -h),
                   obj.xValue(h, h), obj.xValue(h, -h),
                   obj.xValue(-h, h), obj.xValue(-h, -h)]
        maxval = np.max(xvalues)
        #print(N, maxval)
        if maxval > keep_sb_level:
            break
        N /= factor

    return int(N)


class ObjectFlags:
    """
    Class to keep track of the object rendering bit flags. The bits
    will be composed as an int for storing in centroid files.
    """
    def __init__(self, conditions=('skipped', 'simple_sed', 'no_silicon',
                                   'fft_rendered')):
        """
        Parameters
        ----------
        conditions: list or tuple
            The sequence of strings describing the various conditions
            to be tracked.  The order will determine how the bits
            are assigned, so it should be a well-ordered sequence, i.e.,
            specifically a list or a tuple.
        """
        if type(conditions) not in (list, tuple):
            raise TypeError("conditions must be a list or a tuple")
        if len(conditions) != len(set(conditions)):
            raise ValueError("conditions must contain unique entries")
        self.flags = {condition: 1<<shift for shift, condition in
                      enumerate(conditions)}
        self.value = 0

    def set_flag(self, condition):
        """
        Set the bit associated with the specified condition.

        Parameters
        ----------
        condition: str
            A condition not in the known set will raise a ValueError.
        """
        try:
            self.value |= self.flags[condition]
        except KeyError:
            raise ValueError("unknown bit flag: %s" % condition)

    def unset_flag(self, condition):
        """
        Unset the bit associated with the specified condition.

        Parameters
        ----------
        condition: str
            A condition not in the known set will raise a ValueError.
        """
        try:
            self.value &= ~self.flags[condition]
        except KeyError:
            raise ValueError("unknown bit flag: %s" % condition)
