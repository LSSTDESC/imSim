
from functools import lru_cache
import numpy as np
import galsim
from galsim.config import StampBuilder, RegisterStampType

class LSST_SiliconBuilder(StampBuilder):
    """This performs the tasks necessary for building the stamp for a single object.

    It uses the regular Basic functions for most things.
    It specializes the quickSkip, buildProfile, and draw methods.
    """
    _ft_default = galsim.GSParams().folding_threshold
    _pixel_scale = 0.2
    _trivial_sed = galsim.SED(galsim.LookupTable([100, 2000], [1,1], interpolant='linear'),
                              wave_type='nm', flux_type='fphotons')
    _Nmax = 4096  # (Don't go bigger than 4096)

    def setup(self, config, base, xsize, ysize, ignore, logger):
        """
        Do the initialization and setup for building a postage stamp.

        In the base class, we check for and parse the appropriate size and position values in
        config (aka base['stamp'] or base['image'].

        Values given in base['stamp'] take precedence if these are given in both places (which
        would be confusing, so probably shouldn't do that, but there might be a use case where it
        would make sense).

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            xsize:      The xsize of the image to build (if known).
            ysize:      The ysize of the image to build (if known).
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here. i.e. it won't be an error if these parameters are present.
            logger:     A logger object to log progress.

        Returns:
            xsize, ysize, image_pos, world_pos
        """
        gal = galsim.config.BuildGSObject(base, 'gal', logger=logger)[0]
        if gal is None:
            raise galsim.config.SkipThisObject('gal is None (invalid parameters)')
        self.gal = gal

        # Compute or retrieve the realized flux.
        self.rng = galsim.config.GetRNG(config, base, logger, "LSST_Silicon")
        bandpass = base['bandpass']
        if hasattr(gal, 'flux'):
            flux = gal.flux
        else:
            # TODO: This calculation is slow, so we'd like to avoid
            # doing here it for faint objects.  Eventually, this call
            # will be replaced by retrieving precomputed/cached values
            # via skyCatalogs.
            flux = gal.calculateFlux(bandpass)
        self.realized_flux = galsim.PoissonDeviate(self.rng, mean=flux)()

        # Check if the realized flux is 0.
        if self.realized_flux == 0:
            # If so, we'll skip everything after this.
            # The mechanism within GalSim to do this is to raise a special SkipThisObject class.
            raise galsim.config.SkipThisObject('realized flux=0')

        # Otherwise figure out the stamp size
        if self.realized_flux < 10:
            # For really faint things, don't try too hard.  Just use 32x32.
            image_size = 32

        elif 'size' in config:
            # Get the stamp size from the config entry.
            image_size = galsim.config.ParseValue(config, 'size', base, int)[0]

        elif (hasattr(gal, 'original') and isinstance(gal.original, galsim.DeltaFunction)):
            # For bright stars, set the folding threshold for the
            # stamp size calculation.  Use a
            # Kolmogorov_and_Gaussian_PSF since it is faster to
            # evaluate than an AtmosphericPSF.
            base['current_noise_image'] = base['current_image']
            noise_var = galsim.config.CalculateNoiseVariance(base)
            folding_threshold = noise_var/self.realized_flux
            if folding_threshold >= self._ft_default or folding_threshold == 0:
                # a) Don't gratuitously raise folding_threshold above the normal default.
                # b) If sky_level = 0, then folding_threshold=0.  This is bad (stepk=0 below),
                #    but if the user is doing this to avoid sky noise, then they probably care about
                #    other things than detailed large-scale behavior of very bright stars.
                gsparams = None
            else:
                # Every different folding threshold requires a new initialization of Kolmogorov,
                # which takes about a second.  So round down to the nearest e folding to
                # minimize how many of these we need to do.
                folding_threshold = np.exp(np.floor(np.log(folding_threshold)))
                logger.debug('Using folding_threshold %s',folding_threshold)
                logger.debug('From: noise_var = %s, flux = %s',noise_var,self.realized_flux)
                gsparams = galsim.GSParams(folding_threshold=folding_threshold)

            md = galsim.config.GetInputObj('opsim_meta_dict', config, base, 'OpsimMeta')
            psf = self.Kolmogorov_and_Gaussian_PSF(gsparams=gsparams,
                                                   airmass=md.get('airmass'),
                                                   rawSeeing=md.get('rawSeeing'),
                                                   band=md.get('band'))
            image_size = psf.getGoodImageSize(self._pixel_scale)
            # No point in this being larger than a CCD.  Cut back to Nmax if larger than this.
            image_size = min(image_size, self._Nmax)

        else:
            # For extended objects, recreate the object to draw, but
            # convolved with the faster DoubleGaussian PSF.
            psf = self.DoubleGaussian()
            # For Chromatic objects, need to evaluate at the
            # effective wavelength of the bandpass.
            gal_achrom = gal.evaluateAtWavelength(bandpass.effective_wavelength)
            obj = galsim.Convolve(gal_achrom, psf).withFlux(self.realized_flux)

            # Start with GalSim's estimate of a good box size.
            image_size = obj.getGoodImageSize(self._pixel_scale)

            # Find a postage stamp region to draw onto.  Use (sky noise)/3. as the nominal
            # minimum surface brightness for rendering an extended object.
            base['current_noise_image'] = base['current_image']
            noise_var = galsim.config.CalculateNoiseVariance(base)
            keep_sb_level = np.sqrt(noise_var)/3.
            self._large_object_sb_level = 3*keep_sb_level

            gal_at_eff_wl = gal.evaluateAtWavelength(bandpass.effective_wavelength)
            # For bright things, defined as having an average of at least 10 photons per
            # pixel on average, try to be careful about not truncating the surface brightness
            # at the edge of the box.
            if self.realized_flux > 10 * image_size**2:
                image_size = self._getGoodPhotImageSize([gal_at_eff_wl, psf], keep_sb_level,
                                                        pixel_scale=self._pixel_scale)

            # If the above size comes out really huge, scale back to what you get for
            # a somewhat brighter surface brightness limit.
            if image_size > self._Nmax:
                image_size = self._getGoodPhotImageSize([gal_at_eff_wl, psf], self._large_object_sb_level,
                                                        pixel_scale=self._pixel_scale)
                image_size = min(image_size, self._Nmax)

        logger.info('Object %d will use stamp size = %s',base.get('obj_num',0),image_size)

        # Also the position
        world_pos = galsim.config.ParseWorldPos(config, 'world_pos', base, logger)
        image_pos = None  # GalSim will figure this out from the wcs.

        return image_size, image_size, image_pos, world_pos

    def _getGoodPhotImageSize(self, obj_list, keep_sb_level, pixel_scale):
        sizes = [self._getGoodPhotImageSize1(obj, keep_sb_level, pixel_scale)
                 for obj in obj_list]
        return int(np.sqrt(np.sum([size**2 for size in sizes])))

    def _getGoodPhotImageSize1(self, obj, keep_sb_level, pixel_scale):
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

        if (isinstance(obj, galsim.Sum) and
            any([isinstance(_.original, galsim.RandomKnots)
                 for _ in obj.obj_list])):
            # obj is a galsim.Sum object and contains a
            # galsim.RandomKnots component, so make a new obj that's
            # the sum of the non-knotty versions.
            obj_list = []
            for item in obj.obj_list:
                if isinstance(item.original, galsim.RandomKnots):
                    obj_list.append(item.original._profile)
                else:
                    obj_list.append(item)
            obj = galsim.Add(obj_list)
        elif isinstance(obj.original, galsim.RandomKnots):
            # Handle RandomKnots object directly
            obj = obj.original._profile

        # This can be too small for bright stars, so increase it in steps until the edges are
        # all below the requested sb level.
        while N < self._Nmax:
            # Check the edges and corners of the current square
            h = N / 2 * pixel_scale
            xvalues = [ obj.xValue(h,0), obj.xValue(-h,0),
                        obj.xValue(0,h), obj.xValue(0,-h),
                        obj.xValue(h,h), obj.xValue(h,-h),
                        obj.xValue(-h,h), obj.xValue(-h,-h) ]
            maxval = np.max(xvalues)
            if maxval < keep_sb_level:
                break
            N *= factor

        N = min(N, self._Nmax)

        # This can be quite huge for Devauc profiles, but we don't actually have much
        # surface brightness way out in the wings.  So cut it back some.
        # (Don't go below 64 though.)
        while N >= 64 * factor:
            # Check the edges and corners of a square smaller by a factor of N.
            h = N / (2 * factor) * pixel_scale
            xvalues = [ obj.xValue(h,0), obj.xValue(-h,0),
                        obj.xValue(0,h), obj.xValue(0,-h),
                        obj.xValue(h,h), obj.xValue(h,-h),
                        obj.xValue(-h,h), obj.xValue(-h,-h) ]
            maxval = np.max(xvalues)
            if maxval > keep_sb_level:
                break
            N /= factor

        return int(N)

    @lru_cache(maxsize=128)
    def DoubleGaussian(self, fwhm1=0.6, fwhm2=0.12, wgt1=1.0, wgt2=0.1):
        """
        @param [in] fwhm1 is the Full Width at Half Max of the first Gaussian in arcseconds

        @param [in] fwhm2 is the Full Width at Half Max of the second Gaussian in arcseconds

        @param [in] wgt1 is the dimensionless coefficient normalizing the first Gaussian

        @param [in] wgt2 is the dimensionless coefficient normalizing the second Gaussian

        The total PSF will be

        (wgt1 * G(sig1) + wgt2 * G(sig2))/(wgt1 + wgt2)

        where G(sigN) denotes a normalized Gaussian with a standard deviation that gives
        a Full Width at Half Max of fwhmN.  (Integrating a two-dimensional Gaussian, we find
        that sig = fwhm/2.355)

        Because this PSF depends on neither position nor wavelength, this __init__ method
        will instantiate a PSF and cache it.  It is this cached psf that will be returned
        whenever _getPSF is called in this class.
        """

        r1 = fwhm1/2.355
        r2 = fwhm2/2.355
        norm = 1.0/(wgt1 + wgt2)

        gaussian1 = galsim.Gaussian(sigma=r1)
        gaussian2 = galsim.Gaussian(sigma=r2)

        return norm*(wgt1*gaussian1 + wgt2*gaussian2)

    @lru_cache(maxsize=128)
    def Kolmogorov_and_Gaussian_PSF(self, airmass=1.2, rawSeeing=0.7, band='r', gsparams=None):
        """
        This PSF class is based on David Kirkby's presentation to the DESC Survey Simulations
        working group on 23 March 2017.

        https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=LSSTDESC&title=SSim+2017-03-23

        (you will need a SLAC Confluence account to access that link)

        Parameters
        ----------
        airmass

        rawSeeing is the FWHM seeing at zenith at 500 nm in arc seconds
        (provided by OpSim)

        band is the bandpass of the observation [u,g,r,i,z,y]
        """
        # This code was provided by David Kirkby in a private communication

        wlen_eff = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
        # wlen_eff is from Table 2 of LSE-40 (y=y2)

        FWHMatm = rawSeeing * (wlen_eff / 500.) ** -0.3 * airmass ** 0.6
        # From LSST-20160 eqn (4.1)

        FWHMsys = np.sqrt(0.25**2 + 0.3**2 + 0.08**2) * airmass ** 0.6
        # From LSST-20160 eqn (4.2)

        atm = galsim.Kolmogorov(fwhm=FWHMatm, gsparams=gsparams)
        sys = galsim.Gaussian(fwhm=FWHMsys, gsparams=gsparams)
        return galsim.Convolve((atm, sys))

    def buildPSF(self, config, base, gsparams, logger):
        """Build the PSF object.

        For the Basic stamp type, this builds a PSF from the base['psf'] dict, if present,
        else returns None.

        Parameters:
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            gsparams:   A dict of kwargs to use for a GSParams.  More may be added to this
                        list by the galaxy object.
            logger:     A logger object to log progress.

        Returns:
            the PSF
        """
        psf = galsim.config.BuildGSObject(base, 'psf', gsparams=gsparams, logger=logger)[0]

        # For very bright things, we might want to change this for FFT drawing.
        if 'fft_sb_thresh' in config:
            fft_sb_thresh = galsim.config.ParseValue(config,'fft_sb_thresh',base,float)[0]
        else:
            fft_sb_thresh = 0.
        if self.realized_flux < 1.e6 or not fft_sb_thresh or self.realized_flux < fft_sb_thresh:
            self.use_fft = False
            return psf

        # Otherwise (high flux object), we might want to switch to fft.  So be a little careful.
        psf = galsim.config.BuildGSObject(base, 'psf', logger=logger)[0]
        fft_psf = self.make_fft_psf(psf, logger)
        logger.warning('Object %d has flux = %s.  Check if we should switch to FFT',
                       base['obj_num'], self.realized_flux)

        # Now this object should have a much better estimate of the real maximum surface brightness
        # than the original psf did.
        # However, the max_sb feature gives an over-estimate, whereas to be conservative, we would
        # rather an under-estimate.  For this kind of profile, dividing by 2 does a good job
        # of giving us an underestimate of the max surface brightness.
        # Also note that `max_sb` is in photons/arcsec^2, so multiply by pixel_scale**2
        # to get photons/pixel, which we compare to fft_sb_thresh.
        bandpass = base['bandpass']
        gal_achrom = self.gal.evaluateAtWavelength(bandpass.effective_wavelength)
        fft_obj = galsim.Convolve(gal_achrom, fft_psf).withFlux(self.realized_flux)
        max_sb = fft_obj.max_sb/2. * self._pixel_scale**2
        logger.debug('max_sb = %s. cf. %s',max_sb,fft_sb_thresh)
        if max_sb > fft_sb_thresh:
            self.use_fft = True
            logger.warning('Yes. Use FFT for this object.  max_sb = %.0f > %.0f',
                           max_sb, fft_sb_thresh)
            return fft_psf
        else:
            self.use_fft = False
            logger.warning('No. Use photon shooting.  max_sb = %.0f <= %.0f',
                           max_sb, fft_sb_thresh)
            return psf

    def make_fft_psf(self, psf, logger):
        """Swap out any PhaseScreenPSF component with a roughly equivalent analytic approximation.
        """
        if isinstance(psf, galsim.Transformation):
            return galsim.Transformation(self.make_fft_psf(psf.original, logger),
                                         psf.jac, psf.offset, psf.flux_ratio, psf.gsparams)
        elif isinstance(psf, galsim.Convolution):
            obj_list = [self.make_fft_psf(p, logger) for p in psf.obj_list]
            return galsim.Convolution(obj_list, gsparams=psf.gsparams)
        elif isinstance(psf, galsim.PhaseScreenPSF):
            # If psf is a PhaseScreenPSF, then make a simpler one the just convolves
            # a Kolmogorov profile with an OpticalPSF.
            r0_500 = psf.screen_list.r0_500_effective
            atm_psf = galsim.Kolmogorov(lam=psf.lam, r0_500=r0_500,
                                        gsparams=psf.gsparams)

            opt_screens = [s for s in psf.screen_list if isinstance(s, galsim.OpticalScreen)]
            logger.info('opt_screens = %r',opt_screens)
            if len(opt_screens) >= 1:
                # Should never be more than 1, but if there weirdly is, just use the first.
                opt_screen = opt_screens[0]
                optical_psf = galsim.OpticalPSF(
                        lam=psf.lam,
                        diam=opt_screen.diam,
                        aberrations=opt_screen.aberrations,
                        annular_zernike=opt_screen.annular_zernike,
                        obscuration=opt_screen.obscuration,
                        gsparams=psf.gsparams)
                return galsim.Convolve([atm_psf, optical_psf], gsparams=psf.gsparams)
            else:
                return atm_psf
        else:
            return psf

    def getDrawMethod(self, config, base, logger):
        """Determine the draw method to use.

        @param config       The configuration dict for the stamp field.
        @param base         The base configuration dict.
        @param logger       A logger object to log progress.

        @returns method
        """
        method = galsim.config.ParseValue(config,'draw_method',base,str)[0]
        if method not in galsim.config.valid_draw_methods:
            raise galsim.GalSimConfigValueError("Invalid draw_method.", method,
                                                galsim.config.valid_draw_methods)
        if method  == 'auto':
            if  self.use_fft:
                logger.info('Auto -> Use FFT drawing for object %d.',base['obj_num'])
                return 'fft'
            else:
                logger.info('Auto -> Use photon shooting for object %d.',base['obj_num'])
                return 'phot'
        else:
            # If user sets something specific for the method, rather than auto,
            # then respect their wishes.
            logger.info('Use specified method=%s for object %d.',method,base['obj_num'])
            return method

    def draw(self, prof, image, method, offset, config, base, logger):
        """Draw the profile on the postage stamp image.

        Parameters:
            prof:       The profile to draw.
            image:      The image onto which to draw the profile (which may be None).
            method:     The method to use in drawImage.
            offset:     The offset to apply when drawing.
            config:     The configuration dict for the stamp field.
            base:       The base configuration dict.
            logger:     A logger object to log progress.

        Returns:
            the resulting image
        """
        if prof is None:
            # If was decide to do any rejection steps, this could be set to None, in which case,
            # don't draw anything.
            return image

        def fix_seds(prof):
            # If any SEDs are not currently using a LookupTable for the function or if they are
            # using spline interpolation, then the codepath is quite slow.
            # Better to fix them before doing WavelengthSampler.
            if isinstance(prof, galsim.ChromaticObject):
                wave_list, _, _ = galsim.utilities.combine_wave_list(prof.SED, bandpass)
                sed = prof.SED
                # TODO: This bit should probably be ported back to Galsim.
                #       Something like sed.make_tabulated()
                if (not isinstance(sed._spec, galsim.LookupTable)
                    or sed._spec.interpolant != 'linear'):
                    new_spec = galsim.LookupTable(wave_list, sed(wave_list), interpolant='linear')
                    new_sed = galsim.SED(new_spec, 'nm', 'fphotons')
                    prof.SED = new_sed

                # Also recurse onto any components.
                if hasattr(prof, 'obj_list'):
                    for obj in prof.obj_list:
                        fix_seds(obj)
                if hasattr(prof, 'original'):
                    fix_seds(prof.original)

        max_flux_simple = config.get('max_flux_simple', 100)
        faint = self.realized_flux < max_flux_simple
        bandpass = base['bandpass']
        if faint:
            logger.info("Flux = %.0f  Using trivial sed", self.realized_flux)
            prof = prof.evaluateAtWavelength(bandpass.effective_wavelength)
            prof = prof * self._trivial_sed
        else:
            fix_seds(prof)
        prof = prof.withFlux(self.realized_flux, bandpass)

        wcs = base['wcs']

        if method == 'fft':
            # When drawing with FFTs, large offsets can be a problem, since they
            # can blow up the required FFT size.  We'll guard for that below with
            # a try block, but we can minimize how often this happens by making sure
            # the offset is close to 0,0.
            if abs(offset.x) > 2 or abs(offset.y) > 2:
                # Make a larger image that has the object near the center.
                fft_image = galsim.Image(full_bounds, dtype=image.dtype)
                fft_image[image.bounds] = image
                fft_offset = image_pos - full_bounds.true_center
            else:
                fft_image = image.copy()
                fft_offset = offset

            try:
                prof.drawImage(bandpass,
                               method='fft',
                               offset=fft_offset,
                               image=fft_image,
                               wcs=wcs)
            except galsim.errors.GalSimFFTSizeError as e:
                # I think this shouldn't happen with the updates I made to how the image size
                # is calculated, even for extremely bright things.  So it should be ok to
                # just report what happened, give some extra information to diagonose the problem
                # and raise the error.
                logger.error('Caught error trying to draw using FFT:')
                logger.error('%s',e)
                logger.error('You may need to add a gsparams field with maximum_fft_size to')
                logger.error('either the psf or gal field to allow larger FFTs.')
                logger.info('prof = %r',prof)
                logger.info('fft_image = %s',fft_image)
                logger.info('offset = %r',offset)
                logger.info('wcs = %r',wcs)
                raise
            else:
                # Some pixels can end up negative from FFT numerics.  Just set them to 0.
                fft_image.array[fft_image.array < 0] = 0.
                fft_image.addNoise(galsim.PoissonNoise(rng=self.rng))
                # In case we had to make a bigger image, just copy the part we need.
                image += fft_image[image.bounds]

        else:
            if not faint and 'photon_ops' in config:
                photon_ops = galsim.config.BuildPhotonOps(config, 'photon_ops', base, logger)
            else:
                photon_ops = []

            if faint:
                sensor = None
            else:
                sensor = base.get('sensor', None)
                if sensor is not None:
                    sensor.updateRNG(self.rng)

            prof.drawImage(bandpass,
                           method='phot',
                           offset=offset,
                           rng=self.rng,
                           maxN=int(1e6),
                           n_photons=self.realized_flux,
                           image=image,
                           wcs=wcs,
                           sensor=sensor,
                           photon_ops=photon_ops,
                           add_to_image=True,
                           poisson_flux=False)
        return image


# Register this as a valid stamp type
RegisterStampType('LSST_Silicon', LSST_SiliconBuilder())
