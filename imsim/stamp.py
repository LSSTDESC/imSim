from enum import Enum, auto
from functools import lru_cache
from dataclasses import dataclass, fields, MISSING
import numpy as np
import galsim
from galsim.config import StampBuilder, RegisterStampType, GetAllParams, GetInputObj
from lsst.afw import cameraGeom
from lsst.obs.lsst.translators.lsst import SIMONYI_LOCATION as RUBIN_LOC

from .diffraction_fft import apply_diffraction_psf
from .camera import get_camera


class ProcessingMode(Enum):
    FFT = auto()
    PHOT = auto()
    FAINT = auto()


@dataclass
class StellarObject:
    """Cache for quantities of a single object, which need to be computed
    when determining the rendering mode of the object.

    `LSST_PhotonPoolingImage` will store `StellarObject` instances in
    `base._objects` when determining the rendering mode and reuse the data
    in the rendering stage."""
    index: int
    gal: object
    psf: object
    phot_flux: float
    fft_flux: float
    mode: ProcessingMode


@dataclass
class DiffractionFFT:
    """Config subsection to enable diffraction in FFT mode."""
    exptime: float
    azimuth: galsim.Angle
    altitude: galsim.Angle
    rotTelPos: galsim.Angle
    spike_length_cutoff: float = 4000
    brightness_threshold: float = 1.0e5
    latitude: galsim.Angle = RUBIN_LOC.lat
    enabled: bool = True

    def apply(self, image: galsim.Image, wavelength: float) -> None:
        if self.enabled:
            apply_diffraction_psf(
                image.array,
                wavelength=wavelength,
                rottelpos=self.rotTelPos.rad,
                exptime=self.exptime,
                latitude=self.latitude.rad,
                azimuth=self.azimuth.rad,
                altitude=self.altitude.rad,
                brightness_threshold=self.brightness_threshold,
                spike_length_cutoff=self.spike_length_cutoff,
            )

    @classmethod
    def from_config(cls, config: dict, base: dict) -> "DiffractionFFT":
        """Create a DiffractionFFT from config values."""
        req = {f.name: f.type for f in fields(cls) if f.default is MISSING}
        opt = {f.name: f.type for f in fields(cls) if f.default is not MISSING}
        kwargs, _safe = GetAllParams(config, base, req=req, opt=opt)
        return cls(**kwargs)

def build_gal(base, logger):
    gal, _ = galsim.config.BuildGSObject(base, 'gal', logger=logger)
    if gal is None:
        return None
    if not hasattr(gal, 'flux'):
        # In this case, the object flux has not been precomputed
        # or cached by the skyCatalogs code.
        gal.flux = gal.calculateFlux(base['bandpass'])
    if gal.flux == 0.:
        return None
    return gal


def build_obj(stamp_config, base, logger):
    """Precompute all data needed to determine the rendering mode of an
    object."""
    builder = LSST_SiliconBuilder()
    try:
        xsize, ysize, image_pos, world_pos = builder.setup(stamp_config, base, 0.0, 0.0, galsim.config.stamp.stamp_ignore, logger)
    except galsim.SkipThisObject:
        return None
    builder.locateStamp(stamp_config, base, xsize, ysize, image_pos, world_pos, logger)
    psf = builder.buildPSF(stamp_config, base, None, logger)
    max_flux_simple = stamp_config.get('max_flux_simple', 100)
    if builder.use_fft:
        mode = ProcessingMode.FFT
    elif builder.nominal_flux < max_flux_simple:
        mode = ProcessingMode.FAINT
    else:
        mode = ProcessingMode.PHOT

    return StellarObject(base.get('obj_num', 0), builder.gal, psf, phot_flux=builder.phot_flux, fft_flux=builder.fft_flux, mode=mode)


class StampBuilderBase(StampBuilder):
    """Shared functionality between PhotonStampBuilder and LSST_SiliconBuilder."""
    _ft_default = galsim.GSParams().folding_threshold
    _pixel_scale = 0.2
    _trivial_sed = galsim.SED(galsim.LookupTable([100, 2000], [1,1], interpolant='linear'),
                              wave_type='nm', flux_type='fphotons')
    _tiny_flux = 10
    _Nmax = 4096  # (Don't go bigger than 4096)
    _sed_logged = False  # Only log SED linear interpolant warning once.

    @classmethod
    def _fix_seds_24(cls, prof, bandpass, logger):
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
                if not cls._sed_logged:
                    logger.warning(
                            "Warning: Chromatic drawing is most efficient when SEDs have "
                            "interpont='linear'. Switching LookupTables to use 'linear'.")
                    cls._sed_logged = True
                # Workaround for https://github.com/GalSim-developers/GalSim/issues/1228
                f = np.broadcast_to(sed(wave_list), wave_list.shape)
                new_spec = galsim.LookupTable(wave_list, f, interpolant='linear')
                new_sed = galsim.SED(
                    new_spec,
                    'nm',
                    'fphotons' if sed.spectral else '1'
                )
                prof.SED = new_sed

            # Also recurse onto any components.
            if hasattr(prof, 'obj_list'):
                for obj in prof.obj_list:
                    cls._fix_seds_24(obj, bandpass, logger)
            if hasattr(prof, 'original'):
                cls._fix_seds_24(prof.original, bandpass, logger)

    @classmethod
    def _fix_seds_25(cls, prof, bandpass, logger):
        # If any SEDs are not currently using a LookupTable for the function or if they are
        # using spline interpolation, then the codepath is quite slow.
        # Better to fix them before doing WavelengthSampler.

        # In GalSim 2.5, SEDs are not necessarily constructed in most chromatic objects.
        # And really the only ones we need to worry about are the ones that come from
        # SkyCatalog, since they might not have linear interpolants.
        # Those objects are always SimpleChromaticTransformations.  So only fix those.
        if (isinstance(prof, galsim.SimpleChromaticTransformation) and
            (not isinstance(prof._flux_ratio._spec, galsim.LookupTable)
             or prof._flux_ratio._spec.interpolant != 'linear')):
            if not cls._sed_logged:
                logger.warning(
                        "Warning: Chromatic drawing is most efficient when SEDs have "
                        "interpont='linear'. Switching LookupTables to use 'linear'.")
                cls._sed_logged = True
            sed = prof._flux_ratio
            wave_list, _, _ = galsim.utilities.combine_wave_list(sed, bandpass)
            f = np.broadcast_to(sed(wave_list), wave_list.shape)
            new_spec = galsim.LookupTable(wave_list, f, interpolant='linear')
            new_sed = galsim.SED(
                new_spec,
                'nm',
                'fphotons' if sed.spectral else '1'
            )
            prof._flux_ratio = new_sed

        # Also recurse onto any components.
        if isinstance(prof, galsim.ChromaticObject):
            if hasattr(prof, 'obj_list'):
                for obj in prof.obj_list:
                    cls._fix_seds_25(obj, bandpass, logger)
            if hasattr(prof, 'original'):
                cls._fix_seds_25(prof.original, bandpass, logger)

    # Pick the right function to be _fix_seds.
    _fix_seds = _fix_seds_24 if galsim.__version_info__ < (2,5) else _fix_seds_25


class PhotonStampBuilder(StampBuilderBase):
    """StampBuilder which builds photons instead of image stamps."""
    def setup(self, config, base, xsize, ysize, ignore, logger):
        return LSST_SiliconBuilder().setup(config, base, xsize, ysize, ignore, logger)

    def getDrawMethod(self, config, base, logger):
        return "phot"

    def updateOrigin(self, stamp, config, image):
        return

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

        # Prof is normally a convolution here with obj_list being [gal, psf1, psf2,...]
        # for some number of component PSFs.
        gal, *psfs = prof.obj_list if hasattr(prof,'obj_list') else [prof]
        obj_num = base.get('obj_num',0)
        stellar_obj = base.get("_objects", {})[obj_num] # Use cached object
        bandpass = base['bandpass']

        faint = stellar_obj.mode == ProcessingMode.FAINT
        if faint:
            logger.info("Flux = %.0f  Using trivial sed", stellar_obj.gal.flux)
            gal = gal.evaluateAtWavelength(bandpass.effective_wavelength)
            gal = gal * self._trivial_sed
        else:
            self._fix_seds(gal, bandpass, logger)
        gal = gal.withFlux(stellar_obj.phot_flux, bandpass)

        # Put the psfs at the start of the photon_ops.
        # Probably a little better to put them a bit later than the start in some cases
        # (e.g. after TimeSampler, PupilAnnulusSampler), but leave that as a todo for now.
        rng = galsim.config.GetRNG(config, base, logger, "LSST_Silicon")
        gal.drawImage(bandpass,
                      method='phot',
                      offset=offset,
                      rng=rng,
                      maxN=None,
                      n_photons=stellar_obj.phot_flux,
                      image=image,
                      wcs=base['wcs'],
                      sensor=NullSensor(), # Prevent premature photon accumulation
                      photon_ops=psfs,
                      add_to_image=True,
                      poisson_flux=False,
                      save_photons=True)
        img_pos = base["image_pos"]
        image.photons.x += img_pos.x
        image.photons.y += img_pos.y
        return image.photons

# Register this as a valid stamp type
RegisterStampType('PhotonStampBuilder', PhotonStampBuilder())


class LSST_SiliconBuilder(StampBuilderBase):
    """This performs the tasks necessary for building the stamp for a single object.

    It uses the regular Basic functions for most things.
    It specializes the quickSkip, buildProfile, and draw methods.
    """

    fft_flux: float = 0.

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

        # First do a parsing check to make sure that all the options passed to
        # the config are valid, and no required options are missing.

        try:
            self.vignetting = GetInputObj('vignetting', config, base, 'LSST_SiliconBuilder')
        except galsim.config.GalSimConfigError:
            self.vignetting = None

        req = {}
        opt = {'camera': str, 'diffraction_fft': dict,
               'airmass': float, 'rawSeeing': float, 'band': str}
        if self.vignetting:
            req['det_name'] = str
        else:
            opt['det_name'] = str
        # For the optional ones we parse manually, we can put in ignore, and they will
        # still be allowed, but not required.
        ignore = ['fft_sb_thresh', 'max_flux_simple'] + ignore
        params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore)[0]

        obj_num = base.get('obj_num',0)
        # Use cached object, if possible.
        # The cache is currently only used in the `LSST_PhotonPoolingImageBuilder`:
        stellar_obj = base.get("_objects", {}).get(obj_num)
        gal = build_gal(base, logger) if stellar_obj is None else stellar_obj.gal

        if gal is None:
            raise galsim.config.SkipThisObject('gal is None (invalid parameters)')
        self.gal = gal

        if 'diffraction_fft' in config:
            self.diffraction_fft = DiffractionFFT.from_config(config['diffraction_fft'], base)
        else:
            self.diffraction_fft = None

        # Compute or retrieve the realized flux.
        self.rng = galsim.config.GetRNG(config, base, logger, "LSST_Silicon")
        bandpass = base['bandpass']
        self.image = base['current_image']
        camera = get_camera(params.get('camera', 'LsstCam'))
        if self.vignetting:
            self.det = camera[params['det_name']]
        if stellar_obj is not None:
            self.nominal_flux = stellar_obj.gal.flux
            self.phot_flux = stellar_obj.phot_flux
        else:
            self.nominal_flux = gal.flux
            # For photon shooting rendering, precompute the realization of the Poisson variate.
            # Mostly so we can possibly abort early if phot_flux=0.
            self.phot_flux = galsim.PoissonDeviate(self.rng, mean=gal.flux)()

        # Save these later, in case needed for the output catalog.
        base['nominal_flux'] = self.nominal_flux
        base['phot_flux'] = self.phot_flux
        base['fft_flux'] = 0.  # For fft drawing, this will be updated in buildPSF.
        base['realized_flux'] = 0.  # Will update this after drawImage using im.added_flux

        # Check if the Poisson draw for the photon flux is 0.
        if self.phot_flux == 0:
            # If so, we'll skip everything after this.
            # The mechanism within GalSim to do this is to raise a special SkipThisObject class.
            raise galsim.config.SkipThisObject('phot_flux=0')

        # Otherwise figure out the stamp size
        if xsize > 0 and ysize > 0:
            pass  # Already set.

        elif self.nominal_flux < self._tiny_flux:
            # For really faint things, don't try too hard.  Just use 32x32.
            xsize = ysize = 32

        elif 'size' in config:
            # Get the stamp size from the config entry.
            xsize = ysize = galsim.config.ParseValue(config, 'size', base, int)[0]

        else:
            gal_achrom = gal.evaluateAtWavelength(bandpass.effective_wavelength)
            if (hasattr(gal_achrom, 'original')
                and isinstance(gal_achrom.original, galsim.DeltaFunction)):
                # For bright stars, set the folding threshold for the
                # stamp size calculation.  Use a
                # Kolmogorov_and_Gaussian_PSF since it is faster to
                # evaluate than an AtmosphericPSF.
                base['current_noise_image'] = base['current_image']
                noise_var = galsim.config.CalculateNoiseVariance(base)
                folding_threshold = noise_var/self.nominal_flux
                if folding_threshold >= self._ft_default or folding_threshold == 0:
                    # a) Don't gratuitously raise folding_threshold above the normal default.
                    # b) If sky_level = 0, then folding_threshold=0.  This is bad (stepk=0 below),
                    #    but if the user is doing this to avoid sky noise, then they probably care
                    #    about other things than detailed large-scale behavior of very bright stars.
                    gsparams = None
                else:
                    # Every different folding threshold requires a new initialization of Kolmogorov,
                    # which takes about a second.  So round down to the nearest e folding to
                    # minimize how many of these we need to do.
                    folding_threshold = np.exp(np.floor(np.log(folding_threshold)))
                    logger.debug('Using folding_threshold %s',folding_threshold)
                    logger.debug('From: noise_var = %s, flux = %s',noise_var,self.nominal_flux)
                    gsparams = galsim.GSParams(folding_threshold=folding_threshold)

                # Grab the three parameters we need for Kolmogorov_and_Gaussian_PSF.
                keys = ('airmass', 'rawSeeing', 'band')
                kwargs = { k:v for k,v in params.items() if k in keys }
                psf = self.Kolmogorov_and_Gaussian_PSF(gsparams=gsparams, **kwargs)
                image_size = psf.getGoodImageSize(self._pixel_scale)
                # No point in this being larger than a CCD.  Cut back to Nmax if larger than this.
                xsize = ysize = min(image_size, self._Nmax)

            else:
                # For extended objects, recreate the object to draw, but
                # convolved with the faster DoubleGaussian PSF.
                psf = self.DoubleGaussian()
                # For Chromatic objects, need to evaluate at the
                # effective wavelength of the bandpass.
                obj = galsim.Convolve(gal_achrom, psf).withFlux(self.nominal_flux)

                # Start with GalSim's estimate of a good box size.
                image_size = obj.getGoodImageSize(self._pixel_scale)

                # For bright things, defined as having an average of at least 10 photons per
                # pixel on average, or objects for which GalSim's estimate of the image_size is larger
                # than self._Nmax, compute the image_size using the surface brightness limit, trying
                # to be careful about not truncating the surface brightness
                # at the edge of the box.
                if (self.nominal_flux > 10 * image_size**2) or (image_size > self._Nmax):
                    # Find a postage stamp region to draw onto.  Use (sky noise)/8. as the nominal
                    # minimum surface brightness for rendering an extended object.
                    base['current_noise_image'] = base['current_image']
                    noise_var = galsim.config.CalculateNoiseVariance(base)
                    keep_sb_level = np.sqrt(noise_var)/8.
                    self._large_object_sb_level = 3*keep_sb_level
                    image_size = self._getGoodPhotImageSize([gal_achrom, psf], keep_sb_level,
                                                            pixel_scale=self._pixel_scale)

                    # If the above size comes out really huge, scale back to what you get for
                    # a somewhat brighter surface brightness limit.
                    if image_size > self._Nmax:
                        image_size = self._getGoodPhotImageSize([gal_achrom, psf],
                                                                self._large_object_sb_level,
                                                                pixel_scale=self._pixel_scale)
                        image_size = min(image_size, self._Nmax)
                xsize = ysize = image_size

        logger.info('Object %d will use stamp size = %s,%s', base.get('obj_num',0),
                    xsize, ysize)

        # Determine where this object is going to go:
        # This is the same as what the base StampBuilder does:
        if 'image_pos' in config:
            image_pos = galsim.config.ParseValue(config, 'image_pos', base, galsim.PositionD)[0]
        else:
            image_pos = None

        if 'world_pos' in config:
            world_pos = galsim.config.ParseWorldPos(config, 'world_pos', base, logger)
        else:
            world_pos = None

        return xsize, ysize, image_pos, world_pos

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
        # Use cached psf and mode (fft / phot):
        obj_num = base['obj_num']
        stellar_obj = base.get('_objects', {}).get(obj_num)
        if stellar_obj is not None:
            self.use_fft = stellar_obj.mode == ProcessingMode.FFT
            if self.use_fft:
                self.fft_flux = stellar_obj.fft_flux
                base['fft_flux'] = self.fft_flux
                base['phot_flux'] = 0.  # Indicates that photon shooting wasn't done.
            return stellar_obj.psf

        psf = galsim.config.BuildGSObject(base, 'psf', gsparams=gsparams, logger=logger)[0]

        # For very bright things, we might want to change this for FFT drawing.
        if 'fft_sb_thresh' in config:
            fft_sb_thresh = galsim.config.ParseValue(config,'fft_sb_thresh',base,float)[0]
        else:
            fft_sb_thresh = 0.

        if self.nominal_flux < 1.e6 or not fft_sb_thresh or self.nominal_flux < fft_sb_thresh:
            self.use_fft = False
            logger.info('Use photon shooting for object %d. '
                        'nominal flux = %.0f, FFT threshold %.0f',
                        base.get('obj_num'), self.nominal_flux, fft_sb_thresh)
            return psf

        # Otherwise (high flux object), we might want to switch to fft.  So be a little careful.
        bandpass = base['bandpass']
        fft_psf = make_fft_psf(psf.evaluateAtWavelength(bandpass.effective_wavelength), logger)
        logger.info('Object %d has flux = %s.  Check if we should switch to FFT',
                    base['obj_num'], self.nominal_flux)

        # Now this object should have a much better estimate of the real maximum surface brightness
        # than the original psf did.
        # However, the max_sb feature gives an over-estimate, whereas to be conservative, we would
        # rather an under-estimate.  For this kind of profile, dividing by 2 does a good job
        # of giving us an underestimate of the max surface brightness.
        # Also note that `max_sb` is in photons/arcsec^2, so multiply by pixel_scale**2
        # to get photons/pixel, which we compare to fft_sb_thresh.
        gal_achrom = self.gal.evaluateAtWavelength(bandpass.effective_wavelength)
        fft_obj = galsim.Convolve(gal_achrom, fft_psf).withFlux(self.nominal_flux)
        max_sb = fft_obj.max_sb/2. * self._pixel_scale**2
        logger.debug('max_sb = %s. cf. %s',max_sb,fft_sb_thresh)
        if max_sb > fft_sb_thresh:
            self.use_fft = True
            # For FFT-rendered objects, the telescope vignetting isn't
            # emergent as it is for the ray-traced objects, so use the
            # empirical vignetting function, if it's available, to
            # scale the flux to use in FFTs.
            if self.vignetting is not None:
                pix_to_fp = self.det.getTransform(cameraGeom.PIXELS,
                                                  cameraGeom.FOCAL_PLANE)
                self.fft_flux = self.nominal_flux*self.vignetting.at_sky_coord(
                    base['sky_pos'], self.image.wcs, pix_to_fp)
            else:
                self.fft_flux = self.nominal_flux
            base['fft_flux'] = self.fft_flux
            base['phot_flux'] = 0.  # Indicates that photon shooting wasn't done.

            logger.info('Yes. Use FFT for object %d.  max_sb = %.0f > %.0f',
                        base.get('obj_num'), max_sb, fft_sb_thresh)
            return fft_psf
        else:
            self.use_fft = False
            logger.info('No. Use photon shooting for object %d. '
                        'max_sb = %.0f <= %.0f',
                        base.get('obj_num'), max_sb, fft_sb_thresh)
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
            if self.use_fft:
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

        # Prof is normally a convolution here with obj_list being [gal, psf1, psf2,...]
        # for some number of component PSFs.
        gal, *psfs = prof.obj_list if hasattr(prof,'obj_list') else [prof]

        max_flux_simple = config.get('max_flux_simple', 100)
        faint = self.nominal_flux < max_flux_simple
        bandpass = base['bandpass']
        if faint:
            logger.info("Flux = %.0f  Using trivial sed", self.nominal_flux)
            # cosmoDC2 galaxies with z > 2.71 and some SNANA objects
            # may have zero-valued SEDs at the bandpass effective
            # wavelength, so try to evaluate the SED at a few
            # locations across the bandpass and use the first value
            # that returns a non-zero result.
            for profile_wl in (bandpass.effective_wavelength,
                               bandpass.red_limit,
                               bandpass.blue_limit):
                sed_value = gal.sed(profile_wl)
                if sed_value != 0:
                    break
            if sed_value == 0:
                # We can't evalue the profile for this object, so skip it.
                obj_num = base.get('obj_num')
                object_id = base.get('object_id')
                logger.warning("Zero-valued SED for faint object %d, "
                               "object_id %s.  Skipping.", obj_num, object_id)
                return image
            gal = gal.evaluateAtWavelength(profile_wl)
            gal = gal * self._trivial_sed
        else:
            self._fix_seds(gal, bandpass, logger)

        # Normally, wcs is provided as an argument, rather than setting it directly here.
        # However, there is a subtle bug in the ChromaticSum class where it can fail to
        # apply the wcs correctly if the first component has zero realized flux in the band.
        # This line sidesteps that bug.  And it never hurts, so even once we fix this in
        # GalSim, it doesn't hurt to keep this here.
        image.wcs = base['wcs']

        # Set limit on the size of photons batches to consider when
        # calling gsobject.drawImage.
        maxN = int(1e6)
        if 'maxN' in config:
            maxN = galsim.config.ParseValue(config, 'maxN', base, int)[0]

        if method == 'fft':
            # For FFT, we may have adjusted the flux to account for vignetting.
            # So update the flux to self.fft_flux if it's different.
            if self.fft_flux != self.nominal_flux:
                gal = gal.withFlux(self.fft_flux, bandpass)

            fft_image = image.copy()
            fft_offset = offset
            kwargs = dict(
                method='fft',
                offset=fft_offset,
                image=fft_image,
            )
            if not faint and config.get('fft_photon_ops'):
                kwargs.update({
                    "photon_ops": galsim.config.BuildPhotonOps(config, 'fft_photon_ops', base, logger),
                    "maxN": maxN,
                    "rng": self.rng,
                    "n_subsample": 1,
                })

            # Go back to a combined convolution for fft drawing.
            prof = galsim.Convolve([gal] + psfs)
            try:
                fft_image = prof.drawImage(bandpass, **kwargs)
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
                raise
            # Some pixels can end up negative from FFT numerics.  Just set them to 0.
            fft_image.array[fft_image.array < 0] = 0.
            if self.diffraction_fft:
                self.diffraction_fft.apply(fft_image, bandpass.effective_wavelength)
            fft_image.addNoise(galsim.PoissonNoise(rng=self.rng))
            # In case we had to make a bigger image, just copy the part we need.
            image += fft_image[image.bounds]
            base['realized_flux'] = fft_image.added_flux

        else:
            # For photon shooting, use the poisson-realization of the flux
            # and tell GalSim not to redo the Poisson realization.
            gal = gal.withFlux(self.phot_flux, bandpass)

            if not faint and 'photon_ops' in config:
                photon_ops = galsim.config.BuildPhotonOps(config, 'photon_ops', base, logger)
            else:
                photon_ops = []
            # Put the psfs at the start of the photon_ops.
            # Probably a little better to put them a bit later than the start in some cases
            # (e.g. after TimeSampler, PupilAnnulusSampler), but leave that as a todo for now.
            photon_ops = psfs + photon_ops

            if faint:
                sensor = None
            else:
                sensor = base.get('sensor', None)
                if sensor is not None:
                    sensor.updateRNG(self.rng)

            image = gal.drawImage(bandpass,
                                  method='phot',
                                  offset=offset,
                                  rng=self.rng,
                                  maxN=maxN,
                                  n_photons=self.phot_flux,
                                  image=image,
                                  sensor=sensor,
                                  photon_ops=photon_ops,
                                  add_to_image=True,
                                  poisson_flux=False)
            base['realized_flux'] = image.added_flux

        return image


def make_fft_psf(psf, logger):
    """Swap out any PhaseScreenPSF component with a roughly equivalent analytic approximation.
    """
    if isinstance(psf, galsim.Transformation):
        return galsim.Transformation(make_fft_psf(psf.original, logger),
                                     psf.jac, psf.offset, psf.flux_ratio, psf.gsparams)
    if isinstance(psf, galsim.Convolution):
        obj_list = [make_fft_psf(p, logger) for p in psf.obj_list]
        return galsim.Convolution(obj_list, gsparams=psf.gsparams)
    if isinstance(psf, galsim.SecondKick):
        # The Kolmogorov version of the phase screen gets most of the second kick.
        # The only bit that it missing is the Airy part, so convert the SecondKick to that.
        return galsim.Airy(lam=psf.lam, diam=psf.diam, obscuration=psf.obscuration)
    if isinstance(psf, galsim.PhaseScreenPSF):
        # If psf is a PhaseScreenPSF, then make a simpler one the just convolves
        # a Kolmogorov profile with an OpticalPSF.
        r0_500 = psf.screen_list.r0_500_effective
        L0 = psf.screen_list[0].L0
        atm_psf = galsim.VonKarman(lam=psf.lam, r0_500=r0_500, L0=L0, gsparams=psf.gsparams)

        opt_screens = [s for s in psf.screen_list if isinstance(s, galsim.OpticalScreen)]
        logger.info('opt_screens = %r',opt_screens)
        if len(opt_screens) >= 1:
            # Should never be more than 1, but if there weirdly is, just use the first.
            # Note: Technically, if you have both a SecondKick and an optical screen, this
            # will add the Airy part twice, since it's also part of the OpticalPSF.
            # It doesn't usually matter, since we usually set doOpt=False, so we don't usually
            # do this branch. If it is found to matter for someone, it will require a bit
            # of extra logic to do it right.
            opt_screen = opt_screens[0]
            optical_psf = galsim.OpticalPSF(
                    lam=psf.lam,
                    diam=opt_screen.diam,
                    aberrations=opt_screen.aberrations,
                    annular_zernike=opt_screen.annular_zernike,
                    obscuration=opt_screen.obscuration,
                    gsparams=psf.gsparams)
            return galsim.Convolve([atm_psf, optical_psf], gsparams=psf.gsparams)
        return atm_psf
    return psf


class NullSensor(galsim.Sensor):
    """galsim sensor type which does nothing.

    It is used in the photon pooling workflow (`LSST_PhotonPoolingImage`)
    to prevent rendering an
    image when we only want to generate photons."""
    def accumulate(self, photons, image, orig_center=None, resume=False):
        return 0.

# Register this as a valid stamp type
RegisterStampType('LSST_Silicon', LSST_SiliconBuilder())
