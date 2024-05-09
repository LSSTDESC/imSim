import numpy as np
from functools import lru_cache
import galsim
from lsst.afw import cameraGeom


@lru_cache(maxsize=128)
def make_double_gaussian(fwhm1=0.6, fwhm2=0.12, wgt1=1.0, wgt2=0.1):
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
def make_kolmogorov_and_gaussian_psf(
    airmass=None, rawSeeing=None, band=None, gsparams=None,
):
    """
    This PSF class is based on David Kirkby's presentation to the DESC Survey Simulations
    working group on 23 March 2017.

    https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=LSSTDESC&title=SSim+2017-03-23

    (you will need a SLAC Confluence account to access that link)

    Parameters
    ----------
    airmass: float
        Default 1.2
    rawSeeing: float
        Default 0.7
    band: str
        Default 'r'
    gsparams: galsim.GSParams
        Default None

    rawSeeing is the FWHM seeing at zenith at 500 nm in arc seconds
    (provided by OpSim)

    band is the bandpass of the observation [u,g,r,i,z,y]

    This code was provided by David Kirkby in a private communication
    """
    if airmass is None:
        airmass = 1.2

    if rawSeeing is None:
        rawSeeing = 0.7

    if band is None:
        band = 'r'

    wlen_eff = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
    # wlen_eff is from Table 2 of LSE-40 (y=y2)

    FWHMatm = rawSeeing * (wlen_eff / 500.) ** -0.3 * airmass ** 0.6
    # From LSST-20160 eqn (4.1)

    FWHMsys = np.sqrt(0.25**2 + 0.3**2 + 0.08**2) * airmass ** 0.6
    # From LSST-20160 eqn (4.2)

    atm = galsim.Kolmogorov(fwhm=FWHMatm, gsparams=gsparams)
    sys = galsim.Gaussian(fwhm=FWHMsys, gsparams=gsparams)
    return galsim.Convolve((atm, sys))


def make_fft_psf(psf, logger=None):
    """
    Swap out any PhaseScreenPSF component with a roughly equivalent analytic
    approximation.

    Parameters
    ----------
    psf: object representing a PSF
        This can be a variety of forms, Transformation, Convolution,
        SecondKick, PhaseScreenPSF

    Returns
    -------
    New PSF for use with the fft
    """
    if isinstance(psf, galsim.Transformation):
        return galsim.Transformation(make_fft_psf(psf.original, logger),
                                     psf.jac, psf.offset, psf.flux_ratio, psf.gsparams)
    elif isinstance(psf, galsim.Convolution):
        obj_list = [make_fft_psf(p, logger) for p in psf.obj_list]
        return galsim.Convolution(obj_list, gsparams=psf.gsparams)
    elif isinstance(psf, galsim.SecondKick):
        # The Kolmogorov version of the phase screen gets most of the second kick.
        # The only bit that it missing is the Airy part, so convert the SecondKick to that.
        return galsim.Airy(lam=psf.lam, diam=psf.diam, obscuration=psf.obscuration)
    elif isinstance(psf, galsim.PhaseScreenPSF):
        # If psf is a PhaseScreenPSF, then make a simpler one the just convolves
        # a Kolmogorov profile with an OpticalPSF.
        r0_500 = psf.screen_list.r0_500_effective
        L0 = psf.screen_list[0].L0
        atm_psf = galsim.VonKarman(lam=psf.lam, r0_500=r0_500, L0=L0, gsparams=psf.gsparams)

        opt_screens = [s for s in psf.screen_list if isinstance(s, galsim.OpticalScreen)]
        if logger is not None:
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
                gsparams=psf.gsparams,
            )
            return galsim.Convolve([atm_psf, optical_psf], gsparams=psf.gsparams)
        else:
            return atm_psf
    else:
        return psf


def get_fft_psf_maybe(
    obj, nominal_flux, psf, bandpass, wcs, fft_sb_thresh, pixel_scale,
    dm_detector, vignetting=None, sky_pos=None, logger=None,
):
    """
    Get the fft psf if it is deemed necessary to use fft to draw the object,
    otherwise return the input psf

    Parameters
    ----------
    obj: galsim object
        The object being drawn
    nominal_flux: float
        The nominal flux of the object
    psf: galsim psf object
        The current psf
    bandpass: imsim.bandpass.RubinBandpass
        The bandpass for this image
    wcs: galsim.GSFitsWCS
        The WCS for the image
    fft_sb_thresh: float
        Surface brightness threshold for doing FFTs
    pixel_scale: float
        Pixel scale of detector
    vignetting: vignetting object, optional
        Must have an apply(image) method
    dm_detector: lsst.afw.cameraGeom.Detector, optional
        Data management detector object.  Use make_dm_detector(detnum)
        Only needed for vignetting
    sky_pos: obj_coord: galsim.CelestialCoord, optional
        The sky position of the object.  Needed to get vignetting factor
        when doing fft
    logger: python logger, optional
        logger

    Returns
    -------
    psf, draw_method, flux
        The PSF will be the input psf if it is determined that photon
        shooting should be used.  draw_method will be 'phot' or 'fft'
        Flux could be modified via vignetting of fft will be used
    """
    import galsim

    fft_psf = make_fft_psf(
        psf=psf.evaluateAtWavelength(bandpass.effective_wavelength),
        logger=logger,
    )

    # Now this object should have a much better estimate of the real maximum
    # surface brightness than the original psf did.  However, the max_sb
    # feature gives an over-estimate, whereas to be conservative, we would
    # rather an under-estimate.  For this kind of profile, dividing by 2 does a
    # good job of giving us an underestimate of the max surface brightness.
    # Also note that `max_sb` is in photons/arcsec^2, so multiply by
    # pixel_scale**2 to get photons/pixel, which we compare to fft_sb_thresh.
    gal_achrom = obj.evaluateAtWavelength(bandpass.effective_wavelength)
    fft_obj = galsim.Convolve(gal_achrom, fft_psf).withFlux(nominal_flux)
    max_sb = fft_obj.max_sb/2. * pixel_scale**2

    if max_sb > fft_sb_thresh:
        if logger is not None:
            logger.info('max_sb = %s. >  %s', max_sb, fft_sb_thresh)
        # For FFT-rendered objects, the telescope vignetting isn't
        # emergent as it is for the ray-traced objects, so use the
        # empirical vignetting function, if it's available, to
        # scale the flux to use in FFTs.

        if vignetting is not None:
            if sky_pos is None:
                raise ValueError('you must send sky_pos when using vignetting')
            if dm_detector is None:
                raise ValueError('you must send dm_detector when using vignetting')
            pix_to_fp = dm_detector.getTransform(
                cameraGeom.PIXELS,
                cameraGeom.FOCAL_PLANE,
            )
            fft_flux = nominal_flux * vignetting.at_sky_coord(
                sky_pos, wcs, pix_to_fp,
            )
        else:
            fft_flux = nominal_flux

        return fft_psf, 'fft', fft_flux
    else:
        if logger is not None:
            logger.info('max_sb = %s. <  %s', max_sb, fft_sb_thresh)
        return psf, 'phot', nominal_flux
