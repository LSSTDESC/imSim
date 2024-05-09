import numpy as np
import galsim
from .psf_utils import (
    make_kolmogorov_and_gaussian_psf,
    make_double_gaussian,
)


def get_stamp_size(
    obj_achrom, nominal_flux, noise_var, airmass, rawSeeing, band, Nmax,
    pixel_scale, logger=None,
):
    """
    Get a stamp size for the input object

    Parameters
    ----------
    obj_achrom: achromatic version of object
        This can be gotten with
            obj_achrom = obj.evaluateAtWavelength(bandpass.effective_wavelength)
    nominal_flux: float
        The nominal flux of the object
    noise_var: float
        The variance of the noise in the image
    airmass: float
        Airmass for observation
    rawSeeing: float
        Raw seeing value for the observation
    band: str
        E.g. u, g, r, ...
    Nmax: int
        Maximum allowed stamp size
    pixel_scale: float
        The pixel scale of the image
    logger: python logger, optional
        Optional logger

    Returns
    -------
    stamp_size: int
    """

    if (hasattr(obj_achrom, 'original')
            and isinstance(obj_achrom.original, galsim.DeltaFunction)):

        stamp_size = get_star_stamp_size(
            obj_achrom=obj_achrom,
            nominal_flux=nominal_flux,
            noise_var=noise_var,
            airmass=airmass,
            rawSeeing=rawSeeing,
            band=band,
            Nmax=Nmax,
            pixel_scale=pixel_scale,
            logger=logger
        )
    else:
        stamp_size = get_gal_stamp_size(
            obj_achrom=obj_achrom,
            nominal_flux=nominal_flux,
            noise_var=noise_var,
            Nmax=Nmax,
            pixel_scale=pixel_scale,
        )

    return stamp_size


def get_star_stamp_size(
    obj_achrom, nominal_flux, noise_var, airmass, rawSeeing, band, Nmax, pixel_scale,
    logger=None,
):
    """
    Get a stamp size for a star (DeltaFunction) object

    Parameters
    ----------
    obj_achrom: achromatic version of object
        This can be gotten with
            obj_achrom = obj.evaluateAtWavelength(bandpass.effective_wavelength)
    nominal_flux: float
        The nominal flux of the object
    noise_var: float
        The variance of the noise in the image
    airmass: float
        Airmass for observation
    rawSeeing: float
        Raw seeing value for the observation
    band: str
        E.g. u, g, r, ...
    Nmax: int
        Maximum allowed stamp size
    pixel_scale: float
        The pixel scale of the image
    logger: python logger, optional
        Optional logger

    Returns
    -------
    stamp_size: int
    """
    # For bright stars, set the folding threshold for the
    # stamp size calculation.  Use a
    # Kolmogorov and Gaussian PSF since it is faster to
    # evaluate than an AtmosphericPSF.
    # base['current_noise_image'] = base['current_image']
    # noise_var = galsim.config.CalculateNoiseVariance(base)

    folding_threshold = noise_var / nominal_flux
    ft_default = galsim.GSParams().folding_threshold

    if folding_threshold >= ft_default or folding_threshold == 0:
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
        if logger is not None:
            logger.debug('Using folding_threshold %s',folding_threshold)
            logger.debug('From: noise_var = %s, flux = %s', noise_var, nominal_flux)

        gsparams = galsim.GSParams(folding_threshold=folding_threshold)

    psf = make_kolmogorov_and_gaussian_psf(
        airmass=airmass,
        rawSeeing=rawSeeing,
        band=band,
        gsparams=gsparams,
    )
    stamp_size = psf.getGoodImageSize(pixel_scale)
    # No point in this being larger than a CCD.  Cut back to Nmax if larger than this.
    stamp_size = min(stamp_size, Nmax)
    return stamp_size


def get_gal_stamp_size(obj_achrom, nominal_flux, noise_var, Nmax, pixel_scale):
    """
    Get a stamp size for a star (DeltaFunction) object

    Parameters
    ----------
    obj_achrom: achromatic version of object
        This can be gotten with
            obj_achrom = obj.evaluateAtWavelength(bandpass.effective_wavelength)
    nominal_flux: float
        The nominal flux of the object
    noise_var: float
        The variance of the noise in the image
    Nmax: int
        Maximum allowed stamp size
    pixel_scale: float
        The pixel scale of the image

    Returns
    -------
    stamp_size: int
    """

    # For extended objects, recreate the object to draw, but
    # convolved with the faster DoubleGaussian PSF.
    psf = make_double_gaussian()
    # For Chromatic objects, need to evaluate at the
    # effective wavelength of the bandpass.
    convolved_obj = galsim.Convolve(obj_achrom, psf).withFlux(nominal_flux)

    # Start with GalSim's estimate of a good box size.
    stamp_size = convolved_obj.getGoodImageSize(pixel_scale)

    # For bright things, defined as having an average of at least 10 photons
    # per pixel on average, or objects for which GalSim's estimate of the
    # stamp_size is larger than Nmax, compute the stamp_size using the surface
    # brightness limit, trying to be careful about not truncating the surface
    # brightness at the edge of the box.
    if (nominal_flux > 10 * stamp_size**2) or (stamp_size > Nmax):
        # Find a postage stamp region to draw onto.  Use (sky noise)/8. as the
        # nominal minimum surface brightness for rendering an extended object.

        keep_sb_level = np.sqrt(noise_var)/8.

        stamp_size = get_good_phot_stamp_size(
            obj_list=[obj_achrom, psf],
            keep_sb_level=keep_sb_level,
            pixel_scale=pixel_scale,
            Nmax=Nmax,
        )

        # If the above size comes out really huge, scale back to what you get
        # for a somewhat brighter surface brightness limit.
        if stamp_size > Nmax:
            large_object_sb_level = 3*keep_sb_level
            stamp_size = get_good_phot_stamp_size(
                obj_list=[obj_achrom, psf],
                keep_sb_level=large_object_sb_level,
                pixel_scale=pixel_scale,
                Nmax=Nmax,
            )
            stamp_size = min(stamp_size, Nmax)
    return stamp_size


def get_good_phot_stamp_size(obj_list, keep_sb_level, pixel_scale, Nmax):
    """
    Get a postage stamp size (appropriate for photon-shooting) given a
    minimum surface brightness in photons/pixel out to which to
    extend the stamp region.

    The sizes of objects in the input list are added quadratically

    Parameters
    ----------
    obj_list: [galsim.GSObject]
        A list of GalSim objects
    keep_sb_level: float
        The minimum surface brightness (photons/pixel) out to which to
        extend the postage stamp, e.g., a value of
        sqrt(sky_bg_per_pixel)/3 would be 1/3 the Poisson noise
        per pixel from the sky background.
    pixel_scale: float [0.2]
        The CCD pixel scale in arcsec.
    Nmax: int
        Maximum allowed stamp size

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

    sizes = [
        get_good_phot_stamp_size1(
            obj=obj, keep_sb_level=keep_sb_level, pixel_scale=pixel_scale,
            Nmax=Nmax,
        )
        for obj in obj_list
    ]
    return int(np.sqrt(np.sum([size**2 for size in sizes])))


def get_good_phot_stamp_size1(obj, keep_sb_level, pixel_scale, Nmax):
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
    Nmax: int
        Maximum allowed stamp size

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
    elif hasattr(obj, 'original') and isinstance(obj.original, galsim.RandomKnots):
        # Handle RandomKnots object directly
        obj = obj.original._profile

    # This can be too small for bright stars, so increase it in steps until the edges are
    # all below the requested sb level.
    while N < Nmax:
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

    N = min(N, Nmax)

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
