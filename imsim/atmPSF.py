"""
GalSim realistic atmospheric PSF class
"""

import os
import multiprocessing
import logging
import numpy as np
from scipy.optimize import bisect
import pickle
import galsim
from galsim.config import InputLoader, RegisterInputType, RegisterObjectType

from .optical_system import OpticalZernikes, mock_deviations


def save_psf(psf, outfile):
    """
    Save the psf as a pickle file.
    """
    # Set any logger attribute to None since loggers cannot be persisted.
    if hasattr(psf, 'logger'):
        psf.logger = None
    with open(outfile, 'wb') as output:
        with galsim.utilities.pickle_shared():
            pickle.dump(psf, output)

def load_psf(psf_file):
    """
    Load a psf from a pickle file.
    """
    with open(psf_file, 'rb') as fd:
        psf = pickle.load(fd)
    return psf


class OptWF(object):
    def __init__(self, rng, wavelength, gsparams=None):
        u = galsim.UniformDeviate(rng)
        # Fudge factor below comes from an attempt to force the PSF ellipticity distribution to
        # match more closely the targets in the SRD (not be too round).  (See the discussion at
        # https://github.com/LSSTDESC/DC2-production/issues/259).  Since the values in the
        # mock_deviations function currently rely on a small set of simulations (7), this was deemed
        # reasonable.
        deviationsFudgeFactor = 3.0
        self.deviations = deviationsFudgeFactor*mock_deviations(seed=int(u()*2**31))
        self.oz = OpticalZernikes(self.deviations)
        self.dynamic = False
        self.reversible = True

        # Compute stepk once and store
        obj = galsim.Airy(lam=wavelength, diam=8.36, obscuration=0.61, gsparams=gsparams)
        self.stepk = obj.stepk

    def __eq__(self, rhs):
        return (isinstance(rhs, OptWF)
                and np.array_equal(self.deviations, rhs.deviations)
                and self.stepk == rhs.stepk)

    # Note: No _wavefront method.  This would be required if we wanted to draw this in fft
    #       mode.  But we don't want to ever do that.  Without _wavefront defined, we get an
    #       AttributeError if we accidentally don't switch to something analytic for fft drawing.

    def _wavefront_gradient(self, u, v, t, theta):
        # remap theta to prevent extrapolation beyond a radius of 1.708 degrees, which is the
        # radius of the outermost sampling point.
        fudgeFactor = 1.708/2.04

        z = self.oz.cartesian_coeff(theta[0]/galsim.degrees*fudgeFactor,
                                    theta[1]/galsim.degrees*fudgeFactor)
        Z = galsim.OpticalScreen(diam=8.36, obscuration=0.61, aberrations=[0]*4+list(z),
                                 annular_zernike=True)
        return Z._wavefront_gradient(u, v, t, theta)

    def _getStepK(self, **kwargs):
        return self.stepk

    # galsim v1.60 version:
    def _stepK(self, **kwargs):
        return self.stepk


class AtmosphericPSF(object):
    """Class representing an Atmospheric PSF.

    A random realization of the atmosphere will be produced when this class is
    instantiated.

    @param airmass      Airmass of observation
    @param rawSeeing    The wavelength=500nm, zenith FWHM of the seeing
    @param band         One of ['u','g','r','i','z','y']
    @param boresight    The CelestialCoord of the boresight of the observation.
    @param rng          galsim.BaseDeviate
    @param t0           Exposure time start in seconds.  default: 0.
    @param exptime      Exposure time in seconds.  default: 30.
    @param kcrit        Critical Fourier mode at which to split first and second kicks
                        in units of (1/r0).  default: 0.2
    @param screen_size  Size of the phase screens in meters.  default: 819.2
    @param screen_scale Size of phase screen "pixels" in meters.  default: 0.1
    @param doOpt        Add in optical phase screens?  default: False
    @param exponent     Chromatic seeing exponent.  The atmospheric component of
                        the PSF will dilate as (wavelength/500 nm)**exponent.  For
                        Kolmogorov turbulence, this should be -0.2.  For Von Karman
                        turbulence, this is somewhere around -0.3 (depends on the
                        outer scale).  default: -0.3
    @param logger       Optional logger.  default: None
    @param nproc        Number of processes to use in creating screens. If None (default),
                        then allocate one process per phase screen, of which there are 6,
                        nominally.
    @param save_file    A file name to use for saving the built atmosphere.  If the file already
                        exists, then the atmosphere is read in from this file, rather than be
                        rebuilt.  [default: None]
    """
    def __init__(self, airmass, rawSeeing, band, boresight, rng,
                 t0=0.0, exptime=30.0, kcrit=0.2,
                 screen_size=819.2, screen_scale=0.1, doOpt=False, exponent=-0.3,
                 logger=None, nproc=None, save_file=None,
                 _no2k=False # for testing
    ):
        self.airmass = airmass
        self.rawSeeing = rawSeeing
        self.boresight = boresight
        self.logger = galsim.config.LoggerWrapper(logger)

        self.wlen_eff = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
        # wlen_eff is from Table 2 of LSE-40 (y=y2)

        self.targetFWHM = rawSeeing * airmass**0.6 * (self.wlen_eff/500)**(-0.3)

        self.rng = rng
        self.t0 = t0
        self.exptime = exptime
        self.screen_size = screen_size
        self.screen_scale = screen_scale
        self.exponent = exponent
        self.opt = None
        self.second_kick = None

        if save_file and os.path.isfile(save_file):
            self.logger.warning(f'Reading atmospheric PSF from {save_file}')
            self.load_psf(save_file)
        else:
            self.logger.warning('Building atmospheric PSF')
            self._build_atm(kcrit, doOpt, nproc, _no2k)
            if save_file:
                self.logger.warning(f'Saving atmospheric PSF to {save_file}')
                self.save_psf(save_file)

    def save_psf(self, save_file):
        """
        Save the psf as a pickle file.
        """
        with open(save_file, 'wb') as fd:
            with galsim.utilities.pickle_shared():
                pickle.dump((self.atm, self.aper, self.second_kick), fd)

    def load_psf(self, save_file):
        """
        Load a psf from a pickle file.
        """
        with open(save_file, 'rb') as fd:
            self.atm, self.aper, self.second_kick = pickle.load(fd)

    def _build_atm(self, kcrit, doOpt, nproc, _no2k):

        ctx = multiprocessing.get_context('fork')
        self.atm = galsim.Atmosphere(mp_context=ctx, **self._getAtmKwargs())
        self.aper = galsim.Aperture(diam=8.36, obscuration=0.61,
                                    lam=self.wlen_eff, screen_list=self.atm)

        # Instantiate screens now instead of delaying until after multiprocessing
        # has started.
        r0_500 = self.atm.r0_500_effective
        r0 = r0_500 * (self.wlen_eff/500.0)**(6./5)
        kmax = kcrit / r0

        self.logger.info("Instantiating atmospheric screens")

        if nproc is None:
            nproc = len(self.atm)

        if nproc == 1:
            self.atm.instantiate(kmax=kmax, check='phot')
        else:
            self.logger.warning(f"Using {nproc} processes to build the phase screens")
            with galsim.utilities.single_threaded():
                with ctx.Pool(nproc, initializer=galsim.phase_screens.initWorker,
                              initargs=galsim.phase_screens.initWorkerArgs()) as pool:
                    self.atm.instantiate(pool=pool, kmax=kmax, check='phot')

        self.logger.info("Finished building atmosphere")
        self.logger.debug("GSScreenShare keys = %s",list(galsim.phase_screens._GSScreenShare.keys()))
        self.logger.debug("id(self) = %s",id(self))

        if not _no2k:
            self.second_kick = galsim.SecondKick(
                self.wlen_eff,
                r0,
                self.aper.diam,
                self.aper.obscuration,
                kcrit=kcrit
            )

        if doOpt:
            self.opt = galsim.PhaseScreenList(OptWF(self.rng, self.wlen_eff))

    def __eq__(self, rhs):
        return (isinstance(rhs, AtmosphericPSF)
                and self.airmass == rhs.airmass
                and self.rawSeeing == rhs.rawSeeing
                and self.wlen_eff == rhs.wlen_eff
                and self.t0 == rhs.t0
                and self.exptime == rhs.exptime
                and self.atm == rhs.atm
                and self.aper == rhs.aper)

    @staticmethod
    def _vkSeeing(r0_500, wavelength, L0):
        # von Karman profile FWHM from fitting formula in eqn 19 of
        # Tokovinin 2002, PASP, v114, p1156
        # https://dx.doi.org/10.1086/342683
        kolm_seeing = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
        r0 = r0_500 * (wavelength/500)**(6./5)
        arg = 1. - 2.183*(r0/L0)**0.356
        factor = np.sqrt(arg) if arg > 0.0 else 0.0
        return kolm_seeing*factor

    @staticmethod
    def _seeingResid(r0_500, wavelength, L0, targetSeeing):
        return AtmosphericPSF._vkSeeing(r0_500, wavelength, L0) - targetSeeing

    @staticmethod
    def _r0_500(wavelength, L0, targetSeeing):
        """Returns r0_500 to use to get target seeing."""
        r0_500_max = min(1.0, L0*(1./2.183)**(-0.356)*(wavelength/500.)**(6./5))
        r0_500_min = 0.01
        return bisect(
            AtmosphericPSF._seeingResid,
            r0_500_min,
            r0_500_max,
            args=(wavelength, L0, targetSeeing)
        )

    def _getAtmKwargs(self):
        ud = galsim.UniformDeviate(self.rng)
        gd = galsim.GaussianDeviate(self.rng)

        # Use values measured from Ellerbroek 2008.
        altitudes = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]
        # Elevate the ground layer though.  Otherwise, PSFs come out too correlated
        # across the field of view.
        altitudes[0] = 0.2

        # Use weights from Ellerbroek too, but add some random perturbations.
        weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
        weights = [np.abs(w*(1.0 + 0.1*gd())) for w in weights]
        weights = np.clip(weights, 0.01, 0.8)  # keep weights from straying too far.
        weights /= np.sum(weights)  # renormalize

        # Draw outer scale from truncated log normal
        L0 = 0
        while L0 < 10.0 or L0 > 100:
            L0 = np.exp(gd() * 0.6 + np.log(25.0))
        # Given the desired targetFWHM and randomly selected L0, determine appropriate
        # r0_500
        if self.logger:
            self.logger.debug("target FWHM: {}".format(self.targetFWHM))
        r0_500 = AtmosphericPSF._r0_500(self.wlen_eff, L0, self.targetFWHM)
        if self.logger:
            self.logger.debug("Found r0_500, L0: {}, {}".format(r0_500, L0))
            self.logger.debug(
                "yields vonKarman FWHM: {}".format(
                    AtmosphericPSF._vkSeeing(r0_500, self.wlen_eff, L0)))

        # Broadcast common outer scale across all layers
        L0 = [L0]*6

        # Uniformly draw layer speeds between 0 and max_speed.
        maxSpeed = 20.0
        speeds = [ud()*maxSpeed for _ in range(6)]
        # Isotropically draw directions.
        directions = [ud()*360.0*galsim.degrees for _ in range(6)]

        if self.logger:
            self.logger.debug("airmass = {}".format(self.airmass))
            self.logger.debug("wlen_eff = {}".format(self.wlen_eff))
            self.logger.debug("r0_500 = {}".format(r0_500))
            self.logger.debug("L0 = {}".format(L0))
            self.logger.debug("speeds = {}".format(speeds))
            self.logger.debug("directions = {}".format(directions))
            self.logger.debug("altitudes = {}".format(altitudes))
            self.logger.debug("weights = {}".format(weights))

        return dict(r0_500=r0_500, L0=L0, speed=speeds, direction=directions,
                    altitude=altitudes, r0_weights=weights, rng=self.rng,
                    screen_size=self.screen_size, screen_scale=self.screen_scale)

    def getPSF(self, field_pos, gsparams=None):
        """
        Return a PSF to be convolved with sources.

        @param [in] field position of the object relative to the boresight direction.
        """
        theta = (field_pos.x*galsim.arcsec, field_pos.y*galsim.arcsec)
        psfs = [
            galsim.ChromaticAtmosphere(
                self.atm.makePSF(
                    self.wlen_eff,
                    aper=self.aper,
                    theta=theta,
                    t0=self.t0,
                    exptime=self.exptime,
                    gsparams=gsparams,
                    second_kick=False
                ),
                alpha=self.exponent,
                base_wavelength=self.wlen_eff,
                zenith_angle=0*galsim.degrees  # Turns off DCR, since we apply that later using PhotonDCR.
            )
        ]
        if self.second_kick is not None:
            psfs.append(self.second_kick.withGSParams(gsparams))
        if self.opt is not None:
            psfs.append(
                self.opt.makePSF(
                    self.wlen_eff,
                    aper=self.aper,
                    theta=theta,
                    t0=self.t0,
                    exptime=self.exptime,
                    gsparams=gsparams,
                    second_kick=False
                )
            )
        out = galsim.Convolve(psfs)
        return out


class AtmLoader(InputLoader):
    """Custom AtmosphericPSF loader that only loads the atmosphere once per exposure.

    Note: For now, this just loads the atmosphere once for an entire imsim run.
          If we ever decide we want to have a single config processing run handle multiple
          exposures (rather than just multiple CCDs for a single exposure), we'll need to
          reconsider this implementation.
    """
    def __init__(self):
        # Override some defaults in the base init.
        super().__init__(init_func=AtmosphericPSF,
                         takes_logger=True, use_proxy=False,
                         worker_init=galsim.phase_screens.initWorker,
                         worker_initargs=galsim.phase_screens.initWorkerArgs)

    def getKwargs(self, config, base, logger):
        logger.debug("Get kwargs for AtmosphericPSF")

        req_params = { 'airmass' : float,
                       'rawSeeing' : float,
                       'band' : str,
                       'boresight' : galsim.CelestialCoord,
                     }
        opt_params = { 't0' : float,
                       'exptime' : float,
                       'kcrit' : float,
                       'screen_size' : float,
                       'screen_scale' : float,
                       'doOpt' : bool,
                       'exponent': float,
                       'nproc' : int,
                       'save_file' : str,
                       '_no2k': bool
                     }

        # Temporary fix until GalSim 2.5 to make sure atm_psf can be built once and shared,
        # even if the opsim_data that it often needs is later in the list of inputs.
        try:
            kwargs, _ = galsim.config.GetAllParams(config, base, req=req_params, opt=opt_params)
        except galsim.GalSimError as e:
            if str(e).startswith("No input opsim_data"):
                galsim.config.LoadInputObj(base, 'opsim_data', 0, True, logger)
                kwargs, _ = galsim.config.GetAllParams(config, base, req=req_params, opt=opt_params)
            else:
                raise

        logger.debug("kwargs = %s",kwargs)

        # Check that we're not including the optics twice:
        if 'doOpt' in kwargs and kwargs['doOpt']:
            if 'stamp' in base:
                if 'photon_ops' in base['stamp']:
                    ops_types = [op['type'] for op in base['stamp']['photon_ops']]
                    for op in ops_types:
                        if op in ['RubinOptics', 'RubinDiffractionOptics']:
                            import warnings
                            warnings.warn(
                                f"You have included the optics twice!  Once via the "
                                f"photon operator '{op}' "
                                f"and once via 'atm_psf' with 'doOpt=True'. "
                                f"This is likely a mistake!"
                            )

        # We want this to be set up right at the beginning of the run, before the config
        # stuff has even set up the RNG yet.  So make an rng ourselves based on the
        # random seed in image.random_seed.

        seed = base['image'].get('random_seed', None)
        if seed is None:
            raise RuntimeError("AtmLoader requires a seed in config['image']['random_seed']")
        if isinstance(seed, list):
            seed = seed[0]  # If random_seed is a list, just use the first one.
        # Parse the value in case it is an eval or something.
        seed = galsim.config.ParseValue({'seed': seed}, 'seed', base, int)[0]
        # Somewhat gratuitously add an aribtary value to this to avoid any correlations with
        # other uses of this random seed elsewhere in the config processing.
        seed += 271828
        rng = galsim.BaseDeviate(seed)
        kwargs['rng'] = rng
        logger.debug("seed for atm = %s",seed)

        # Include the logger
        kwargs['logger'] = logger

        # safe=True means this will be used for the whole run.
        safe = True

        return kwargs, safe


def BuildAtmosphericPSF(config, base, ignore, gsparams, logger):
    """Build an AtmosphericPSF from the information in the config file.
    """
    atm = galsim.config.GetInputObj('atm_psf', config, base, 'AtmosphericPSF')
    image_pos = base['image_pos']
    boresight = atm.boresight
    field_pos = base['wcs'].posToWorld(image_pos, project_center=boresight)
    if gsparams: gsparams = GSParams(**gsparams)
    else: gsparams = None

    #logger.debug("Making PSF for pos %s",image_pos)
    #logger.debug("GSScreenShare keys = %s",list(galsim.phase_screens._GSScreenShare.keys()))
    #logger.debug("type(atm) = %s",str(type(atm)))
    #logger.debug("id(atm) = %s",id(atm))
    psf = atm.getPSF(field_pos, gsparams)
    return psf, False

# These next two are approximations to the above atmospheric PSF, which might be
# useful in contexts where accuracy of the PSF isn't so important.
def BuildDoubleGaussianPSF(config, base, ignore, gsparams, logger):
    """
    This is an example implementation of a wavelength- and
    position-independent Double Gaussian PSF.  See the documentation
    in PSFbase to learn how it is used.

    This specific PSF comes from equation(30) of the signal-to-noise
    document (LSE-40), which can be found at

    www.astro.washington.edu/users/ivezic/Astr511/LSST_SNRdoc.pdf

    The required fwhm parameter is the Full Width at Half Max of the total
    PSF.  This is given in arcseconds.
    """
    req = {'fwhm': float}
    opt = {'pixel_scale': float}

    params, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    fwhm = params['fwhm']
    pixel_scale = params.get('pixel_scale', 0.2)
    if gsparams: gsparams = GSParams(**gsparams)
    else: gsparams = None

    # the expression below is derived by solving equation (30) of
    # the signal-to-noise document
    # (www.astro.washington.edu/uses/ivezic/Astr511/LSST_SNRdoc.pdf)
    # for r at half the maximum of the PSF
    alpha = fwhm/2.3835

    eff_pixel_sigma_sq = pixel_scale*pixel_scale/12.0

    sigma = np.sqrt(alpha*alpha - eff_pixel_sigma_sq)
    gaussian1 = galsim.Gaussian(sigma=sigma, gsparams=gsparams)

    sigma = np.sqrt(4.0*alpha*alpha - eff_pixel_sigma_sq)
    gaussian2 = galsim.Gaussian(sigma=sigma, gsparams=gsparams)

    psf = 0.909*(gaussian1 + 0.1*gaussian2)

    return psf, safe


def BuildKolmogorovPSF(config, base, ignore, gsparams, logger):
    """
    This PSF class is based on David Kirkby's presentation to the DESC
    Survey Simulations working group on 23 March 2017.

    https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=LSSTDESC&title=SSim+2017-03-23

    (you will need a SLAC Confluence account to access that link)

    Parameters
    ----------
    airmass

    rawSeeing is the FWHM seeing at zenith at 500 nm in arc seconds
    (provided by OpSim)

    band is the bandpass of the observation [u,g,r,i,z,y]
    """

    req = {
            'airmass': float,
            'rawSeeing': float,
            'band': str,
          }

    params, safe = galsim.config.GetAllParams(config, base, req=req)
    airmass = params['airmass']
    rawSeeing = params['rawSeeing']
    band = params['band']
    if gsparams: gsparams = GSParams(**gsparams)
    else: gsparams = None

    # This code was provided by David Kirkby in a private communication

    wlen_eff = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21,
                    y=991.66)[band]
    # wlen_eff is from Table 2 of LSE-40 (y=y2)

    FWHMatm = rawSeeing*(wlen_eff/500.)**-0.3*airmass**0.6
    # From LSST-20160 eqn (4.1)

    FWHMsys = np.sqrt(0.25**2 + 0.3**2 + 0.08**2)*airmass**0.6
    # From LSST-20160 eqn (4.2)

    atm = galsim.Kolmogorov(fwhm=FWHMatm, gsparams=gsparams)
    sys = galsim.Gaussian(fwhm=FWHMsys, gsparams=gsparams)
    psf = galsim.Convolve((atm, sys))

    return psf, safe

RegisterInputType('atm_psf', AtmLoader())
RegisterObjectType('AtmosphericPSF', BuildAtmosphericPSF, input_type='atm_psf')
RegisterObjectType('DoubleGaussianPSF', BuildDoubleGaussianPSF)
RegisterObjectType('KolmogorovPSF', BuildKolmogorovPSF)
