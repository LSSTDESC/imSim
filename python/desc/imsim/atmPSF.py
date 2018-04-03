"""
GalSim realistic atmospheric PSF class
"""

import numpy as np
from scipy.optimize import fsolve

import galsim
from lsst.sims.GalSimInterface import PSFbase

from .optical_system import OpticalZernikes, mock_deviations

class OptWF(object):
    def __init__(self, rng):
        u = galsim.UniformDeviate(rng)
        self.deviations = mock_deviations(seed=int(u()*2**31))
        self.oz = OpticalZernikes(self.deviations)
        self.dynamic = False
        self.reversible = True

    def _wavefront_gradient(self, u, v, t, theta):
        z = self.oz.cartesian_coeff(theta[0].rad, theta[1].rad)
        Z = galsim.OpticalScreen(diam=8.36, obscuration=0.61, aberrations=[0]*4+list(z))
        return Z._wavefront_gradient(u, v, t, theta)


class AtmosphericPSF(PSFbase):
    """Class representing an Atmospheric PSF.

    A random realization of the atmosphere will be produced when this class is
    instantiated.

    @param airmass      Airmass of observation
    @param rawSeeing    The wavelength=500nm, zenith FWHM of the seeing
    @param band         One of ['u','g','r','i','z','y']
    @param rng          galsim.BaseDeviate
    @param t0           Exposure time start in seconds.  default: 0.
    @param exptime      Exposure time in seconds.  default: 30.
    @param kcrit        Critical Fourier mode at which to split first and second kicks
                        in units of (1/r0).  default: 0.2
    @param doOpt        Add in optical phase screens?  default: True
    @param logger       Optional logger.  default: None
    """
    def __init__(self, airmass, rawSeeing, band, rng, t0=0.0, exptime=30.0, kcrit=0.2,
                 doOpt=True, logger=None):
        self.airmass = airmass
        self.rawSeeing = rawSeeing

        self.wlen_eff = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
        # wlen_eff is from Table 2 of LSE-40 (y=y2)

        self.seeing500 = rawSeeing * airmass ** 0.6

        self.rng = rng
        self.t0 = t0
        self.exptime = exptime
        self.logger = logger

        self.atm = galsim.Atmosphere(**self._getAtmKwargs())
        self.aper = galsim.Aperture(diam=8.36, obscuration=0.61,
                                    lam=self.wlen_eff, screen_list=self.atm)

        # Instantiate screens now instead of delaying until after multiprocessing
        # has started.
        r0_500 = self.atm.r0_500_effective
        r0 = r0_500 * (self.wlen_eff/500.0)**(6./5)
        kmax = kcrit / r0
        self.atm.instantiate(kmax=kmax, check='phot')

        if doOpt:
            self.atm.append(OptWF(rng))

    @staticmethod
    def _seeing_resid(r0_500, wavelength, L0, target):
        """Residual function to use with `_r0_500` below."""
        r0_500 = r0_500[0]
        kolm_seeing = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
        r0 = r0_500 * (wavelength/500)**1.2
        factor = np.sqrt(1. - 2.183*(r0/L0)**0.356)
        return kolm_seeing*factor - target

    @staticmethod
    def _r0_500(wavelength, L0, seeing):
        """Returns r0_500 to use to get target seeing."""
        guess = wavelength*1e-9/(seeing/206265)*0.976
        result = fsolve(AtmosphericPSF._seeing_resid, guess, (wavelength, L0, seeing))
        return result[0]

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

        # Draw a single common outer scale for all layers from a log normal
        L0 = 0
        while L0 < 10.0 or L0 > 100:
            L0 = np.exp(gd() * 0.6 + np.log(25.0))
        L0 = [L0 for _ in range(6)]

        # Uniformly draw layer speeds between 0 and max_speed.
        maxSpeed = 20.0
        speeds = [ud()*maxSpeed for _ in range(6)]
        # Isotropically draw directions.
        directions = [ud()*360.0*galsim.degrees for _ in range(6)]

        # Given the desired seeing500 and randomly selected L0, determine appropriate
        # r0_500
        r0_500 = AtmosphericPSF._r0_500(500.0, L0, self.seeing500)

        if self.logger:
            self.logger.debug("airmass = {}".format(self.airmass))
            self.logger.debug("seeing500 = {}".format(self.seeing500))
            self.logger.debug("wlen_eff = {}".format(self.wlen_eff))
            self.logger.debug("r0_500 = {}".format(r0_500))
            self.logger.debug("L0 = {}".format(L0))
            self.logger.debug("speeds = {}".format(speeds))
            self.logger.debug("directions = {}".format(directions))
            self.logger.debug("altitudes = {}".format(altitudes))
            self.logger.debug("weights = {}".format(weights))

        return dict(r0_500=r0_500, L0=L0, speed=speeds, direction=directions,
                    altitude=altitudes, r0_weights=weights, rng=self.rng,
                    screen_size=819.2, screen_scale=0.1)

    def _getPSF(self, xPupil=None, yPupil=None):
        """
        Return a PSF to be convolved with sources.

        @param [in] xPupil the x coordinate on the pupil in arc seconds

        @param [in] yPupil the y coordinate on the pupil in arc seconds
        """
        theta = (xPupil*galsim.arcsec, yPupil*galsim.arcsec)
        psf = self.atm.makePSF(
            self.wlen_eff,
            aper=self.aper,
            theta=theta,
            t0=self.t0,
            exptime=self.exptime)
        return psf
