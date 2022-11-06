
import copy
import warnings
import numpy as np
import galsim
from galsim.config import InputLoader, RegisterInputType, RegisterValueType


RUBIN_AREA = 0.25 * np.pi * 649**2  # cm^2


__all__ = ['SkyModel', 'SkyGradient']


class SkyModel:
    """Interface to rubin_sim.skybrightness model."""
    def __init__(self, exptime, mjd, bandpass, eff_area=RUBIN_AREA, logger=None):
        """
        Parameters
        ----------
        exptime : `float`
            Exposure time in seconds.
        mjd : `float`
            MJD of observation.
        bandpass : `galsim.Bandpass`
            Bandpass to use for flux calculation.
        eff_area : `float`
            Collecting area of telescope in cm^2. Default: Rubin value from
            https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
        logger : `logging.Logger`
            Logger object.
        """
        from rubin_sim import skybrightness
        self.exptime = exptime
        self.mjd = mjd
        self.eff_area = eff_area
        self.bandpass = bandpass
        self.logger = logger
        self._rubin_sim_sky_model = skybrightness.SkyModel()

    def get_sky_level(self, skyCoord):
        """
        Return the sky level in units of photons/arcsec^2 at the
        specified coordinate.

        Parameters
        ----------
        skyCoord : `galsim.CelestialCoord`
            Sky coordinate at which to compute the sky background level.

        Returns
        -------
        `float` : sky level in photons/arcsec^2
        """
        # Make a copy of the skybrightness.SkyModel object to avoid
        # collisions with other threads running this code.
        rubin_sim_sky_model = copy.deepcopy(self._rubin_sim_sky_model)

        # Set the ra, dec, mjd for the sky SED calculation
        with warnings.catch_warnings():
            # Silence astropy IERS warnings.
            warnings.simplefilter('ignore')
            rubin_sim_sky_model.setRaDecMjd(skyCoord.ra.deg, skyCoord.dec.deg,
                                            self.mjd, degrees=True)

        # Compute the flux in units of photons/cm^2/s/arcsec^2
        wave, spec = rubin_sim_sky_model.returnWaveSpec()
        lut = galsim.LookupTable(wave, spec[0])
        sed = galsim.SED(lut, wave_type='nm', flux_type='flambda')
        flux = sed.calculateFlux(self.bandpass)

        # Return photons/arcsec^2
        value = flux * self.eff_area * self.exptime

        if self.logger is not None:
            self.logger.info("Setting sky level to %.2f photons/arcsec^2 "
                             "at (ra, dec) = %s, %s", value,
                             skyCoord.ra.deg, skyCoord.dec.deg)
        return value


class SkyGradient:
    """
    Functor class that computes the plane containing three input
    points: the x, y positions of the CCD center, the lower left
    corner, and the lower right corner, and the z-value for each set
    to the corresponding sky background level.

    The function call operator returns the sky background level as a
    function of pixel coordinates relative to the value at the CCD
    center.
    """
    def __init__(self, sky_model, wcs, world_center, image_xsize,
                 pixel_scale=0.2):
        self.pixel_scale = pixel_scale
        sky_level_center = sky_model.get_sky_level(world_center)
        center = wcs.toImage(world_center)
        llc = galsim.PositionD(0, 0)
        lrc = galsim.PositionD(image_xsize, 0)
        M = np.array([[center.x, center.y, 1],
                      [llc.x, llc.y, 1],
                      [lrc.x, lrc.y, 1]])
        Minv = np.linalg.inv(M)
        z = np.array([sky_level_center,
                      sky_model.get_sky_level(wcs.toWorld(llc)),
                      sky_model.get_sky_level(wcs.toWorld(lrc))])
        self.a, self.b, self.c = np.dot(Minv, z)

        # Subtract the sky level at the center of the CCD
        self.c -= sky_level_center

    def __call__(self, x, y):
        """Return photons/pixel."""
        return (self.a*x + self.b*y + self.c)*self.pixel_scale**2


class SkyModelLoader(InputLoader):
    """
    Class to load a SkyModel object.
    """
    def getKwargs(self, config, base, logger):
        req = {'exptime': float,
               'mjd': float}
        opt = {'eff_area': float}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        bandpass, safe1 = galsim.config.BuildBandpass(base['image'], 'bandpass', base, logger)
        safe = safe and safe1
        kwargs['bandpass'] = bandpass
        kwargs['logger'] = galsim.config.GetLoggerProxy(logger)
        return kwargs, safe


def SkyLevel(config, base, value_type):
    """
    Use the rubin_sim skybrightness model to return the sky level in
    photons/arcsec^2 at the center of the image.
    """
    sky_model = galsim.config.GetInputObj('sky_model', config, base, 'SkyLevel')

    kwargs, safe = galsim.config.GetAllParams(config, base)

    value = sky_model.get_sky_level(base['world_center'])

    return value, safe


RegisterInputType('sky_model', SkyModelLoader(SkyModel, takes_logger=True))
RegisterValueType('SkyLevel', SkyLevel, [float], input_type='sky_model')
