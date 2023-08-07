
import copy
import warnings
import numpy as np
import galsim
from galsim.config import InputLoader, RegisterInputType, RegisterValueType, RegisterBandpassType

RUBIN_AREA = 0.25 * np.pi * 649**2  # cm^2


__all__ = ['SkyModel', 'SkyGradient', 'RubinBandpass']


class SkyModel:
    """Interface to rubin_sim.skybrightness model."""
    def __init__(self, exptime, mjd, bandpass, eff_area=RUBIN_AREA):
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
        """
        from rubin_sim import skybrightness
        self.exptime = exptime
        self.mjd = mjd
        self.eff_area = eff_area
        self.bandpass = bandpass
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
            try:
                rubin_sim_sky_model.set_ra_dec_mjd(skyCoord.ra.deg, skyCoord.dec.deg,
                                                   self.mjd, degrees=True)
                wave, spec = rubin_sim_sky_model.return_wave_spec()
            except AttributeError:
                # Assume pre-v1.0.0 rubin_sim interfaces.
                rubin_sim_sky_model.setRaDecMjd(skyCoord.ra.deg, skyCoord.dec.deg,
                                                self.mjd, degrees=True)
                wave, spec = rubin_sim_sky_model.returnWaveSpec()

        # Compute the flux in units of photons/cm^2/s/arcsec^2
        lut = galsim.LookupTable(wave, spec[0])
        sed = galsim.SED(lut, wave_type='nm', flux_type='flambda')
        flux = sed.calculateFlux(self.bandpass)

        # Return photons/arcsec^2
        value = flux * self.eff_area * self.exptime
        return value


class SkyGradient:
    """
    Functor class that computes the plane containing three input
    points: the x, y positions of the CCD center, the lower left
    corner, and the lower right corner, and the z-value for each set
    to the corresponding sky background level.

    The function call operator returns the fractional sky background
    level as a function of pixel coordinates relative to the value at
    the CCD center.
    """
    def __init__(self, sky_model, wcs, world_center, image_xsize):
        self.sky_level_center = sky_model.get_sky_level(world_center)
        center = wcs.toImage(world_center)
        llc = galsim.PositionD(0, 0)
        lrc = galsim.PositionD(image_xsize, 0)
        M = np.array([[center.x, center.y, 1],
                      [llc.x, llc.y, 1],
                      [lrc.x, lrc.y, 1]])
        Minv = np.linalg.inv(M)
        z = np.array([self.sky_level_center,
                      sky_model.get_sky_level(wcs.toWorld(llc)),
                      sky_model.get_sky_level(wcs.toWorld(lrc))])
        self.a, self.b, self.c = np.dot(Minv, z)

    def __call__(self, x, y):
        """
        Return the ratio of the sky level at the desired pixel wrt the
        value at the CCD center.
        """
        return (self.a*x + self.b*y + self.c)/self.sky_level_center


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
        return kwargs, safe


def SkyLevel(config, base, value_type):
    """
    Use the rubin_sim skybrightness model to return the sky level in
    photons/arcsec^2 at the center of the image.
    """
    sky_model = galsim.config.GetInputObj('sky_model', config, base, 'SkyLevel')
    value = sky_model.get_sky_level(base['world_center'])
    return value, False

class RubinBandpass(galsim.Bandpass):
    """One of the Rubin bandpasses, specified by the single-letter name.

    The zeropoint is automatically set to the AB zeropoint normalization.
    """

    def __init__(self, band):
        """
        Parameters
        ----------
        band : `str`
            The name of the bandpass.  One of u,g,r,i,z,Y.
        """
        # TODO: Should switch this to use lsst.sims versions.  GalSim files are pretty old.
        super().__init__('LSST_%s.dat'%band, wave_type='nm')
        self.zeropoint = self.withZeropoint('AB').zeropoint

class RubinBandpassBuilder(galsim.config.BandpassBuilder):
    """A class for building a RubinBandpass in the config file
    """
    def buildBandpass(self, config, base, logger):
        """Build the Bandpass object based on the LSST filter name.

        Parameters:
            config:     The configuration dict for the bandpass type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed Bandpass object.
        """
        req = { 'band' : str }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
        bp = RubinBandpass(**kwargs)
        logger.debug('bandpass = %s', bp)
        return bp, safe

RegisterInputType('sky_model', SkyModelLoader(SkyModel))
RegisterValueType('SkyLevel', SkyLevel, [float], input_type='sky_model')
RegisterBandpassType('RubinBandpass', RubinBandpassBuilder())
