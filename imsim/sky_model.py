
import copy
import warnings
import numpy as np
import galsim
from galsim.config import InputLoader, RegisterInputType, RegisterValueType
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
import os
from .meta_data import data_dir


RUBIN_AREA = 0.25 * np.pi * 649**2  # cm^2


__all__ = ['SkyModel', 'SkyGradient']


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
    
    def print(self):
        print('This is a test')


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
    
class CCD_Fringe:
    """
    Class generates normalized fringing map. 
    
    The function call operator returns the normlized fringing level
    as a function of pixel coordinates relative to the value at the 
    CCD center.
    """
    def __init__(self):
        
        # Load the 4k x 4k normalized map
        fringing_filename = os.path.join(data_dir, 'fringing_data',
                                    'e2v-321-fringe-sim-norm-center.fits.gz')
        
        fringe_im = fits.open(fringing_filename)[0].data
        
        # Interpolate
        x = np.linspace(0,fringe_im.shape[-1]-1,fringe_im.shape[-1],dtype= int)
        y = np.linspace(0, fringe_im.shape[0]-1,fringe_im.shape[0],dtype = int )
        self.interp_func = RegularGridInterpolator((x, y), fringe_im.T)
        
    def fringe_variation_level(self):
        '''
        This is a place holder for the function that will be used to 
        characterize temporal and spatial variation of fringing. 
        
        The function will return a multiplicative factor that modifies
        the fringing amplitude from sensor to sensor based on the time
        and its location on the focal plane.
        Right now just return 1 since we are still waiting for the data.
        '''
        
        level = 1.
        
        return(level)

    def __call__(self, x, y):
        """
        Return the normalized fringing amplitude at the desired pixel 
        wrt the value at the CCD center.
        """
        level = self.fringe_variation_level()
        return (self.interp_func((x,y))*level)


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


RegisterInputType('sky_model', SkyModelLoader(SkyModel))
RegisterValueType('SkyLevel', SkyLevel, [float], input_type='sky_model')
