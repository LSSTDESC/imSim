from astropy.io import fits
import copy
import warnings
import numpy as np
import galsim
import pickle
from galsim.config import InputLoader, RegisterInputType, RegisterValueType
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
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

class CCD_Fringing:
    """
    Class generates normalized fringing map.

    The function call operator returns the normlized fringing level
    as a function of pixel coordinates relative to the value at the
    CCD center.
    """
    def __init__(self, true_center, boresight, seed, spatial_vary):
        """
        Parameters
        ----------
        true_center :
            Center of the current image.
        boresight : `CelestialCoord`
            Boresight of the current pointing.
            This is projected to the true_center to get the angular offset
            from FoV center for each CCD.
        seed : `int`
            Random seed number for each CCD.
            This is the hash value computed for the current sensor based on
            its serial number.
        spatial_vary : `bool`
            Parameter controls whether to account for Sky line variation
            in the fringing model or not. Default is set to True.
        """
        self.true_center = true_center
        self.boresight = boresight
        self.spatial_vary = spatial_vary
        self.seed = seed

    def generate_heightfield(self, fractal_dimension=2.5, n=4096):
        '''
        # Use the spectral sythesis method to generate a heightfield
        '''
        H = 1 - (fractal_dimension - 2)
        kpow = -(H + 1.0) / 1.2

        A = np.zeros((n, n), complex)

        kvec = np.fft.fftfreq(n)
        k0 = kvec[n // 64]
        kx, ky = np.meshgrid(kvec, kvec, sparse=True, copy=False)
        ksq = kx ** 2 + ky ** 2
        m = ksq > 0
        random_seed = self.seed
        gen = galsim.BaseDeviate(random_seed).np
        phase = 2 * np.pi * gen.uniform(size=(n, n))
        A[m] = ksq[m] ** kpow * gen.normal(size=(n, n))[m] * np.exp(1.j * phase[m]) * np.exp(-ksq[m] / k0 ** 2)

        return np.fft.ifft2(A)


    def simulate_fringes(self, amp=0.002):
        '''
        # Generate random fringing pattern from a heightfield
        Parameters
        ------------------------
        amp: `float`
            Fringing amplitude. Default is set to 0.002 (0.02%)
        '''
        n = 1.2
        n1 = 1.5
        nwaves_rms=10.
        X = self.generate_heightfield(n, 4096)
        X *= nwaves_rms / np.std(X.real)
        Z =  amp * np.cos(2 * n1 * (X.real))
        # Scale up to unity
        #Z  += 1
        return(Z)


    def fringe_variation_level(self):
        '''
        Function implementing temporal and spatial variation of fringing.

        The function will return a multiplicative factor that modifies
        the fringing amplitude from sensor to sensor based on the
        time (not implemented yet) and its location on the focal plane.

        Set spatial = True to turn on spatial variation implementation.
        Otherwise, this function will return a unity value.
        '''

        if self.spatial_vary:
            # Load 2d interpolator for OH spatial variation
            filename = os.path.join(
                data_dir,
                'fringing_data',
                'skyline_var.fits'
            )
            hdu = fits.open(filename)[0]
            z = hdu.data
            nx, ny = z.shape
            x = np.linspace(hdu.header['XMIN'], hdu.header['XMAX'], nx)
            y = np.linspace(hdu.header['YMIN'], hdu.header['YMAX'], ny)
            interp = RectBivariateSpline(x, y, z)
            dx, dy = self.boresight.project(self.true_center)
            # calculated OH flux level wrst the center of focal plane.
            level = interp(dx.deg,dy.deg)/interp(0,0)
            return(level)
        else:
            return 1

    def calculate_fringe_amplitude(self,x,y,amplitude = 0.002):
        """
        Return the normalized fringing amplitude at the desired pixel
        wrt the value at the CCD center.

        Parameters
        ------------------------
        amplitude: `float`
            Fringing amplitude. Default is set to 0.002 (0.02%)
        """
        level = self.fringe_variation_level()
        fringe_im = self.simulate_fringes(amp=amplitude*level)
        if (np.all(fringe_im) != True) or (True in np.isnan(fringe_im)):
            raise ValueError(" 0 or nan value in the fringe map!")
        fringe_im += 1
        xx = np.linspace(0, fringe_im.shape[-1]-1, fringe_im.shape[-1], dtype= int)
        yy = np.linspace(0, fringe_im.shape[0]-1, fringe_im.shape[0], dtype=int )
        interp_func = RegularGridInterpolator((xx, yy), fringe_im.T)

        return (interp_func((x,y)))

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
