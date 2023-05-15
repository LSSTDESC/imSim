# These need conda (via stackvana).  Not pip-installable
import os
from lsst.afw import cameraGeom

# This is not on conda yet, but is pip installable.
# We'll need to get Matt to add this to conda-forge probably.
import batoid

import numpy as np
import erfa  # Installed as part of astropy.

import astropy.time
import galsim
from galsim.config import WCSBuilder, RegisterWCSType, GetInputObj
from lsst.obs.lsst.translators.lsst import SIMONYI_LOCATION as RUBIN_LOC
from .camera import get_camera
from .utils import pixel_to_focal, focal_to_pixel


# There are 5 coordinate systems to handle.  In order:
#   ICRF (rc, dc)
#     catalog positions of stars
#   observed (rob, dob)
#     position after applying precession, nutation, aberration, refraction
#   field (thx, thy)
#     position wrt boresight using a gnomonic tangent plane projection.  The
#     axes are defined such that moving towards zenith increases thy, and moving
#     towards positive azimuth increases thx.
#   focal (fpx, fpy)
#     Millimeters on a hypothetical focal plane.  Axes are aligned to the
#     "data visualization coordinate system".  See https://lse-349.lsst.io/.
#   pixel (x, y)
#     Pixels on an individual detector.


def det_z_offset(det):
    ccd_orientation = det.getOrientation()
    if hasattr(ccd_orientation, 'getHeight'):
        return ccd_orientation.getHeight()*1.0e-3  # Convert to meters.
    return 0.


class BatoidWCSFactory:
    """
    Factory for constructing WCS's.  Currently hard-coded for Rubin Observatory.

    Parameters
    ----------
    boresight : galsim.CelestialCoord
        The ICRF coordinate of light that reaches the boresight.  Note that this
        is distinct from the spherical coordinates of the boresight with respect
        to the ICRF axes.
    obstime : astropy.time.Time
        Mean time of observation.
    telescope : batoid.Optic
        Telescope instance. Should include any camera rotation.
    wavelength : float
        Nanometers
    camera : lsst.afw.cameraGeom.Camera
    temperature : float
        Ambient temperature in Kelvin
    pressure : float
        Ambient pressure in kPa
    H2O_pressure : float
        Water vapor pressure in kPa
    """
    def __init__(
        self,
        boresight,
        obstime,
        telescope,
        wavelength,
        camera,
        temperature,
        pressure,
        H2O_pressure
    ):
        self.boresight = boresight
        self.obstime = obstime
        self.telescope = telescope
        self.wavelength = wavelength
        self.camera = camera
        self.temperature = temperature
        self.pressure = pressure
        self.H2O_pressure = H2O_pressure
        # Rubin Observatory lat/lon/height parameters (in ERFA speak)
        # Wikipedia says
        # self.phi = -np.deg2rad(30 + 14/60 + 40.7/3600)
        # self.elong = -np.deg2rad(70 + 44/60 + 57.9/3600)
        # self.hm = 2663.0  # meters
        # Opsim seems to use:
        self.phi = RUBIN_LOC.lat.rad
        self.elong = RUBIN_LOC.lon.rad
        self.hm = RUBIN_LOC.height.value

        # Various conversions required for ERFA functions
        self.utc1, self.utc2 = erfa.dtf2d("UTC", *self.obstime.utc.ymdhms)
        self.dut1 = self.obstime.delta_ut1_utc
        self.phpa = self.pressure * 10 # kPa -> mbar
        self.tc = self.temperature - 273.14  # K -> C
        # https://earthscience.stackexchange.com/questions/9868/convert-air-vapor-pressure-to-relative-humidity
        es = 6.11 * np.exp(17.27 * self.tc / (237.3 + self.tc))  # mbar
        self.rh = self.H2O_pressure/es  # relative humidity
        self.wl = self.wavelength * 1e-3  # nm -> micron

    def _ICRF_to_observed(self, rc, dc, all=False):
        """
        Parameters
        ----------
        rc, dc : array
            right ascension and declination in ICRF in radians
        all : bool
            If False, then just return observed ra/dec. If True, then also
            return azimuth, zenith angle, hour angle, equation-of-origins

        Returns
        -------
        aob : array
            Azimuth in radians (from N through E)
        zob : array
            Zenith angle in radians
        hob : array
            Hour angle in radians
        dob : array
            Observed declination in radians
        rob : array
            Observed right ascension in radians.  This is a CIO-based right
            ascension.  Add `eo` below to get an equinox-based right ascension.
        eo : array
            Equation of the origins (ERA - GST) in radians
        """
        # ERFA function with 0 proper motion, parallax, rv, polar motion
        aob, zob, hob, dob, rob, eo = erfa.atco13(
            rc, dc,  # ICRF radec
            0.0, 0.0,  # proper motion
            0.0, 0.0,  # parallax, radial velocity
            self.utc1, self.utc2,  # [seconds]
            self.dut1,  # [sec]
            self.elong, self.phi, self.hm, # [observatory location]
            0.0, 0.0,  # polar motion [rad]
            self.phpa,  # pressure [hPa = mbar]
            self.tc,  # temperature [C]
            self.rh,  # relative humidity [0-1]
            self.wl  # wavelength [micron]
        )
        if all:
            return aob, zob, hob, dob, rob, eo
        return rob, dob

    def _observed_to_ICRF(self, rob, dob):
        """
        Parameters
        ----------
        rob, dob : array
            Observed ra/dec in radians (CIO-based)

        Returns
        -------
        rc, dc : array
            ICRF ra/dec in radians
        """
        # ERFA function with 0 proper motion, parallax, rv, polar motion
        return erfa.atoc13(
            "R",
            rob, dob,
            self.utc1, self.utc2,
            self.dut1,
            self.elong, self.phi, self.hm,
            0.0, 0.0,
            self.phpa,
            self.tc,
            self.rh,
            self.wl
        )

    def _observed_hadec_to_ICRF(self, hob, dob):
        """
        Parameters
        ----------
        hob : array
            Hour angle in radians
        dob : array
            Declination in radians

        Returns
        -------
        rc, dc : array
            ICRF ra/dec in radians
        """
        return erfa.atoc13(
            "H",
            hob, dob,
            self.utc1, self.utc2,
            self.dut1,
            self.elong, self.phi, self.hm,
            0.0, 0.0,
            self.phpa,
            self.tc,
            self.rh,
            self.wl
        )

    def _observed_az_to_ICRF(self, aob, zob):
        """
        Parameters
        ----------
        aob : array
            Azimuth in radians (from N through E)
        zob : array
            Zenith angle in radians

        Returns
        -------
        rc, dc : array
            ICRF ra/dec in radians
        """
        return erfa.atoc13(
            "A",
            aob, zob,
            self.utc1, self.utc2,
            self.dut1,
            self.elong, self.phi, self.hm,
            0.0, 0.0,
            self.phpa,
            self.tc,
            self.rh,
            self.wl
        )

    @galsim.utilities.lazy_property
    def obs_boresight(self):
        """ Observed ra/dec of light that reaches the boresight.
        """
        rob, dob = self._ICRF_to_observed(
            self.boresight.ra.rad,
            self.boresight.dec.rad
        )
        return galsim.CelestialCoord(rob*galsim.radians, dob*galsim.radians)

    @galsim.utilities.lazy_property
    def q(self):
        """Parallactic angle.
        Should be equal to rotTelPos - rotSkyPos.
        """
        # Position angle of zenith measured from _observed_ North through East
        aob, zob, hob, dob, rob, eo = self._ICRF_to_observed(
            self.boresight.ra.rad,
            self.boresight.dec.rad,
            all=True
        )
        return erfa.hd2pa(hob, dob, self.phi)

    @galsim.utilities.lazy_property
    def _field_wcs(self):
        """WCS for converting between observed position and field angle.
        """
        cq, sq = np.cos(self.q), np.sin(self.q)
        affine = galsim.AffineTransform(cq, sq, sq, -cq)
        return galsim.TanWCS(
            affine, self.obs_boresight, units=galsim.radians
        )

    def _observed_to_field(self, rob, dob):
        """
        Field axes are defined such that if the zenith angle of an object
        decreases, thy increases (it moves "up" in the sky).  If the azimuth of
        an object increases, thx increases (it moves to the right on the sky).

        Parameters
        ----------
        rob, dob : array
            Observed ra/dec in radians

        Returns
        -------
        thx, thy : array
            Field angle in radians
        """
        return self._field_wcs.radecToxy(rob, dob, units="rad")

    def _field_to_observed(self, thx, thy):
        """
        Parameters
        ----------
        thx, thy : array
            Field angle in radians

        Returns
        -------
        rob, dob : array
            Observed ra/dec in radians
        """
        return self._field_wcs.xyToradec(thx, thy, units="rad")

    def _field_to_focal(self, thx, thy, z_offset=0.0, _telescope=None):
        """
        Parameters
        ----------
        thx, thy : array
            Field angle in radians

        Returns
        -------
        fpx, fpy : array
            Focal plane position in millimeters in DVCS.
            See https://lse-349.lsst.io/
        """
        rv = batoid.RayVector.fromFieldAngles(
            thx, thy, projection='gnomonic',
            optic=self.telescope,
            wavelength=self.wavelength*1e-9
        )
        if _telescope is not None:
            det_telescope = _telescope
        else:
            det_telescope = self.telescope
            if z_offset != 0.0:
                det_telescope = det_telescope.withLocallyShiftedOptic(
                    "Detector", [0.0, 0.0, -z_offset]  # batoid convention is opposite of DM
                )
        det_telescope.trace(rv)
        # x/y transpose to convert from EDCS to DVCS
        return rv.y*1e3, rv.x*1e3

    def _focal_to_field(self, fpx, fpy, z_offset=0.0):
        """
        Parameters
        ----------
        fpx, fpy : array
            Focal plane position in millimeters in DVCS.
            See https://lse-349.lsst.io/

        Returns
        -------
        thx, thy : array
            Field angle in radians
        """
        det_telescope = self.telescope
        if z_offset != 0.0:
            det_telescope = det_telescope.withLocallyShiftedOptic(
                "Detector", [0.0, 0.0, -z_offset]  # batoid convention is opposite of DM
            )

        fpx = np.atleast_1d(fpx)
        fpy = np.atleast_1d(fpy)
        N = len(fpx)
        # Iteratively solve.
        from scipy.optimize import least_squares
        def resid(p):
            thx = p[:N]
            thy = p[N:]
            x, y = self._field_to_focal(thx, thy, z_offset=z_offset, _telescope=det_telescope)
            return np.concatenate([x-fpx, y-fpy])
        result = least_squares(resid, np.zeros(2*N))
        return result.x[:N], result.x[N:]

    def get_field_samples(self, det):
        z_offset = det_z_offset(det)

        # Get field angle of detector center.
        fpx, fpy = det.getCenter(cameraGeom.FOCAL_PLANE)
        thx, thy = self._focal_to_field(fpx, fpy, z_offset=z_offset)
        # Hard-coding detector dimensions for Rubin science sensors for now.
        # detector width ~ 4000 px * 0.2 arcsec/px ~ 800 arcsec ~ 0.22 deg
        # max radius is about sqrt(2) smaller, ~0.16 deg
        # make hexapolar grid with that radius for test points
        rs = [0.0]
        ths = [0.0]
        Nrings = 5
        for r in np.linspace(0.01, 0.16, Nrings):
            nth = (int(r/0.16*6*Nrings)//6+1)*6
            rs.extend([r]*nth)
            ths.extend([ith/nth*2*np.pi for ith in range(nth)])
        thxs = thx + np.deg2rad(np.array(rs) * np.cos(ths))
        thys = thy + np.deg2rad(np.array(rs) * np.sin(ths))
        return thxs, thys

    def getWCS(self, det, order=3):
        """
        Parameters
        ----------
        det : lsst.afw.cameraGeom.Detector
            Detector of interest.
        order : int
            SIP order for fit.

        Returns
        -------
        wcs : galsim.fitswcs.GSFitsWCS
            WCS transformation between ICRF <-> pixels.
        """
        thxs, thys = self.get_field_samples(det)
        z_offset = det_z_offset(det)

        # trace both directions (field -> ICRF and field -> pixel)
        # then fit TanSIP to ICRF -> pixel.
        fpxs, fpys = self._field_to_focal(thxs, thys, z_offset=z_offset)
        xs, ys = focal_to_pixel(fpxs, fpys, det)
        rob, dob = self._field_to_observed(thxs, thys)
        rc, dc = self._observed_to_ICRF(rob, dob)

        return galsim.FittedSIPWCS(xs, ys, rc, dc, order=order)

    def ICRF_to_pixel(self, rc, dc, det):
        """
        Parameters
        ----------
        rc, dc : array
            right ascension and declination in ICRF in radians
        det : lsst.afw.cameraGeom.Detector
            Detector of interest.

        Returns
        -------
        x, y : array
            Pixel coordinates.
        """
        z_offset = det_z_offset(det)

        rob, dob = self._ICRF_to_observed(rc, dc)
        thx, thy = self._observed_to_field(rob, dob)
        fpx, fpy = self._field_to_focal(thx, thy, z_offset=z_offset)
        x, y = focal_to_pixel(fpx, fpy, det)
        return x, y

    def pixel_to_ICRF(self, x, y, det):
        """
        Parameters
        ----------
        x, y : array
            Pixel coordinates.
        det : lsst.afw.cameraGeom.Detector
            Detector of interest.

        Returns
        -------
        rc, dc : array
            right ascension and declination in ICRF in radians
        """
        z_offset = det_z_offset(det)

        fpx, fpy = pixel_to_focal(x, y, det)
        thx, thy = self._focal_to_field(fpx, fpy, z_offset=z_offset)
        rob, dob = self._field_to_observed(thx, thy)
        rc, dc = self._observed_to_ICRF(rob, dob)
        return rc, dc

    def get_icrf_to_field(self, det, order=3):
        thxs, thys = self.get_field_samples(det)

        rob, dob = self._field_to_observed(thxs, thys)
        rc, dc = self._observed_to_ICRF(rob, dob)

        return galsim.FittedSIPWCS(thxs, thys, rc, dc, order=order)


class BatoidWCSBuilder(WCSBuilder):

    def __init__(self):
        self._camera = None

    @property
    def camera(self):
        if self._camera is None:
            self._camera = get_camera(self._camera_name)
        return self._camera

    def buildWCS(self, config, base, logger):
        """Build a Tan-SIP WCS based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the wcs type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed WCS object (a galsim.GSFitsWCS instance)
        """
        req = {
                "boresight": galsim.CelestialCoord,
                "obstime": None,  # Either str or astropy.time.Time instance
                "det_name": str,
              }
        opt = {
                "camera": str,
                "telescope": str,
                "temperature": float,
                "pressure": float,
                "H2O_pressure": float,
                "wavelength": float,
                "order": int,
              }

        # Make sure the bandpass is built, since we are likely to need it to get the
        # wavelength (either in user specification or the default behavior below).
        if 'bandpass' not in base and 'bandpass' in base.get('image',{}):
            bp = galsim.config.BuildBandpass(base['image'], 'bandpass', base, logger)[0]
            base['bandpass'] = bp

        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        kwargs['bandpass'] = base.get('bandpass', None)
        kwargs['camera'] = kwargs.pop('camera', 'LsstCam')
        if (self._camera is not None and self._camera_name != kwargs['camera']):
            self._camera_name = kwargs['camera']
            self._camera = get_camera(self._camera_name)
        order = kwargs.pop('order', 3)
        det_name = kwargs.pop('det_name')
        kwargs['telescope'] = GetInputObj('telescope', config, base, 'telescope').fiducial
        factory = self.makeWCSFactory(**kwargs)
        det = self.camera[det_name]
        logger.info("Building Batoid WCS for %s and %s on pid=%d", det_name,
                    self.camera.getName(), os.getpid())
        wcs = factory.getWCS(det, order=order)
        base['_icrf_to_field'] = factory.get_icrf_to_field(det, order=order)
        return wcs

    def makeWCSFactory(
        self, boresight, obstime, telescope,
        camera='LsstCam',
        temperature=None, pressure=None, H2O_pressure=None,
        wavelength=None, bandpass=None
    ):
        """Make the WCS factory given the parameters explicitly rather than via a config dict.

        It mostly just constructs BatoidWCSFactory, but it has sensible defaults for many
        parameters.

        Parameters
        ----------
        boresight : galsim.CelestialCoord
            The ICRF coordinate of light that reaches the boresight.  Note that this
            is distinct from the spherical coordinates of the boresight with respect
            to the ICRF axes.
        obstime : astropy.time.Time or str
            Mean time of observation.  Note: if this is a string, it is assumed to be in TAI scale,
            which seems to be standard in the Rubin project.
        telescope : batoid.Optic
            Telescope instance. Should include any camera rotation.
        temperature : float
            Ambient temperature in Kelvin [default: 280 K]
        pressure : float
            Ambient pressure in kPa [default: based on LSST heigh of 2715 meters]
        H2O_pressure : float
            Water vapor pressure in kPa [default: 1.0 kPa]
        wavelength : float
            Nanometers.  One of wavelength, bandpass is required.
        bandpass : galsim.Bandpass or str
            If wavelength is None, use this to get the effective wavelength. If a string, then load
            the associated LSST bandpass.

        Returns:
            the constructed WCSFactory
        """

        # If a string, convert it to astropy.time.Time.
        if isinstance(obstime, str):
            obstime = astropy.time.Time(obstime, scale='tai')

        self._camera_name = camera

        # Update optional kwargs

        if wavelength is None:
            if isinstance(bandpass, str):
                bandpass = galsim.Bandpass('LSST_%s.dat'%bandpass, wave_type='nm')
            wavelength = bandpass.effective_wavelength

        if temperature is None:
            # cf. https://www.meteoblue.com/en/weather/historyclimate/climatemodelled/Cerro+Pachon
            # Average minimum temp is around 45 F = 7 C, but obviously varies a lot.
            temperature = 280 # Kelvin

        if pressure is None:
            # cf. https://www.engineeringtoolbox.com/air-altitude-pressure-d_462.html
            # p = 101.325 kPa (1 - 2.25577e-5 (h / 1 m))**5.25588
            # Cerro Pachon  altitude = 2715 m
            h = 2715
            pressure = 101.325 * (1-2.25577e-5*h)**5.25588

        if H2O_pressure is None:
            # I have no idea what a good default is, but this seems like a minor enough effect
            # that we should not require the user to pick something.
            H2O_pressure = 1.0 # kPa

        # Finally, build the WCS.
        return BatoidWCSFactory(
            boresight, obstime, telescope, wavelength, self.camera, temperature,
            pressure, H2O_pressure
        )


RegisterWCSType('Batoid', BatoidWCSBuilder(), input_type="telescope")
