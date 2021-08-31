
# These need conda (via stackvana).  Not pip-installable
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCamMapper

# This is not on conda yet, but is pip installable.
# We'll need to get Matt to add this to conda-forge probably.
import batoid

import numpy as np
from astropy.time import Time
import erfa  # Installed as part of astropy.

import galsim


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


class BatoidWCSFactory:
    """
    Parameters
    ----------
    boresight : galsim.CelestialCoord
        The ICRF coordinate of light that reaches the boresight.  Note that this
        is distinct from the spherical coordinates of the boresight with respect
        to the ICRF axes.
    rotTelPos : galsim.Angle
        Camera rotator angle.
    obstime : astropy.Time
        Mean time of observation.
    fiducial_telescope : batoid.Optic
        Telescope instance without applying the rotator.
    wavelength : float
        Meters
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
        rotTelPos,
        obstime,
        fiducial_telescope,
        wavelength,
        camera,
        temperature,
        pressure,
        H2O_pressure
    ):
        self.boresight = boresight
        self.rotTelPos = rotTelPos
        self.obstime = obstime
        self.fiducial_telescope = fiducial_telescope
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
        self.phi = -np.deg2rad(30.2444)
        self.elong = -np.deg2rad(70.7494)
        self.hm = 2650.0

        # Various conversions required for ERFA functions
        self.utc1, self.utc2 = erfa.dtf2d("UTC", *self.obstime.utc.ymdhms)
        self.dut1 = self.obstime.delta_ut1_utc
        self.phpa = self.pressure * 10 # kPa -> mbar
        self.tc = self.temperature - 273.14  # K -> C
        # https://earthscience.stackexchange.com/questions/9868/convert-air-vapor-pressure-to-relative-humidity
        es = 6.11 * np.exp(17.27 * self.tc / (237.3 + self.tc))  # mbar
        self.rh = self.H2O_pressure/es  # relative humidity
        self.wl = self.wavelength * 1e6  # m -> micron

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
        affine = galsim.AffineTransform(cq, -sq, sq, cq)
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

    @galsim.utilities.lazy_property
    def _telescope(self):
        """Telescope including camera rotation as batoid.Optic
        """
        return self.fiducial_telescope.withLocallyRotatedOptic(
            "LSSTCamera",
            batoid.RotZ(-self.rotTelPos)
        )

    def _field_to_focal(self, thx, thy):
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
            optic=self._telescope,
            wavelength=self.wavelength
        )
        self._telescope.trace(rv)
        # x -> -x to map batoid x to EDCS x.
        # x/y transpose to convert from EDCS to DVCS
        return rv.y*1e3, -rv.x*1e3

    def _focal_to_field(self, fpx, fpy):
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
        fpx = np.atleast_1d(fpx)
        fpy = np.atleast_1d(fpy)
        N = len(fpx)
        # Iteratively solve.
        from scipy.optimize import least_squares
        def resid(p):
            thx = p[:N]
            thy = p[N:]
            x, y = self._field_to_focal(thx, thy)
            return np.concatenate([x-fpx, y-fpy])
        result = least_squares(resid, np.zeros(2*N))
        return result.x[:N], result.x[N:]

    def _focal_to_pixel(self, fpx, fpy, det):
        """
        Parameters
        ----------
        fpx, fpy : array
            Focal plane position in millimeters in DVCS
            See https://lse-349.lsst.io/
        det : lsst.afw.cameraGeom.Detector
            Detector of interest.

        Returns
        -------
        x, y : array
            Pixel coordinates.
        """
        tx = det.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
        x, y = np.vsplit(
            tx.getMapping().applyForward(
                np.vstack((fpx, fpy))
            ),
            2
        )
        return x.ravel(), y.ravel()

    def _pixel_to_focal(self, x, y, det):
        """
        Parameters
        ----------
        x, y : array
            Pixel coordinates.
        det : lsst.afw.cameraGeom.Detector
            Detector of interest.

        Returns
        -------
        fpx, fpy : array
            Focal plane position in millimeters in DVCS
            See https://lse-349.lsst.io/
        """
        tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
        fpx, fpy = np.vsplit(
            tx.getMapping().applyForward(
                np.vstack((x, y))
            ),
            2
        )
        return fpx.ravel(), fpy.ravel()

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
        # Get field angle of detector center.
        fpx, fpy = det.getCenter(cameraGeom.FOCAL_PLANE)
        thx, thy = self._focal_to_field(fpx, fpy)
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

        # trace both directions (field -> ICRF and field -> pixel)
        # then fit TanSIP to ICRF -> pixel.
        fpxs, fpys = self._field_to_focal(thxs, thys)
        xs, ys = self._focal_to_pixel(fpxs, fpys, det)
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
        rob, dob = self._ICRF_to_observed(rc, dc)
        thx, thy = self._observed_to_field(rob, dob)
        fpx, fpy = self._field_to_focal(thx, thy)
        x, y = self._focal_to_pixel(fpx, fpy, det)
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
        fpx, fpy = self._pixel_to_focal(x, y, det)
        thx, thy = self._focal_to_field(fpx, fpy)
        rob, dob = self._field_to_observed(thx, thy)
        rc, dc = self._observed_to_ICRF(rob, dob)
        return rc, dc


def sphere_dist(ra1, dec1, ra2, dec2):
    # Vectorizing CelestialCoord.distanceTo()
    # ra/dec in rad
    x1 = np.cos(dec1)*np.cos(ra1)
    y1 = np.cos(dec1)*np.sin(ra1)
    z1 = np.sin(dec1)
    x2 = np.cos(dec2)*np.cos(ra2)
    y2 = np.cos(dec2)*np.sin(ra2)
    z2 = np.sin(dec2)
    dsq = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
    dist = 2*np.arcsin(0.5*np.sqrt(dsq))
    w = dsq >= 3.99
    if np.any(w):
        cross = np.cross(np.array([x1, y1, z1])[w], np.array([x2, y2, z2])[w])
        crosssq = cross[0]**2 + cross[1]**2 + cross[2]**2
        dist[w] = np.pi - np.arcsin(np.sqrt(crosssq))
    return dist


def test_wcs_fit():
    """Check that fitted WCS transformation is close to actual
    ICRF <-> pixel transformation.
    """
    import astropy.units as u
    rng = np.random.default_rng(57721)
    fiducial_telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    wavelength = 620e-9
    camera = LsstCamMapper().camera

    for _ in range(30):
        # Random spherepoint for boresight
        z = rng.uniform(-1, 1)
        th = rng.uniform(0, 2*np.pi)
        x = np.sqrt(1-z**2) * np.cos(th)
        y = np.sqrt(1-z**2) * np.sin(th)
        boresight = galsim.CelestialCoord(
            np.arctan2(y, x) * galsim.radians,
            np.arcsin(z) * galsim.radians
        )

        # Random obstime.  No attempt to make sky dark.
        obstime = Time("J2020") + rng.uniform(0, 1)*u.year

        # Rotator
        rotTelPos = rng.uniform(-np.pi/2, np.pi/2)

        # Ambient conditions
        temperature = rng.uniform(270, 300)
        pressure = rng.uniform(66, 72)
        H2O_pressure = rng.uniform(0.1, 10)

        factory = BatoidWCSFactory(
            boresight, rotTelPos, obstime, fiducial_telescope, wavelength,
            camera, temperature, pressure, H2O_pressure
        )

        aob, zob, hob, dob, rob, eo = factory._ICRF_to_observed(
            boresight.ra.rad, boresight.dec.rad, all=True
        )

        # If zenith angle > 70 degrees, try again
        if np.rad2deg(zob) > 70:
            continue

        # Pick a few detectors randomly
        for det in rng.choice(camera, 3):
            wcs = factory.getWCS(det, order=3)

            # center of detector:
            xc, yc = det.getCenter(cameraGeom.PIXELS)
            x = xc + rng.uniform(-2000, 2000, 100)
            y = yc + rng.uniform(-2000, 2000, 100)
            rc, dc = wcs.xyToradec(x, y, units='radians')
            rc1, dc1 = factory.pixel_to_ICRF(x, y, det)

            dist = sphere_dist(rc, dc, rc1, dc1)
            np.testing.assert_allclose(  # sphere dist < 1e-5 arcsec
                0,
                np.rad2deg(np.max(np.abs(dist)))*3600,
                rtol=0,
                atol=1e-5
            )
            print(
                "ICRF dist (arcsec)  ",
                np.rad2deg(np.mean(dist))*3600,
                np.rad2deg(np.max(np.abs(dist)))*3600,
                np.rad2deg(np.std(dist))*3600
            )
            x, y = wcs.radecToxy(rc, dc, units='radians')
            x1, y1 = factory.ICRF_to_pixel(rc, dc, det)
            np.testing.assert_allclose(  # pix dist < 2e-3
                0,
                np.max(np.abs(x-x1)),
                rtol=0,
                atol=2e-3
            )
            np.testing.assert_allclose(
                0,
                np.max(np.abs(y-y1)),
                rtol=0,
                atol=2e-3
            )
            print(
                "x-x1 (pixel)  ",
                np.mean(x-x1),
                np.max(np.abs(x-x1)),
                np.std(x-x1)
            )
            print(
                "y-y1 (pixel)  ",
                np.mean(y-y1),
                np.max(np.abs(y-y1)),
                np.std(y-y1)
            )
            print("\n"*3)


def test_imsim():
    """Check that we can reproduce a collection of WCS's previously output by
    ImSim.
    """
    import yaml
    import astropy.units as u
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # Need these for `eval` below
    from numpy import array
    import coord

    with open("wcs_466749.yaml", 'r') as f:
        wcss = yaml.safe_load(f)

    cmds = {}
    with open("phosim_cat_466749.txt", 'r') as f:
        for line in f:
            k, v = line.split()
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            cmds[k] = v

    # Values below (and others) from phosim_cat_466749.txt
    rc = cmds['rightascension']
    dc = cmds['declination']
    boresight = galsim.CelestialCoord(
        rc*galsim.degrees,
        dc*galsim.degrees
    )
    obstime = Time(cmds['mjd'], format='mjd', scale='tai')
    obstime -= 15*u.s
    band = "ugrizy"[cmds['filter']]
    fiducial_telescope = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
    wavelength_dict = dict(
        u=365.49e-9,
        g=480.03e-9,
        r=622.20e-9,
        i=754.06e-9,
        z=868.21e-9,
        y=991.66e-9
    )
    wavelength = wavelength_dict[band]
    camera = LsstCamMapper().camera

    rotTelPos =  cmds['rottelpos'] * galsim.degrees

    # Ambient conditions
    # These are a guess.
    temperature = 293.
    pressure = 69.0
    H2O_pressure = 1.0

    # Start by constructing a refractionless factory, which we can use to
    # cross-check some of the other values in the phosim cmd file.
    factory = BatoidWCSFactory(
        boresight, rotTelPos, obstime, fiducial_telescope, wavelength,
        camera,
        temperature=temperature,
        pressure=0.0,
        H2O_pressure=H2O_pressure
    )

    aob, zob, hob, dob, rob, eo = factory._ICRF_to_observed(
        boresight.ra.rad, boresight.dec.rad, all=True
    )
    np.testing.assert_allclose(
        np.rad2deg(aob)*3600, cmds['azimuth']*3600,
        rtol=0, atol=2.0
    )
    np.testing.assert_allclose(
        (90-np.rad2deg(zob))*3600, cmds['altitude']*3600,
        rtol=0, atol=5.0,
    )
    q = factory.q * galsim.radians
    rotSkyPos = rotTelPos - q
    # Hmmm.. Seems like we ought to be able to do better than 30 arcsec on the
    # rotator?  Maybe this is defined at a different point in time? Doesn't seem
    # to affect the final WCS much though.
    np.testing.assert_allclose(
        rotSkyPos.deg*3600, cmds['rotskypos']*3600,
        rtol=0, atol=30.0,
    )

    # For actual WCS check, we use a factory that _does_ know about refraction.
    factory = BatoidWCSFactory(
        boresight, rotTelPos, obstime, fiducial_telescope, wavelength,
        camera,
        temperature=temperature,
        pressure=pressure,
        H2O_pressure=H2O_pressure
    )

    do_plot = False
    my_centers = []
    imsim_centers = []
    if do_plot:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    i = 0
    r1 = []
    d1 = []
    r2 = []
    d2 = []
    rng = np.random.default_rng(1234)
    for k, v in tqdm(wcss.items()):
        name = k[18:25].replace('-', '_')
        det = camera[name]
        cpix = det.getCenter(cameraGeom.PIXELS)

        wcs = factory.getWCS(det, order=2)
        wcs1 = eval(v)
        # Need to adjust ab parameters to new GalSim convention
        wcs1.ab[0,1,0] = 1.0
        wcs1.ab[1,0,1] = 1.0

        my_centers.append(wcs.posToWorld(galsim.PositionD(cpix.x, cpix.y)))
        imsim_centers.append(wcs1.posToWorld(galsim.PositionD(cpix.x, cpix.y)))

        corners = det.getCorners(cameraGeom.PIXELS)
        xs = np.array([corner.x for corner in corners])
        ys = np.array([corner.y for corner in corners])
        ra1, dec1 = wcs.xyToradec(xs, ys, units='radians')
        ra2, dec2 = wcs1.xyToradec(xs, ys, units='radians')
        if i == 0:
            labels = ['batoid', 'PhoSim']
        else:
            labels = [None]*2
        if do_plot:
            ax.plot(ra1, dec1, c='r', label=labels[0])
            ax.plot(ra2, dec2, c='b', label=labels[1])

        # add corners to ra/dec check lists
        r1.extend(ra1)
        d1.extend(dec1)
        r2.extend(ra2)
        d2.extend(dec2)
        # Add some random points as well
        xs = rng.uniform(0, 4000, 100)
        ys = rng.uniform(0, 4000, 100)
        ra1, dec1 = wcs.xyToradec(xs, ys, units='radians')
        ra2, dec2 = wcs1.xyToradec(xs, ys, units='radians')
        r1.extend(ra1)
        d1.extend(dec1)
        r2.extend(ra2)
        d2.extend(dec2)
        i += 1

    if do_plot:
        ax.legend()
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[1], xlim[0])
        plt.show()

    dist = sphere_dist(r1, d1, r2, d2)
    print("sphere dist mean, max, std")
    print(
        np.rad2deg(np.mean(dist))*3600,
        np.rad2deg(np.max(dist))*3600,
        np.rad2deg(np.std(dist))*3600,
    )
    np.testing.assert_array_less(
        np.rad2deg(np.mean(dist))*3600,
        5.0
    )
    if do_plot:
        plt.hist(np.rad2deg(dist)*3600, bins=100)
        plt.show()

    if do_plot:
        r1 = np.array([c.ra.rad for c in my_centers])
        d1 = np.array([c.dec.rad for c in my_centers])
        r2 = np.array([c.ra.rad for c in imsim_centers])
        d2 = np.array([c.dec.rad for c in imsim_centers])
        cd = np.cos(np.deg2rad(cmds['declination']))
        q = plt.quiver(r1, d1, np.rad2deg(r1-r2)*3600*cd, np.rad2deg(d1-d2)*3600)
        plt.quiverkey(q, 0.5, 1.1, 5.0, "5 arcsec", labelpos='E')
        plt.show()


def test_intermediate_coord_sys():
    import yaml
    import astropy.units as u
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from numpy import array
    import coord

    with open("wcs_466749.yaml", "r") as f:
        wcss = yaml.safe_load(f)

    cmds = {}
    with open("phosim_cat_466749.txt", 'r') as f:
        for line in f:
            k, v = line.split()
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            cmds[k] = v

    # Values below (and others) from phosim_cat_466749.txt
    rc = cmds['rightascension']
    dc = cmds['declination']
    boresight = galsim.CelestialCoord(
        rc*galsim.degrees,
        dc*galsim.degrees
    )
    obstime = Time(cmds['mjd'], format='mjd', scale='tai')
    obstime -= 15*u.s
    band = "ugrizy"[cmds['filter']]
    fiducial_telescope = batoid.Optic.fromYaml(f"LSST_{band}.yaml")
    wavelength_dict = dict(
        u=365.49e-9,
        g=480.03e-9,
        r=622.20e-9,
        i=754.06e-9,
        z=868.21e-9,
        y=991.66e-9
    )
    wavelength = wavelength_dict[band]
    camera = LsstCamMapper().camera

    rotTelPos =  cmds['rottelpos'] * galsim.degrees

    # Ambient conditions
    temperature = 293.
    pressure = 69.0
    H2O_pressure = 1.0

    factory = BatoidWCSFactory(
        boresight, rotTelPos, obstime, fiducial_telescope, wavelength,
        camera, temperature, pressure, H2O_pressure
    )

    # How do thx, thy move when zob, aob are perturbed?
    aob, zob, hob, dob, rob, eo = factory._ICRF_to_observed(
        boresight.ra.rad, boresight.dec.rad, all=True
    )
    # Verify that we start at thx=thy=0
    thx, thy = factory._observed_to_field(rob, dob)
    np.testing.assert_allclose(thx, 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(thy, 0.0, rtol=0, atol=1e-12)

    # Check roundtrip of alt az
    rc, dc = factory._observed_az_to_ICRF(aob, zob)
    np.testing.assert_allclose(
        rc, boresight.ra.rad,
        rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        dc, boresight.dec.rad,
        rtol=0, atol=1e-10
    )
    # Now decrease zenith angle, so moves higher in the sky.
    # thx should stay the same and thy should increase.
    dz = 0.001
    rc, dc = factory._observed_az_to_ICRF(aob, zob-dz)
    rob, dob = factory._ICRF_to_observed(rc, dc)
    thx, thy = factory._observed_to_field(rob, dob)
    np.testing.assert_allclose(thx, 0.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(thy, dz, rtol=0, atol=1e-8)

    # Now increase azimuth angle, so moves towards the ground East.  What happens to thx, thy?
    dz = 0.001
    rc, dc = factory._observed_az_to_ICRF(aob+dz, zob)
    rob, dob = factory._ICRF_to_observed(rc, dc)
    thx, thy = factory._observed_to_field(rob, dob)
    np.testing.assert_allclose(thx/np.sin(zob), dz, rtol=0, atol=1e-8)
    np.testing.assert_allclose(thy, 0.0, rtol=0, atol=1e-6)


if __name__ == "__main__":
    test_wcs_fit()
    test_imsim()
    test_intermediate_coord_sys()
