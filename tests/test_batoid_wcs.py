from pathlib import Path
import numpy as np
import batoid
import galsim
from lsst.afw import cameraGeom
from astropy.time import Time

import imsim

DATA_DIR = Path(__file__).parent / 'data'


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
    camera = imsim.get_camera()

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

        wavelength = 620. # nm
        telescope = imsim.load_telescope(
            "LSST_r.yaml", rotTelPos=rotTelPos*galsim.radians
        )

        factory = imsim.BatoidWCSFactory(
            boresight, obstime, telescope, wavelength,
            camera, temperature, pressure, H2O_pressure
        )

        aob, zob, hob, dob, rob, eo = factory._ICRF_to_observed(
            boresight.ra.rad, boresight.dec.rad, all=True
        )

        # If zenith angle > 70 degrees, try again
        if np.rad2deg(zob) > 70:
            continue

        # Pick a few detectors randomly
        for idet in rng.choice(len(camera), 3):
            det = camera[idet]
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

    with open(DATA_DIR / "wcs_466749.yaml", 'r') as f:
        wcss = yaml.safe_load(f)

    cmds = {}
    with open(DATA_DIR / "phosim_cat_466749.txt", 'r') as f:
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
    wavelength_dict = dict(
        u=365.49,
        g=480.03,
        r=622.20,
        i=754.06,
        z=868.21,
        y=991.66
    )
    wavelength = wavelength_dict[band]
    camera = imsim.get_camera()

    rotTelPos =  cmds['rottelpos'] * galsim.degrees
    telescope = imsim.load_telescope(f"LSST_{band}.yaml", rotTelPos=rotTelPos)
    # Ambient conditions
    # These are a guess.
    temperature = 293.
    pressure = 69.0
    H2O_pressure = 1.0

    # Start by constructing a refractionless factory, which we can use to
    # cross-check some of the other values in the phosim cmd file.
    factory = imsim.BatoidWCSFactory(
        boresight, obstime, telescope, wavelength,
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
        rtol=0, atol=6.0,
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

    # We accidentally simulated DC2 with the camera rotated 180 degrees too far.
    # That includes the regression test data here.  So to fix the WCS code, but
    # still use the same regression data, we need to add 180 degrees here.  Just
    # rotate the camera by another 180 degrees
    telescope = telescope.withLocallyRotatedOptic(
        "LSSTCamera", batoid.RotZ(np.deg2rad(180))
    )

    # For actual WCS check, we use a factory that _does_ know about refraction.
    factory = imsim.BatoidWCSFactory(
        boresight, obstime, telescope, wavelength,
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
    from numpy import array

    with open(DATA_DIR / "wcs_466749.yaml", "r") as f:
        wcss = yaml.safe_load(f)

    cmds = {}
    with open(DATA_DIR / "phosim_cat_466749.txt", 'r') as f:
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
    wavelength_dict = dict(
        u=365.49,
        g=480.03,
        r=622.20,
        i=754.06,
        z=868.21,
        y=991.66
    )
    wavelength = wavelength_dict[band]
    camera = imsim.get_camera()

    rotTelPos =  cmds['rottelpos'] * galsim.degrees
    telescope = imsim.load_telescope(f"LSST_{band}.yaml", rotTelPos=rotTelPos)

    # Ambient conditions
    temperature = 293.
    pressure = 69.0
    H2O_pressure = 1.0

    factory = imsim.BatoidWCSFactory(
        boresight, obstime, telescope, wavelength,
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
    # thx should stay the same and thy should decrease.
    dz = 0.001
    rc, dc = factory._observed_az_to_ICRF(aob, zob-dz)
    rob, dob = factory._ICRF_to_observed(rc, dc)
    thx, thy = factory._observed_to_field(rob, dob)
    np.testing.assert_allclose(thx, 0.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(thy, -dz, rtol=0, atol=1e-8)

    # Now increase azimuth angle, so moves towards the ground East.  What happens to thx, thy?
    dz = 0.001
    rc, dc = factory._observed_az_to_ICRF(aob+dz, zob)
    rob, dob = factory._ICRF_to_observed(rc, dc)
    thx, thy = factory._observed_to_field(rob, dob)
    np.testing.assert_allclose(thx/np.sin(zob), dz, rtol=0, atol=1e-8)
    np.testing.assert_allclose(thy, 0.0, rtol=0, atol=1e-6)

def test_config():
    """Check the config interface to BatoidWCS.
    """
    import yaml
    import astropy.units as u
    from tqdm import tqdm
    # Need these for `eval` below
    from numpy import array

    # Same test suite as used in test_imsim above.
    # This time, we just use this for the det names.
    with open(DATA_DIR / "wcs_466749.yaml", 'r') as f:
        wcss = yaml.safe_load(f)

    cmds = {}
    with open(DATA_DIR / "phosim_cat_466749.txt", 'r') as f:
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
    obstime = Time(cmds['mjd'], format='mjd', scale='utc')
    obstime -= 15*u.s
    band = "ugrizy"[cmds['filter']]
    wavelength_dict = dict(
        u=365.49,
        g=480.03,
        r=622.20,
        i=754.06,
        z=868.21,
        y=991.66
    )
    wavelength = wavelength_dict[band]
    camera = imsim.get_camera()

    rotTelPos =  cmds['rottelpos'] * galsim.degrees
    telescope = imsim.load_telescope(f"LSST_{band}.yaml", rotTelPos=rotTelPos)
    # Non-default values.
    temperature = 293.
    pressure = 69.0
    H2O_pressure = 2.0

    factory = imsim.BatoidWCSFactory(
        boresight, obstime, telescope, wavelength,
        camera,
        temperature=temperature,
        pressure=pressure,
        H2O_pressure=H2O_pressure
    )

    config = {
        'input': {
            'telescope': {
                'file_name':f"LSST_{band}.yaml",
                'rotTelPos': rotTelPos
            }
        },
        'image': {
            'wcs': {
                'type': 'Batoid',
                'boresight': boresight,
                'camera': 'LsstCam',
                'obstime': obstime,
                'wavelength': wavelength,
                'temperature': temperature,
                'pressure': pressure,
                'H2O_pressure': H2O_pressure,
                'order': 2,
            }
        }
    }

    rng = np.random.default_rng(1234)
    for k in tqdm(wcss.keys()):
        name = k[18:25].replace('-', '_')
        det = camera[name]

        wcs1 = factory.getWCS(det, order=2)
        config['image']['wcs']['det_name'] = name
        galsim.config.RemoveCurrent(config['image']['wcs'])
        galsim.config.ProcessInput(config)
        wcs2 = galsim.config.BuildWCS(config['image'], 'wcs', config)

        # Test points
        xs = rng.uniform(0, 4000, 100)
        ys = rng.uniform(0, 4000, 100)
        ra1, dec1 = wcs1.xyToradec(xs, ys, units='radians')
        ra2, dec2 = wcs2.xyToradec(xs, ys, units='radians')
        np.testing.assert_allclose(ra1, ra2)
        np.testing.assert_allclose(dec1, dec2)

    # Test == when identical
    galsim.config.RemoveCurrent(config['image']['wcs'])
    wcs3 = galsim.config.BuildWCS(config['image'], 'wcs', config)
    assert wcs3 == wcs2

    # Test that pressure and temperature matter.
    config['image']['wcs']['temperature'] = 250
    galsim.config.RemoveCurrent(config['image']['wcs'])
    wcs4 = galsim.config.BuildWCS(config['image'], 'wcs', config)
    assert wcs4 != wcs2

    config['image']['wcs']['temperature'] = temperature
    config['image']['wcs']['pressure'] = 55
    galsim.config.RemoveCurrent(config['image']['wcs'])
    wcs5 = galsim.config.BuildWCS(config['image'], 'wcs', config)
    assert wcs5 != wcs2

    config['image']['wcs']['pressure'] = pressure
    config['image']['wcs']['H2O_pressure'] = 10
    galsim.config.RemoveCurrent(config['image']['wcs'])
    wcs6 = galsim.config.BuildWCS(config['image'], 'wcs', config)
    assert wcs6 != wcs2

    # Test defaults
    del config['image']['wcs']['temperature']
    del config['image']['wcs']['pressure']
    del config['image']['wcs']['H2O_pressure']
    galsim.config.RemoveCurrent(config['image']['wcs'])
    config = galsim.config.CleanConfig(config)
    galsim.config.ProcessInput(config)
    wcs7 = galsim.config.BuildWCS(config['image'], 'wcs', config)
    default_pressure = 101.325 * (1-2.25577e-5*2715)**5.25588
    wcs7a = imsim.BatoidWCSFactory(
        boresight, obstime, telescope, wavelength, camera,
        temperature=280,
        pressure=default_pressure,
        H2O_pressure=1.0,
    ).getWCS(det, order=2)
    assert wcs7 == wcs7a

    # Default wavelength from bandpass
    del config['image']['wcs']['wavelength']
    config['bandpass'] = galsim.Bandpass('LSST_r.dat', 'nm')
    galsim.config.RemoveCurrent(config['image']['wcs'])
    config = galsim.config.CleanConfig(config)
    galsim.config.ProcessInput(config)
    wcs8 = galsim.config.BuildWCS(config['image'], 'wcs', config)
    wcs8a = imsim.BatoidWCSFactory(
        boresight, obstime, telescope,
        wavelength=config['bandpass'].effective_wavelength,
        camera=camera,
        temperature=280,
        pressure=default_pressure,
        H2O_pressure=1.0,
    ).getWCS(det, order=2)
    assert wcs8 == wcs8a

    del config['bandpass']
    config['image']['bandpass'] = {
        'file_name' : 'LSST_r.dat',
        'wave_type' : 'nm',
    }
    galsim.config.RemoveCurrent(config['image']['wcs'])
    config = galsim.config.CleanConfig(config)
    galsim.config.ProcessInput(config)
    wcs8b = galsim.config.BuildWCS(config['image'], 'wcs', config)
    assert wcs8b == wcs8a

    # Obstime can be a string
    print('obstime = ',obstime.to_value('iso'), type(obstime.to_value('iso')))
    config['image']['wcs']['obstime'] = obstime.to_value('iso')
    # Doesn't quite roundtrip perfectly.  But within a millisecond.
    obstime = Time(obstime.to_value('iso'), scale='tai')
    print('obstime => ',obstime)
    galsim.config.RemoveCurrent(config['image']['wcs'])
    config = galsim.config.CleanConfig(config)
    galsim.config.ProcessInput(config)
    wcs9 = galsim.config.BuildWCS(config['image'], 'wcs', config)
    wcs9a = imsim.BatoidWCSFactory(
        boresight, obstime, telescope,
        wavelength=config['bandpass'].effective_wavelength,
        camera=camera,
        temperature=280,
        pressure=default_pressure,
        H2O_pressure=1.0,
    ).getWCS(det, order=2)
    assert wcs9 == wcs9a


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
