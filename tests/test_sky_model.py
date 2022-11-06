import os
from pathlib import Path
import numpy as np
import json
import galsim
from imsim import SkyModel, SkyGradient, make_batoid_wcs


DATA_DIR = Path(__file__).parent / 'data'


def test_sky_model():
    """
    Test the sky_model code for a random viable pointing, comparing to
    sky background values computed by hand using the
    rubin_sim.skybrightness code.
    """
    # Pointing info for observationId=11873 from the
    # baseline_v2.0_10yrs.db cadence file at
    # http://astro-lsst-01.astro.washington.edu:8080/
    ra = 54.9348753510528
    dec = -35.8385705255579
    skyCoord = galsim.CelestialCoord(ra*galsim.degrees, dec*galsim.degrees)
    mjd = 60232.3635999295
    exptime = 30.

    # Load expected sky bg values obtained from running the
    # rubin_sim.skybrightness code using sky_level_reference_values.py script.
    with open(os.path.join(DATA_DIR, 'reference_sky_levels.json')) as fobj:
        expected_sky_levels = json.load(fobj)

    for band in 'ugrizy':
        bandpass = galsim.Bandpass(f'LSST_{band}.dat', wave_type='nm')
        sky_model = SkyModel(exptime, mjd, bandpass)
        sky_level = sky_model.get_sky_level(skyCoord)
        np.testing.assert_approx_equal(sky_level, expected_sky_levels[band],
                                       significant=5)


def test_sky_gradient():
    # Pointing info for observationId=11873 from the
    # baseline_v2.0_10yrs.db cadence file at
    # http://astro-lsst-01.astro.washington.edu:8080/
    ra = 54.9348753510528
    dec = -35.8385705255579
    skyCoord = galsim.CelestialCoord(ra*galsim.degrees, dec*galsim.degrees)
    mjd = 60232.3635999295
    rottelpos = 350.946271812373
    exptime = 30.

    band = 'i'
    bandpass = galsim.Bandpass(f'LSST_{band}.dat', wave_type='nm')
    sky_model = SkyModel(exptime, mjd, bandpass)

    wcs = make_batoid_wcs(ra, dec, rottelpos, mjd, band, 'LsstCam')

    world_center = galsim.CelestialCoord(ra*galsim.degrees, dec*galsim.degrees)
    image_xsize = 4096  # Size in pixels of R22_S11 in x-direction
    # Set pixel scale to 1 arcsec since the SkyGradient functor returns
    # photons/pixel, and we are comparing to photons/arcsec**2.
    pixel_scale = 1
    sky_gradient = SkyGradient(sky_model, wcs, world_center, image_xsize,
                               pixel_scale=pixel_scale)

    # Compute the sky levels at the three positions used by SkyGradient
    # for evaluating the planar approximation across the CCD.
    center = wcs.toImage(world_center)
    llc = galsim.PositionD(0, 0)
    lrc = galsim.PositionD(image_xsize, 0)
    sky_level_center = sky_model.get_sky_level(world_center)
    rvals = [(center.x, center.y, sky_level_center),
             (llc.x, llc.y, sky_model.get_sky_level(wcs.toWorld(llc))),
             (lrc.x, lrc.y, sky_model.get_sky_level(wcs.toWorld(lrc)))]

    for r in rvals:
        np.testing.assert_approx_equal(r[2],
                                       sky_gradient(*r[:2]) + sky_level_center,
                                       significant=7)
