import os
from pathlib import Path
import numpy as np
import json
import logging
import galsim
from imsim import SkyModel, SkyGradient, make_batoid_wcs, RubinBandpass

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
        bandpass = RubinBandpass(band)
        sky_model = SkyModel(exptime, mjd, bandpass)
        sky_level = sky_model.get_sky_level(skyCoord)
        np.testing.assert_approx_equal(sky_level, expected_sky_levels[band],
                                       significant=4)

    # Repeat explicitly setting the airmass to 1.2.  This _ought_ to be the same
    # as the default bandpass files, but is slightly different for unknown but
    # presumably innocuous reasons (makes a difference ~part per 10000).
    # Test still passes at significant=3.
    for band in 'ugrizy':
        bandpass = RubinBandpass(band, airmass=1.2)
        sky_model = SkyModel(exptime, mjd, bandpass)
        sky_level = sky_model.get_sky_level(skyCoord)
        np.testing.assert_approx_equal(sky_level, expected_sky_levels[band],
                                       significant=3)


def test_sky_gradient():
    # Pointing info for observationId=11873 from the
    # baseline_v2.0_10yrs.db cadence file at
    # http://astro-lsst-01.astro.washington.edu:8080/
    ra = 54.9348753510528
    dec = -35.8385705255579
    mjd = 60232.3635999295
    rottelpos = 350.946271812373
    exptime = 30.

    band = 'i'
    bandpass = RubinBandpass(band)
    sky_model = SkyModel(exptime, mjd, bandpass)

    wcs = make_batoid_wcs(ra, dec, rottelpos, mjd, band, 'LsstCam')

    world_center = galsim.CelestialCoord(ra*galsim.degrees, dec*galsim.degrees)
    image_xsize = 4096  # Size in pixels of R22_S11 in x-direction
    sky_gradient = SkyGradient(sky_model, wcs, world_center, image_xsize)

    # Compute the sky levels at the three positions used by SkyGradient
    # for evaluating the planar approximation across the CCD.
    center = wcs.toImage(world_center)
    llc = galsim.PositionI(1, 1)
    lrc = galsim.PositionI(image_xsize, 1)
    ulc = galsim.PositionI(1, image_xsize)
    urc = galsim.PositionI(image_xsize, image_xsize)
    sky_level_center = sky_model.get_sky_level(world_center)
    rvals = [(center.x, center.y, sky_level_center),
             (llc.x, llc.y, sky_model.get_sky_level(wcs.toWorld(llc))),
             (lrc.x, lrc.y, sky_model.get_sky_level(wcs.toWorld(lrc))),
             (ulc.x, ulc.y, sky_model.get_sky_level(wcs.toWorld(ulc))),
             (urc.x, urc.y, sky_model.get_sky_level(wcs.toWorld(urc)))]

    for r in rvals[:3]:
        print(r[2]/sky_level_center, sky_gradient(*r[:2]))
        np.testing.assert_approx_equal(r[2]/sky_level_center,
                                       sky_gradient(*r[:2]),
                                       significant=7)
    # The upper left and right aren't as close.
    for r in rvals[3:]:
        print(r[2]/sky_level_center, sky_gradient(*r[:2]))
        np.testing.assert_approx_equal(r[2]/sky_level_center,
                                       sky_gradient(*r[:2]),
                                       significant=4)

    # Check that it gets applied by LSST_Image
    config = {
        'input': {
            'sky_model': {
                'exptime': exptime,
                'mjd': mjd
            },
        },
        'image': {
            'type': 'LSST_Image',
            'det_name': 'R22_S11',
            'xsize': image_xsize,
            'ysize': image_xsize,
            'wcs': wcs,
            'bandpass': { "type": "RubinBandpass", "band": band },
            'apply_sky_gradient': True,
            'sky_level': {'type': 'SkyLevel'},
            'nobjects': 0,
        },
    }
    galsim.config.ProcessInput(config)
    image = galsim.config.BuildImage(config)
    for pos, rv in zip((center.round(), llc, lrc, ulc, urc), rvals):
        print(image[pos], rv[2], wcs.pixelArea(image_pos=pos))
        np.testing.assert_allclose(image[pos], rv[2] * wcs.pixelArea(image_pos=pos), rtol=1.e-4)

    # Check that logging doesn't cause problems (since it used to).
    logger = logging.getLogger('test_sky_gradient')
    logger.setLevel(logging.INFO)
    del config['image']['_current_sky']  # GalSim caches this.  Make sure it remakes it.
    del config['image']['_current_sky_tag']
    image2 = galsim.config.BuildImage(config, logger=logger)
    assert image2 == image


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
