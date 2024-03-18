import numpy as np
import hashlib
import pytest
from imsim import make_batoid_wcs, CCD_Fringing, get_camera
import galsim

def test_fringing():
    """
    Test the fringing model.
    """
    # Set a random center ra/dec
    cra = 54.9348753510528
    cdec = -35.8385705255579
    world_center = galsim.CelestialCoord(cra*galsim.degrees, cdec*galsim.degrees)

    mjd = 60232.3635999295
    rottelpos = 350.946271812373
    band = 'y'
    camera = get_camera()
    det_name = 'R22_S11'
    serial_num = camera[det_name].getSerial()
    seed = int(hashlib.sha256(serial_num.encode('UTF-8')).hexdigest(), 16) & 0xFFFFFFFF

    xarr, yarr = np.meshgrid(range(4096), range(4004))

    # Testing a CCD with an arbitrary location on the focal plane.
    ra = 54.86
    dec = -35.76
    wcs = make_batoid_wcs(ra, dec, rottelpos, mjd, band, 'LsstCam')

    config = {
        'image': {
            'type': 'LSST_Image',
            'xsize': 4096,
            'ysize': 4004,
            'wcs': wcs,
            'nobjects': 0,
            'det_name': 'R22_S11',
        },
    }

    image = galsim.config.BuildImage(config)

    ccd_fringing = CCD_Fringing(true_center=image.wcs.toWorld(image.true_center),
                                boresight=world_center,
                                seed=seed, spatial_vary=True)
    # Test zero value error
    with pytest.raises(ValueError):
        ccd_fringing.calculate_fringe_amplitude(xarr, yarr, amplitude=0)

    # Test when spatial vary is True.
    fringe_map = ccd_fringing.calculate_fringe_amplitude(xarr,yarr)

    # Check std of the diagnoal of fringe map.
    np.testing.assert_approx_equal(np.std(np.diag(fringe_map)), 0.0014, significant=2)

    # Check the min/max of fringing varaition for the current offset.
    np.testing.assert_approx_equal(fringe_map.max(), 1.00205, significant=4)
    np.testing.assert_approx_equal(fringe_map.min(), 0.99794, significant=4)

    # Actually make a fringing map with this:
    sky_level = 1000
    sky_image = galsim.Image(bounds=image.bounds, wcs=image.wcs, init_value=sky_level)
    sky_image *= fringe_map

    # Check that this is the same image that the config processing makes
    config = {
        'image': {
            'type': 'LSST_Image',
            'xsize': 4096,
            'ysize': 4004,
            'wcs': wcs,
            'nobjects': 0,
            'sky_level_pixel': sky_level,
            'apply_fringing': True,
            'boresight': world_center,
            'det_name': 'R22_S11',
        },
        'det_name': 'R22_S11'
    }
    config_sky_image = galsim.config.BuildImage(config)

    np.testing.assert_allclose(config_sky_image.array, sky_image.array)

    # If boresight is missing, it raises an exception
    config = galsim.config.CleanConfig(config)
    del config['image']['boresight']
    with np.testing.assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)

    # Test when spatial vary is False. The fringe amplitude should be the same for
    # sensors at different locations

    ccd_fringing_1 = CCD_Fringing(true_center=image.wcs.toWorld(image.true_center),
                                boresight=world_center,
                                seed=seed, spatial_vary=False)
    fringe_map1 = ccd_fringing_1.calculate_fringe_amplitude(xarr,yarr)

    # Try another random location on the focal plane.
    ra = 58.86
    dec = -38.76
    wcs = make_batoid_wcs(ra, dec, rottelpos, mjd, band, 'LsstCam')

    ccd_fringing_2 = CCD_Fringing(true_center=image.wcs.toWorld(image.true_center),
                                boresight=world_center,
                                seed=seed, spatial_vary=False)
    fringe_map2 = ccd_fringing_2.calculate_fringe_amplitude(xarr,yarr)
    # Check if the two fringing maps are indentical.
    if np.array_equal(fringe_map1,fringe_map2) != True:
        raise ValueError("Fringe amplitude should be the same for sensors when spatial vary is False.")


def test_fringing_variation_level():
    # Regression test for pkl => fits conversion.
    for ra, dec, level in [
        (0, 0.1, 1.056503042318907),
        (0, 0.2, 1.1207294877266138),
        (0.2, -0.1, 1.0044602251026102),
        (1.1, 0.2, 1.0166040509448886),
        (-1.2, 0.5, 1.0389039410245318),
        (1.2, -0.4, 1.0204232685215646),
    ]:
        true_center = galsim.CelestialCoord(
            ra*galsim.degrees, dec*galsim.degrees
        )
        fringing = CCD_Fringing(
            true_center=true_center,
            boresight=galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees),
            seed=0,
            spatial_vary=True
        )
        np.testing.assert_allclose(
            fringing.fringe_variation_level(),
            level,
            atol=1e-10, rtol=1e-10
        )


if __name__ == '__main__':
    test_fringing()
    test_fringing_variation_level()