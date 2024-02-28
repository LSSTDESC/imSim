import os
from pathlib import Path
import numpy as np
import astropy
import galsim
from lsst.afw import cameraGeom
import imsim


TEST_DATA_DIR = str(Path(__file__).parent / 'data')


def get_wcs_factory(opsim_db_file, visit):
    """Make a WCSFactory for a visit in the test opsim db file."""
    opsim_data = imsim.OpsimDataLoader(opsim_db_file, visit=visit)
    boresight = galsim.CelestialCoord(
        ra=opsim_data["fieldRA"] * galsim.degrees,
        dec=opsim_data["fieldDec"] * galsim.degrees,
    )
    mjd = opsim_data["mjd"]
    rottelpos = opsim_data["rotTelPos"] * galsim.degrees
    obstime = astropy.time.Time(mjd, format="mjd", scale="tai")
    band = opsim_data["band"]
    wcs_builder = imsim.BatoidWCSBuilder()
    telescope = imsim.load_telescope(f"LSST_{band}.yaml", rotTelPos=rottelpos)
    return wcs_builder.makeWCSFactory(boresight, obstime, telescope,
                                      bandpass=band)


def test_vignetting():
    """
    Test the vignetting function for several CCDs in the LSSTCam focal plane
    """
    vignetting = imsim.Vignetting('LSSTCam_vignetting_data.json')

    opsim_db_file = os.path.join(TEST_DATA_DIR, "small_opsim_9683.db")
    visit = 449053
    wcs_factory = get_wcs_factory(opsim_db_file, visit)

    camera = imsim.get_camera()

    # Test guider, wavefront, and science sensors, in order to
    # consider different CCD orientations and locations in the LSSTCam
    # focal plane.
    det_names = ["R00_SG0", "R40_SW0", "R30_S11", "R22_S11"]
    for det_name in det_names:
        det = camera[det_name]
        pix_to_fp = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
        wcs = wcs_factory.getWCS(det)

        # Vignetting function evaluated over the entire CCD:
        radii = imsim.Vignetting.get_pixel_radii(det)
        image_vignetting = vignetting.apply_to_radii(radii)

        # Compare with the values at the detector corners, using the
        # corresponding sky coordinates obtained from the WCS for this
        # detector to cross-check the .at_sky_coord(...) function.
        corners = [(int(_.x), int(_.y)) for _ in
                   det.getCorners(cameraGeom.PIXELS)]
        for corner in corners:
            image_pos = galsim.PositionD(*corner)
            sky_coord = wcs.toWorld(image_pos)
            sky_value = vignetting.at_sky_coord(sky_coord, wcs, pix_to_fp)
            test_values = sky_value, image_vignetting[corner[1], corner[0]]
            np.testing.assert_almost_equal(*test_values)

    # Repeat for ComCam
    camera = imsim.get_camera("LsstComCamSim")
    vignetting = imsim.Vignetting('LSSTComCamSim_vignetting_data.json')
    det_names = ["R22_S00", "R22_S01", "R22_S11"]

    for det_name in det_names:
        det = camera[det_name]
        pix_to_fp = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
        wcs = wcs_factory.getWCS(det)

        # Vignetting function evaluated over the entire CCD:
        radii = imsim.Vignetting.get_pixel_radii(det)
        image_vignetting = vignetting.apply_to_radii(radii)

        # Compare with the values at the detector corners, using the
        # corresponding sky coordinates obtained from the WCS for this
        # detector to cross-check the .at_sky_coord(...) function.
        corners = [(int(_.x), int(_.y)) for _ in
                   det.getCorners(cameraGeom.PIXELS)]
        for corner in corners:
            image_pos = galsim.PositionD(*corner)
            sky_coord = wcs.toWorld(image_pos)
            sky_value = vignetting.at_sky_coord(sky_coord, wcs, pix_to_fp)
            test_values = sky_value, image_vignetting[corner[1], corner[0]]
            print(test_values)
            np.testing.assert_almost_equal(*test_values)


if __name__ == '__main__':
    test_vignetting()
