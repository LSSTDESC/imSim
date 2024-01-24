"""
Unit tests for electronics readout simulation code.
"""
import os
import copy
import numpy as np
from pathlib import Path
import itertools
import unittest
import astropy.io.fits as fits
import lsst.geom
from lsst.obs.base import createInitialSkyWcsFromBoresight
import imsim
import galsim

DATA_DIR = Path(__file__).parent / 'data'


class ImageSourceTestCase(unittest.TestCase):
    "TestCase class for ImageSource."

    def setUp(self):
        self.eimage_file = str(DATA_DIR / 'eimage_00449053-1-r-R22_S11-det094.fits')
        instcat_file = str(DATA_DIR / 'tiny_instcat.txt')
        self.image = galsim.fits.read(self.eimage_file)
        self.image.header = galsim.FitsHeader(file_name=self.eimage_file)
        self.config = {
            'image': {'random_seed': 1234},
            'input': {
                'opsim_data': {'file_name': instcat_file}
            },
            'output': {
                'readout' : {
                    'file_name': 'amp.fits',
                    'readout_time': 3,
                    'dark_current': 0.02,
                    'bias_level': 1000.,
                    'pcti': 1.e-6,
                    'scti': 1.e-6,
                    'added_keywords' : {
                        'TESTKEY1': 'TESTVAL1',
                        'SOMEMATH': '$1+2'
                    }
                }
            },
            'index_key': 'image_num',
            'image_num': 0,
            'det_name': 'R22_S11',
            'exptime': 30,
        }

        galsim.config.SetupConfigRNG(self.config)
        self.logger = galsim.config.LoggerWrapper(None)
        self.readout = imsim.CameraReadout()
        self.readout_config = self.config['output']['readout']
        galsim.config.ProcessInput(self.config)
        self.readout.initialize(None,None, self.readout_config, self.config, self.logger)
        self.readout.ensureFinalized(self.readout_config, self.config, [self.image], self.logger)

    def tearDown(self):
        pass

    def test_create_from_eimage(self):
        "Test the .create_from_eimage static method."
        hdus = self.readout.final_data
        self.assertAlmostEqual(hdus[0].header['EXPTIME'], 30.)
        self.assertTupleEqual(self.image.array.shape, (4004, 4096))
        for i in range(1,16):
            self.assertTupleEqual(hdus[i].data.shape, (2048, 576))

    def test_bias_level_override(self):
        """Test that the .bias_level attribute is None when bias_levels_file
        is provided"""
        bias_levels_file = 'LSSTCam_bias_levels_run_13421.json'
        ccd_readout = imsim.CcdReadout(self.image, self.logger,
                                       bias_level=1234.,
                                       bias_levels_file=bias_levels_file)
        self.assertIsNone(ccd_readout.bias_level)

    def test_get_amplifier_hdu(self):
        "Test the .get_amplifier_hdu method."
        hdus = self.readout.final_data
        hdu = hdus[1]
        self.assertEqual(hdu.header['EXTNAME'], "Segment10")
        self.assertEqual(hdu.header['DATASEC'], "[11:522,1:2002]")
        self.assertEqual(hdu.header['DETSEC'], "[512:1,4004:2003]")

        hdu = hdus[8]
        self.assertEqual(hdu.header['EXTNAME'], "Segment17")
        self.assertEqual(hdu.header['DATASEC'], "[11:522,1:2002]")
        self.assertEqual(hdu.header['DETSEC'], "[4096:3585,4004:2003]")

    def test_raw_file_headers(self):
        "Test contents of raw file headers."
        outfile = 'raw_file_test.fits'
        self.readout.writeFile(outfile, self.readout_config, self.config, self.logger)
        with fits.open(outfile) as hdus:
            self.assertEqual(hdus[0].header['IMSIMVER'], imsim.__version__)
            # Test added_keywords are included correctly
            self.assertEqual(hdus[0].header['TESTKEY1'], 'TESTVAL1')
            self.assertEqual(hdus[0].header['SOMEMATH'], '3')
        os.remove(outfile)

    def test_no_opsim(self):
        "Test running readout without OpsimData (e.g. for flats)"
        outfile = 'raw_no_opsim_test.fits'
        config = {  # Same as above, but no input field.
            'image': {'random_seed': 1234},
            'output': {
                'readout' : {
                    'file_name': 'amp.fits',
                    'readout_time': 3,
                    'dark_current': 0.02,
                    'bias_level': 1000.,
                    'pcti': 1.e-6,
                    'scti': 1.e-6,
                }
            },
            'index_key': 'image_num',
            'image_num': 0,
            'det_name': 'R20_S00',
            'exptime': 30,
        }
        # Use an image that starts as all 0.
        image = copy.copy(self.image)
        image.setZero()

        galsim.config.SetupConfigRNG(config)
        readout = imsim.CameraReadout()
        readout_config = config['output']['readout']
        readout.initialize(None,None, readout_config, config, self.logger)
        readout.ensureFinalized(readout_config, config, [image], self.logger)

        # Mean value in eimage should be ~ dark_current * dark_time
        dark_time = image.header['EXPTIME'] + 3
        eimage_mean = np.mean(image.array)
        eimage_var = np.var(image.array)
        assert np.isclose(eimage_mean, 0.02 * dark_time, rtol=1.e-3)
        assert np.isclose(eimage_var, 0.02 * dark_time, rtol=1.e-3)

        # Amp images also have the bias applied.
        camera_name = image.header['CAMERA']
        det_name = image.header['DET_NAME']
        ccd = imsim.Camera(camera_name)[det_name]
        for amp in range(16):
            read_noise = list(ccd.values())[amp].read_noise
            gain = list(ccd.values())[amp].gain
            amp_mean = np.mean(readout.final_data[amp+1].data)
            amp_var = np.var(readout.final_data[amp+1].data)
            assert np.isclose(amp_mean, 1000 + 0.02*dark_time, rtol=1.e-3)
            print(amp_var, 0.02*dark_time/gain + read_noise**2, (0.02*dark_time/gain + read_noise**2-amp_var)/amp_var)
            assert np.isclose(amp_var, 0.02*dark_time/gain + read_noise**2, rtol=4.e-3)

        readout.writeFile(outfile, self.readout_config, self.config, self.logger)
        with fits.open(outfile) as hdus:
            self.assertEqual(hdus[0].header['IMSIMVER'], imsim.__version__)
            self.assertEqual(hdus[0].header['FILTER'], 'r_57')
            self.assertEqual(hdus[0].header['MJD-OBS'], 61017.0451099272)
        os.remove(outfile)

        # Filter is option in the readout section to show up in the header.
        readout_config['filter'] = 'r'
        readout.initialize(None,None, readout_config, config, self.logger)
        readout.ensureFinalized(readout_config, config, [image], self.logger)
        readout.writeFile(outfile, self.readout_config, self.config, self.logger)
        with fits.open(outfile) as hdus:
            self.assertEqual(hdus[0].header['FILTER'], 'r_57')

        # Make sure it parses it, not just gets it.
        readout_config['filter'] = '$"Happy Birthday!"[8]'
        readout.initialize(None,None, readout_config, config, self.logger)
        readout.ensureFinalized(readout_config, config, [image], self.logger)
        readout.writeFile(outfile, self.readout_config, self.config, self.logger)
        with fits.open(outfile) as hdus:
            self.assertEqual(hdus[0].header['FILTER'], 'r_57')
        os.remove(outfile)

        # If bias level, dark_current, and read_noise are all 0, then output should be 0.
        readout_config = galsim.config.CleanConfig(readout_config)
        readout_config['bias_level'] = 0.
        readout_config['dark_current'] = 0.
        readout_config['read_noise'] = 0.
        readout.initialize(None,None, readout_config, config, self.logger)
        image.setZero()
        readout.ensureFinalized(readout_config, config, [image], self.logger)
        np.testing.assert_array_equal(image.array, 0.)
        for amp in range(16):
            amp_image = readout.final_data[amp+1].data
            np.testing.assert_array_equal(amp_image, 0.)


def sky_coord(ra, dec, units=lsst.geom.degrees):
    """Sky coordinate object in Rubin DM code."""
    return lsst.geom.SpherePoint(lsst.geom.Angle(ra, units),
                                 lsst.geom.Angle(dec, units))


def test_compute_rotSkyPos():
    """
    Test rotSkyPos (= ROTANGLE in FITS header) calculation.  This
    angle is used by LSST code to find the initial WCS from raw
    images.
    """
    # Pointing info for observationId=11873 from the
    # baseline_v2.0_10yrs.db cadence file at
    # http://astro-lsst-01.astro.washington.edu:8080/
    ra0 = 54.9348753510528
    dec0 = -35.8385705255579
    rottelpos = 341.776422048124
    obsmjd = 60232.3635999295
    band = 'i'
    camera_name = 'LsstCamImSim'
    detector = 94

    batoid_wcs = imsim.readout.make_batoid_wcs(ra0, dec0, rottelpos, obsmjd,
                                               band, camera_name)

    # Compute rotSkyPos and undo the sign change and 90 deg rotation
    # needed for compatibility with the imsim config in
    # astro_metadata_translator.
    rotSkyPos = 90 - imsim.readout.compute_rotSkyPos(ra0, dec0, rottelpos,
                                                     obsmjd, band,
                                                     camera_name=camera_name)

    # Create an initial WCS using the LSST code, given the boresight
    # direction and computed rotSkyPos value.
    boresight = sky_coord(ra0, dec0)
    orientation = lsst.geom.Angle(rotSkyPos, lsst.geom.degrees)
    camera = imsim.get_camera(camera_name)

    lsst_wcs = createInitialSkyWcsFromBoresight(boresight, orientation,
                                                camera[detector])

    # Compare coordinates for locations on the CCD near each corner
    # and compute the sum of the offsets between WCSs in pixel units.
    value = 0
    for x, y in ((0, 0), (4000, 0), (4000, 4000), (0, 4000)):
        ra, dec = batoid_wcs.xyToradec(x, y, units='degrees')
        batoid_coord = sky_coord(ra, dec)
        lsst_coord = lsst_wcs.pixelToSky(x, y)
        value += batoid_coord.separation(lsst_coord).asDegrees()*3600./0.2

    # Check that the summed offsets are less than about 1 pixel per
    # location.
    assert(value < 5.)


if __name__ == '__main__':
    unittest.main()
