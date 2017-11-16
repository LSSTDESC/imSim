"""
Unit tests for cosmic ray code.
"""
import os
import unittest
import numpy as np
import astropy.io.fits as fits
from desc.imsim import CosmicRays, write_cosmic_ray_catalog


class CosmicRaysTestCase(unittest.TestCase):
    "TestCase class for the cosmic ray code."
    def setUp(self):
        self.test_catalog = 'tmp_cr_catalog.fits'
        self.test_image = np.array([[0, 10, 0], [20, 30, 20], [0, 40, 0]])
        hdu_list = fits.HDUList([fits.PrimaryHDU()])
        fp_id = (0, 0, 0, 1, 2)
        x0 = (10, 10, 10, 0, 5)
        y0 = (20, 21, 22, 100, 4000)
        pixel_values = [x for x in self.test_image] + [[100], [50]]
        write_cosmic_ray_catalog(fp_id, x0, y0, pixel_values, 1., 100,
                                 outfile=self.test_catalog)

    def tearDown(self):
        try:
            os.remove(self.test_catalog)
        except OSError:
            pass

    def test_read_catalog(self):
        "Test the catalog contents."
        crs = CosmicRays()
        crs.read_catalog(self.test_catalog)
        self.assertEqual(len(crs), 3)
        self.assertEqual(len(crs[0]), 3)
        self.assertEqual(crs[0][0].x0, 10)
        self.assertEqual(crs[0][0].y0, 20)
        self.assertEqual(tuple(crs[0][0].pixel_values), (0, 10, 0))

    def test_paint_cr(self):
        "Test the painting of a CR into an input image array."
        imarr = np.zeros((3, 3))
        crs = CosmicRays()
        crs.read_catalog(self.test_catalog)
        imarr = crs.paint_cr(imarr, index=0, pixel=(0, 0))
        np.testing.assert_array_equal(self.test_image, imarr)


if __name__ == '__main__':
    unittest.main()
