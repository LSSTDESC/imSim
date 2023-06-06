"""
Unit tests for PSF-related quantities
"""
import unittest
import numpy as np
import imsim

class FWHMgeomTestCase(unittest.TestCase):
    """
    Class to test FWHMgeom calculation from rawSeeing, band, and
    altitude.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_airmass(self):
        """Test the calculation of airmass from altitude in degrees."""
        altitude = 52.542
        opsim = imsim.OpsimDataLoader.from_dict({})
        self.assertAlmostEqual(opsim.getAirmass(altitude), 1.24522984, places=7)

        opsim = imsim.OpsimDataLoader.from_dict(dict(altitude=altitude))
        self.assertAlmostEqual(opsim.getAirmass(), 1.24522984, places=7)

    def test_FWHMeff(self):
        """
        Test the calculation of the effective FWHM for a single Gaussian
        describing the PSF.
        """
        # Values from visit 197356, DC2, Run1.2p
        rawSeeing = 0.5059960
        band = 'r'
        altitude = 52.54199126195116065
        opsim = imsim.OpsimDataLoader.from_dict({})
        self.assertLess(np.abs(opsim.FWHMeff(rawSeeing, band, altitude) - 0.8300650), 0.03)

        opsim = imsim.OpsimDataLoader.from_dict(
                dict(rawSeeing=0.5059960,
                     band='r', altitude=52.54199126195116065))
        self.assertLess(np.abs(opsim.FWHMeff() - 0.8300650), 0.03)

    def test_FWHMgeom(self):
        """
        Test the calculation of FWHMgeom.
        """
        # Values from visit 197356, DC2, Run1.2p
        rawSeeing = 0.5059960
        band = 'r'
        altitude = 52.54199126195116065
        opsim = imsim.OpsimDataLoader.from_dict({})
        self.assertLess(np.abs(opsim.FWHMgeom(rawSeeing, band, altitude) - 0.7343130), 0.03)

        opsim = imsim.OpsimDataLoader.from_dict(
                dict(rawSeeing=0.5059960,
                     band='r', altitude=52.54199126195116065))
        self.assertLess(np.abs(opsim.FWHMgeom() - 0.7343130), 0.03)

if __name__ == '__main__':
    unittest.main()
