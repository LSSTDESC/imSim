"""
Unit tests for InstCatTrimmer class.
"""
import os
import unittest
import imsim
import galsim
import astropy.time


class InstCatTrimmerTestCase(unittest.TestCase):
    """
    TestCase class for InstCatTrimmer.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def make_wcs(self, instcat_file, det_name):
        obs_md = imsim.OpsimMetaDict(instcat_file)
        boresight = galsim.CelestialCoord(ra=obs_md['rightascension'] * galsim.degrees,
                                          dec=obs_md['declination'] * galsim.degrees)
        rotTelPos = obs_md['rottelpos'] * galsim.degrees
        obstime = astropy.time.Time(obs_md['mjd'], format='mjd', scale='tai')
        band = obs_md['band']

        builder = imsim.BatoidWCSBuilder()
        return builder.makeWCS(boresight, rotTelPos, obstime, det_name, band)

    def test_InstCatTrimmer(self):
        """Unit test for InstCatTrimmer class."""
        instcat_file = os.path.join(os.path.dirname(__file__), 'tiny_instcat.txt')
        sensor = 'R22_S11'
        wcs = self.make_wcs(instcat_file, sensor)

        # Note: some objects in instcat are up to ~600 pixels off the image.
        # So need to use a largish edge_pix to keep them all.
        instcat = imsim.InstCatalog(instcat_file, wcs, edge_pix=1000, skip_invalid=False)
        self.assertEqual(instcat.nobjects, 24)

        # With the default edge_pix=100, only 17 make the cut.
        instcat = imsim.InstCatalog(instcat_file, wcs, skip_invalid=False)
        self.assertEqual(instcat.nobjects, 17)

        # Check the application of min_source.
        instcat = imsim.InstCatalog(instcat_file, wcs, edge_pix=1000, min_source=10,
                                    skip_invalid=False)
        self.assertEqual(instcat.nobjects, 24)

        instcat = imsim.InstCatalog(instcat_file, wcs, edge_pix=1000, min_source=12,
                                    skip_invalid=False)
        self.assertEqual(instcat.nobjects, 0)

    def test_inf_filter(self):
        """
        Test filtering of the ` inf ` string (i.e., bracked by spaces)
        appearing anywhere in the instance catalog entry to avoid
        underflows, floating point exceptions, etc.. from badly formed
        entries.
        """
        instcat_file = os.path.join(os.path.dirname(__file__), 'bad_instcat.txt')
        sensor = 'R22_S11'
        wcs = self.make_wcs(instcat_file, sensor)

        instcat = imsim.InstCatalog(instcat_file, wcs, edge_pix=1000, min_source=10,
                                    skip_invalid=False)
        self.assertEqual(instcat.nobjects, 26)


if __name__ == '__main__':
    unittest.main()
