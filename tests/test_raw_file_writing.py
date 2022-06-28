"""
Unit test code for raw file writing.
"""
import os
import unittest
from astropy.io import fits
import imsim


class RawFileOutputTestCase(unittest.TestCase):
    """
    TestCase subclass to test direct writing of `raw` files from a
    galsim Image produced by a GalSimInterpreter object.
    """
    def setUp(self):
        self.outfile = None

    def tearDown(self):
        if os.path.isfile(self.outfile):
            os.remove(self.outfile)

    def test_raw_file_writing(self):
        """
        Test the writing of raw files directly from a galsim.Image.
        This is mostly an operational test that the raw files can be
        written from a galsim image.
        """
        opsim_md = imsim.OpsimMetaDict.from_dict(
            dict(fieldRA=31.1133844,
                 fieldDec=-10.0970060,
                 rotSkyPos=146.24369132422518,
                 rotTelPos=1.,
                 mjd=59797.2854090,
                 band='r',
                 observationId=161899,
                 FWHMgeom=0.7,
                 altitude=43.6990272,
                 rawSeeing=0.6))

        det_name = "R22_S11"
        self.outfile = 'lsst_a_{}_r.fits'.format(det_name)

        # Add keyword values via an optional dict.
        added_keywords = {'GAUSFWHM': 0.4,
                          'FOOBAR': 'hello, world'}

        hdu = imsim.get_primary_hdu(opsim_md, det_name, added_keywords=added_keywords)
        hdr = hdu.header

        # Test some keywords.
        self.assertAlmostEqual(hdr['RATEL'], opsim_md['fieldRA'])
        self.assertAlmostEqual(hdr['DECTEL'], opsim_md['fieldDec'])
        self.assertAlmostEqual(hdr['ROTANGLE'], opsim_md['rotSkyPos'])
        self.assertEqual(hdr['CHIPID'], det_name)

        # Ensure the following keywords are set.
        # XXX: These used to be different.  Used lsst.sims to compute AMEND from mjd_end
        #      Do we really care about getting this correct?
        self.assertEqual(hdr['AMSTART'], hdr['AMEND'])
        self.assertNotEqual(hdr['HASTART'], hdr['HAEND'])
        self.assertEqual(hdr['DATE-OBS'], '2022-08-06T06:50:59.338')
        self.assertEqual(hdr['DATE-END'], '2022-08-06T06:51:29.338')
        self.assertEqual(hdr['TIMESYS'], 'TAI')
        self.assertEqual(hdr['OBSTYPE'], 'SKYEXP')

        # Test the added_keywords.
        for key, value in added_keywords.items():
            self.assertEqual(hdr[key], value)


if __name__ == '__main__':
    unittest.main()
