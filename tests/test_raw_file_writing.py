"""
Unit test code for raw file writing.
"""
import os
from pathlib import Path
import logging
import unittest
from astropy.io import fits
import galsim
import imsim

DATA_DIR = Path(__file__).parent / 'data'


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
        eimage_file = str(DATA_DIR / 'eimage_00449053-1-r-R22_S11-det094.fits.gz')
        eimage = galsim.fits.read(eimage_file)
        eimage.header = galsim.FitsHeader(file_name=eimage_file)

        det_name = "R22_S11"
        self.outfile = 'lsst_a_{}_r.fits'.format(det_name)

        # Add keyword values via an optional dict.
        added_keywords = {'GAUSFWHM': 0.4,
                          'FOOBAR': 'hello, world'}

        logger = logging.getLogger()
        ccd_readout = imsim.CcdReadout(eimage, logger, added_keywords=added_keywords)
        rng = galsim.BaseDeviate()
        hdus = ccd_readout.prepare_hdus(rng)
        hdr = hdus[0].header

        # Test some keywords.
        self.assertAlmostEqual(hdr['RA'], eimage.header['RATEL'])
        self.assertAlmostEqual(hdr['DEC'], eimage.header['DECTEL'])

        self.assertTrue(hdr["OBSID"].startswith('MC_S'))

        self.assertEqual(hdr['CHIPID'], det_name)

        # Ensure the following keywords are set.
        # XXX: These used to be different.  Used lsst.sims to compute AMEND from mjd_end
        #      Do we really care about getting this correct?
        self.assertEqual(hdr['AMSTART'], hdr['AMEND'])
        self.assertNotEqual(hdr['HASTART'], hdr['HAEND'])
        self.assertEqual(hdr['DATE-OBS'], '2025-12-08T01:04:57.498')
        self.assertEqual(hdr['DATE-END'], '2025-12-08T01:05:27.498')
        self.assertEqual(hdr['TIMESYS'], 'TAI')
        self.assertEqual(hdr['OBSTYPE'], 'SKYEXP')

        # Test the added_keywords.
        for key, value in added_keywords.items():
            self.assertEqual(hdr[key], value)

        # Write the file.
        ccd_readout.write_raw_file(hdus, self.outfile)


if __name__ == '__main__':
    unittest.main()
