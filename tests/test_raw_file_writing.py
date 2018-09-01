"""
Unit test code for raw file writing.
"""
import os
import unittest
from astropy.io import fits
import lsst.sims.GalSimInterface as sims_gsi
from lsst.sims.photUtils import PhotometricParameters
from lsst.sims.utils import ObservationMetaData
import desc.imsim


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
        camera_wrapper = sims_gsi.LSSTCameraWrapper()
        phot_params = PhotometricParameters()
        obs_md = ObservationMetaData(pointingRA=23.0,
                                     pointingDec=12.0,
                                     rotSkyPos=13.2,
                                     mjd=59580.,
                                     bandpassName='r')
        obs_md.OpsimMetaData = {'obshistID': 219976}
        detname = "R:2,2 S:1,1"
        chipid = 'R{}_S{}'.format(detname[2:5:2], detname[8:11:2])
        detector = sims_gsi.make_galsim_detector(camera_wrapper, detname,
                                                 phot_params, obs_md)
        gs_interpreter = sims_gsi.GalSimInterpreter(detectors=[detector])
        gs_image = gs_interpreter.blankImage(detector)
        raw_image = desc.imsim.ImageSource.create_from_galsim_image(gs_image)
        self.outfile = 'lsst_a_{}_r.fits'.format(chipid)
        raw_image.write_fits_file(self.outfile, overwrite=True)
        # Test some keywords.
        raw_file = fits.open(self.outfile)
        self.assertAlmostEqual(raw_file[0].header['RATEL'], obs_md.pointingRA)
        self.assertAlmostEqual(raw_file[0].header['DECTEL'], obs_md.pointingDec)
        self.assertAlmostEqual(raw_file[0].header['ROTANGLE'], obs_md.rotSkyPos)
        self.assertEqual(raw_file[0].header['CHIPID'], chipid)


if __name__ == '__main__':
    unittest.main()
