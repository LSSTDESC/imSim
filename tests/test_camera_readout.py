"""
Unit tests for electronics readout simulation code.
"""
from __future__ import absolute_import, print_function
import os
import itertools
import unittest
import astropy.io.fits as fits
import lsst.obs.lsstCam as obs_lsstCam
import lsst.utils as lsstUtils
import desc.imsim

desc.imsim.read_config()

class ImageSourceTestCase(unittest.TestCase):
    "TestCase class for ImageSource."
    imsim_dir = lsstUtils.getPackageDir('imsim')
    eimage_file = os.path.join(imsim_dir, 'tests', 'data',
                               'lsst_e_161899_R22_S11_r.fits.gz')
    image_source \
        = desc.imsim.ImageSource.create_from_eimage(eimage_file, 'R22_S11')
    def setUp(self):
        imsim_dir = lsstUtils.getPackageDir('imsim')
        self.eimage_file = os.path.join(imsim_dir, 'tests', 'data',
                                        'lsst_e_161899_R22_S11_r.fits.gz')
        seg_file = os.path.join(imsim_dir, 'data', 'segmentation_itl.txt')
        self.image_source \
            = desc.imsim.ImageSource.create_from_eimage(self.eimage_file,
                                                        'R22_S11',
                                                        seg_file=seg_file)

    def tearDown(self):
        pass

    def test_create_from_eimage(self):
        "Test the .create_from_eimage static method."
        self.assertAlmostEqual(self.image_source.exptime, 30.)
        self.assertTupleEqual(self.image_source.eimage_data.shape, (4072, 4000))
        self.assertTupleEqual(
            self.image_source.amp_images['R22_S11_C00'].getArray().shape,
            (2048, 544))

    def test_get_amplifier_hdu(self):
        "Test the .get_amplifier_hdu method."
        hdu = self.image_source.get_amplifier_hdu('R22_S11_C10')
        self.assertEqual(hdu.header['DATASEC'], "[4:512,1:2000]")
        self.assertEqual(hdu.header['DETSEC'], "[509:1,1:2000]")

        hdu = self.image_source.get_amplifier_hdu('R22_S11_C17')
        self.assertEqual(hdu.header['DATASEC'], "[4:512,1:2000]")
        self.assertEqual(hdu.header['DETSEC'], "[4072:3564,1:2000]")

    def test_get_amp_image(self):
        "Test the .get_amp_image method."
        camera = obs_lsstCam.LsstCamMapper().camera
        det = camera['R22_S11']
        amp_info_record = desc.imsim.set_itl_bboxes(det['C03'])
        image = self.image_source.get_amp_image(amp_info_record)
        self.assertTupleEqual(image.getArray().shape, (2048, 544))


class NoaoKeywordTestCase(unittest.TestCase):
    "TestCase class for raft-level NOAO mosaicking keywords"

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_set_noao_keywords(self):
        "Test the set_noao_keywords function."
        # Test that DETSIZE has been set so that appropriate vendor
        # geometry can be used.
        hdu = fits.ImageHDU()
        hdu.name = 'Segment00'
        self.assertRaises(RuntimeError, desc.imsim.set_noao_keywords,
                          hdu, 'S00')

        # Test keyword values written by this code against real TS8
        # raft level headers.
        exclude = ('NAXIS BITPIX CHECKSUM HEADVER CHANNEL EXTNAME '
                   'CCDSUM AVERAGE AVGBIAS STDVBIAS STDEV BSCALE BZERO '
                   'DATASUM SLOT')
        ccd_geoms = {'E2V': '[1:4096,1:4004]',
                     'ITL': '[1:4072,1:4000]'}
        for vendor, detsize in ccd_geoms.items():
            for slot in ('S%i%i' % x for x in
                         itertools.product(range(3), range(3))):
                ref_file \
                    = os.path.join(lsstUtils.getPackageDir('imsim'), 'tests',
                                   'data', '%s_raft_example_%s.fits.gz'
                                   % (vendor, slot))
                with fits.open(ref_file) as ref:
                    for amp in ('%i%i' % chan for chan in
                                itertools.product((0, 1), range(8))):
                        hdu = fits.ImageHDU()
                        hdu.name = 'Segment%s' % amp
                        hdu.header['DETSIZE'] = detsize
                        hdu = desc.imsim.set_noao_keywords(hdu, slot)
                        for keyword in hdu.header.keys():
                            if keyword not in exclude:
                                self.assertEqual(hdu.header[keyword],
                                                 ref[hdu.name].header[keyword])


if __name__ == '__main__':
    unittest.main()
