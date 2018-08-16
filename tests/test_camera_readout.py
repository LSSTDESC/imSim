"""
Unit tests for electronics readout simulation code.
"""
from __future__ import absolute_import, print_function
import os
import itertools
import unittest
import astropy.io.fits as fits
import lsst.obs.lsstSim as lsstSim
import lsst.utils as lsstUtils
import desc.imsim


class ImageSourceTestCase(unittest.TestCase):
    "TestCase class for ImageSource."

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
        del self.image_source

    def test_create_from_eimage(self):
        "Test the .create_from_eimage static method."
        self.assertAlmostEqual(self.image_source.exptime, 30.)
        self.assertTupleEqual(self.image_source.eimage_data.shape, (4072, 4000))
        self.assertTupleEqual(
            self.image_source.amp_images['R22_S11_C00'].getArray().shape,
            (2020, 532))

    def test_get_amplifier_hdu(self):
        "Test the .get_amplifier_hdu method."
        hdu = self.image_source.get_amplifier_hdu('R22_S11_C10')
        self.assertEqual(hdu.header['DATASEC'], "[4:512,1:2000]")
        self.assertEqual(hdu.header['DETSEC'], "[509:1,1:2000]")
        self.assertEqual(hdu.header['BIASSEC'], "[513:532,1:2000]")
        self.assertAlmostEqual(hdu.header['GAIN'], 1.7)

        hdu = self.image_source.get_amplifier_hdu('R22_S11_C17')
        self.assertEqual(hdu.header['DATASEC'], "[4:512,1:2000]")
        self.assertEqual(hdu.header['DETSEC'], "[4072:3564,1:2000]")
        self.assertEqual(hdu.header['BIASSEC'], "[513:532,1:2000]")
        self.assertAlmostEqual(hdu.header['GAIN'], 1.7)

    def test_get_amp_image(self):
        "Test the .get_amp_image method."
        mapper = lsstSim.LsstSimMapper()
        camera = mapper.camera
        raft = 'R:2,2'
        ccd = 'S:1,1'
        sensor = camera[' '.join((raft, ccd))]
        amp_info_record = desc.imsim.set_itl_bboxes(sensor['0,3'])
        image = self.image_source.get_amp_image(amp_info_record)
        self.assertTupleEqual(image.getArray().shape, (2020, 532))


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


class FocalPlaneInfoTestCase(unittest.TestCase):
    "TestCase class for FocalPlaneInfo."

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_read_segmentation_txt(self):
        "Test reading of the segmentation.txt file."
        seg_file = os.path.join(lsstUtils.getPackageDir('imsim'), 'data',
                                'segmentation_itl.txt')
        readout_props = desc.imsim.FocalPlaneInfo.read_phosim_seg_file(seg_file)
        sensor_props = readout_props.get_sensor('R22_S11')
        self.assertEqual(sensor_props.width, 4072)
        self.assertEqual(sensor_props.height, 4000)
        self.assertEqual(sensor_props.num_amps, 16)
        amp_num = 0
        for col in '01':
            for row in '01234567':
                amp_num += 1
                amp_name = 'R22_S11_C' + col + row
                amp_props = readout_props.get_amp(amp_name)
                self.assertEqual(amp_props.gain, 1.7)
                self.assertEqual(amp_props.bias_level, 1000.)
                self.assertEqual(amp_props.read_noise, 7.)
                self.assertEqual(amp_props.dark_current, 0.02)
                crosstalk = [0]*16
                crosstalk[amp_num - 1] = 1.0
                self.assertListEqual(amp_props.crosstalk.tolist(), crosstalk)

        amp_name = 'R22_S11_C00'
        amp_props = readout_props.get_amp(amp_name)
        self.assertEqual(amp_props.flip_x, True)
        self.assertEqual(amp_props.flip_y, True)
        bbox = amp_props.mosaic_section
        self.assertEqual(bbox.getMinX(), 0)
        self.assertEqual(bbox.getMinY(), 2000)

        amp_name = 'R22_S11_C10'
        amp_props = readout_props.get_amp(amp_name)
        self.assertEqual(amp_props.flip_x, True)
        self.assertEqual(amp_props.flip_y, False)
        bbox = amp_props.mosaic_section
        self.assertEqual(bbox.getMinX(), 0)
        self.assertEqual(bbox.getMinY(), 0)


if __name__ == '__main__':
    unittest.main()
