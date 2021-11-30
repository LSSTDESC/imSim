"""
Unit tests for electronics readout simulation code.
"""
import os
import itertools
import unittest
import astropy.io.fits as fits
import imsim
import galsim

def transpose(image):
    # TODO: This should be a method of the galsim.Image class.
    b1 = image.bounds
    b2 = galsim.BoundsI(b1.ymin, b1.ymax, b1.xmin, b1.xmax)
    return galsim._Image(image.array.T, b2, image.wcs)

class ImageSourceTestCase(unittest.TestCase):
    "TestCase class for ImageSource."

    def setUp(self):
        # Note: This e-image was originally made using the sizes for ITL sensors.
        #       Now this raft is set to be e2v, so we need to fake the numbers below
        #       to pretend it is still an ITL raft.
        #       This is why we use R20_S11 for the det_name below.
        self.eimage_file = os.path.join('data', 'lsst_e_161899_R22_S11_r.fits.gz')
        self.image = galsim.fits.read(self.eimage_file)
        # Also, this file was made in phosim convention, which swaps x,y relative
        # to the normal convention.  So we need to transpose the image after reading it.
        self.image = transpose(self.image)
        self.config = {
            'image': {'random_seed': 1234},
            'input': {
                'opsim_meta_dict': {'file_name': 'tiny_instcat.txt'},
            },
            'output': {
                'readout' : {
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
            'exp_time': 30,
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
        self.assertTupleEqual(self.image.array.shape, (4000, 4072))
        for i in range(1,16):
            self.assertTupleEqual(hdus[i].data.shape, (2048, 544))

    def test_get_amplifier_hdu(self):
        "Test the .get_amplifier_hdu method."
        hdus = self.readout.final_data
        hdu = hdus[1]
        self.assertEqual(hdu.header['EXTNAME'], "Segment10")
        self.assertEqual(hdu.header['DATASEC'], "[4:512,1:2000]")
        # XXX: This test is currently failing.  This DETSEC now seems to correspond to
        #      hdu 17, which is C00, not C10.  C10 has DETSEC = [509:1,4000:2001]
        #      I don't know what should be changed to fix this.  Is it an error in the code?
        #      Or is the test wrong now after some change to the camera layout?
        #      Or did I screw something up in trying to adapt the test from the old code?
        self.assertEqual(hdu.header['DETSEC'], "[509:1,1:2000]")

        hdu = hdus[7]
        self.assertEqual(hdu.header['EXTNAME'], "Segment17")
        self.assertEqual(hdu.header['DATASEC'], "[4:512,1:2000]")
        self.assertEqual(hdu.header['DETSEC'], "[4072:3564,1:2000]")

    def test_raw_file_headers(self):
        "Test contents of raw file headers."
        outfile = 'raw_file_test.fits'
        self.readout.writeFile(outfile, self.readout_config, self.config, self.logger)
        with fits.open(outfile) as hdus:
            self.assertEqual(hdus[0].header['IMSIMVER'], imsim.__version__)
        os.remove(outfile)


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
