import os
import unittest
import lsst.utils as lsstUtils
from desc.imsim import FocalPlaneReadout

class FocalPlaneReadoutTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_read_segmentation_txt(self):
        seg_file = os.path.join(lsstUtils.getPackageDir('imsim'),
                                'data', 'segmentation_itl.txt')
        readout_props = FocalPlaneReadout.read_phosim_seg_file(seg_file)
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
