"""
Unit tests for imSim_class_factory module.
"""
from __future__ import absolute_import
import os
import unittest
import desc.imsim

class ImSimClassFactoryTestCase(unittest.TestCase):
    "TestCase class for imSim_class_factory."
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_imSim_class_factory(self):
        "Test code for imSim_class_factory."
        instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                                    'tiny_instcat.txt')
        commands, objects = desc.imsim.parsePhoSimInstanceFile(instcat_file)
        obs_md = desc.imsim.phosim_obs_metadata(commands)
        stars = \
            desc.imsim.ImSimStars(objects.query("uniqueId==1046817878020"),
                                  obs_md)
        self.assertEqual(stars.column_by_name('galSimType')[0], 'pointSource')
        self.assertAlmostEqual(stars.column_by_name('x_pupil')[0], -0.0008283)
        self.assertAlmostEqual(stars.column_by_name('y_pupil')[0], -0.00201296)

if __name__ == '__main__':
    unittest.main()
