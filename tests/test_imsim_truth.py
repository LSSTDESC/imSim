from __future__ import print_function
import os
import unittest
import desc.imsim
import desc.imsim.imsim_truth as imsim_truth

class ApparentMagnitudesTestCase(unittest.TestCase):
    "Test case class for ApparentMagnitudes class."
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_magnitudes(self):
        instcat = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                               'tiny_instcat.txt')
        commands, objects = desc.imsim.parsePhoSimInstanceFile(instcat)


        obj = objects.iloc[0]
        app_mags = imsim_truth.ApparentMagnitudes(obj.sedFilepath)
        mags = app_mags(obj)
        self.assertAlmostEqual(mags['u'], 25.948024179976176)
        self.assertAlmostEqual(mags['r'], 22.275029634051265)

        obj = objects.iloc[8]
        app_mags = imsim_truth.ApparentMagnitudes(obj.sedFilepath)
        mags = app_mags(obj)
        for band in mags:
            self.assertAlmostEqual(mags[band], 1000.)

if __name__ == '__main__':
    unittest.main()
