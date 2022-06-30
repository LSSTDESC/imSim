"""
Unit tests for tree rings code.  Just tests that the tree ring data was found and could be read.
"""
import os
from pathlib import Path
import unittest
import numpy as np
import galsim
import imsim

DATA_DIR = Path(__file__).parent / 'data'


class TreeRingsTestCase(unittest.TestCase):
    """TestCase class for the tree rings code."""
    def setUp(self):
        self.sensors = ['R:2,2 S:1,1', 'R:3,4 S:2,2']
        self.detnames = ['R22_S11', 'R34_S22']
        self.instcat_file = str(DATA_DIR / 'tiny_instcat.txt')
        self.rtest = 5280.0 # Just a value to test the radial function at
        self.rvalues = [.0030205, -.0034135] # Expected results
        self.centers = [(-3026.3, -3001.0), (3095.5, -2971.3)] # Input center values

    def test_read_tree_rings(self):
        """Check reading of tree_ring_parameters file"""
        obs_md = imsim.OpsimMetaDict(self.instcat_file)
        band = obs_md['band']
        bp = galsim.Bandpass('LSST_%s.dat'%band, wave_type='nm')

        tr_filename = os.path.join(imsim.data_dir, 'tree_ring_data',
                                   'tree_ring_parameters_19mar18.txt')
        tree_rings = imsim.TreeRings(tr_filename, only_dets=self.detnames)

        for i, detname in enumerate(self.detnames):
            center = tree_rings.get_center(detname)
            print('center = ',center)
            print('cf. ',self.centers)
            shifted_center = (center.x - 2048.5,
                              center.y - 2048.5)
            self.assertAlmostEqual(shifted_center, self.centers[i], 1)
            r_value_test = tree_rings.get_func(detname)(self.rtest)
            self.assertAlmostEqual(r_value_test, self.rvalues[i], 6)

        # Can also just give the file name, and imSim will find it in the data dir.
        tr_filename = 'tree_ring_parameters_19mar18.txt'
        tree_rings2 = imsim.TreeRings(tr_filename, only_dets=self.detnames)

        for i, detname in enumerate(self.detnames):
            center = tree_rings2.get_center(detname)
            print('center = ',center)
            print('cf. ',self.centers)
            shifted_center = (center.x - 2048.5,
                              center.y - 2048.5)
            self.assertAlmostEqual(shifted_center, self.centers[i], 1)
            r_value_test = tree_rings2.get_func(detname)(self.rtest)
            self.assertAlmostEqual(r_value_test, self.rvalues[i], 6)

        # If file not found, OSError
        np.testing.assert_raises(OSError, imsim.TreeRings, 'invalid.txt')

if __name__ == '__main__':
    unittest.main()


