"""
Unit tests for tree rings code.  Just tests that the tree ring data was found and could be read.
"""
import os
from pathlib import Path
import unittest
import numpy as np
import imsim

DATA_DIR = Path(__file__).parent / 'data'


class TreeRingsTestCase(unittest.TestCase):
    """TestCase class for the tree rings code."""
    def setUp(self):
        self.detnames = ['R22_S11', 'R34_S22']
        self.instcat_file = str(DATA_DIR / 'tiny_instcat.txt')
        self.rtest = 5280.0  # Just a value to test the radial function at
        self.rvalues = [.0030205, -.0034135]  # Expected results
        self.centers = [(-3026.3, -3001.0), (3095.5, -2971.3)]  # Input center values
        self.tr_filename = 'tree_ring_parameters_19mar18.txt'

    def test_read_tree_rings(self):
        """Check reading of tree_ring_parameters file"""
        tr_filename = os.path.join(imsim.data_dir, 'tree_ring_data',
                                   self.tr_filename)
        tree_rings = imsim.TreeRings(tr_filename, only_dets=self.detnames,
                                     defer_load=False)

        for i, detname in enumerate(self.detnames):
            center = tree_rings.get_center(detname)
            print('center = ', center)
            print('cf. ', self.centers)
            shifted_center = (center.x - 2048.5,
                              center.y - 2048.5)
            self.assertAlmostEqual(shifted_center, self.centers[i], 1)
            r_value_test = tree_rings.get_func(detname)(self.rtest)
            self.assertAlmostEqual(r_value_test, self.rvalues[i], 6)

        # Can also just give the file name, and imSim will find it in the data dir.
        tree_rings2 = imsim.TreeRings(self.tr_filename,
                                      only_dets=self.detnames,
                                      defer_load=False)

        for i, detname in enumerate(self.detnames):
            center = tree_rings2.get_center(detname)
            print('center = ', center)
            print('cf. ', self.centers)
            shifted_center = (center.x - 2048.5,
                              center.y - 2048.5)
            self.assertAlmostEqual(shifted_center, self.centers[i], 1)
            r_value_test = tree_rings2.get_func(detname)(self.rtest)
            self.assertAlmostEqual(r_value_test, self.rvalues[i], 6)

        # If file not found, OSError
        np.testing.assert_raises(OSError, imsim.TreeRings, 'invalid.txt')

    def test_deferred_read(self):
        """Test deferred reading of tree ring data."""
        tree_rings = imsim.TreeRings(self.tr_filename, defer_load=True)
        self.assertEqual(len(tree_rings.info), 0)
        det_name = "R22_S00"
        _ = tree_rings.get_center(det_name)
        self.assertEqual(len(tree_rings.info), 1)
        _ = tree_rings.get_func(det_name)
        self.assertEqual(len(tree_rings.info), 1)
        self.assertIn(det_name, tree_rings.info)

        det_name = "R22_S11"
        _ = tree_rings.get_center(det_name)
        self.assertEqual(len(tree_rings.info), 2)
        _ = tree_rings.get_func(det_name)
        self.assertEqual(len(tree_rings.info), 2)
        self.assertIn(det_name, tree_rings.info)


if __name__ == '__main__':
    unittest.main()
