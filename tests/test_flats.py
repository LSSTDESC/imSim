"""
Unit test code for flat production code.
"""
from collections import namedtuple
import unittest
import numpy as np
import galsim
import desc.imsim

TreeRingInfo = namedtuple('TreeRingInfo', ['center', 'func'])

class GsDetector:
    """
    Minimal implementation of an interface-compatible version
    of GalSimDetector for testing the flat production code.
    """
    def __init__(self):
        self.xMaxPix = 256
        self.yMaxPix = 256
        self.xMinPix = 1
        self.yMinPix = 1
        self.wcs = galsim.wcs.PixelScale(0.2)
        self.tree_rings \
            = TreeRingInfo(galsim.PositionD(0, 0),
                           galsim.SiliconSensor.simple_treerings(0.26, 47))


class FlatTestCase(unittest.TestCase):
    """TestCase class for flat production code."""
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_make_flat(self):
        """Test of make_flat function."""
        gs_det = GsDetector()
        counts_per_iter = 1000
        niter = 10
        rng = galsim.UniformDeviate()
        logger = desc.imsim.get_logger('WARN')
        flat = desc.imsim.make_flat(gs_det, counts_per_iter, niter, rng,
                                    logger=logger)
        self.assertLess(np.abs(niter*counts_per_iter - flat.array.mean()),
                        flat.array.std())


if __name__ == '__main__':
    unittest.main()
