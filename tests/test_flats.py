"""
Unit test code for flat production code.
"""
from collections import namedtuple
import unittest
import numpy as np
import logging
import galsim
import imsim

class FlatTestCase(unittest.TestCase):
    """TestCase class for flat production code."""
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple_flat(self):
        """Test of basic LSST_Flat functionality."""

        counts_per_iter = 1_000
        niter = 10
        tot_counts = counts_per_iter * niter
        config = {
            'image': {
                'type': 'LSST_Flat',
                'random_seed': 1234,
                'xsize': 256,
                'ysize': 256,
                'counts_per_pixel': tot_counts,
                'max_counts_per_iter': counts_per_iter,
                'wcs': galsim.PixelScale(0.2),
                'noise': {'type': 'Poisson'},
            }
        }
        flat = galsim.config.BuildImage(config)
        self.assertLess(np.abs(niter*counts_per_iter - flat.array.mean()),
                        flat.array.std())
        flat.write('output/simple_flat.fits')

        # Without any BFE or tree rings, the variance should be close to the mean,
        # and the covariance beteen pixels should be negligible.
        print('mean = ',np.mean(flat.array))
        print('var = ',np.var(flat.array))
        np.testing.assert_allclose(np.mean(flat.array), tot_counts, rtol=1.e-2)
        np.testing.assert_allclose(np.var(flat.array), tot_counts, rtol=1.e-2)

        flat.array[:,:] -= np.mean(flat.array)
        cov10 = np.mean(flat.array[1:,:] * flat.array[:-1,:])
        cov01 = np.mean(flat.array[:,1:] * flat.array[:,:-1])
        cov11 = np.mean(flat.array[1:,1:] * flat.array[:-1,:-1])
        print('cov10 01 11 = ', cov10, cov01, cov11)
        assert np.abs(cov10) < 1.e-2*tot_counts
        assert np.abs(cov01) < 1.e-2*tot_counts
        assert np.abs(cov11) < 1.e-2*tot_counts

    def test_silicon_flat(self):
        """Test LSST_Flat with Silicon sensor but no treerings."""

        counts_per_iter = 4_000
        niter = 20
        tot_counts = counts_per_iter * niter
        config = {
            'image': {
                'type': 'LSST_Flat',
                'random_seed': 1234,
                'xsize': 256,
                'ysize': 256,
                'counts_per_pixel': tot_counts,
                'max_counts_per_iter': counts_per_iter,
                'wcs': galsim.PixelScale(0.2),
                'noise': {'type': 'Poisson'},
                'sensor': { 'type': 'Silicon', },
            },
        }
        flat = galsim.config.BuildImage(config)
        self.assertLess(np.abs(niter*counts_per_iter - flat.array.mean()),
                        flat.array.std())
        flat.write('output/silicon_flat.fits')

        # With BFE, the neighboring pixels are correlated and the variance is slightly lowered.
        print('mean = ',np.mean(flat.array))
        print('var = ',np.var(flat.array))
        np.testing.assert_allclose(np.mean(flat.array), tot_counts, rtol=1.e-2)
        # It's smaller, but still relatively similar, so rtol=1e-1 still works.
        np.testing.assert_allclose(np.var(flat.array), tot_counts, rtol=1.e-1)
        assert np.var(flat.array) < tot_counts

        flat.array[:,:] -= np.mean(flat.array)
        cov10 = np.mean(flat.array[1:,:] * flat.array[:-1,:])
        cov01 = np.mean(flat.array[:,1:] * flat.array[:,:-1])
        cov11 = np.mean(flat.array[1:,1:] * flat.array[:-1,:-1])
        print('cov10 01 11 = ', cov10, cov01, cov11)
        # These are now all significantly non-zero (and positive).
        assert cov10 > 1.e-2*tot_counts
        assert cov01 > 5.e-3*tot_counts
        assert cov11 > 2.e-3*tot_counts
        # Also, 11 is the smallest, and there is more covariance in the y direction (10).
        assert cov10 > cov01 > cov11

    def test_treerings_flat(self):
        """Test LSST_Flat with Silicon sensor and treerings."""

        counts_per_iter = 1_000
        niter = 10
        tot_counts = counts_per_iter * niter
        tree_amp = 0.26
        tree_period = 87
        config = {
            'image': {
                'type': 'LSST_Flat',
                'random_seed': 1234,
                'xsize': 256,
                'ysize': 256,
                'counts_per_pixel': tot_counts,
                'max_counts_per_iter': counts_per_iter,
                'wcs': galsim.PixelScale(0.2),
                'noise': {'type': 'Poisson'},
                'sensor': {
                    'type': 'Silicon',
                    'treering_center': galsim.PositionD(-100,-100),
                    'treering_func': galsim.SiliconSensor.simple_treerings(tree_amp, tree_period),
                },
            },
        }
        flat = galsim.config.BuildImage(config)
        self.assertLess(np.abs(niter*counts_per_iter - flat.array.mean()),
                        flat.array.std())
        # Visually, you can see the tree rings quite clearly here
        flat.write('output/treering_flat.fits')

        # Now with tree rings and BFE, the variance is dominated by the large scale tree rings.
        print('mean = ',np.mean(flat.array))
        print('var = ',np.var(flat.array))
        print('min/max = ',np.min(flat.array),np.max(flat.array))
        # The relative effect of the tree ring on the pixel area is ~df/dr = A 2pi/P sin(2pi r/P)
        # The predicted variance of the tree-ring pattern is then 1/2 (counts A 2pi/P)^2
        # The regular Poisson noise also contributes, so the net prediction is:
        pred_var = 0.5 * (tot_counts * tree_amp * 2*np.pi / tree_period)**2 + tot_counts
        print('pred_var = ',pred_var)
        np.testing.assert_allclose(np.mean(flat.array), tot_counts, rtol=1.e-2)
        np.testing.assert_allclose(np.var(flat.array), pred_var, rtol=3.e-2)

        # The covariance is mostly due to the tree rings now, so cov11 is about the same
        # as cov10 and cov01.
        flat.array[:,:] -= np.mean(flat.array)
        cov10 = np.mean(flat.array[1:,:] * flat.array[:-1,:])
        cov01 = np.mean(flat.array[:,1:] * flat.array[:,:-1])
        cov11 = np.mean(flat.array[1:,1:] * flat.array[:-1,:-1])
        print('cov10 01 11 = ', cov10, cov01, cov11)
        assert cov10 > 0.5*tot_counts
        assert cov01 > 0.5*tot_counts
        assert cov11 > 0.5*tot_counts


if __name__ == '__main__':
    unittest.main()
