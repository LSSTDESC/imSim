#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

import numpy as np

from desc.imsim.optical_system import OpticalZernikes, mock_deviations


class OpticalDeviations(unittest.TestCase):
    """Tests the generation of mock optical deviations"""

    def test_seed_handeling(self):
        """Tests that random deviations are seed dependant"""

        # Checks that seed is set explicitly and not time dependant
        fixed_seed_equal = np.array_equal(mock_deviations(0), mock_deviations(0))
        self.assertTrue(fixed_seed_equal)

        # checks that default seeds are time dependant and not persistant
        time_seed_equal = np.array_equal(mock_deviations(), mock_deviations())
        self.assertFalse(time_seed_equal)

    def test_shape(self):
        """Tests that mock optical deviations are the correct shape"""

        shape = mock_deviations().shape
        err_msg = 'Expected shape (50,) but received {} instead.'
        self.assertEqual(shape, (50,), err_msg.format(shape))


class FocalPlaneModeling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Instantiate a model fo the focal plane"""

        cls.opt_state = OpticalZernikes()

    def test_compare_coord_systems(self):
        """Tests that polar and cartesian sampling coordinates agree"""

        samp_x, samp_y = self.opt_state.cartesian_coords
        samp_r, samp_theta = self.opt_state.polar_coords

        polar_x = np.multiply(samp_r, np.cos(samp_theta))
        polar_y = np.multiply(samp_r, np.sin(samp_theta))

        err_msg = 'Polar and cartesian sampling coordinates do not match'
        self.assertTrue(all(np.isclose(samp_x, polar_x)), err_msg)
        self.assertTrue(all(np.isclose(samp_y, polar_y)), err_msg)

    def test_coeff_shape(self):
        """Tests that the expected number of coefficients is returned"""

        num_coeff = len(self.opt_state.polar_coeff(1, 3.14))
        err_msg = 'Expected 19 coefficients, received {}.'
        self.assertEqual(num_coeff, 19, err_msg.format(num_coeff))

    def test_zero_deviations(self):
        """Tests that zernike deviations are zero for zero optical deviations"""

        moc_deviation = np.zeros((50, ))
        zern_deviations = OpticalZernikes(moc_deviation).deviation_coeff
        is_zeros = not np.count_nonzero(zern_deviations)
        self.assertTrue(is_zeros, "Received nonzero zernike coefficients")
