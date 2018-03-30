#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

import numpy as np

from desc.imsim.cartesian_zernikes import gen_superposition as gen_cart_op
from desc.imsim.cartesian_zernikes import gen_superposition_unop as gen_cart_unop
from desc.imsim.polar_zernikes import gen_superposition as gen_pol_op
from desc.imsim.polar_zernikes import gen_superposition_unop as gen_pol_unop
from desc.imsim.optical_system import OpticalZernikes, mock_deviations


class ZernikePolynomial(unittest.TestCase):
    """Tests generator functions in zernike_cartesian.py / zernike_polar.py"""

    @classmethod
    def setUpClass(cls):
        """Defines a set of test superposition weights and coordinates"""

        cls.coeff = np.ones(22)
        cls.r = 1.5
        cls.theta = 2.2  # radians
        cls.x = cls.r * np.cos(cls.r)
        cls.y = cls.r * np.sin(cls.r)

    def test_polar_algebra(self):
        """Test optimized and unoptimized superpositions agree in polar"""

        optim = gen_pol_op(self.coeff)(self.r, self.theta)
        un_optim = gen_pol_unop(self.coeff)(self.r, self.theta)
        values_close = np.isclose(optim, un_optim)
        err_msg = 'Values not close: {},  {}'
        self.assertTrue(values_close, err_msg.format(optim, un_optim))

    def test_cartesian_algebra(self):
        """Test optimized and unoptimized superpositions agree in cartesian"""

        optim = gen_cart_op(self.coeff)(self.x, self.y)
        un_optim = gen_cart_unop(self.coeff)(self.x, self.y)
        values_close = np.isclose(optim, un_optim)
        err_msg = 'Values not close: {},  {}'
        self.assertTrue(values_close, err_msg.format(optim, un_optim))

    def test_compare_coord_systems(self):
        """Test optimized cartesian and polar superpositions agree"""

        polar = gen_pol_op(self.coeff)(self.r, self.theta)
        cartesian = gen_cart_op(self.coeff)(self.x, self.y)
        values_close = np.isclose(polar, cartesian, .005)
        err_msg = 'Values not close: {},  {}'
        self.assertTrue(values_close, err_msg.format(polar, cartesian))


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

        moc_deviation = np.zeros((35, 50))
        zern_deviations = OpticalZernikes(moc_deviation).deviation_coeff
        is_zeros = not np.count_nonzero(zern_deviations)
        self.assertTrue(is_zeros, "Received nonzero zernike coefficients")
