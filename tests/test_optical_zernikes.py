#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

import numpy as np

from desc.imsim.cartesian_zernikes import gen_superposition as gen_cart_op
from desc.imsim.cartesian_zernikes import gen_superposition_unop as gen_cart_unop
from desc.imsim.polar_zernikes import gen_superposition as gen_pol_op
from desc.imsim.polar_zernikes import gen_superposition_unop as gen_pol_unop
from desc.imsim import OpticalZernikes


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
        values_close = np.isclose(polar, cartesian)
        err_msg = 'Values not close: {},  {}'
        self.assertTrue(values_close, err_msg.format(polar, cartesian))


class FocalPlaneModeling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Instantiate a model fo the focal plane"""

        cls.focal_model = OpticalZernikes()

    def test_compare_coord_systems(self):

        samp_x, samp_y = self.focal_model.cartesian_coords
        samp_r, samp_theta = self.focal_model.polar_coords
        polar_x = np.multiply(samp_r, np.cos(samp_theta))
        polar_y = np.multiply(samp_r, np.sin(samp_theta))
        polar_cart = np.rec.fromarrays([polar_x, polar_y])
        polar_cart.sort()

        err_msg = 'Polar and cartesian sampling coordinates do not match'
        self.assertTrue(all(np.isclose(samp_x, polar_x)), err_msg)
        self.assertTrue(all(np.isclose(samp_y, polar_y)), err_msg)
