"""
Code to determine deviations in the Zernike coefficients determined by the
AOS closed loop control system. Results do not include contributions from the
open loop lookup table.
"""

import os

import numpy as np
from scipy.interpolate import interp2d
from scipy.optimize import leastsq

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MATRIX_PATH = os.path.join(FILE_DIR, 'sensitivity_matrix.txt')


def _gen_fit_func(p):
    """
    Uses given coefficients to generate a function representing a 2d polynomial

    @param [in] p is an array of 6 polynomial coefficients

    @param [out] A function representing the polynomial
        f(x, y) = p[0]x^2 + p[1]x + p[2]y^2 + p[3]y + p[4]xy + p[5]
    """

    def fit_func(x_arr, y_arr):
        x_terms = p[0] * x_arr * x_arr + p[1] * x_arr
        y_terms = p[2] * y_arr * y_arr + p[3] * y_arr
        cross_terms = p[4] * x_arr * y_arr + p[5]
        return x_terms + y_terms + cross_terms

    return fit_func


def _error_fit_func(p, x_arr, y_arr, z_arr):
    """
    Calculates the residuals of a 2d polynomial fit function

    @param [in] p is an array of 6 polynomial coefficients for the fit

    @param [in] x_arr is an array of x coordinates

    @param [in] y_arr is an array of y coordinates

    @param [in] z_val is an array of expected or measured values

    @param [out] An array of the residuals
    """

    return _gen_fit_func(p)(x_arr, y_arr) - z_arr


def moc_deviations(spread):
    """
    Returns an array of mock optical deviations as a (35, 50) array. Mock

    For each optical degree of freedom in LSST, deviations are chosen randomly
    at 35 positions in the focal plane using a uniform distribution. Maximum
    values for each degree of freedom are chosen and hardcoded based on Angeli
    et al. 2014. The minimum value is set using the deviation argument as
    (1 - <deviation>) * <max distortion>.

    @param [in] deviation defines the range in size of randomly generated

    @param [out] A numpy array representing mock optical distortions
    """

    max_distortion = np.array([
        # M2: Piston (microns), x/y decenter (microns), x/y tilt (arcsec)
        15.0, 5.0, 5.0, 0.75, 0.75,

        # Camera: Piston (microns), x/y decenter (microns), x/y tilt (arcsec)
        30.0, 2.0, 2.5, 1.5, 1.5,

        # M1M3: bending modes (microns)
        0.5, 0.2, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,

        # M2: bending modes (microns)
        0.4, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
    ])

    min_distortion = (1 - spread) * max_distortion
    random_signs = np.random.choice([-1, 1], size=50)
    distortion = np.zeros((35, 50))
    for i in range(35):
        rand_distortion = np.random.uniform(min_distortion, max_distortion)
        distortion[i] = random_signs * rand_distortion

    return distortion


class ClosedLoopZ:

    sensitivity = np.genfromtxt(MATRIX_PATH).reshape((35, 19, 50))

    def __init__(self, deviations):
        """
        @param [in] deviations is a (35, 50) array representing deviations in
        each optical degree of freedom at the 35 sampling coordinates
        """

        self._sampling_coeff = self._calc_sampling_coeff(deviations)
        self._fit_functions = self._gen_fit_functions()

    def _calc_sampling_coeff(self, deviations):
        """
        Calculates 19 zernike coefficients at 35 positions in the focal plane

        @param [in] deviations is a (35, 50) array representing deviations in
        each optical degree of freedom at the 35 sampling coordinates
        """

        num_sampling_coords = self.sensitivity.shape[0]
        num_zernike_coeff = self.sensitivity.shape[1]

        coefficients = np.zeros((num_sampling_coords, num_zernike_coeff))
        for i in range(num_sampling_coords):
            coefficients[i] = self.sensitivity[i].dot(deviations[i])

        return coefficients.transpose()

    @staticmethod
    def cartesian_samp_coords():
        """
        Return 35 cartesian sampling coordinates in the focal plane

        @param [out] an array of 35 x coordinates

        @param [out] an array of 35 y coordinates
        """

        # Initialize with central point
        x_list = [0.]
        y_list = [0.]

        # Loop over points on spines
        radii = [0.379, 0.841, 1.237, 1.535, 1.708]
        angles = [0, 60, 120, 180, 240, 300]
        for radius in radii:
            for angle in angles:
                x_list.append(radius * np.cos(np.deg2rad(angle)))
                y_list.append(radius * np.sin(np.deg2rad(angle)))

        # Add Corner raft points by hand
        x_list.extend([1.185, -1.185, -1.185, 1.185])
        y_list.extend([1.185, 1.185, -1.185, -1.185])

        return np.array(x_list), np.array(y_list)

    def _gen_fit_functions(self):
        """
        Generate a separate fit function for each zernike coefficient

        @param [out] A list of 19 functions
        """

        out = []
        x, y = self.cartesian_samp_coords()
        for i, zernike_coeff in enumerate(self._sampling_coeff):
            optimal = leastsq(_error_fit_func, np.arange(0, 6),
                              args=(x, y, zernike_coeff))

            fit_coeff = optimal[0]
            out.append(_gen_fit_func(fit_coeff))

        return out

    def interp_deviations(self, fp_x, fp_y, kind='linear'):
        """
        Determine the zernike coefficients at given coordinates by interpolating

        @param [in] fp_x is the desired focal plane x coordinate

        @param [in] fp_y is the desired focal plane y coordinate

        @param [in] kind is the type of interpolation to perform. (eg. "cubic")

        @param [out] An array of 19 zernike coefficients
        """

        x, y = self.cartesian_samp_coords()
        num_zernike_coeff = self._sampling_coeff.shape[1]

        out_arr = np.zeros((num_zernike_coeff,))
        for i, coeff in enumerate(self._sampling_coeff):
            interp_func = interp2d(x, y, coeff[i], kind=kind)
            out_arr[i] = interp_func(fp_x, fp_y)[0]

        return out_arr

    def fit_coefficients(self, fp_x, fp_y):
        """
        Determine the zernike coefficients at given coordinates by fitting a 2d polynomial

        @param [in] fp_x is the desired focal plane x coordinate

        @param [in] fp_y is the desired focal plane y coordinate

        @param [out] An array of 19 zernike coefficients
        """

        return np.array([f(fp_x, fp_y) for f in self._fit_functions])

