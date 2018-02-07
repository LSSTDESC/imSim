"""
Code to determine deviations in the Zernike coefficients determined by the
LSST AOS closed loop control system. Results do not include contributions from
the open loop lookup table.
"""

import os

import numpy as np
from scipy.interpolate import interp2d
from scipy.optimize import leastsq

from zernike_cartesian import *

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MATRIX_PATH = os.path.join(FILE_DIR, 'sensitivity_matrix.txt')


def _gen_fit_func(p):
    """
    A generator function for a superposition of zernike polynomials

    Uses given coefficients to generate a function representing a superposition
    of the first 22 Zernike polynomials in the Noll basis:
        f(x, y) = p[i-1] * z_i

    @param [in] p is an array of 22 polynomial coefficients

    @param [out] a function object
    """

    def fit_func(x_arr, y_arr):
        fit = p[0] * z_1(x_arr, y_arr) + p[1] * z_2(x_arr, y_arr) \
              + p[2] * z_3(x_arr, y_arr) + p[3] * z_4(x_arr, y_arr) \
              + p[4] * z_5(x_arr, y_arr) + p[5] * z_6(x_arr, y_arr) \
              + p[6] * z_7(x_arr, y_arr) + p[7] * z_8(x_arr, y_arr) \
              + p[8] * z_9(x_arr, y_arr) + p[9] * z_10(x_arr, y_arr) \
              + p[10] * z_11(x_arr, y_arr) + p[11] * z_12(x_arr, y_arr) \
              + p[12] * z_13(x_arr, y_arr) + p[13] * z_14(x_arr, y_arr) \
              + p[14] * z_15(x_arr, y_arr) + p[15] * z_16(x_arr, y_arr) \
              + p[16] * z_17(x_arr, y_arr) + p[17] * z_18(x_arr, y_arr) \
              + p[18] * z_19(x_arr, y_arr) + p[19] * z_20(x_arr, y_arr) \
              + p[20] * z_21(x_arr, y_arr) + p[21] * z_22(x_arr, y_arr)

        return fit

    return fit_func


def _calc_fit_error(p, x_arr, y_arr, z_arr):
    """
    Calculates the residuals of a function returned by _gen_fit_func

    @param [in] p is an array of 22 polynomial coefficients for the fit

    @param [in] x_arr is an array of x coordinates

    @param [in] y_arr is an array of y coordinates

    @param [in] z_val is an array of expected or measured values

    @param [out] An array of the residuals
    """

    return _gen_fit_func(p)(x_arr, y_arr) - z_arr


def get_cartesian_sampling():
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


def moc_deviations(spread):
    """
    Returns an array of random mock optical deviations as a (35, 50) array.

    For each optical degree of freedom in LSST, deviations are chosen randomly
    at 35 positions in the focal plane using a uniform distribution. Maximum
    values for each degree of freedom are chosen and hardcoded based on Angeli
    et al. 2014. The minimum value is set using the deviation argument as
    (1 - <deviation>) * <max distortion>.

    @param [in] spread is a float thet defines the range in size of the
    randomly generated deviations

    @param [out] A (35, 50) array representing mock optical distortions
    """

    # Todo: Modify this function to pull from a gaussian distribution
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
    """
    This class provides fit functions for the zernike coefficients returned by
    the LSST AOS closed loop control system. It includes 19 functions that map
    a cartesian position in the focal plane to the coefficient of zernike 4
    through zernike 22 (in the NOLL indexing scheme)
    """

    sensitivity = np.genfromtxt(MATRIX_PATH).reshape((35, 19, 50))

    def __init__(self, deviations):
        """
        @param [in] deviations is a (35, 50) array representing deviations in
        each of LSST's optical degrees of freedom at 35 sampling coordinates
        """

        self.sampling_coeff = self._calc_sampling_coeff(deviations)
        self._fit_functions = self._optimize_fits()

    def _calc_sampling_coeff(self, deviations):
        """
        Calculates 19 zernike coefficients at 35 positions in the focal plane

        @param [in] deviations is a (35, 50) array representing deviations in
        each optical degree of freedom at the 35 sampling coordinates

        @param [out] a (19, 35) array of zernike coefficients
        """

        num_sampling_coords = self.sensitivity.shape[0]
        num_zernike_coeff = self.sensitivity.shape[1]

        coefficients = np.zeros((num_sampling_coords, num_zernike_coeff))
        for i in range(num_sampling_coords):
            coefficients[i] = self.sensitivity[i].dot(deviations[i])

        return coefficients.transpose()

    @property
    def cartesian_sampling(self):
        """
        Return 35 cartesian sampling coordinates in the focal plane

        @param [out] an array of 35 x coordinates

        @param [out] an array of 35 y coordinates
        """

        return get_cartesian_sampling()

    def _optimize_fits(self):
        """
        Generate a separate fit function for each zernike coefficient

        @param [out] A list of 19 functions
        """

        out = []
        x, y = self.cartesian_sampling
        for coefficient in self.sampling_coeff:
            optimal = leastsq(_calc_fit_error, np.ones((22,)),
                              args=(x, y, coefficient))

            fit_coeff = optimal[0]
            fit_func = _gen_fit_func(fit_coeff)
            out.append(fit_func)

        return out

    def interp_deviations(self, fp_x, fp_y, kind='cubic'):
        """
        Determine the zernike coefficients at given coordinates by interpolating

        @param [in] fp_x is the desired focal plane x coordinate

        @param [in] fp_y is the desired focal plane y coordinate

        @param [in] kind is the type of interpolation to perform. (eg. "linear")

        @param [out] An array of 19 zernike coefficients for z=4 through z=22
        """

        x, y = self.cartesian_sampling
        num_zernike_coeff = self.sampling_coeff.shape[0]
        out_arr = np.zeros(num_zernike_coeff)

        for i, coeff in enumerate(self.sampling_coeff):
            interp_func = interp2d(x, y, coeff, kind=kind)
            out_arr[i] = interp_func(fp_x, fp_y)[0]

        return out_arr

    def fit_coefficients(self, fp_x, fp_y):
        """
        Determine the zernike coefficients using a fir of zernike polynomials

        @param [in] fp_x is the desired focal plane x coordinate

        @param [in] fp_y is the desired focal plane y coordinate

        @param [out] An array of 19 zernike coefficients for z=4 through z=22
        """

        return np.array([f(fp_x, fp_y) for f in self._fit_functions])


# Check run times for CloosedLoopZ
if __name__ == '__main__':
    n_runs = 10
    n_coords = 1000

    optical_deviations = moc_deviations(spread=.4)
    x_coords = np.random.uniform(-1.5, 1.5, size=(n_coords,))
    y_coords = np.random.uniform(-1.5, 1.5, size=(n_coords,))

    from timeit import timeit

    init_time = timeit('ClosedLoopZ(optical_deviations)',
                       globals=globals(),
                       number=n_runs)

    print('init time:', init_time / n_runs)

    closed_loop = ClosedLoopZ(optical_deviations)
    runtime = timeit('closed_loop.fit_coefficients(x_coords, y_coords)',
                     globals=globals(),
                     number=n_runs)

    print('run time for', n_coords, 'coords:', runtime / n_runs)
