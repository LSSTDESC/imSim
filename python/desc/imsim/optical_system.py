"""
Code to determine deviations in the Zernike coefficients determined by the
LSST AOS closed loop control system. Results do not include contributions from
the open loop lookup table.
"""

import os

import numpy as np
from scipy.interpolate import interp2d
from scipy.optimize import leastsq
from timeit import timeit

from polar_zernikes import gen_superposition

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MATRIX_PATH = os.path.join(FILE_DIR, 'sensitivity_matrix.txt')
AOS_PATH = os.path.join(FILE_DIR, 'aos_sim_results.txt')


def _calc_fit_error(p, r_arr, t_arr, z_arr):
    """
    Calculates the residuals of a superposition of zernike polynomials

    Generates a function representing a superposition of 22 zernike polynomials
    using given coefficients and returns the residuals.

    @param [in] p is an array of 22 polynomial coefficients for the superposition

    @param [in] r_arr is an array of rho coordinates

    @param [in] t_arr is an array of theta coordinates

    @param [in] z_val is an array of expected or measured values

    @param [out] An array of the residuals
    """

    return gen_superposition(p)(r_arr, t_arr) - z_arr


def cartesian_coords():
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
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    for radius in radii:
        for angle in angles:
            x_list.append(radius * np.cos(angle))
            y_list.append(radius * np.sin(angle))

    # Add Corner raft points by hand
    x_list.extend([1.185, -1.185, -1.185, 1.185])
    y_list.extend([1.185, 1.185, -1.185, -1.185])

    return np.array(x_list), np.array(y_list)


def polar_coords():
    """
    Return 35 polar sampling coordinates in the focal plane.

    Angular values are returned in radians

    @param [out] an array of 35 r coordinates

    @param [out] an array of 35 theta coordinates
    """

    # Initialize with central point
    r_list = [0.]
    theta_list = [0.]

    # Loop over points on spines
    radii = [0.379, 0.841, 1.237, 1.535, 1.708]
    angles = [0, 60, 120, 180, 240, 300]
    for radius in radii:
        for angle in angles:
            r_list.append(radius)
            theta_list.append(np.deg2rad(angle))

    # Add Corner raft points
    x_raft_coords = [1.185, -1.185, -1.185, 1.185]
    y_raft_coords = [1.185, 1.185, -1.185, -1.185]
    for x, y in zip(x_raft_coords, y_raft_coords):
        theta_list.append(np.arctan2(y, x))
        r_list.append(np.sqrt(x * x + y * y))

    return np.array(r_list), np.array(theta_list)


def moc_deviations():
    """
    Returns an array of random mock optical deviations as a (35, 50) array.

    For each optical degree of freedom in LSST, deviations are chosen randomly
    at 35 positions in the focal plane using a normal distribution. Parameters
    for each distribution are hardcoded based on Angeli et al. 2014.

    @param [out] A (35, 50) array representing mock optical distortions
    """

    aos_sim_results = np.genfromtxt(AOS_PATH)
    assert aos_sim_results.shape[0] == 50
    avg = np.average(aos_sim_results, axis=1)
    std = np.std(aos_sim_results, axis=1)

    distortion = np.zeros((35, 50))
    for i in range(35):
        distortion[i] = np.random.normal(avg, std)

    return distortion


def test_runtime(n_runs, n_coords, verbose=False):
    """
    Determines average runtimes to both instantiate the OpticalZernikes class
    and to evaluate the cartesian_coeff method.

    @param [in] n_runs is the total number of runs to average runtimes over

    @param [in] n_coords is the total number of cartesian coordinates
        to average runtimes over

    @param [in] verbose is a boolean specifying whether to print results
        (default = false)

    @param [out] The average initialization time in seconds

    @param [out] The average evaluation time of cartesian_coeff in seconds
    """

    init_time = timeit('OpticalZernikes()', globals=globals(), number=n_runs)

    optical_state = OpticalZernikes()
    x_coords = np.random.uniform(-1.5, 1.5, size=(n_coords,))
    y_coords = np.random.uniform(-1.5, 1.5, size=(n_coords,))
    runtime = timeit('optical_state.cartesian_coeff(x_coords, y_coords)',
                     globals=locals(), number=n_runs)

    if verbose:
        print('Averages over {} runs:'.format(n_runs))
        print('Init time (s):', init_time / n_runs)
        print('Run time for (s)', n_coords, 'cartesian coords:',
              runtime / n_runs)

    return


class OpticalZernikes:
    """
    This class provides fit functions for the zernike coefficients returned by
    the LSST AOS closed loop control system. It includes 19 functions that map
    a cartesian position in the focal plane to the coefficient of zernike 4
    through zernike 22 (in the NOLL indexing scheme)
    """

    sensitivity = np.genfromtxt(MATRIX_PATH).reshape((35, 19, 50))
    polar_coords = polar_coords()
    _cartesian_cords = None

    def __init__(self, deviations=None):
        """
        @param [in] deviations is a (35, 50) array representing deviations in
        each of LSST's optical degrees of freedom at 35 sampling coordinates
        """

        if deviations is None:
            deviations = moc_deviations()

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
    def cartesian_coords(self):
        """
        Lazy loads 35 cartesian sampling coordinates in the focal plane

        @param [out] an array of 35 x coordinates

        @param [out] an array of 35 y coordinates
        """

        if self._cartesian_cords is None:
            self._cartesian_cords = cartesian_coords()

        return self._cartesian_cords

    def _optimize_fits(self):
        """
        Generate a separate fit function for each zernike coefficient

        @param [out] A list of 19 functions
        """

        out = []
        r, t = self.polar_coords
        for coefficient in self.sampling_coeff:
            optimal = leastsq(_calc_fit_error, np.ones((22,)),
                              args=(r, t, coefficient))

            fit_coeff = optimal[0]
            fit_func = gen_superposition(fit_coeff)
            out.append(fit_func)

        return out

    def _interp_deviations(self, fp_x, fp_y, kind='cubic'):
        """
        Determine the zernike coefficients at given coordinates by interpolating

        @param [in] fp_x is the desired focal plane x coordinate

        @param [in] fp_y is the desired focal plane y coordinate

        @param [in] kind is the type of interpolation to perform. (eg. "linear")

        @param [out] An array of 19 zernike coefficients for z=4 through z=22
        """

        x, y = self.cartesian_coords
        num_zernike_coeff = self.sampling_coeff.shape[0]
        out_arr = np.zeros(num_zernike_coeff)

        for i, coeff in enumerate(self.sampling_coeff):
            interp_func = interp2d(x, y, coeff, kind=kind)
            out_arr[i] = interp_func(fp_x, fp_y)[0]

        return out_arr

    def polar_coeff(self, fp_r, fp_t):
        """
        Determine the zernike coefficients using a fit of zernike polynomials

        @param [in] fp_r is the desired focal plane radial coordinate in rads

        @param [in] fp_t is the desired focal plane angular coordinate

        @param [out] An array of 19 zernike coefficients for z=4 through z=22
        """

        return np.array([f(fp_r, fp_t) for f in self._fit_functions])

    def cartesian_coeff(self, fp_x, fp_y):
        """
        Determine the zernike coefficients using a fit of zernike polynomials

        @param [in] fp_x is the desired focal plane x coordinate

        @param [in] fp_y is the desired focal plane y coordinate

        @param [out] An array of 19 zernike coefficients for z=4 through z=22
        """

        fp_r = np.sqrt(fp_x ** 2 + fp_y ** 2)
        fp_t = np.arctan2(fp_y, fp_x)
        return self.polar_coeff(fp_r, fp_t)


if __name__ == '__main__':
    # Check run times for OpticalZernikes
    test_runtime(100, 1000, verbose=True)
