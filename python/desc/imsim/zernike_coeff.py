"""
Code to determine deviations in the Zernike coefficients determined by the
LSST AOS closed loop control system. Results do not include contributions from
the open loop lookup table.
"""

import os
from warnings import warn

import numpy as np
from scipy.interpolate import interp2d
from scipy.optimize import leastsq

from zernike_polar import gen_superposition

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MATRIX_PATH = os.path.join(FILE_DIR, 'sensitivity_matrix.txt')


def _calc_fit_error(p, x_arr, y_arr, z_arr):
    """
    Calculates the residuals of a superposition of zernike polynomials

    Generates a function representing a superposition of 22 zernike polynomials
    using given coefficients and returns the residuals.

    @param [in] p is an array of 22 polynomial coefficients for the superposition

    @param [in] x_arr is an array of x coordinates

    @param [in] y_arr is an array of y coordinates

    @param [in] z_val is an array of expected or measured values

    @param [out] An array of the residuals
    """

    return gen_superposition(p)(x_arr, y_arr) - z_arr


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
    angles = [0, 60, 120, 180, 240, 300]
    for radius in radii:
        for angle in angles:
            x_list.append(radius * np.cos(np.deg2rad(angle)))
            y_list.append(radius * np.sin(np.deg2rad(angle)))

    # Add Corner raft points by hand
    x_list.extend([1.185, -1.185, -1.185, 1.185])
    y_list.extend([1.185, 1.185, -1.185, -1.185])

    return np.array(x_list), np.array(y_list)


def polar_coords():
    """
    Return 35 polar sampling coordinates in the focal plane

    @param [out] an array of 35 x coordinates

    @param [out] an array of 35 y coordinates
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
            theta_list.append(angle)

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
    at 35 positions in the focal plane using a uniform distribution. Maximum
    values for each degree of freedom are chosen and hardcoded based on Angeli
    et al. 2014. The minimum value is set using the deviation argument as
    (1 - <deviation>) * <max distortion>.

    @param [in] spread is a float thet defines the range in size of the
    randomly generated deviations

    @param [out] A (35, 50) array representing mock optical distortions
    """

    warn('Deviation values are generated using place holder values. \n'
         'They are not an accurate representation of real, physical values.')

    # [average, standard deviation]
    deviation_params = np.array([
        # M2: Piston (microns), x/y decenter (microns), x/y tilt (arcsec)
        [0.0, 15.0], [0.0, 5.0], [0.0, 5.0], [0.0, 0.75], [0.0, 0.75],

        # Camera: Piston (microns), x/y decenter (microns), x/y tilt (arcsec)
        [0.0, 30.0], [0.0, 2.0], [0.0, 2.5], [0.0, 1.5], [0.0, 1.5],

        # M1M3: bending modes (microns)
        [0.0, 0.5], [0.0, 0.2], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1],
        [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1],
        [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1],
        [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1],

        # M2: bending modes (microns)
        [0.0, 0.4], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1],
        [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1],
        [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1],
        [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1]
    ])

    distortion = np.zeros((35, 50))
    for i in range(35):
        distortion[i] = np.random.normal(deviation_params[i][0],
                                         deviation_params[i][1])

    return distortion


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
        x, y = self.polar_coords
        for coefficient in self.sampling_coeff:
            optimal = leastsq(_calc_fit_error, np.ones((22,)),
                              args=(x, y, coefficient))

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

        @param [in] fp_r is the desired focal plane radial coordinate

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


# Check run times for OpticalZernikes
if __name__ == '__main__':
    n_runs = 20
    n_coords = 1000

    optical_deviations = moc_deviations()
    x_coords = np.random.uniform(-1.5, 1.5, size=(n_coords,))
    y_coords = np.random.uniform(-1.5, 1.5, size=(n_coords,))

    from timeit import timeit

    init_time = timeit('OpticalZernikes(optical_deviations)',
                       globals=globals(),
                       number=n_runs)

    print('Averages over {} runs:'.format(n_runs))
    print('Init time (s):', init_time / n_runs)

    closed_loop = OpticalZernikes(optical_deviations)
    runtime = timeit('closed_loop.cartesian_coeff(x_coords, y_coords)',
                     globals=globals(), number=n_runs)

    print('Run time for (s)', n_coords, 'coords:', runtime / n_runs)
