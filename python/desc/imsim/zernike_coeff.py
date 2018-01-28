"""
Code to determine deviations in the Zernike coefficients as determined by the
AOS open loop control system. Results do not include contributions from the
closed loop lookup table.
"""

import os

import numpy as np
from scipy.interpolate import interp2d
from scipy import optimize

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MATRIX_PATH = os.path.join(FILE_DIR, 'sensitivity_matrix.txt')

DEBUG = False


def get_sensitivity_matrix():
    """
    Returns the sensitivity matrix as a (35, 19, 50) array

    Dimension 0 represents 35 sampling positions in the focal plane.

    Dimension 1 represents 19 Zernike polynomials from z=4 (Focus) to
        z=22 (2nd Spherical).

    Dimension 3 represents the 50 degrees of freedom in the optical system.

    @param [out] a numpy array with shape (35, 19, 50)
    """

    return np.genfromtxt(MATRIX_PATH).reshape((35, 19, 50))


def mock_distortions():
    """
    Returns an array of mock optical distortions as a (35, 50) array

    @param [out] A numpy array representing mock optical distortions
    """

    distortion_sizes = [
        # M2: Piston (microns), x/y decenter (microns), x/y tilt (arcsec)
        -15.0, 0.1, -2.0, 1.0, -0.1,

        # Camera: Piston (microns), x/y decenter (microns), x/y tilt (arcsec)
        -30.0, -1.0, -0.5, 0.1, -1.5,

        # M1M3: bending modes (microns)
        -0.5, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,

        # M2: bending modes (microns)
        -0.4, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
    ]

    distortion = np.zeros((35, 50))
    for i in range(35):
        distortion[i] = distortion_sizes[i]

    return distortion


def cartesian_samp_coords():
    """
    Returns two lists with 35 cartesian sampling coordinates in the focal plane

    @param [out] a list of 35 x coordinates

    @param [out] a list of 35 y coordinates
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


def polar_samp_coords():
    """
    Returns two lists with 35 polar sampling coordinates in the focal plane

    @param [out] a list of 35 r coordinates

    @param [out] a list of 35 theta coordinates
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


def _raise_zernike_deviations(fp_x, fp_y, distortion_vectors):
    """
    Type and value check arguments for `zernike_deviations`

    @param [in] fp_x should be a float or int type

    @param [in] fp_y should be a float or int type

    @param [in] distortion_vectors should be an array with shape (35, 50)
    """

    type_err = "Argument {} should be {} type."
    if not isinstance(fp_x, (float, int)):
        raise TypeError(type_err.format('fp_x', 'float'))

    if not isinstance(fp_y, (float, int)):
        raise TypeError(type_err.format('fp_y', 'float'))

    # Todo: Change place holder value (100) to focal plane radius
    r_focal_plane = 100
    r_coordinates = fp_x * fp_x + fp_y + fp_y
    if r_coordinates >= r_focal_plane:
        raise ValueError('Provided coordinates are outside LSST focal plane')

    if not isinstance(distortion_vectors, np.ndarray):
        raise TypeError(type_err.format('distortion_vectors', 'array'))

    distortions_shape = distortion_vectors.shape
    if distortions_shape != (35, 50):
        dist_val_err = 'Argument distortion_vector should have shape ' \
                       '(35, 50). Found {} instead.'

        raise ValueError(dist_val_err.format(distortions_shape))


def interp_z_deviations(fp_x, fp_y, distortion_vectors):
    """
    Return the interpolated zernike coefficients at a focal plane position

    @param [in] fp_x is the desired focal plane x coordinate

    @param [in] fp_y is the desired focal plane y coordinate

    @param [in] distortion_vectors is an array describing deviations in
    each optical degree of freedom sampled at 35 positions.
    It should have shape (35, 50).

    @param [out] A list of 19 zernike coefficients
    """

    _raise_zernike_deviations(fp_x, fp_y, distortion_vectors)

    sensitivity_matrix = get_sensitivity_matrix()
    num_sampling_positions = sensitivity_matrix.shape[0]
    num_zernike_coefficients = sensitivity_matrix.shape[1]

    coefficients = np.zeros((num_sampling_positions, num_zernike_coefficients))
    for i in range(num_sampling_positions):
        coefficients[i] = sensitivity_matrix[i].dot(distortion_vectors[i])

    out = []
    x, y = cartesian_samp_coords()
    for i in range(num_zernike_coefficients):
        interp_func = interp2d(x, y, coefficients[:, i], kind='cubic')
        out.extend(interp_func(fp_x, fp_y))

    return out


def _error_fit_func(p, x_arr, y_arr, z_arr):
    """
    Calculates the residual of a 2d fit using as:
        _fit_func`(p, x_arr, y_arr) - z_val

    @param [in] p is an array of the coefficients for the fit

    @param [in] x_arr is an array of x coordinates

    @param [in] y_arr is an array of y coordinates

    @param [in] z_val is an array of the expected value of the fit evaluated
        at x_arr and y_arr

    @param [out] An array of the residuals
    """

    return _fit_func(p, x_arr, y_arr) - z_arr


def _fit_func(p, x_arr, y_arr):
    """
    Calculates the value of a two dimensional, second order polynomial

    @param [in] p is an array of 7 polynomial coefficients

    @param [in] x_arr is an array of x coordinates

    @param [in] y_arr is an array of y coordinates

    @param [out] An array of the polynomial evaluated at x_arr and y_arr
    """

    x_terms = p[0] * x_arr * x_arr + p[1] * x_arr + p[2]
    y_terms = p[3] * y_arr * y_arr + p[4] * y_arr + p[5]
    cross_terms = p[6] * x_arr * y_arr
    return x_terms + y_terms + cross_terms


def fit_z_deviations(fp_x, fp_y, distortion_vectors):
    """
    Return the zernike coefficients at a focal plane position using a 2d fit

    @param [in] fp_x is the desired focal plane x coordinate

    @param [in] fp_y is the desired focal plane y coordinate

    @param [in] distortion_vectors is an array describing deviations in
    each optical degree of freedom sampled at 35 positions.
    It should have shape (35, 50).

    @param [out] A list of 19 zernike coefficients
    """

    _raise_zernike_deviations(fp_x, fp_y, distortion_vectors)

    sensitivity_matrix = get_sensitivity_matrix()
    num_sampling_positions = sensitivity_matrix.shape[0]
    num_zernike_coefficients = sensitivity_matrix.shape[1]

    coefficients = np.zeros((num_sampling_positions, num_zernike_coefficients))
    for i in range(num_sampling_positions):
        coefficients[i] = sensitivity_matrix[i].dot(distortion_vectors[i])

    x, y = cartesian_samp_coords()
    out = []
    for i in range(num_zernike_coefficients):
        fit_coeff = optimize.leastsq(_error_fit_func, np.arange(0, 7),
                                     args=(x, y, coefficients[:, i]))

        fit_eval = _fit_func(fit_coeff[0], fp_x, fp_y)
        out.append(fit_eval)

    return out


def _calc_residuals(func, distortions):

    sensitivity_matrix = get_sensitivity_matrix()

    num_positions = sensitivity_matrix.shape[0]
    num_coefficients = sensitivity_matrix.shape[1]

    # Determine the coefficients at the sampling points
    coefficients = np.zeros((num_positions, num_coefficients))
    for i in range(num_positions):
        coefficients[i] = sensitivity_matrix[i].dot(distortions[i])

    # Determine the residuals from a 2d fit
    x, y = cartesian_samp_coords()
    residuals = np.zeros((num_positions, num_coefficients))
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        fit_coefficients = func(x_i, y_i, distortions)
        residuals[i] = np.subtract(coefficients[i], fit_coefficients)

    return residuals.transpose()


def _plot_residuals(func, distortions, path, format):

    # This is a throw away function so we encapsulate this import
    from matplotlib import pyplot as plt

    x, y = cartesian_samp_coords()
    residuals = _calc_residuals(func, distortions)

    fig = plt.figure(figsize=(18, 15))
    for i in range(19):
        axis = fig.add_subplot(4, 5, i + 1)
        axis.scatter(x, y, c=residuals[i], cmap='bwr',
                     label='Z_{}'.format(i + 4))
        axis.legend()

    plt.tight_layout()
    plt.savefig(path, format=format)


if __name__ == "__main__" and DEBUG:

    # Todo: Residuals may be calculated using an incorrect array slice
    distortions = mock_distortions()
    _plot_residuals(fit_z_deviations, distortions, 'fit_resids.jpg', 'jpg')
    _plot_residuals(interp_z_deviations, distortions, 'interp_resids.jpg', 'jpg')
