"""
Code to determine deviations in the Zernike coefficients as determined by the
AOS open loop control system. Results do not include contributions from the
closed loop lookup table.
"""
import os

import numpy as np
from scipy.interpolate import *

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MATRIX_PATH = os.path.join(FILE_DIR, 'sensitivity_matrix.txt')

DEBUG = True


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


def sampling_coordinates():
    """
    Returns two lists with 35 sampling coordinates in the focal plane

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

    return x_list, y_list


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

    r_focal_plane = 100  # Todo: Change place holder value for focal plane radius
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


def zernike_deviations(fp_x, fp_y, distortion_vectors):
    """
    Return the interpolated zernike coefficients at a focal plane position

    @param [in] fp_x is the desired focal plane x coordinate

    @param [in] fp_y is the desired focal plane y coordinate

    @param [in] distortion_vectors is an array describing deviations in
    each optical degree of freedom sampled at 35 positions.
    It should have shape (35, 50).
    """

    _raise_zernike_deviations(fp_x, fp_y, distortion_vectors)

    sensitivity_matrix = get_sensitivity_matrix()
    num_sampling_positions = sensitivity_matrix.shape[0]
    num_zernike_coefficients = sensitivity_matrix.shape[1]

    coefficients = np.zeros((num_sampling_positions, num_zernike_coefficients))
    for i in range(num_sampling_positions):
        coefficients[i] = sensitivity_matrix[i].dot(distortion_vectors[i])

    out = []
    x, y = sampling_coordinates()
    for i in range(num_zernike_coefficients):
        interp_func = interp2d(x, y, coefficients[:, i], kind='cubic')
        out.extend(interp_func(fp_x, fp_y))

    return out


if __name__ == "__main__" and DEBUG:
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

    test_distortions = np.zeros((35, 50))
    for i in range(35):
        test_distortions[i] = distortion_sizes[i]

    coef = zernike_deviations(0, 0, test_distortions)
    print(coef)
