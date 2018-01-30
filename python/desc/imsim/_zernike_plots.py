"""
Code to plot residuals for fits of deviations in the Zernike coefficients
determined by the AOS open loop control system.
"""

import os

import numpy as np

from zernike_coeff import cartesian_samp_coords
from zernike_coeff import fit_z_deviations
from zernike_coeff import get_sensitivity_matrix
from zernike_coeff import interp_z_deviations
from zernike_coeff import mock_distortions

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MATRIX_PATH = os.path.join(FILE_DIR, 'sensitivity_matrix.txt')


def _coeff_at_sampling_points(distortions):
    """Determines the zernike coefficients at a set of sampling coordinates

    Zernike coefficients are calculated by multiplying the sensitivity matrix
    by an array of optical distortions.

    @param [in] distortions is a (35, 50) array of mock optical deviations in
    each of the 50 optical degrees of freedom at 35 sampling coordinates.

    @param [out] A (35, 19) array of zernike coefficients for z=4 through z=22
    """

    sensitivity_matrix = get_sensitivity_matrix()
    num_positions = sensitivity_matrix.shape[0]
    num_coefficients = sensitivity_matrix.shape[1]

    # Determine the coefficients at the sampling points
    coefficients = np.zeros((num_positions, num_coefficients))
    for i in range(num_positions):
        coefficients[i] = sensitivity_matrix[i].dot(distortions[i])

    return coefficients


def _plot_coeff(distortions, path, frmt):
    """Writes to file a plot of zernike coefficients at 35 sampling locations

    @param [in] distortions is a (35, 50) array of mock optical deviations in
    each of the 50 optical degrees of freedom at 35 sampling coordinates.

    @param [in] path is the desired output location of the image

    @param [in] format is the desired format of the image (eg. jpeg or eps)
    """

    # _plot_coeff is a throw away function so we encapsulate this import
    from matplotlib import pyplot as plt

    x, y = cartesian_samp_coords()
    coeff = _coeff_at_sampling_points(distortions)
    coeff = coeff.transpose()

    fig = plt.figure(figsize=(18, 15))
    for i in range(19):
        # Format figure
        axis = fig.add_subplot(4, 5, i + 1)
        axis.set_xlim(-1.5, 1.5)
        axis.xaxis.set_ticks(np.arange(-1.5, 2.0, .5))
        if i % 5:
            axis.set_yticklabels([])

        if i < 14:
            axis.set_xticklabels([])

        # Plot data
        label = 'Z_{}'.format(i + 4)
        vlim = max(np.abs(np.amin(coeff)), np.abs(np.amax(coeff)))
        scatter = axis.scatter(x, y,
                               c=coeff[i],
                               cmap='bwr',
                               label=label,
                               vmin=-vlim,
                               vmax=vlim)
        axis.legend()

    cb_ax = fig.add_axes([0.93, 0.09, 0.02, 0.8])
    fig.colorbar(scatter, cax=cb_ax)
    plt.savefig(path, format=frmt)


def _calc_residuals(func, distortions):
    """Calculates the residuals for a fit of the zernike coefficients

    @param [in] func is the fit function to calculate residuals for

    @param [in] distortions is a (35, 50) array of mock optical deviations in
    each of the 50 optical degrees of freedom at 35 sampling coordinates.

    @param [out] a (35, 19) array of residuals in the zernike coefficients for
    z=4 through z=22
    """

    coefficients = _coeff_at_sampling_points(distortions)
    num_positions = coefficients.shape[0]
    num_coefficients = coefficients.shape[1]

    x, y = cartesian_samp_coords()
    residuals = np.zeros((num_positions, num_coefficients))
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        fit_coefficients = func(x_i, y_i, distortions)
        residuals[i] = np.subtract(coefficients[i], fit_coefficients)

    return residuals


def _plot_residuals(func, distortions, path, frmt):
    """Calculates the residuals for a fit of the zernike coefficients

    @param [in] func is the fit function to calculate residuals for

    @param [in] distortions is a (35, 50) array of mock optical deviations in
    each of the 50 optical degrees of freedom at 35 sampling coordinates.

    @param [in] path is the desired output location of the image

    @param [in] format is the desired format of the image (eg. jpeg or eps)
    """

    # _plot_residuals is a throw away function so we encapsulate this import
    from matplotlib import pyplot as plt

    x, y = cartesian_samp_coords()
    residuals = _calc_residuals(func, distortions)
    residuals = residuals.transpose()

    fig = plt.figure(figsize=(18, 15))
    for i in range(19):
        # Format figure
        axis = fig.add_subplot(4, 5, i + 1)
        axis.set_xlim(-1.5, 1.5)
        axis.xaxis.set_ticks(np.arange(-1.5, 2.0, .5))
        if i % 5:
            axis.set_yticklabels([])

        if i < 14:
            axis.set_xticklabels([])

        # Plot data
        label = 'Z_{}'.format(i + 4)
        vlim = max(np.abs(np.amin(residuals)), np.abs(np.amax(residuals)))
        scatter = axis.scatter(x, y,
                               c=residuals[i],
                               cmap='bwr',
                               label=label,
                               vmin=-vlim,
                               vmax=vlim)
        axis.legend()

    cb_ax = fig.add_axes([0.93, 0.09, 0.02, 0.8])
    fig.colorbar(scatter, cax=cb_ax)
    plt.savefig(path, format=frmt)


if __name__ == "__main__":

    fig_dir = os.path.join(FILE_DIR, 'figs/')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # Note that the jpg format is not supported in Python 2.7
    moc_distort = mock_distortions()
    _plot_coeff(moc_distort, os.path.join(fig_dir, 'coeff.jpg'), 'jpg')
    _plot_residuals(fit_z_deviations, moc_distort,
                    os.path.join(fig_dir, 'fit_resids.jpg'), 'jpg')
