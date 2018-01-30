"""
Code to plot residuals for fits of deviations in the Zernike coefficients
determined by the AOS open loop control system.
"""

import os

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np

from zernike_coeff import cartesian_samp_coords
from zernike_coeff import fit_z_deviations
from zernike_coeff import get_sensitivity_matrix
from zernike_coeff import interp_z_deviations
from zernike_coeff import mock_distortions

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
MATRIX_PATH = os.path.join(FILE_DIR, 'sensitivity_matrix.txt')


def coeff_at_sampling_points(distortions):
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


def calc_residuals(func, distortions):
    """Calculates the residuals for a fit of the zernike coefficients

    @param [in] func is the fit function to calculate residuals for

    @param [in] distortions is a (35, 50) array of mock optical deviations in
    each of the 50 optical degrees of freedom at 35 sampling coordinates.

    @param [out] a (35, 19) array of residuals in the zernike coefficients for
    z=4 through z=22
    """

    coefficients = coeff_at_sampling_points(distortions)
    num_positions = coefficients.shape[0]
    num_coefficients = coefficients.shape[1]

    x, y = cartesian_samp_coords()
    residuals = np.zeros((num_positions, num_coefficients))
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        fit_coefficients = func(x_i, y_i, distortions)
        residuals[i] = np.subtract(coefficients[i], fit_coefficients)

    return residuals


def plot_array(data_array, path, frmt, match_colorbar=False):
    """Write to file a plot of data from a (35, 19) array

    @param [in] data_array is a (35, 19) array

    @param [in] path is the desired output location of the image

    @param [in] format is the desired format of the image (eg. jpeg or eps)
    """

    x, y = cartesian_samp_coords()
    ticks = np.arange(-2, 2.5, .5)
    tick_labels = ['-2', '', '-1', '', '0', '', '1', '', '2']

    fig = plt.figure(figsize=(22, 15))
    for i in range(19):
        # Format figure
        axis = fig.add_subplot(4, 5, i + 1)
        axis.set_ylim(-2, 2)
        axis.set_xlim(-2, 2)
        for sub_axis in (axis.yaxis, axis.xaxis):
            sub_axis.set_ticks(ticks)
            sub_axis.set_ticklabels(tick_labels, fontsize=15)

        if i % 5:
            axis.set_yticklabels([])

        if i < 14:
            axis.set_xticklabels([])

        # Plot data
        if match_colorbar:
            vlim_data = data_array

        else:
            vlim_data = data_array[:, i]

        vlim = max(np.abs(np.amin(vlim_data)),
                       np.abs(np.amax(vlim_data)))

        scatter = axis.scatter(x, y,
                               c=data_array[:, i],
                               cmap='bwr',
                               label='Z_{}'.format(i + 4),
                               vmin=-vlim,
                               vmax=vlim)
        axis.legend()
        fig.colorbar(scatter)

    #cb_ax = fig.add_axes([0.93, 0.09, 0.02, 0.8])

    plt.savefig(path, format=frmt)


def plot_nominal_zernikes(path, frmt):
    """Write to file a plot of the nominal Zernike polynomials

    @param [in] path is the desired output location of the image

    @param [in] format is the desired format of the image (eg. jpeg or eps)
    """

    # plot nominal Zernike maps
    zernike_data = fits.open("nominal_zernike_coefs.fits")[0].data

    fig, axes = plt.subplots(5, 4, figsize=(18, 15))
    flat_ax = axes.flatten()
    for i in range(19):
        im = flat_ax[i].imshow(zernike_data[:, :, i + 3],
                               aspect='auto',
                               origin='lower',
                               extent=(-2., 2., -2., 2.))

        fig.colorbar(im, ax=flat_ax[i])
        flat_ax[i].set_title('Zernike {}'.format(i + 4))

    fig.suptitle("Zernike by Focal Plane position", fontsize=20)
    plt.savefig(path, format=frmt)


if __name__ == "__main__":

    fig_dir = os.path.join(FILE_DIR, 'figs/')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    moc_distort = mock_distortions(.2)

    coeff = coeff_at_sampling_points(moc_distort)
    coeff_path = os.path.join(fig_dir, 'coeff.jpg')
    plot_array(coeff, coeff_path, 'jpg')

    fit_residuals = calc_residuals(fit_z_deviations, moc_distort)
    fit_path = os.path.join(fig_dir, 'fit_resids.jpg')
    plot_array(fit_residuals, fit_path, 'jpg', True)

    interp_residuals = calc_residuals(interp_z_deviations, moc_distort)
    interp_path = os.path.join(fig_dir, 'interp_resids.jpg')
    plot_array(interp_residuals, interp_path, 'jpg', True)

    nominal_path = os.path.join(fig_dir, 'nominal_zernikes.jpg')
    plot_nominal_zernikes(nominal_path, 'jpg')
