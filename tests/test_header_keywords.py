import os
import glob
import sys
from pathlib import Path
import logging
from astropy.io import fits
import numpy as np
import galsim


def run_imsim(camera):
    imsim_dir = os.path.dirname(os.path.abspath(str(Path(__file__).parent)))
    os.environ['SIMS_SED_LIBRARY_DIR'] \
        = os.path.join(imsim_dir, 'tests', 'data', 'test_sed_library')
    template = os.path.join(imsim_dir, 'config', 'imsim-config.yaml')
    instcat_file = os.path.join(imsim_dir, 'tests', 'data',
                                'instcat_object_positions_test.txt')

    logger = logging.getLogger('test_header_keywords')
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.CRITICAL)  # silence the log messages

    only_dets = ['R22_S11']

    config = {'template': template,
              'input.instance_catalog.file_name': instcat_file,
              'input.opsim_meta_dict.file_name': instcat_file,
              'input.tree_rings.only_dets': only_dets,
              'input.atm_psf': '',
              'input.checkpoint': '',
              'image.sky_level': 0,
              'image.random_seed': 42,
              'image.sensor': '',
              'stamp.fft_sb_thresh': '1e5',
              'stamp.size': 48,
              'psf.items': '',
              'psf.type': 'Gaussian',
              'psf.fwhm': 0.7,
              'output.camera': camera,
              'output.cosmic_ray_rate': 0,
              'output.only_dets': only_dets,
              'output.det_num.first': 0,
              'output.nfiles': 1,
              'output.readout': '',
              'output.dir': 'fits_header_test',
              'output.truth': ''}

    galsim.config.Process(config, logger=logger)


def test_header_keywords():
    run_imsim('LsstCam')
    fits_dir = 'fits_header_test'
    eimage_file = glob.glob(os.path.join(fits_dir, 'eimage*.fits'))[0]
    with fits.open(eimage_file) as hdus:
        mjd = hdus[0].header['MJD']
        exptime = hdus[0].header['EXPTIME']
        mjd_obs = mjd + exptime/2./86400.
        np.testing.assert_approx_equal(hdus[0].header['MJD-OBS'], mjd_obs,
                                       significant=7)
    os.remove(os.path.join(eimage_file))
    os.removedirs(os.path.dirname(eimage_file))


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
