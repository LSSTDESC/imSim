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
    template = os.path.join(imsim_dir, 'config', 'imsim-config-instcat.yaml')
    instcat_file = os.path.join(imsim_dir, 'tests', 'data',
                                'instcat_object_positions_test.txt')

    logger = logging.getLogger('test_header_keywords')
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.CRITICAL)  # silence the log messages

    only_dets = ['R22_S11']

    config = {'template': template,
              'input.instance_catalog.file_name': instcat_file,
              'input.opsim_data.file_name': instcat_file,
              'input.tree_rings.only_dets': only_dets,
              'input.atm_psf': '',
              'input.checkpoint': '',
              'input.opsim_data.image_type': 'BIAS',
              'input.opsim_data.reason': 'calibration',
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
              'output.dir': 'fits_header_test',
              'output.truth': '',
              'output.header': {
                  'fieldRA': '$boresight.ra.deg',
                  'fieldDec': '$boresight.dec.deg',
                  'test1': '$1+2.3',
                  'test2': '@output.det_num',
                  'test3': 'banana'
              }
             }

    galsim.config.Process(config, logger=logger)


def test_header_keywords():
    run_imsim('LsstCam')
    fits_dir = 'fits_header_test'
    eimage_file = glob.glob(os.path.join(fits_dir, 'eimage*.fits'))[0]
    with fits.open(eimage_file) as hdus:
        mjd = hdus[0].header['MJD']
        exptime = hdus[0].header['EXPTIME']
        assert hdus[0].header['IMGTYPE'] == 'BIAS'
        assert hdus[0].header['REASON'] == 'calibration'
        mjd_obs = mjd + exptime/2./86400.
        np.testing.assert_approx_equal(hdus[0].header['MJD-OBS'], mjd_obs,
                                       significant=7)
        assert hdus[0].header['TEST1'] == 3.3
        assert hdus[0].header['TEST2'] == 0
        assert hdus[0].header['TEST3'] == 'banana'
    raw_file = glob.glob(os.path.join(fits_dir, 'amp*.fits.fz'))[0]
    with fits.open(raw_file) as hdus:
        assert hdus[0].header['RUNNUM'] == 182850
        assert hdus[0].header['LSST_NUM'] == 'E2V-CCD250-382'
        assert hdus[0].header['FILTER'] == 'i_39'
    for item in glob.glob(os.path.join(fits_dir, '*')):
        os.remove(item)
    os.removedirs(fits_dir)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
