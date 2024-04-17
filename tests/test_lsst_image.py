import os
import shutil
import sys
import logging
from pathlib import Path
import numpy as np
import galsim


def test_image_nobjects():
    """
    Test that the minimum of the optional config setting of
    image.nobjects and the number of objects available is used to
    avoid repeated object renderings.
    """
    imsim_dir = os.path.dirname(os.path.abspath(str(Path(__file__).parent)))
    os.environ['SIMS_SED_LIBRARY_DIR'] \
        = os.path.join(imsim_dir, 'tests', 'data', 'test_sed_library')
    template = os.path.join(imsim_dir, 'config', 'imsim-config-instcat.yaml')
    instcat_file = os.path.join(imsim_dir, 'tests', 'data',
                                'test_multiproc_instcat.txt')

    logger = logging.getLogger('test_lsst_image')
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.CRITICAL)  # silence the log messages

    camera = "LsstCam"
    output_dir = f"fits_{camera}"

    # Loop over values of image.nobjects that bracket the expected number
    # from the instance catalog to test the number of objects actually
    # rendered.
    for nobjects, nexpected in zip((1, 5, ""), (1, 3, 3)):
        config = {'modules': ['imsim'],
                  'template': template,
                  'input.instance_catalog.file_name': instcat_file,
                  'input.opsim_data.file_name': instcat_file,
                  'input.tree_rings': '',
                  'input.atm_psf': '',
                  'input.checkpoint': '',
                  'image.sky_level': 0,
                  'image.random_seed': 42,
                  'image.sensor': '',
                  'image.nobjects': nobjects,
                  'stamp.fft_sb_thresh': '1e5',
                  'stamp.size': 48,
                  'psf.items': '',
                  'psf.type': 'Gaussian',
                  'psf.fwhm': 0.7,
                  'output.camera': camera,
                  'output.cosmic_ray_rate': 0,
                  'output.det_num.first': 94,
                  'output.nfiles': 1,
                  'output.readout': '',
                  'output.dir': output_dir,
                  'output.truth.dir': output_dir,
                  'output.truth.file_name.format': 'centroid_%08d-%1d-%s-%s-det%03d.txt',
        }

        galsim.config.Process(config, logger=logger)
        data = np.genfromtxt(f"{output_dir}/centroid_00161899-0-r-R22_S11-det094.txt",
                             names=True)
        assert nexpected == data.size

        shutil.rmtree(output_dir)

if __name__ == '__main__':
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
