import os
import sys
from pathlib import Path
import logging
import unittest
import galsim


class MultiprocTestCase(unittest.TestCase):
    """TestCase class to test multiprocessing with imSim"""
    def setUp(self):
        self.output_dir = 'fits_multiproc_test'
        self.only_dets = ['R22_S11', 'R22_S12']
        self.expected_files = []
        for det_num, det_name in enumerate(self.only_dets):
            self.expected_files.extend(
                [os.path.join(self.output_dir,
                              f'{prefix}_00161899-0-r-{det_name}'
                              f'-det{det_num:03d}.{suffix}')
                 for prefix, suffix in [('amp', 'fits.fz'),
                                        ('eimage', 'fits')]])

    def tearDown(self):
        """Clean up test output files, if they exist."""
        for item in self.expected_files:
            if os.path.isfile(item):
                os.remove(item)
        if os.path.isdir(self.output_dir):
            os.removedirs(self.output_dir)

    def run_imsim(self):
        imsim_dir = os.path.dirname(os.path.abspath(str(Path(__file__).parent)))
        if 'SIMS_SED_LIBRARY_DIR' not in os.environ:
            os.environ['SIMS_SED_LIBRARY_DIR'] \
                = os.path.join(imsim_dir, 'tests', 'data', 'test_sed_library')
        template = os.path.join(imsim_dir, 'config',
                                'imsim-config-instcat.yaml')
        instcat_file = os.path.join(imsim_dir, 'tests', 'data',
                                    'test_multiproc_instcat.txt')
        logger = logging.getLogger('test_multiproc')
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.CRITICAL)  # silence the log messages

        config = {'modules': ['imsim'],
                  'template': template,
                  'input.instance_catalog.file_name': instcat_file,
                  'input.opsim_data.file_name': instcat_file,
                  'input.tree_rings.only_dets': self.only_dets,
                  'input.checkpoint': '',
                  'input.atm_psf': '',
                  'psf': {'type': 'Convolve',
                          'items': [{'type': 'Gaussian',
                                     'fwhm': 0.8},
                                    {'type': 'Gaussian',
                                     'fwhm': 0.3}]
                          },
                  'image.random_seed': 42,
                  'stamp.fft_sb_thresh': '1e5',
                  'output.only_dets': self.only_dets,
                  'output.det_num.first': 0,
                  'output.nfiles': 2,
                  'output.dir': self.output_dir,
                  'output.truth': '',
                  'output.nproc': 2,
                  }

        galsim.config.Process(config, logger=logger, except_abort=True)

    def test_multiproc(self):
        """Run the 2-process test"""
        self.run_imsim()
        # Check that expected files exist.
        for item in self.expected_files:
            self.assertTrue(os.path.isfile(item))


if __name__ == "__main__":
    unittest.main()
