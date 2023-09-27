import os
import sys
import glob
import shutil
import unittest
import logging
from pathlib import Path
import psutil
import numpy as np
import galsim


class ProcessInfoTestCase(unittest.TestCase):
    """TestCase class for ProcessInfo code."""
    def setUp(self):
        pass

    def tearDown(self):
        for output_dir in (self.config['output.process_info']['dir'],
                           self.config['output.dir']):
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)

    def run_imsim(self):
        imsim_dir = os.path.dirname(os.path.abspath(str(Path(__file__).parent)))
        if 'SIMS_SED_LIBRARY_DIR' not in os.environ:
            os.environ['SIMS_SED_LIBRARY_DIR'] \
                = os.path.join(imsim_dir, 'tests', 'data', 'test_sed_library')
        template = os.path.join(imsim_dir, 'config',
                                'imsim-config-instcat.yaml')
        instcat_file = os.path.join(imsim_dir, 'tests', 'data',
                                    'test_multiproc_instcat.txt')
        logger = logging.getLogger('test_process_info')
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.CRITICAL)

        self.config = {'modules': ['imsim'],
                       'template': template,
                       'input.instance_catalog.file_name': instcat_file,
                       'input.opsim_data.file_name': instcat_file,
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
                       'output.cosmic_ray_rate': 0,
                       'output.det_num.first': 94,
                       'output.nfiles': 1,
                       'output.readout': '',
                       'output.truth': '',
                       'output.dir': 'process_info_fits',
                       'output.process_info': {'dir': 'process_info_test',
                                               'file_name': {'type': 'FormattedStr',
                                                             'format': 'process_info_%08d-%1d-%s-%s-det%03d.txt.gz',
                                                             'items': [{'type': 'OpsimData', 'field': 'observationId'},
                                                                       {'type': 'OpsimData', 'field': 'snap'},
                                                                       '$band',
                                                                       '$det_name',
                                                                       '@output.det_num']
                                                             }
                                               }
                       }
        galsim.config.Process(self.config, logger=logger, except_abort=True)

    def test_process_info(self):
        """Test the process_info outputs."""
        pid = os.getpid()
        proc = psutil.Process(pid)
        user_time_0 = proc.cpu_times().user
        self.run_imsim()
        user_time_1 = proc.cpu_times().user
        pattern = os.path.join(self.config['output.process_info']['dir'],
                               "process_info_*-R22_S11-det094.txt.gz")
        process_info_file = glob.glob(pattern)[0]
        data = np.genfromtxt(process_info_file,
                             names=['object_id', 'pid', 'rss',
                                    'uss', 'user_time', 'unix_time'])
        self.assertTrue(all(user_time_0 < data['user_time']))
        self.assertTrue(all(user_time_1 > data['user_time']))
        self.assertTrue(all(pid == data['pid']))


if __name__ == "__main__":
    unittest.main()
