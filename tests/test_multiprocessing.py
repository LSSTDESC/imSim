"""
Script to test that the sky catalog object lists match the
CCD being simulated in multi-CCD runs.
"""
import os
import sys
import glob
import argparse
from pathlib import Path
import numpy as np
import logging
import galsim

def run_imsim(nfiles):
    """
    Run imSim with sky catalog input for 1 or 2 CCDs.

    Parameters
    ----------
    nfiles : int
          For nfiles=1, just simulate the central CCD, R22_S11.
          For nfiles=2, simulate R22_S11 and R22_S12.
    """
    imsim_dir = os.path.dirname(str(Path(__file__).parent))
    template = os.path.join(imsim_dir, 'config', 'imsim-config.yaml')
    skycatalog_file = os.path.join(imsim_dir, 'tests', 'data',
                                   'sky_cat_multiproc_test.yaml')
    opsim_db_file = os.path.join(imsim_dir, 'tests', 'data',
                                 'small_opsim_9683.db')

    logger = logging.getLogger('test_multiprocessing')
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    config = {'template': template,
              'input.instance_catalog': '',
              'input.sky_catalog': {'file_name': skycatalog_file,
                                    'obj_types': ['galaxy']},
              'input.opsim_meta_dict.file_name': opsim_db_file,
              'input.opsim_meta_dict.visit': 449053,
              'input.tree_rings.only_dets': ['R22_S11', 'R22_S12'],
              'image.random_seed': 42,
              'gal.type': 'SkyCatObj',
              'stamp.world_pos.type': 'SkyCatWorldPos',
              'stamp.fft_sb_thresh': '1e5',
              'output.camera': 'LsstCam',
              'output.det_num.first': 94,
              'output.nfiles': nfiles,
              'output.truth.dir': 'fits',
              'output.truth.file_name.format': 'centroid_%08d-%1d-%s-%s-det%03d.txt',
            }

    args = argparse.Namespace(config_file='', variables=[],
                              verbosity=2, log_file=None, file_type=None,
                              module=None, profile=False, njobs=1, job=1,
                              except_abort=True, version=False)

    galsim.config.Process(config, logger=logger)

def cleanup():
    # Clean up the test output.
    for item in glob.glob('fits/*'):
        os.remove(item)
    os.rmdir('fits')

np.testing.assert_(run_imsim(1) is None)
cleanup()

np.testing.assert_(run_imsim(2) is None)
cleanup()

