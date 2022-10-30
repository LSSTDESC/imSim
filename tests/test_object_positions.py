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
    template = os.path.join(imsim_dir, 'config', 'imsim-config.yaml')
    instcat_file = os.path.join(imsim_dir, 'tests', 'data',
                                'instcat_object_positions_test.txt')

    logger = logging.getLogger('test_object_positions')
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    only_dets = ['R22_S11', 'R01_S00', 'R42_S21', 'R34_S22', 'R03_S02']

    config = {'template': template,
              'input.instance_catalog.file_name': instcat_file,
              'input.opsim_meta_dict.file_name': instcat_file,
              'input.tree_rings.only_dets': only_dets,
              'image.sky_level': 0,
              'image.random_seed': 42,
              'stamp.fft_sb_thresh': '1e5',
              'stamp.size': 48,
              'output.camera': camera,
              'output.cosmic_ray_rate': 0,
              'output.only_dets': only_dets,
              'output.det_num.first': 0,
              'output.nfiles': 5,
              'output.readout': '',
              'output.dir': f'fits_{camera}',
              'output.truth.dir': f'fits_{camera}',
              'output.truth.file_name.format': 'centroid_%08d-%1d-%s-%s-det%03d.txt',
            }

    galsim.config.Process(config, logger=logger)

def compute_pixel_offset(eimage_file):
    # Assuming there is just one object rendered on the eimage,
    # estmate its position by taking the weighted mean of the x and y
    # pixel coordinates.
    with fits.open(eimage_file) as hdus:
        eimage = hdus[0].data
    ny, nx = eimage.shape

    xarr = np.ones((ny, nx))
    yarr = np.ones((ny, nx))
    for row in range(ny):
        yarr[row,:] *= row
    for col in range(nx):
        xarr[:,col] *= col

    total_counts = np.sum(eimage)
    x_avg = np.sum(xarr*eimage)/total_counts
    y_avg = np.sum(yarr*eimage)/total_counts

    # Compute offset wrt centroid coordinates.
    centroid_file = eimage_file.replace('eimage', 'centroid')\
                               .replace('.fits', '.txt')
    data = np.genfromtxt(centroid_file, names='id ra dec x y'.split())

    return np.sqrt((x_avg - data['x'])**2 + (y_avg - data['y'])**2)

for camera in ('LsstCam', 'LsstCamImSim'):
    print(camera)
    run_imsim(camera)
    eimage_files = glob.glob(f'fits_{camera}/eimage*')
    for eimage_file in eimage_files:
        offset = compute_pixel_offset(eimage_file)
        print(eimage_file, offset)
        np.testing.assert_(offset < 2)