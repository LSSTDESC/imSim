import os
from pathlib import Path
import numpy as np
import logging
import galsim
import time
import logging
import imsim
import fitsio

DATA_DIR = Path(__file__).parent / 'data'


def test_checkpoint_image():
    """Test using checkpointing in LSST_Image type.
    """
    wcs = galsim.PixelScale(0.2)
    bandpass = galsim.Bandpass('LSST_r.dat', 'nm')
    if os.path.exists('output/checkpoint_0.hdf'):
        os.remove('output/checkpoint_0.hdf')
    config = {
        'input': {
            'checkpoint': {
                'dir': 'output',
                'file_name': '$"checkpoint_%d.hdf"%image_num',
            },
        },
        'gal': {
            'type': 'Exponential',
            'half_light_radius': {'type': 'Random', 'min': 0.3, 'max': 1.2},
            'ellip': {
                'type': 'EBeta',
                'e': {'type': 'Random', 'min': 0.0, 'max': 0.3},
                'beta': {'type': 'Random'},
            }
        },
        'image': {
            'type': 'LSST_Image',
            'xsize': 2048,
            'ysize': 2048,
            'wcs': wcs,
            'bandpass': bandpass,
            'random_seed': 12345,
            'nobjects': 500,
        },
    }
    # This first pass should write checkpointing updates, but not really use them for anything.
    config['image_num'] = 0
    galsim.config.ProcessInput(config)
    t0 = time.time()
    image0 = galsim.config.BuildImage(config)
    t1 = time.time()
    print('no checkpointing: time = ',t1-t0)  # around 0.4 sec

    # Running it again should use the checkpointing, which tells it that all the rendering
    # is already complete.  So it just returns the already completed image.
    config = galsim.config.CleanConfig(config) # prevents GalSim using "current" items.
    t2 = time.time()
    image1 = galsim.config.BuildImage(config)
    t3 = time.time()
    print('use checkpointing: time = ',t3-t2)  # around 0.008 sec

    assert image0 == image1
    assert t3-t2 < t1-t0

    # If there is an extra_builder item, that will get checkpointed as well.
    os.remove('output/checkpoint_0.hdf')
    config = galsim.config.CleanConfig(config)
    config['output'] = {
        'dir': 'output',
        'file_name': '$"test_checkpoint_image_%d.fits.fz"%image_num',
        'truth': {
            'file_name': '$"test_checkpoint_centroid_%d.fits"%image_num',
            'columns': {
                'hlr': "gal.half_light_radius",
                'e': "gal.ellip.e",
                'beta': "gal.ellip.beta",
                'pos': "image_pos",
            },
        },
    }

    t0 = time.time()
    galsim.config.Process(config)
    t1 = time.time()
    print('with centroid: time = ',t1-t0)  # around 0.5 sec
    image2 = galsim.fits.read('output/test_checkpoint_image_0.fits.fz')
    centroid2 = fitsio.read('output/test_checkpoint_centroid_0.fits')

    # And again from teh checkpoint file.
    config = galsim.config.CleanConfig(config)
    t2 = time.time()
    galsim.config.Process(config)
    t3 = time.time()
    print('use checkpointing: time = ',t3-t2)  # around 0.09 sec

    image3 = galsim.fits.read('output/test_checkpoint_image_0.fits.fz')
    centroid3 = fitsio.read('output/test_checkpoint_centroid_0.fits')

    assert image2 == image3
    np.testing.assert_array_equal(centroid2, centroid3)
    assert t3-t2 < t1-t0

    # Finally, test a series of runs where the jobs time out and resume using the checkpoint file.
    # Rather than actually timing out, we include a function that has some probability of
    # calling exit().
    def timeout_one(config, base, value_type):
        p = config['p']
        rng = galsim.config.GetRNG(config, base)
        u = galsim.UniformDeviate(rng)()
        if u < p:
            raise TimeoutError("u = %f < %f at obj_num = %d"%(u,p,base['obj_num']))
        return 1
    galsim.config.RegisterValueType('timeout_one', timeout_one, [float])

    os.remove('output/checkpoint_0.hdf')
    os.remove('output/test_checkpoint_image_0.fits.fz')
    os.remove('output/test_checkpoint_centroid_0.fits')
    del galsim.config.valid_extra_outputs['truth'].cat

    logger = logging.getLogger('test_checkpoint')
    logger.setLevel(logging.INFO)

    p = 0.1 # Start fairly large, so "times out" a lot, but decrease this each iteration.
    while not os.path.exists('output/test_checkpoint_image_0.fits.fz'):
        print('p = ',p)
        config['gal']['scale_flux'] = {'type': 'timeout_one', 'p': p}
        config = galsim.config.CleanConfig(config)
        config.pop('extra_builder',None)
        try:
            galsim.config.Process(config, logger)
        except Exception as e:
            print('Caught ',e)

        chk = imsim.Checkpointer('output/checkpoint_0.hdf').load('buildImage_')
        if chk is not None:
            print('nobj complete = ',chk[2])
        p /= 2

    image4 = galsim.fits.read('output/test_checkpoint_image_0.fits.fz')
    centroid4 = fitsio.read('output/test_checkpoint_centroid_0.fits')
    assert image4 == image2
    np.testing.assert_array_equal(centroid4, centroid2)

    # Finally, all of these should be the same as when there is no checkpointing in input.
    del config['input']
    del config['gal']['scale_flux']
    config = galsim.config.CleanConfig(config)

    galsim.config.Process(config)
    image5 = galsim.fits.read('output/test_checkpoint_image_0.fits.fz')
    centroid5 = fitsio.read('output/test_checkpoint_centroid_0.fits')
    assert image5 == image2
    np.testing.assert_array_equal(centroid5, centroid2)



if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
