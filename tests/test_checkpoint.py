import os
from pathlib import Path
import numpy as np
import hashlib
import logging
import galsim
import time
import logging
import imsim
import fitsio
import shutil

DATA_DIR = Path(__file__).parent / 'data'

# Used below in both test_checkpoint_image and test_checkpoint_flatten
def timeout_one(config, base, value_type):
    p = config['p']
    rng = galsim.config.GetRNG(config, base).duplicate()
    u = galsim.UniformDeviate(rng)()
    if u < p:
        raise TimeoutError("u = %f < %f at obj_num = %d"%(u,p,base['obj_num']))
    return 1

galsim.config.RegisterValueType('timeout_one', timeout_one, [float])


def test_checkpoint_base():
    """Test various aspects of the Checkpointer class directly.
    """
    # Check file names
    if os.path.exists('output/chk.hdf'):
        os.remove('output/chk.hdf')
    chk = imsim.Checkpointer('output/chk.hdf')
    assert chk.file_name == 'output/chk.hdf'
    assert chk.file_name_bak == 'output/chk.hdf_bak'
    assert chk.file_name_new == 'output/chk.hdf_new'

    # Test save
    data = np.arange(5)
    chk.save('test1', data)
    chk.save('test2', (data, 17))

    # Test load
    chk2 = imsim.Checkpointer('chk.hdf', dir='output')
    assert chk2.file_name == 'output/chk.hdf'
    data2 = chk2.load('test1')
    np.testing.assert_equal(data, data2)
    np.testing.assert_equal(chk2.load('test2'), (data, 17))
    assert chk2.load('test3') is None

    # Test overwrite save
    chk.save('test2', (data, 32))
    np.testing.assert_equal(chk2.load('test2'), (data, 32))

    # Test recover from backup
    os.rename(chk.file_name, chk.file_name_bak)
    chk3 = imsim.Checkpointer('output/chk.hdf')
    np.testing.assert_equal(chk3.load('test2'), (data, 32))
    assert not os.path.isfile(chk.file_name_bak)

    shutil.copy(chk.file_name, chk.file_name_bak)
    chk3 = imsim.Checkpointer('output/chk.hdf')
    np.testing.assert_equal(chk3.load('test2'), (data, 32))
    assert not os.path.isfile(chk.file_name_bak)

    shutil.copy(chk.file_name, chk.file_name_bak)
    os.rename(chk.file_name, chk.file_name_new)
    chk3 = imsim.Checkpointer('output/chk.hdf')
    np.testing.assert_equal(chk3.load('test2'), (data, 32))
    assert not os.path.isfile(chk.file_name_bak)
    assert not os.path.isfile(chk.file_name_new)

    # Test loading from new file
    chk4 = imsim.Checkpointer('output/chk4.hdf')
    assert chk4.load('test2') is None


def test_checkpoint_image():
    """Test using checkpointing in LSST_Image type.
    """
    wcs = galsim.PixelScale(0.2)
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
            'det_name': 'R22_S11',
            'xsize': 2048,
            'ysize': 2048,
            'wcs': wcs,
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
    # raising a TimeoutError.  (Defined above, since we also use it in test_checkpoint_flatten.)

    os.remove('output/checkpoint_0.hdf')
    os.remove('output/test_checkpoint_image_0.fits.fz')
    os.remove('output/test_checkpoint_centroid_0.fits')
    del galsim.config.valid_extra_outputs['truth'].final_data
    try:
        del galsim.config.valid_extra_outputs['truth'].cat
    except AttributeError:
        # This is only on GalSim 2.4 series.
        pass

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
            print('nobj complete = ',chk[3])
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


def test_nbatch_per_checkpoint():
    """Test that the final checkpoint files written with two different values
    of nbatch_per_checkpoint, both not factors of nbatch, produce the same
    checkpoint output."""
    wcs = galsim.PixelScale(0.2)
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
            'det_name': 'R22_S11',
            'xsize': 2048,
            'ysize': 2048,
            'wcs': wcs,
            'random_seed': 12345,
            'nobjects': 500,
            'nbatch': 100,
        },
    }

    checkpoint_0 = 'output/checkpoint_0.hdf'
    if os.path.exists(checkpoint_0):
        os.remove(checkpoint_0)
    config['image_num'] = 0
    config['image']['nbatch_per_checkpoint'] = 11
    galsim.config.ProcessInput(config)
    image0 = galsim.config.BuildImage(config)
    with open(checkpoint_0, 'rb') as fobj:
        md5_0 = hashlib.md5(fobj.read()).hexdigest()

    checkpoint_1 = 'output/checkpoint_1.hdf'
    if os.path.exists(checkpoint_1):
        os.remove(checkpoint_1)
    config['image_num'] = 1
    config['image']['nbatch_per_checkpoint'] = 13
    galsim.config.ProcessInput(config)
    image1 = galsim.config.BuildImage(config)
    with open(checkpoint_1, 'rb') as fobj:
        md5_1 = hashlib.md5(fobj.read()).hexdigest()

    assert md5_0 == md5_1


def test_checkpoint_flatten():
    """Test the flatten() step of buildImage when using checkpointing
    """
    # If LSST_Image is used with RealGalaxy objects, then we have to "flatten" the noise.
    # This is a bit tricky to do correctly in conjunction with checkpointing, so this
    # test verifies that this is done correctly.
    real_gal_dir = DATA_DIR
    real_gal_cat = 'real_galaxy_catalog_23.5_example.fits'
    config = {
        'input': {
            'real_catalog': {
                'dir': real_gal_dir,
                'file_name': real_gal_cat,
            },
        },
        'gal': {
            'type': 'RealGalaxy',
            'index': 79,
            'flux': {'type': 'Random', 'min': 100, 'max': 500}
        },
        'image': {
            'type': 'LSST_Image',
            'det_name': 'R22_S11',
            'xsize': 2048,
            'ysize': 2048,
            'pixel_scale': 0.2,
            'random_seed': 12345,
            'nobjects': 20,
            'noise': {
                'type': 'Gaussian',
                'sigma': 30,
                'whiten': True,
            },
        },
        'stamp': {
            # Somewhat gratuitously test two other features in LSST_Image while we're at it.
            # 1. some objects being skipped
            # 2. some objects being off the image.
            'skip': {'type': 'Random', 'p': 0.1},
            'image_pos': {
                'type' : 'XY' ,
                'x' : { 'type' : 'Random' , 'min' : -300, 'max' : 2350 },
                'y' : { 'type' : 'Random' , 'min' : -300, 'max' : 2350 }
            },
        },
    }

    # First without any checkpointing or batching
    config['image']['nbatch'] = 0
    galsim.config.ProcessInput(config)
    im1 = galsim.config.BuildImage(config)

    # Second with nbatch > 1, but not yet checkpointing.
    config = galsim.config.CleanConfig(config)
    config['image']['nbatch'] = 5
    im2 = galsim.config.BuildImage(config)
    assert im1 == im2

    # Finally, with checkpointing
    if os.path.exists('output/checkpoint_flat_0.hdf'):
        os.remove('output/checkpoint_flat_0.hdf')
    config['input']['checkpoint'] = {
        'dir': 'output',
        'file_name': '$"checkpoint_flat_%d.hdf"%image_num',
    }
    p = 0.1
    while True:
        print('p = ',p)
        config['gal']['scale_flux'] = { 'type': 'timeout_one', 'p': p }
        config = galsim.config.CleanConfig(config)
        try:
            im3 = galsim.config.BuildImage(config)
            break
        except Exception as e:
            print('Caught ',e)
            p /= 2
    assert im1 == im3

    # Lastly, one additional test to complete the coverage of the buildImage method in
    # LSST_ImageBuilder.
    config['image']['world_pos'] = config['image']['image_pos'] = config['stamp'].pop('image_pos')
    with np.testing.assert_raises(galsim.GalSimConfigError):
        galsim.config.BuildImage(config)



if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
