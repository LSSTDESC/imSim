"""
Unit tests for PSF-related code.
"""
import os
import numpy as np
from pathlib import Path
import glob
import unittest
import time
import imsim
import galsim

DATA_DIR = Path(__file__).parent / 'data'


class PsfTestCase(unittest.TestCase):
    """
    TestCase class for PSF-related functions.
    """
    def setUp(self):
        self.test_dir = 'psf_tests_dir'
        instcat = DATA_DIR / 'tiny_instcat.txt'
        self.obs_md = imsim.OpsimMetaDict(str(instcat))
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def tearDown(self):
        for item in glob.glob(os.path.join(self.test_dir, '*')):
            os.remove(item)
        os.rmdir(self.test_dir)

    def test_save_and_load_psf(self):
        """
        Test that the different imSim PSFs are saved and retrieved
        correctly.
        """
        config = {
            'DoubleGaussian': {
                'type': 'DoubleGaussianPSF',
                'fwhm': self.obs_md['FWHMgeom'],
                'pixel_scale': 0.2,
            },
            'Kolmogorov': {
                'type': 'KolmogorovPSF',
                'airmass': self.obs_md['airmass'],
                'rawSeeing': self.obs_md['rawSeeing'],
                'band': self.obs_md['band'],
            },
            'Atmospheric': {
                'type': 'Convolve',
                'items': [
                    { 'type': 'AtmosphericPSF' },
                    { 'type': 'Gaussian', 'fwhm': 0.3 },
                ],
            },
            'input': {
                'atm_psf': {
                    'airmass': self.obs_md['airmass'],
                    'rawSeeing': self.obs_md['rawSeeing'],
                    'band':  self.obs_md['band'],
                    'screen_scale': 6.4,
                    'boresight': {
                        'type': 'RADec',
                        'ra': { 'type': 'Degrees', 'theta': self.obs_md['rightascension'], },
                        'dec': { 'type': 'Degrees', 'theta': self.obs_md['declination'], }
                    }
                }
            },
            'image_pos': galsim.PositionD(0,0),  # This would get set appropriately during
                                                 # normal config processing.
            'image' : {
                'random_seed': 1234,
                'wcs': {
                    'type' : 'Tan',
                    'dudx' : 0.2,
                    'dudy' : 0.,
                    'dvdx' : 0.,
                    'dvdy' : 0.2,
                    'ra' : '@input.atm_psf.boresight.ra',
                    'dec' : '@input.atm_psf.boresight.dec',
                }
            }
        }
        galsim.config.ProcessInput(config)
        config['wcs'] = galsim.config.BuildWCS(config['image'], 'wcs', config)
        for psf_name in ("DoubleGaussian", "Kolmogorov", "Atmospheric"):
            psf = galsim.config.BuildGSObject(config, psf_name)
            psf_file = os.path.join(self.test_dir, '{}.pkl'.format(psf_name))
            imsim.save_psf(psf, psf_file)
            psf_retrieved = imsim.load_psf(psf_file)
            self.assertEqual(psf, psf_retrieved)

    def test_atm_psf_save_file(self):
        """Test using save_file in  AtmosphericPSF
        """
        psf_file = os.path.join(self.test_dir, 'save_atm_psf.pkl')
        config = {
            'psf': {
                'type': 'AtmosphericPSF'
            },
            'input': {
                'atm_psf': {
                    'airmass': self.obs_md['airmass'],
                    'rawSeeing': self.obs_md['rawSeeing'],
                    'band':  self.obs_md['band'],
                    'screen_scale': 6.4,
                    'boresight': {
                        'type': 'RADec',
                        'ra': { 'type': 'Degrees', 'theta': self.obs_md['rightascension'], },
                        'dec': { 'type': 'Degrees', 'theta': self.obs_md['declination'], }
                    },
                    'save_file': psf_file
                }
            },
            'image_pos': galsim.PositionD(0,0),  # This would get set appropriately during
                                                 # normal config processing.
            'image' : {
                'random_seed': 1234,
                'wcs': {
                    'type' : 'Tan',
                    'dudx' : 0.2,
                    'dudy' : 0.,
                    'dvdx' : 0.,
                    'dvdy' : 0.2,
                    'ra' : '@input.atm_psf.boresight.ra',
                    'dec' : '@input.atm_psf.boresight.dec',
                }
            }
        }

        if os.path.isfile(psf_file):
            os.remove(psf_file)

        config['wcs'] = galsim.config.BuildWCS(config['image'], 'wcs', config)
        config1 = galsim.config.CopyConfig(config)
        config2 = galsim.config.CopyConfig(config)

        # The first time, it will build the psf from scratch and save the screens.
        t0 = time.time()
        galsim.config.ProcessInput(config1)
        t1 = time.time()

        assert os.path.isfile(psf_file)

        # The second time, it will be faster, since it loads the screens from the file.
        t2 = time.time()
        galsim.config.ProcessInput(config2)
        t3 = time.time()

        print('Times = ',t1-t0,t3-t2)
        assert t1-t0 > t3-t2

        # Both input objects will make the same PSF at the same location:
        psf1 = galsim.config.BuildGSObject(config1, 'psf')[0]
        psf2 = galsim.config.BuildGSObject(config2, 'psf')[0]
        assert psf1 == psf2


    def test_atm_psf_config(self):
        """
        Test that the psf delivered by make_psf correctly applies the
        config file value for gaussianFWHM as input to the AtmosphericPSF.
        """
        config = {
            'psf': {
                'type': 'Convolve',
                'items': [
                    { 'type': 'AtmosphericPSF' },
                    { 'type': 'Gaussian', 'fwhm': 0.3 },
                ],
            },
            'input': {
                'atm_psf': {
                    'airmass': self.obs_md['airmass'],
                    'rawSeeing': self.obs_md['rawSeeing'],
                    'band':  self.obs_md['band'],
                    'screen_scale': 6.4,
                    'boresight': {
                        'type': 'RADec',
                        'ra': { 'type': 'Degrees', 'theta': self.obs_md['rightascension'], },
                        'dec': { 'type': 'Degrees', 'theta': self.obs_md['declination'], }
                    }
                }
            },
            'image_pos': galsim.PositionD(0,0),  # This would get set appropriately during
                                                 # normal config processing.
            'image' : {
                'random_seed': 1234,
                'wcs': {
                    'type' : 'Tan',
                    'dudx' : 0.2,
                    'dudy' : 0.,
                    'dvdx' : 0.,
                    'dvdy' : 0.2,
                    'ra' : '@input.atm_psf.boresight.ra',
                    'dec' : '@input.atm_psf.boresight.dec',
                }
            }
        }
        config['wcs'] = galsim.config.BuildWCS(config['image'], 'wcs', config)
        config2 = galsim.config.CopyConfig(config)

        galsim.config.ProcessInput(config)

        # PSF without gaussian
        config1 = galsim.config.CopyConfig(config)
        del config1['psf']['items'][1]
        psf_0 = galsim.config.BuildGSObject(config1, 'psf')[0]

        # PSF with gaussian
        psf_c = galsim.config.BuildGSObject(config, 'psf')[0]

        # PSF made manually convolving Atm with Gaussian.
        psf_a = galsim.config.BuildGSObject(config['psf']['items'], 0, config)[0]
        psf = galsim.Convolve(psf_a, galsim.Gaussian(fwhm=0.3))

        self.assertEqual(psf, psf_c)
        self.assertNotEqual(psf, psf_0)

        # random_seed is required
        del config2['image']['random_seed']
        with np.testing.assert_raises(RuntimeError):
            galsim.config.ProcessInput(config2)

        # random_seed can be a list
        # And the first item can be something that needs to be evaluated.
        config2['image']['random_seed'] = [
            '$run_id',
            {'type': 'Sequence', 'first': '$run_id', 'repeat': 189}
        ]
        config2['eval_variables'] = {'irun_id': 1234}
        galsim.config.ProcessInput(config2)
        psf_d = galsim.config.BuildGSObject(config, 'psf')[0]
        assert psf == psf_d


    def test_r0_500(self):
        """Test that inversion of the Tokovinin fitting formula for r0_500 works."""
        np.random.seed(57721)
        for _ in range(100):
            airmass = np.random.uniform(1.001, 1.5)
            rawSeeing = np.random.uniform(0.5, 1.5)
            band = 'ugrizy'[np.random.randint(6)]
            boresight = galsim.CelestialCoord(0 * galsim.radians, 0 * galsim.radians)
            rng = galsim.BaseDeviate(np.random.randint(2**32))
            atmPSF = imsim.AtmosphericPSF(airmass, rawSeeing, band, boresight, rng, screen_size=6.4)

            wlen = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
            targetFWHM = rawSeeing * airmass**0.6 * (wlen/500)**(-0.3)

            r0_500 = atmPSF.atm.r0_500_effective
            L0 = atmPSF.atm[0].L0
            vkFWHM = imsim.AtmosphericPSF._vkSeeing(r0_500, wlen, L0)

            np.testing.assert_allclose(targetFWHM, vkFWHM, atol=1e-3, rtol=0)


if __name__ == '__main__':
    unittest.main()
