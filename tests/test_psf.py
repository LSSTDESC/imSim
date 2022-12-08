"""
Unit tests for PSF-related code.
"""
import os
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
        galsim.config.ProcessInput(config)
        config['wcs'] = galsim.config.BuildWCS(config['image'], 'wcs', config)

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


if __name__ == '__main__':
    unittest.main()
