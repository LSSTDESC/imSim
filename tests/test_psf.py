"""
Unit tests for PSF-related code.
"""
import os
import glob
import unittest
import imsim
import galsim

class PsfTestCase(unittest.TestCase):
    """
    TestCase class for PSF-related functions.
    """
    def setUp(self):
        self.test_dir = 'psf_tests_dir'
        instcat = 'tiny_instcat.txt'
        self.obs_md = imsim.OpsimMetaDict(instcat)
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
                    'boresight': {
                        'type': 'RADec',
                        'ra': { 'type': 'Degrees', 'theta': self.obs_md['rightascension'], },
                        'dec': { 'type': 'Degrees', 'theta': self.obs_md['declination'], }
                    }
                }
            }
        }
        galsim.config.ProcessInput(config)
        for psf_name in ("DoubleGaussian", "Kolmogorov", "Atmospheric"):
            psf = galsim.config.BuildGSObject(config, psf_name)
            psf_file = os.path.join(self.test_dir, '{}.pkl'.format(psf_name))
            imsim.save_psf(psf, psf_file)
            psf_retrieved = imsim.load_psf(psf_file)
            self.assertEqual(psf, psf_retrieved)

    def test_atm_psf_config(self):
        """
        Test that the psf delivered by make_psf correctly applies the
        config file value for gaussianFWHM as input to the AtmosphericPSF.
        """
        config = {
            'psf': {
                'type': 'Convolve',
                'items': [
                    { 'type': 'AtmosphericPSF',
                      'screen_scale': 6.4,
                    },
                    { 'type': 'Gaussian', 'fwhm': 0.3 },
                ],
            },
            'input': {
                'atm_psf': {
                    'airmass': self.obs_md['airmass'],
                    'rawSeeing': self.obs_md['rawSeeing'],
                    'band':  self.obs_md['band'],
                    'boresight': {
                        'type': 'RADec',
                        'ra': { 'type': 'Degrees', 'theta': self.obs_md['rightascension'], },
                        'dec': { 'type': 'Degrees', 'theta': self.obs_md['declination'], }
                    }
                }
            }
        }
        galsim.config.ProcessInput(config)
        config1 = galsin.config.CopyConfig(config)
        # PSF with gaussianFWHM explicitly set to zero.
        config1['psf']['items'][1]['fwhm'] = 0.
        psf_0 = galsim.config.BuildGSObject(config1, 'psf')

        # PSF with gaussianFWHM explicitly set to config file value.
        psf_c = galsim.config.BuildGSObject(config, 'psf')

        # PSF made manually convolving Atm with Gaussian.
        psf_a = galsim.config.BuildGSObject(config['psf']['items'], 0, config)
        psf = galsim.Convolve(psf_a, galsim.Gaussian(fwhm=0.3))

        self.assertEqual(psf, psf_c)
        self.assertNotEqual(psf, psf_0)


if __name__ == '__main__':
    unittest.main()
