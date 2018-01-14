"""
Unit tests for skyModel code.
"""
from __future__ import absolute_import
import os
import unittest
import time
try:
    import configparser
except ImportError:
    # python 2 backwards-compatibility
    import ConfigParser as configparser
import numpy as np
import galsim
import desc.imsim


class SkyModelTestCase(unittest.TestCase):
    """
    TestCase class for skyModel module code.
    """
    def setUp(self):
        self.test_config_file = 'test_config.txt'
        self.zp_u = 0.282598538804
        cp = configparser.ConfigParser()
        cp.optionxform = str
        section = 'skyModel_params'
        cp.add_section(section)
        cp.set(section, 'B0', '24.')
        cp.set(section, 'u', str(self.zp_u))
        with open(self.test_config_file, 'w') as output:
            cp.write(output)

    def tearDown(self):
        try:
            os.remove(self.test_config_file)
        except OSError:
            pass

    def test_get_skyModel_params(self):
        "Test the get_skyModel_params function."
        desc.imsim.read_config(self.test_config_file)
        pars = desc.imsim.get_skyModel_params()
        self.assertAlmostEqual(pars['B0'], 24.)
        self.assertAlmostEqual(pars['u'], self.zp_u)


    def _apply_sky_background_tests(self, image_1, image_2):
        self.assertNotEqual(image_1.array[0, 0], 0)
        nphot_1 = np.mean(image_1.array.ravel())
        nphot_2 = np.mean(image_2.array.ravel())
        self.assertLess(abs(2.*nphot_1 - nphot_2), 5)
        self.assertLess(abs(nphot_2 - 850), 20)

    def test_nexp_scaling(self):
        """
        Test that the sky background level is proportional to nexp*exptime
        in the combined image for multiple nsnaps.
        """
        desc.imsim.read_config()
        instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                                    'tiny_instcat.txt')
        commands, objects = desc.imsim.parsePhoSimInstanceFile(instcat_file)
        obs_md = desc.imsim.phosim_obs_metadata(commands)
        photPars_2 = desc.imsim.photometricParameters(commands)
        self.assertEqual(photPars_2.nexp, 2)
        self.assertEqual(photPars_2.exptime, 15.)

        commands['nsnap'] = 1
        commands['vistime'] = 15.
        photPars_1 = desc.imsim.photometricParameters(commands)
        self.assertEqual(photPars_1.nexp, 1)
        self.assertEqual(photPars_1.exptime, 15.)

        seed = 100
        nx, ny = 30, 30
        # Making an ESOSkyModel is expensive, so do it once and set
        # the .fast_background attribute by hand.
        skymodel = desc.imsim.ESOSkyModel(obs_md, seed=seed, addNoise=False,
                                          addBackground=True, fast_background=True)

        # Check fast background model, i.e., with sensor effects turned off.
        t0 = time.time()
        image_2 = galsim.Image(nx, ny)
        image_2 = skymodel.addNoiseAndBackground(image_2, photParams=photPars_2)
        image_1 = galsim.Image(nx, ny)
        image_1 = skymodel.addNoiseAndBackground(image_1, photParams=photPars_1)
        dt_fast = time.time() - t0
        self._apply_sky_background_tests(image_1, image_2)

        # Test with default background including sensor model and
        # bundled photons.
        skymodel.fast_background = False
        t0 = time.time()
        image_2 = galsim.Image(nx, ny)
        image_2 = skymodel.addNoiseAndBackground(image_2, photParams=photPars_2)
        image_1 = galsim.Image(nx, ny)
        image_1 = skymodel.addNoiseAndBackground(image_1, photParams=photPars_1)
        dt_bundled = time.time() - t0
        self._apply_sky_background_tests(image_1, image_2)

        # Test with unbundled photons.
        skymodel_unbundled \
            = desc.imsim.ESOSkyModel(obs_md, seed=seed, addNoise=False,
                                     addBackground=True, bundles_per_pix=1)
        t0 = time.time()
        image_2 = galsim.Image(nx, ny)
        image_2 = skymodel.addNoiseAndBackground(image_2, photParams=photPars_2)
        image_1 = galsim.Image(nx, ny)
        image_1 = skymodel.addNoiseAndBackground(image_1, photParams=photPars_1)
        dt_unbundled = time.time() - t0
        self._apply_sky_background_tests(image_1, image_2)


if __name__ == '__main__':
    unittest.main()
