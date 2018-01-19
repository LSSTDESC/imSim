"""
Unit tests for skyModel code.
"""
from __future__ import absolute_import
import os
import unittest
try:
    import configparser
except ImportError:
    # python 2 backwards-compatibility
    import ConfigParser as configparser
import galsim
import desc.imsim
import lsst.sims.skybrightness as skybrightness
import numpy as np


class SkyModelTestCase(unittest.TestCase):
    """
    TestCase class for skyModel module code.
    """

    def setUp(self):
        self.test_config_file = 'test_config.txt'
        self.zp_u = 0.28259860576333784
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

        skymodel = desc.imsim.ESOSkyModel(obs_md, addNoise=False,
                                          addBackground=True)
        image_2 = galsim.Image(100, 100)
        image_2 = skymodel.addNoiseAndBackground(
            image_2, photParams=photPars_2)

        image_1 = galsim.Image(100, 100)
        image_1 = skymodel.addNoiseAndBackground(
            image_1, photParams=photPars_1)

        self.assertNotEqual(image_1.array[0, 0], 0)
        self.assertAlmostEqual(2 * image_1.array[0, 0], image_2.array[0, 0])

    def test_skycounts_function(self):
        """
        Test that the sky counts per sec function gives the right result for the previously
        calculated zero points. (Defined as the number of electrons per second for
        a 24 magnitude source, the default for skyCountsperSec will be set to u band and 24 mag.)
        """

        desc.imsim.read_config()
        instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                                    'tiny_instcat.txt')
        commands, objects = desc.imsim.parsePhoSimInstanceFile(instcat_file)
        obs_md = desc.imsim.phosim_obs_metadata(commands)
        photPars_2 = desc.imsim.photometricParameters(commands)
        skyModel = skybrightness.SkyModel(mags=False)
        skyModel.setRaDecMjd(0., 90., 58000, azAlt=True, degrees=True)
        skycounts_persec_u = desc.imsim.skyCountsPerSec(
            skyModel, photPars_2, 'u', 24)
        self.assertAlmostEqual(skycounts_persec_u.value, self.zp_u)


if __name__ == '__main__':
    unittest.main()
