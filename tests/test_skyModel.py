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
import lsst.sims.skybrightness as skybrightness
from lsst.sims.photUtils import BandpassDict


class SkyModelTestCase(unittest.TestCase):
    """
    TestCase class for skyModel module code.
    """

    def setUp(self):
        self.test_config_file = 'test_config.txt'
        self.zp_u = 0.36626526294988776
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
        self.assertLess(abs(nphot_2 - 760), 20)

    def test_nexp_scaling(self):
        """
        Test that the sky background level is proportional to nexp*exptime
        in the combined image for multiple nsnaps.
        """
        desc.imsim.read_config()
        instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                                    'tiny_instcat.txt')
        commands = desc.imsim.metadata_from_file(instcat_file)
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
        skymodel1 = desc.imsim.make_sky_model(obs_md, photPars_1, seed=seed,
                                              addNoise=False,
                                              addBackground=True,
                                              apply_sensor_model=False)
        skymodel2 = desc.imsim.make_sky_model(obs_md, photPars_2, seed=seed,
                                              addNoise=False,
                                              addBackground=True,
                                              apply_sensor_model=False)

        # Check fast background model, i.e., with sensor effects turned off.
        t0 = time.time()
        image_2 = galsim.Image(nx, ny)
        image_2 = skymodel2.addNoiseAndBackground(image_2,
                                                  photParams=photPars_2)
        image_1 = galsim.Image(nx, ny)
        image_1 = skymodel1.addNoiseAndBackground(image_1,
                                                  photParams=photPars_1)
        dt_fast = time.time() - t0
        self._apply_sky_background_tests(image_1, image_2)

        # Test with default background including sensor model and
        # bundled photons.
        skymodel1.fast_background = False
        skymodel2.fast_background = False
        t0 = time.time()
        image_2 = galsim.Image(nx, ny)
        image_2 = skymodel2.addNoiseAndBackground(image_2,
                                                  photParams=photPars_2)
        image_1 = galsim.Image(nx, ny)
        image_1 = skymodel1.addNoiseAndBackground(image_1,
                                                  photParams=photPars_1)
        dt_bundled = time.time() - t0
        self._apply_sky_background_tests(image_1, image_2)

        # Test with unbundled photons.
        skymodel_unbundled1 \
            = desc.imsim.make_sky_model(obs_md, photPars_1, seed=seed,
                                        addNoise=False, addBackground=True,
                                        apply_sensor_model=True,
                                        bundles_per_pix=1)
        skymodel_unbundled2 \
            = desc.imsim.make_sky_model(obs_md, photPars_2, seed=seed,
                                        addNoise=False, addBackground=True,
                                        apply_sensor_model=True,
                                        bundles_per_pix=1)
        t0 = time.time()
        image_2 = galsim.Image(nx, ny)
        image_2 = skymodel_unbundled2.addNoiseAndBackground(image_2)
        image_1 = galsim.Image(nx, ny)
        image_1 = skymodel_unbundled1.addNoiseAndBackground(image_1)
        dt_unbundled = time.time() - t0
        self._apply_sky_background_tests(image_1, image_2)


    def test_skycounts_function(self):
        """
        Test that the SkyCountsPerSec class gives the right result for the previously
        calculated zero points. (This is defined as the number of counts per second for
        a 24 magnitude source.)  Here we set magNorm=24 to calculate the zero points
        but when calculating the sky background from the sky brightness
        model magNorm=None as above.
        """
        desc.imsim.read_config()
        instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                                    'tiny_instcat.txt')
        obs_md, photPars, sources = desc.imsim.parsePhoSimInstanceFile(instcat_file)
        skyModel = skybrightness.SkyModel(mags=False)
        skyModel.setRaDecMjd(0., 90., 58000, azAlt=True, degrees=True)

        bandPassdic = BandpassDict.loadTotalBandpassesFromFiles(['u','g','r','i','z','y'])
        skycounts_persec = desc.imsim.skyModel.SkyCountsPerSec(skyModel, photPars, bandPassdic)

        skycounts_persec_u = skycounts_persec('u', 24)
        self.assertAlmostEqual(skycounts_persec_u.value, self.zp_u)

if __name__ == '__main__':
    unittest.main()
