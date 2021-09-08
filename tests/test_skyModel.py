"""
Unit tests for skyModel code.
"""
import os
import copy
import unittest
import pickle
import configparser
import numpy as np
import numpy.random as random
import galsim
from desc.imsim.sims_GalSimInterface import make_galsim_detector, LSSTCameraWrapper
import desc.imsim


class DC2SkyModel:
    """
    Minimal SkyModel class to provide the sky SED evaluated at
    skyModel.setRaDecMjd(0., 90., 58000, azAlt=True, degrees=True)
    using data from the DC2Production sims_skybrightness_data tag.
    """
    def __init__(self):
        # Read the SED associated with the persisted DC2Production
        # dataset from the pickle file.
        with open(os.path.join(os.environ['IMSIM_DIR'], 'data',
                               'sky_model_sed_DC2Production.pkl'), 'rb') as fd:
            self.wave, self.spec = pickle.load(fd)

    def returnWaveSpec(self):
        """Return the SED for use by the sky_counts_per_sec function."""
        return self.wave, self.spec


class SkyModelTestCase(unittest.TestCase):
    """
    TestCase class for skyModel module code.
    """

    def setUp(self):
        desc.imsim.read_config()
        instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                                    'tiny_instcat.txt')
        self.commands = desc.imsim.metadata_from_file(instcat_file)
        self.obs_md = desc.imsim.phosim_obs_metadata(self.commands)
        self.phot_params = desc.imsim.photometricParameters(self.commands)

        self.camera_wrapper = LSSTCameraWrapper()

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

    def detector(self, chip_name="R:2,2 S:1,1"):
        """
        Factory to create a GalSimDetector instance.
        """
        return make_galsim_detector(self.camera_wrapper, chip_name,
                                    self.phot_params, self.obs_md)

    def _apply_sky_background_tests(self, image_1, image_2):
        self.assertNotEqual(image_1.array[0, 0], 0)
        nphot_1 = np.mean(image_1.array.ravel())
        nphot_2 = np.mean(image_2.array.ravel())
        self.assertLess(abs(2.*nphot_1 - nphot_2), 5)

    def test_nexp_scaling(self):
        """
        Test that the sky background level is proportional to nexp*exptime
        in the combined image for multiple nsnaps.
        """
        commands = copy.deepcopy(self.commands)
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

        # Check fast background model, i.e., with sensor effects turned off.
        skymodel1 = desc.imsim.make_sky_model(self.obs_md, photPars_1,
                                              seed=seed,
                                              addNoise=False,
                                              addBackground=True,
                                              apply_sensor_model=False)
        skymodel2 = desc.imsim.make_sky_model(self.obs_md, photPars_2,
                                              seed=seed,
                                              addNoise=False,
                                              addBackground=True,
                                              apply_sensor_model=False)

        image_2 = galsim.Image(nx, ny)
        image_2 = skymodel2.addNoiseAndBackground(image_2,
                                                  photParams=photPars_2,
                                                  detector=self.detector())
        image_1 = galsim.Image(nx, ny)
        image_1 = skymodel1.addNoiseAndBackground(image_1,
                                                  photParams=photPars_1,
                                                  detector=self.detector())
        self._apply_sky_background_tests(image_1, image_2)

        # Test with sensor effects turned on, using fast silicon model.
        skymodel1 = desc.imsim.make_sky_model(self.obs_md, photPars_1,
                                              seed=seed,
                                              apply_sensor_model=True,
                                              fast_silicon=True)
        skymodel2 = desc.imsim.make_sky_model(self.obs_md, photPars_2,
                                              seed=seed,
                                              apply_sensor_model=True,
                                              fast_silicon=True)

        image_2 = galsim.Image(nx, ny)
        image_2 = skymodel2.addNoiseAndBackground(image_2,
                                                  photParams=photPars_2,
                                                  detector=self.detector())
        image_1 = galsim.Image(nx, ny)
        image_1 = skymodel1.addNoiseAndBackground(image_1,
                                                  photParams=photPars_1,
                                                  detector=self.detector())
        self._apply_sky_background_tests(image_1, image_2)

    def test_sky_variation(self):
        """
        Test that the sky background varies over the focal plane.
        """
        instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests', 'data',
                                    'phosim_stars.txt')
        obs_md, phot_params, _ \
            = desc.imsim.parsePhoSimInstanceFile(instcat_file, ())
        skymodel = desc.imsim.ESOSkyModel(obs_md, phot_params, addNoise=False,
                                          addBackground=True)
        camera = desc.imsim.get_obs_lsstSim_camera()
        chip_names = random.choice([chip.getName() for chip in camera],
                                   size=10, replace=False)
        sky_bg_values = set()
        for chip_name in chip_names:
            image = galsim.Image(1, 1)
            skymodel.addNoiseAndBackground(image, photParams=phot_params,
                                           detector=self.detector(chip_name))
            sky_bg_values.add(image.array[0][0])
        self.assertEqual(len(sky_bg_values), len(chip_names))

    def test_skycounts_function(self):
        """
        Test that the SkyCountsPerSec class gives the right result for the
        previously calculated zero points. (This is defined as the
        number of counts per second for a 24 magnitude source.)  Here
        we set magNorm=24 to calculate the zero points but when
        calculating the sky background from the sky brightness model
        magNorm=None as above.
        """
        desc.imsim.read_config()
        instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                                    'tiny_instcat.txt')
        _, phot_params, _ = desc.imsim.parsePhoSimInstanceFile(instcat_file, ())
        with open(os.path.join(os.environ['IMSIM_DIR'], 'data',
                               'bp_dict_DC2Production.pkl'), 'rb') as fd:
            bp_dict = pickle.load(fd)
        skyModel = DC2SkyModel()

        bandpass = bp_dict['u']
        countrate = desc.imsim.skyModel.sky_counts_per_sec(skyModel, phot_params,
                                                           bandpass, magNorm=24)
        self.assertAlmostEqual(countrate, self.zp_u)

    def test_NullSkyModel(self):
        """Test that the NullSkyModel adds zero photons."""
        commands = copy.deepcopy(self.commands)
        photParams = desc.imsim.photometricParameters(commands)
        null_sky_model = desc.imsim.make_sky_model(self.obs_md, photParams,
                                                   disable_sky_model=True)
        nx, ny = 30, 30
        image = galsim.Image(nx, ny)
        image = null_sky_model.addNoiseAndBackground(image,
                                                     photParams=photParams,
                                                     detector=self.detector())
        self.assertEqual(0, np.sum(image.array.ravel()))

if __name__ == '__main__':
    unittest.main()
