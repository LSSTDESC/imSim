"""
Unit tests for skyModel code.
"""
from __future__ import absolute_import
import os
import unittest
from collections import namedtuple
try:
    import configparser
except ImportError:
    # python 2 backwards-compatibility
    import ConfigParser as configparser
import numpy.random as random
import galsim
import lsst.sims.skybrightness as skybrightness
from lsst.sims.photUtils import BandpassDict
import desc.imsim

def detector(chip_name):
    "Proxy for a GalSimDetector object."
    Detector = namedtuple('Detector', ['name'])
    return Detector(chip_name)

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

    def test_nexp_scaling(self):
        """
        Test that the sky background level is proportional to nexp*exptime
        in the combined image for multiple nsnaps.
        """
        desc.imsim.read_config()
        instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests', 'data',
                                    'phosim_stars.txt')
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

        skymodel = desc.imsim.ESOSkyModel(obs_md, addNoise=False,
                                          addBackground=True)
        image_2 = galsim.Image(100, 100)
        image_2 = skymodel.addNoiseAndBackground(image_2, photParams=photPars_2,
                                                 detector=detector('R:4,2 S:1,0'))
        image_1 = galsim.Image(100, 100)
        image_1 = skymodel.addNoiseAndBackground(image_1, photParams=photPars_1,
                                                 detector=detector('R:4,2 S:1,0'))

        self.assertNotEqual(image_1.array[0, 0], 0)
        self.assertAlmostEqual(2*image_1.array[0, 0], image_2.array[0, 0])

    def test_sky_variation(self):
        """
        Test that the sky background varies over the focal plane.
        """
        instcat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests', 'data',
                                    'phosim_stars.txt')
        obs_md, phot_params, _ \
            = desc.imsim.parsePhoSimInstanceFile(instcat_file)
        skymodel = desc.imsim.ESOSkyModel(obs_md, addNoise=False,
                                          addBackground=True)
        camera = desc.imsim.get_obs_lsstSim_camera()
        chip_names = random.choice([chip.getName() for chip in camera],
                                   size=10, replace=False)
        sky_bg_values = set()
        for chip_name in chip_names:
            image = galsim.Image(1, 1)
            skymodel.addNoiseAndBackground(image, photParams=phot_params,
                                           detector=detector(chip_name))
            sky_bg_values.add(image.array[0][0])
        self.assertEqual(len(sky_bg_values), len(chip_names))

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
        _, phot_params, _ = desc.imsim.parsePhoSimInstanceFile(instcat_file)
        skyModel = skybrightness.SkyModel(mags=False)
        skyModel.setRaDecMjd(0., 90., 58000, azAlt=True, degrees=True)

        bandPassdic = BandpassDict.loadTotalBandpassesFromFiles(['u', 'g', 'r', 'i', 'z', 'y'])
        skycounts_persec = desc.imsim.skyModel.SkyCountsPerSec(skyModel, phot_params, bandPassdic)

        skycounts_persec_u = skycounts_persec('u', 24)
        self.assertAlmostEqual(skycounts_persec_u.value, self.zp_u)


if __name__ == '__main__':
    unittest.main()
