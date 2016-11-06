"""
Unit tests for instance catalog parsing code.
"""
from __future__ import absolute_import, print_function
import os
import unittest
import warnings
import desc.imsim

class InstanceCatalogParserTestCase(unittest.TestCase):
    """
    TestCase class for instance catalog parsing code.
    """
    @classmethod
    def setUpClass(cls):
        cls.config = desc.imsim.read_config()

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.command_file = os.path.join(os.environ['IMSIM_DIR'],
                                         'tests', 'tiny_instcat.txt')
        self.extra_commands = 'instcat_extra.txt'
        with open(self.extra_commands, 'w') as output:
            for line in open(self.command_file).readlines()[:20]:
                output.write(line)
            output.write('extra_command 1\n')

    def tearDown(self):
        os.remove(self.extra_commands)

    def test_parsePhoSimInstanceFile(self):
        "Test code for parsePhoSimInstanceFile."
        instcat_contents = desc.imsim.parsePhoSimInstanceFile(self.command_file)
        # Test a handful of values directly:
        self.assertEqual(instcat_contents.commands['obshistid'], 161899)
        self.assertEqual(instcat_contents.commands['filter'], 2)
        self.assertAlmostEqual(instcat_contents.commands['altitude'],
                               43.6990272)
        self.assertAlmostEqual(instcat_contents.commands['vistime'], 33.)
        self.assertAlmostEqual(instcat_contents.commands['bandpass'], 'r')

        self.assertEqual(len(instcat_contents.objects), 21)

        star = \
            instcat_contents.objects.query("galSimType=='pointSource'").iloc[0]
        self.assertEqual(star['objectID'], 1046817878020)
        self.assertAlmostEqual(star['ra'], 31.2400746)
        self.assertAlmostEqual(star['dec'], -10.09365)
        self.assertEqual(star['sedName'], 'starSED/phoSimMLT/lte033-4.5-1.0a+0.4.BT-Settl.spec.gz')

        galaxy = instcat_contents.objects.query("galSimType=='sersic'").iloc[0]
        self.assertEqual(galaxy['objectID'], 34308924793883)
        self.assertAlmostEqual(galaxy['positionAngle'], 2.77863669)
        self.assertEqual(galaxy['sersicIndex'], 1)

        self.assertRaises(desc.imsim.PhosimInstanceCatalogParseError,
                          desc.imsim.parsePhoSimInstanceFile,
                          self.command_file, 10)

    def test_parsePhoSimInstanceFile_warning(self):
        "Test the warnings emitted by the instance catalog parser."
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(UserWarning,
                              desc.imsim.parsePhoSimInstanceFile,
                              self.extra_commands)

    def test_photometricParameters(self):
        "Test the photometricParameters function."
        instcat_contents = \
            desc.imsim.parsePhoSimInstanceFile(self.command_file, 40)
        phot_params = \
            desc.imsim.photometricParameters(instcat_contents.commands)
        self.assertEqual(phot_params.gain, 1)
        self.assertEqual(phot_params.bandpass, 'r')
        self.assertEqual(phot_params.nexp, 2)
        self.assertAlmostEqual(phot_params.exptime, 15.)
        self.assertEqual(phot_params.readnoise, 0)
        self.assertEqual(phot_params.darkcurrent, 0)

    def test_validate_phosim_object_list(self):
        "Test the validation of the rows of the phoSimObjects DataFrame."
        instcat_contents = desc.imsim.parsePhoSimInstanceFile(self.command_file)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Omitted', UserWarning)
            accepted, rejected = \
                desc.imsim.validate_phosim_object_list(instcat_contents.objects)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(len(accepted), 19)
        self.assertEqual(len(rejected.query("halfLightSemiMinor > halfLightSemiMajor")), 1)
        self.assertEqual(len(rejected.query("magNorm>50")), 1)

if __name__ == '__main__':
    unittest.main()
