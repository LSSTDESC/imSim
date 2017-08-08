"""
Unit tests for instance catalog parsing code.
"""
from __future__ import absolute_import, print_function
import os
import unittest
import warnings
import numpy as np
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
            for line in open(self.command_file).readlines()[:23]:
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

        star = instcat_contents.objects.query("uniqueId==1046817878020").iloc[0]
        self.assertEqual(star['galSimType'], 'pointSource')
        self.assertAlmostEqual(star['raJ2000'], 31.2400746)
        self.assertAlmostEqual(star['decJ2000'], -10.09365)
        self.assertEqual(star['sedFilepath'], 'starSED/phoSimMLT/lte033-4.5-1.0a+0.4.BT-Settl.spec.gz')

        galaxy = instcat_contents.objects.query("uniqueId==34308924793883").iloc[0]
        self.assertEqual(galaxy['galSimType'], 'sersic')
        self.assertAlmostEqual(galaxy['positionAngle'], 2.77863669*np.pi/180.)
        self.assertEqual(galaxy['sindex'], 1)

        self.assertRaises(desc.imsim.PhosimInstanceCatalogParseError,
                          desc.imsim.parsePhoSimInstanceFile,
                          self.command_file, 10)

    def test_extinction_parsing(self):
        "Test the parsing of the extinction parameters."
        instcat_contents = desc.imsim.parsePhoSimInstanceFile(self.command_file)
        star = instcat_contents.objects.query("uniqueId==1046817878020").iloc[0]
        self.assertEqual(star['internalAv'], 0)
        self.assertEqual(star['internalRv'], 0)
        self.assertAlmostEqual(star['galacticAv'], 0.0635117705)
        self.assertAlmostEqual(star['galacticRv'], 3.1)

        star = instcat_contents.objects.query("uniqueId==956090372100").iloc[0]
        self.assertAlmostEqual(star['internalAv'], 0.0651282621)
        self.assertAlmostEqual(star['internalRv'], 3.1)
        self.assertAlmostEqual(star['galacticAv'], 0.0651282621)
        self.assertAlmostEqual(star['galacticRv'], 3.1)

        star = instcat_contents.objects.query("uniqueId==956090392580").iloc[0]
        self.assertAlmostEqual(star['internalAv'], 0.0639515271)
        self.assertAlmostEqual(star['internalRv'], 3.1)
        self.assertEqual(star['galacticAv'], 0)
        self.assertEqual(star['galacticRv'], 0)

        star = instcat_contents.objects.query("uniqueId==811883374596").iloc[0]
        self.assertEqual(star['internalAv'], 0)
        self.assertEqual(star['internalRv'], 0)
        self.assertEqual(star['galacticAv'], 0)
        self.assertEqual(star['galacticRv'], 0)

        galaxy = instcat_contents.objects.query("uniqueId==34308924793883").iloc[0]
        self.assertAlmostEqual(galaxy['internalAv'], 0.100000001)
        self.assertAlmostEqual(galaxy['internalRv'], 3.0999999)
        self.assertAlmostEqual(galaxy['galacticAv'], 0.0594432589)
        self.assertAlmostEqual(galaxy['galacticRv'], 3.1)

        galaxy = instcat_contents.objects.query("uniqueId==34314197354523").iloc[0]
        self.assertEqual(galaxy['internalAv'], 0)
        self.assertEqual(galaxy['internalRv'], 0)
        self.assertAlmostEqual(galaxy['galacticAv'], 0.0595998126)
        self.assertAlmostEqual(galaxy['galacticRv'], 3.1)

        galaxy = instcat_contents.objects.query("uniqueId==34307989098523").iloc[0]
        self.assertAlmostEqual(galaxy['internalAv'], 0.300000012)
        self.assertAlmostEqual(galaxy['internalRv'], 3.0999999)
        self.assertEqual(galaxy['galacticAv'], 0)
        self.assertEqual(galaxy['galacticRv'], 0)

        galaxy = instcat_contents.objects.query("uniqueId==34307989098524").iloc[0]
        self.assertEqual(galaxy['internalAv'], 0)
        self.assertEqual(galaxy['internalRv'], 0)
        self.assertEqual(galaxy['galacticAv'], 0)
        self.assertEqual(galaxy['galacticRv'], 0)

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
<<<<<<< HEAD
        self.assertAlmostEqual(phot_params.exptime, 33/2.)
=======
        self.assertAlmostEqual(phot_params.exptime, 15.)
>>>>>>> master
        self.assertEqual(phot_params.readnoise, 0)
        self.assertEqual(phot_params.darkcurrent, 0)

    def test_validate_phosim_object_list(self):
        "Test the validation of the rows of the phoSimObjects DataFrame."
        instcat_contents = desc.imsim.parsePhoSimInstanceFile(self.command_file)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '\nOmitted', UserWarning)
            accepted, rejected = \
                desc.imsim.validate_phosim_object_list(instcat_contents.objects)
        self.assertEqual(len(rejected), 5)
        self.assertEqual(len(accepted), 16)
        self.assertEqual(len(rejected.query("minorAxis > majorAxis")), 1)
        self.assertEqual(len(rejected.query("magNorm > 50")), 1)
        self.assertEqual(len(rejected.query("galacticAv==0 and galacticRv==0")), 4)


if __name__ == '__main__':
    unittest.main()
