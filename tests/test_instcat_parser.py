"""
Unit tests for instance catalog parsing code.
"""
from __future__ import absolute_import, print_function
import os
import unittest
import desc.imsim

class InstanceCatalogParserTestCase(unittest.TestCase):
    """
    TestCase class for instance catalog parsing code.
    """
    def setUp(self):
        self.command_file = os.path.join(os.environ['IMSIM_DIR'],
                                         'tests', 'tiny_instcat.txt')

    def tearDown(self):
        pass

    def test_parsePhoSimInstanceFile(self):
        "Test code for parsePhoSimInstanceFile."
        instcat_contents = \
            desc.imsim.parsePhoSimInstanceFile(self.command_file, 40)
        # Test a handful of values directly:
        self.assertEqual(instcat_contents.commands['obshistid'][0], 161899)
        self.assertEqual(instcat_contents.commands['filter'][0], 2)
        self.assertAlmostEqual(instcat_contents.commands['altitude'][0],
                               43.6990272)
        self.assertAlmostEqual(instcat_contents.commands['vistime'][0], 33.)

        self.assertEqual(len(instcat_contents.objects), 19)

        star = \
            instcat_contents.objects.query("galSimType=='pointSource'").iloc[0]
        self.assertEqual(star['objectID'], 1046817878020)
        self.assertAlmostEqual(star['ra'], 31.2400746)
        self.assertAlmostEqual(star['dec'], -10.09365)

        galaxy = instcat_contents.objects.query("galSimType=='sersic'").iloc[0]
        self.assertEqual(galaxy['objectID'], 34308924793883)
        self.assertAlmostEqual(galaxy['positionAngle'], 2.77863669)
        self.assertEqual(galaxy['sersicIndex'], 1)

        self.assertRaises(desc.imsim.PhosimInstanceCatalogParseError,
                          desc.imsim.parsePhoSimInstanceFile,
                          self.command_file, 10)

if __name__ == '__main__':
    unittest.main()
