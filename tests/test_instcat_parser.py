import os
import unittest
import desc.imsim

class InstanceCatalogParserTestCase(unittest.TestCase):
    def setUp(self):
        self.command_file = os.path.join(os.environ['IMSIM_DIR'],
                                         'tests', 'tiny_instcat.txt')

    def tearDown(self):
        pass

    def test_parsePhoSimInstanceFile(self):
        instcat_contents = \
            desc.imsim.parsePhoSimInstanceFile(self.command_file, 30)
        # Test a handful of values directly:
        self.assertEqual(instcat_contents.commands['obshistid'][0], 161899)
        self.assertEqual(instcat_contents.commands['filter'][0], 2)
        self.assertAlmostEqual(instcat_contents.commands['altitude'][0],
                               43.6990272)
        self.assertAlmostEqual(instcat_contents.commands['vistime'][0], 33.)
        self.assertEqual(len(instcat_contents.objects), 9)

        self.assertRaises(desc.imsim.PhosimInstanceCatalogParseError,
                          desc.imsim.parsePhoSimInstanceFile,
                          self.command_file, 10)

if __name__ == '__main__':
    unittest.main()
