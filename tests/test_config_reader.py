"""
Unit tests for imSim configuration parameter code.
"""
from __future__ import print_function, absolute_import
import os
import unittest
import ConfigParser
import desc.imsim

class ImSimConfigurationTestCase(unittest.TestCase):
    """
    TestCase class for configuration parameter code.
    """
    def setUp(self):
        self.test_config_file = 'test_config.txt'
        cp = ConfigParser.SafeConfigParser()
        section = 'electronics_readout'
        cp.add_section(section)
        cp.set(section, 'readout_time', '2')
        with open(self.test_config_file, 'wb') as output:
            cp.write(output)

    def tearDown(self):
        try:
            os.remove(self.test_config_file)
        except OSError:
            pass

    def test_read_config(self):
        "Test the read_config function."
        # Read the default config.
        config = desc.imsim.read_config()
        self.assertAlmostEqual(config['readout_time'], 3.)
        self.assertEqual(config['eimage_prefix'], 'lsst_e')

        # Read a different config file and show that the previous
        # instance reflects the new configuration.
        desc.imsim.read_config(self.test_config_file)
        self.assertAlmostEqual(config['readout_time'], 2.)

if __name__ == '__main__':
    unittest.main()
