"""
Unit tests for imSim configuration parameter code.
"""
from __future__ import print_function, absolute_import
import os
import unittest
try:
    import configparser
except ImportError:
    # python 2 backwards-compatibility
    import ConfigParser as configparser
import desc.imsim

class ImSimConfigurationTestCase(unittest.TestCase):
    """
    TestCase class for configuration parameter code.
    """
    def setUp(self):
        self.test_config_file = 'test_config.txt'
        cp = configparser.ConfigParser()
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
        self.assertAlmostEqual(config['electronics_readout']['readout_time'], 3.)
        self.assertEqual(config['persistence']['eimage_prefix'], 'lsst_e_')

        # Read a different config file and show that the previous
        # instance reflects the new configuration.
        desc.imsim.read_config(self.test_config_file)
        self.assertAlmostEqual(config['electronics_readout']['readout_time'], 2.)

    def test_get_config(self):
        "Test the get_config function."
        # Read the default config.
        desc.imsim.read_config()

        # Get an instance without re-reading the data.
        config = desc.imsim.get_config()
        self.assertAlmostEqual(config['electronics_readout']['readout_time'], 3.)
        self.assertEqual(config['persistence']['eimage_prefix'], 'lsst_e_')

if __name__ == '__main__':
    unittest.main()
