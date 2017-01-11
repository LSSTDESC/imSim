"""
Unit tests for skyModel code.
"""
from __future__ import absolute_import
import os
import unittest
import ConfigParser
import desc.imsim


class SkyModelTestCase(unittest.TestCase):
    """
    TestCase class for skyModel module code.
    """
    def setUp(self):
        self.test_config_file = 'test_config.txt'
        self.zp_u = 0.282598538804
        cp = ConfigParser.SafeConfigParser()
        cp.optionxform = str
        section = 'skyModel_params'
        cp.add_section(section)
        cp.set(section, 'B0', '24.')
        cp.set(section, 'u', str(self.zp_u))
        with open(self.test_config_file, 'wb') as output:
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

if __name__ == '__main__':
    unittest.main()
