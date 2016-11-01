"""
Test the setting of the logging configuration.
"""
from __future__ import absolute_import, print_function
import unittest
import desc.imsim

class LoggingTestCase(unittest.TestCase):
    "TestCase class for logging."

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_logging_config(self):
        "Test logging level configuration."
        for level, log_level in zip(list(range(10, 60, 10)),
                                    "DEBUG INFO WARN ERROR CRITICAL".split()):
            logger = desc.imsim.get_logger(log_level)
            self.assertEqual(logger.level, level)

if __name__ == '__main__':
    unittest.main()
