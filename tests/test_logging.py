"""
Test the setting of the logging configuration.
"""
import unittest
import imsim

# XXX: I think this is probably not needed anymore.  Now use GalSim logger.

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
            logger = imsim.get_logger(log_level)
            self.assertEqual(logger.level, level)


if __name__ == '__main__':
    unittest.main()
