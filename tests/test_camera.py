"""
Tests of the Camera class.
"""
import os
import unittest
import json
from pathlib import Path
import imsim


DATA_DIR = str(Path(__file__).parent.parent / 'data')


class CameraTestCase(unittest.TestCase):
    """TestCase class for the Camera class."""
    def test_bias_levels(self):
        # Check that per-amp bias levels can be set from the json file.
        bias_levels_file = 'LSSTCam_bias_levels_run_13421.json'

        with open(os.path.join(DATA_DIR, bias_levels_file)) as fobj:
            bias_levels = json.load(fobj)

        camera = imsim.Camera('LsstCamSim', bias_levels_file=bias_levels_file)
        for det_name, ccd in camera.items():
            for amp_name, amp in ccd.items():
                self.assertEqual(bias_levels[det_name][amp_name],
                                 amp.bias_level)

        # Check that a full path to the bias_levels_file can be used.
        bias_levels_file = os.path.join(DATA_DIR,
                                        'LSSTCam_bias_levels_run_13421.json')
        camera = imsim.Camera('LsstCamSim', bias_levels_file=bias_levels_file)

        for det_name, ccd in camera.items():
            for amp_name, amp in ccd.items():
                self.assertEqual(bias_levels[det_name][amp_name],
                                 amp.bias_level)

        # Check that a single bias level for all amps can be set.
        bias_level = 1234.
        camera = imsim.Camera('LsstCamSim', bias_level=bias_level)
        for det_name, ccd in camera.items():
            for amp_name, amp in ccd.items():
                self.assertEqual(bias_level, amp.bias_level)


if __name__ == '__main__':
    unittest.main()
