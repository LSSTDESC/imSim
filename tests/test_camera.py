"""
Tests of the Camera class.
"""
import os
import unittest
import json
from pathlib import Path
import imsim
from galsim import GalSimValueError


DATA_DIR = str(Path(__file__).parent.parent / 'data')


class CameraTestCase(unittest.TestCase):
    """TestCase class for the Camera class."""
    def test_bias_levels(self):
        # Check that per-amp bias levels can be set from the json file.
        bias_levels_file = 'LSSTCam_bias_levels_run_13421.json'

        with open(os.path.join(DATA_DIR, bias_levels_file)) as fobj:
            bias_levels = json.load(fobj)

        camera = imsim.Camera('LsstCam', bias_levels_file=bias_levels_file)
        for det_name, ccd in camera.items():
            for amp_name, amp in ccd.items():
                self.assertEqual(bias_levels[det_name][amp_name],
                                 amp.bias_level)

        # Check that a full path to the bias_levels_file can be used.
        bias_levels_file = os.path.join(DATA_DIR,
                                        'LSSTCam_bias_levels_run_13421.json')
        camera = imsim.Camera('LsstCam', bias_levels_file=bias_levels_file)

        for det_name, ccd in camera.items():
            for amp_name, amp in ccd.items():
                self.assertEqual(bias_levels[det_name][amp_name],
                                 amp.bias_level)

        # Check that a single bias level for all amps can be set.
        bias_level = 1234.
        camera = imsim.Camera('LsstCam', bias_level=bias_level)
        for det_name, ccd in camera.items():
            for amp_name, amp in ccd.items():
                self.assertEqual(bias_level, amp.bias_level)

    def check_adjacent_detectors_match(self, camera, det_name, expected_detectors):
        camera = imsim.Camera(camera)
        adjacent_detectors = camera.get_adjacent_detectors(det_name)
        self.assertCountEqual(expected_detectors, adjacent_detectors,
                             f"Adjacent detectors for {det_name} do not match expected values.")

    def test_get_adjacent_detectors_central(self):
        # Case of detectors adjacent to one central in a raft.
        det_name = 'R22_S11'
        expected_detectors = [
            'R22_S00', 'R22_S01', 'R22_S02',
            'R22_S10', 'R22_S11', 'R22_S12',
            'R22_S20', 'R22_S21', 'R22_S22',
        ]
        self.check_adjacent_detectors_match('LsstCam', det_name, expected_detectors)

    def test_get_adjacent_detectors_offset(self):
        # Case of detectors adjacent to one offset to a corner in a raft.
        # Top left corner.
        det_name = 'R22_S20'
        expected_detectors = [
            'R31_S02', 'R32_S00', 'R32_S01',
            'R21_S22', 'R22_S20', 'R22_S21',
            'R21_S12', 'R22_S10', 'R22_S11',
        ]
        self.check_adjacent_detectors_match('LsstCam', det_name, expected_detectors)
        # Bottom right corner.
        det_name = 'R22_S02'
        expected_detectors = [
            'R22_S11', 'R22_S12', 'R23_S10',
            'R22_S01', 'R22_S02', 'R23_S00',
            'R12_S21', 'R12_S22', 'R13_S20',
        ]
        self.check_adjacent_detectors_match('LsstCam', det_name, expected_detectors)

    def test_get_adjacent_detectors_edge(self):
        # Case of detectors adjacent to one on the edge of the camera.
        # Bottom edge.
        det_name = 'R02_S01'
        expected_detectors = [
            'R02_S10', 'R02_S11', 'R02_S12',
            'R02_S00', 'R02_S01', 'R02_S02',
        ]
        self.check_adjacent_detectors_match('LsstCam', det_name, expected_detectors)
        # Left edge.
        det_name = 'R20_S10'
        expected_detectors = [
            'R20_S20', 'R20_S21',
            'R20_S10', 'R20_S11',
            'R20_S00', 'R20_S01',
        ]
        # Top edge.
        det_name = 'R42_S21'
        expected_detectors = [
            'R42_S20', 'R42_S21', 'R42_S22',
            'R42_S10', 'R42_S11', 'R42_S12',
        ]
        # Right edge.
        det_name = 'R24_S12'
        expected_detectors = [
            'R24_S21', 'R24_S22',
            'R24_S11', 'R24_S12',
            'R24_S01', 'R24_S02',
        ]
        self.check_adjacent_detectors_match('LsstCam', det_name, expected_detectors)

    def test_get_adjacent_detectors_corner(self):
        # Case of detectors adjacent to one in a corner of the camera.
        # We don't expect wavefront or guider detectors to be returned.
        # On the side of a corner cutout.
        det_name = 'R01_S10'
        expected_detectors = [
            'R01_S20', 'R01_S21',
            'R01_S10', 'R01_S11',
            'R01_S00', 'R01_S01',
        ]
        self.check_adjacent_detectors_match('LsstCam', det_name, expected_detectors)
        # On the corner of a corner cutout.
        det_name = 'R11_S00'
        expected_detectors = [
            'R10_S12', 'R11_S10', 'R11_S11',
            'R10_S02', 'R11_S00', 'R11_S01',
                       'R01_S20', 'R01_S21',
        ]
        self.check_adjacent_detectors_match('LsstCam', det_name, expected_detectors)

    def test_get_adjacent_detectors_exceptions(self):
        # Ensure that get_adjacent_detectors raises errors for invalid detecor
        # names -- either ones which don't exist, or non science ones.
        camera = imsim.Camera('LsstCam')

        # Non-existent detectors.
        with self.assertRaises(GalSimValueError):
            camera.get_adjacent_detectors('R99_S99')
        with self.assertRaises(GalSimValueError):
            camera.get_adjacent_detectors('R00_S00')

        # Non-science detectors.
        with self.assertRaises(GalSimValueError):
            camera.get_adjacent_detectors('R00_SG0')
        with self.assertRaises(GalSimValueError):
            camera.get_adjacent_detectors('R00_SW0')


if __name__ == '__main__':
    unittest.main()
