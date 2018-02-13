"""
Unit tests for instance catalog parsing code.
"""
from __future__ import absolute_import, print_function
import os
import unittest
import warnings
import numpy as np
import desc.imsim
from lsst.sims.utils import _pupilCoordsFromRaDec
from lsst.sims.utils import arcsecFromRadians


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
        self.data_dir = os.path.join(os.environ['IMSIM_DIR'], 'tests', 'data')
        self.phosim_file = os.path.join(self.data_dir,
                                         'phosim_stars.txt')
        self.extra_commands = 'instcat_extra.txt'
        with open(self.extra_commands, 'w') as output:
            for line in open(self.phosim_file).readlines()[:23]:
                output.write(line)
            output.write('extra_command 1\n')

    def tearDown(self):
        os.remove(self.extra_commands)

    def test_metadata_from_file(self):
        """
        Test methods that get ObservationMetaData
        from InstanceCatalogs.
        """
        metadata = desc.imsim.metadata_from_file(self.phosim_file)
        self.assertAlmostEqual(metadata['rightascension'], 53.0091385, 7)
        self.assertAlmostEqual(metadata['declination'], -27.4389488, 7)
        self.assertAlmostEqual(metadata['mjd'], 59580.1397460, 7)
        self.assertAlmostEqual(metadata['altitude'], 66.3464409, 7)
        self.assertAlmostEqual(metadata['azimuth'], 270.2764762, 7)
        self.assertEqual(metadata['filter'], 2)
        self.assertIsInstance(metadata['filter'], int)
        self.assertEqual(metadata['bandpass'], 'r')
        self.assertAlmostEqual(metadata['rotskypos'], 256.7507532, 7)
        self.assertAlmostEqual(metadata['FWHMeff'], 1.1219680, 7)
        self.assertAlmostEqual(metadata['FWHMgeom'], 0.9742580, 7)
        self.assertAlmostEqual(metadata['dist2moon'], 124.2838277, 7)
        self.assertAlmostEqual(metadata['moonalt'], -36.1323801, 7)
        self.assertAlmostEqual(metadata['moondec'], -23.4960252, 7)
        self.assertAlmostEqual(metadata['moonphase'], 3.8193650, 7)
        self.assertAlmostEqual(metadata['moonra'], 256.4036553, 7)
        self.assertEqual(metadata['nsnap'], 2)
        self.assertIsInstance(metadata['nsnap'], int)
        self.assertEqual(metadata['obshistid'], 230)
        self.assertIsInstance(metadata['obshistid'], int)
        self.assertAlmostEqual(metadata['rawSeeing'], 0.8662850, 7)
        self.assertAlmostEqual(metadata['rottelpos'], 0.0000000, 7)
        self.assertEqual(metadata['seed'], 230)
        self.assertIsInstance(metadata['seed'], int)
        self.assertAlmostEqual(metadata['seeing'], 0.8662850, 7)
        self.assertAlmostEqual(metadata['sunalt'], -32.7358290, 7)
        self.assertAlmostEqual(metadata['vistime'], 33.0000000, 7)

        self.assertEqual(len(metadata), 23)  # 22 lines plus 'bandpass'

        obs = desc.imsim.phosim_obs_metadata(metadata)

        self.assertAlmostEqual(obs.pointingRA, metadata['rightascension'], 7)
        self.assertAlmostEqual(obs.pointingDec, metadata['declination'], 7)
        self.assertAlmostEqual(obs.rotSkyPos, metadata['rotskypos'], 7)
        self.assertAlmostEqual(obs.mjd.TAI, metadata['mjd'], 7)
        self.assertEqual(obs.bandpass, 'r')

    def test_object_extraction(self):
        """
        Test that method to get GalSimCelestialObjects from
        InstanceCatalogs works
        """
        commands = desc.imsim.metadata_from_file(self.phosim_file)
        obs_md = desc.imsim.phosim_obs_metadata(commands)
        phot_params = desc.imsim.photometricParameters(commands)
        (gs_object_arr,
         gs_object_dict) = desc.imsim.sources_from_file(self.phosim_file,
                                                        obs_md,
                                                        phot_params)

        id_arr = np.zeros(len(gs_object_arr), dtype=int)
        for i_obj in range(len(gs_object_arr)):
            id_arr[i_obj] = gs_object_arr[i_obj].uniqueId

        truth_dtype = np.dtype([('uniqueId', int), ('x_pupil', float), ('y_pupil', float),
                                ('sedFilename', str, 200), ('magNorm', float),
                                ('raJ2000', float), ('decJ2000', float),
                                ('pmRA', float), ('pmDec', float),
                                ('parallax', float), ('v_rad', float)])

        truth_data = np.genfromtxt(os.path.join(self.data_dir, 'truth_stars.txt'),
                                   dtype=truth_dtype, delimiter=';')

        np.testing.assert_array_equal(truth_data['uniqueId'], id_arr)

        x_pup_test, y_pup_test = _pupilCoordsFromRaDec(truth_data['raJ2000'],
                                                       truth_data['decJ2000'],
                                                       pm_ra=truth_data['pmRA'],
                                                       pm_dec=truth_data['pmDec'],
                                                       v_rad=truth_data['v_rad'],
                                                       parallax=truth_data['parallax'],
                                                       obs_metadata=obs_md)

        for i_obj, gs_obj in enumerate(gs_object_arr):
            self.assertEqual(truth_data['uniqueId'][i_obj], gs_obj.uniqueId)
            dd = np.sqrt((x_pup_test[i_obj]-gs_obj.xPupilRadians)**2 +
                         (y_pup_test[i_obj]-gs_obj.yPupilRadians)**2)
            dd = arcsecFromRadians(dd)
            self.assertLess(dd, 0.0005)

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
        self.assertAlmostEqual(phot_params.exptime, 15.)
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
