"""
Unit tests for instance catalog parsing code.
"""
from __future__ import absolute_import, print_function
import os
import unittest
import warnings
import tempfile
import shutil
import numpy as np
import desc.imsim
from lsst.sims.utils import _pupilCoordsFromRaDec
from lsst.sims.utils import arcsecFromRadians
from lsst.sims.photUtils import Sed, BandpassDict
from lsst.sims.photUtils import Bandpass, PhotometricParameters
from lsst.sims.coordUtils import chipNameFromPupilCoordsLSST


class InstanceCatalogParserTestCase(unittest.TestCase):
    """
    TestCase class for instance catalog parsing code.
    """
    @classmethod
    def setUpClass(cls):
        cls.config = desc.imsim.read_config()
        cls.data_dir = os.path.join(os.environ['IMSIM_DIR'], 'tests', 'data')
        cls.scratch_dir = tempfile.mkdtemp(prefix=cls.data_dir)


    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.scratch_dir):
            for file_name in os.listdir(cls.scratch_dir):
                os.unlink(os.path.join(cls.scratch_dir, file_name))
            shutil.rmtree(cls.scratch_dir)

    def setUp(self):
        self.phosim_file = os.path.join(self.data_dir,
                                         'phosim_stars.txt')
        self.extra_commands = 'instcat_extra.txt'
        with open(self.phosim_file, 'r') as input_file:
            with open(self.extra_commands, 'w') as output:
                for line in input_file.readlines()[:23]:
                    output.write(line)
                output.write('extra_command 1\n')

    def tearDown(self):
        os.remove(self.extra_commands)

    def test_required_commands_error(self):
        """
        Test that an error is raised if required commands are
        missing from the InstanceCatalog file
        """
        dummy_catalog = tempfile.mktemp(prefix=self.scratch_dir)
        with open(self.phosim_file, 'r') as input_file:
            input_lines = input_file.readlines()
            with open(dummy_catalog, 'w') as output_file:
                for line in input_lines[:8]:
                    output_file.write(line)
                for line in input_lines[24:]:
                    output_file.write(line)

        with self.assertRaises(desc.imsim.PhosimInstanceCatalogParseError) as ee:
            results = desc.imsim.parsePhoSimInstanceFile(dummy_catalog)
        self.assertIn("Required commands", ee.exception.args[0])
        if os.path.isfile(dummy_catalog):
            os.remove(dummy_catalog)

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

    def test_object_extraction_stars(self):
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
                                ('parallax', float), ('v_rad', float),
                                ('Av', float), ('Rv', float)])

        truth_data = np.genfromtxt(os.path.join(self.data_dir, 'truth_stars.txt'),
                                   dtype=truth_dtype, delimiter=';')

        np.testing.assert_array_equal(truth_data['uniqueId'], id_arr)

        ######## test that pupil coordinates are correct to within
        ######## half a milliarcsecond

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

        ######## test that fluxes are correctly calculated

        bp_dict = BandpassDict.loadTotalBandpassesFromFiles()
        imsim_bp = Bandpass()
        imsim_bp.imsimBandpass()
        phot_params = PhotometricParameters(nexp=1, exptime=30.0)

        for i_obj, gs_obj in enumerate(gs_object_arr):
            sed = Sed()
            full_sed_name = os.path.join(os.environ['SIMS_SED_LIBRARY_DIR'],
                                         truth_data['sedFilename'][i_obj])
            sed.readSED_flambda(full_sed_name)
            fnorm = sed.calcFluxNorm(truth_data['magNorm'][i_obj], imsim_bp)
            sed.multiplyFluxNorm(fnorm)
            sed.resampleSED(wavelen_match=bp_dict.wavelenMatch)
            a_x, b_x = sed.setupCCMab()
            sed.addCCMDust(a_x, b_x, A_v=truth_data['Av'][i_obj],
                           R_v=truth_data['Rv'][i_obj])

            for bp in ('u', 'g', 'r', 'i', 'z', 'y'):
                flux = sed.calcADU(bp_dict[bp], phot_params)*phot_params.gain
                self.assertAlmostEqual(flux/gs_obj.flux(bp), 1.0, 10)

        ######## test that objects are assigned to the right chip in
        ######## gs_object_dict

        unique_id_dict = {}
        for chip_name in gs_object_dict:
            local_unique_id_list = []
            for gs_object in gs_object_dict[chip_name]:
                local_unique_id_list.append(gs_object.uniqueId)
            local_unique_id_list = set(local_unique_id_list)
            unique_id_dict[chip_name] = local_unique_id_list

        valid = 0
        valid_chip_names = set()
        for unq, xpup, ypup in zip(truth_data['uniqueId'],
                                   truth_data['x_pupil'],
                                   truth_data['y_pupil']):

            chip_name = chipNameFromPupilCoordsLSST(xpup, ypup)[0]
            if chip_name is not None:
               self.assertIn(unq, unique_id_dict[chip_name])
               valid_chip_names.add(chip_name)
               valid += 1

        self.assertGreater(valid, 10)
        self.assertGreater(len(valid_chip_names), 5)

    def test_object_extraction_galaxies(self):
        """
        Test that method to get GalSimCelestialObjects from
        InstanceCatalogs works
        """
        galaxy_phosim_file = os.path.join(self.data_dir, 'phosim_galaxies.txt')
        commands = desc.imsim.metadata_from_file(galaxy_phosim_file)
        obs_md = desc.imsim.phosim_obs_metadata(commands)
        phot_params = desc.imsim.photometricParameters(commands)
        (gs_object_arr,
         gs_object_dict) = desc.imsim.sources_from_file(galaxy_phosim_file,
                                                        obs_md,
                                                        phot_params)

        id_arr = np.zeros(len(gs_object_arr), dtype=int)
        for i_obj in range(len(gs_object_arr)):
            id_arr[i_obj] = gs_object_arr[i_obj].uniqueId

        truth_dtype = np.dtype([('uniqueId', int), ('x_pupil', float), ('y_pupil', float),
                                ('sedFilename', str, 200), ('magNorm', float),
                                ('raJ2000', float), ('decJ2000', float),
                                ('redshift', float), ('gamma1', float),
                                ('gamma2', float), ('kappa', float),
                                ('galacticAv', float), ('galacticRv', float),
                                ('internalAv', float), ('internalRv', float),
                                ('minorAxis', float), ('majorAxis', float),
                                ('positionAngle', float), ('sindex', float)])

        truth_data = np.genfromtxt(os.path.join(self.data_dir, 'truth_galaxies.txt'),
                                   dtype=truth_dtype, delimiter=';')

        np.testing.assert_array_equal(truth_data['uniqueId'], id_arr)

        ######## test that galaxy parameters are correctly read in

        g1 = truth_data['gamma1']/(1.0-truth_data['kappa'])
        g2 = truth_data['gamma2']/(1.0-truth_data['kappa'])
        mu = 1.0/((1.0-truth_data['kappa'])**2 - (truth_data['gamma1']**2 + truth_data['gamma2']**2))
        for i_obj, gs_obj in enumerate(gs_object_arr):
            self.assertAlmostEqual(gs_obj.mu/mu[i_obj], 1.0, 6)
            self.assertAlmostEqual(gs_obj.g1/g1[i_obj], 1.0, 6)
            self.assertAlmostEqual(gs_obj.g2/g2[i_obj], 1.0, 6)
            self.assertGreater(np.abs(gs_obj.mu), 0.0)
            self.assertGreater(np.abs(gs_obj.g1), 0.0)
            self.assertGreater(np.abs(gs_obj.g2), 0.0)

            self.assertAlmostEqual(gs_obj.halfLightRadiusRadians,
                                   truth_data['majorAxis'][i_obj], 13)
            self.assertAlmostEqual(gs_obj.minorAxisRadians,
                                   truth_data['minorAxis'][i_obj], 13)
            self.assertAlmostEqual(gs_obj.majorAxisRadians,
                                   truth_data['majorAxis'][i_obj], 13)
            self.assertAlmostEqual(gs_obj.positionAngleRadians,
                                   truth_data['positionAngle'][i_obj], 7)
            self.assertAlmostEqual(gs_obj.sindex,
                                   truth_data['sindex'][i_obj], 10)

        ######## test that pupil coordinates are correct to within
        ######## half a milliarcsecond

        x_pup_test, y_pup_test = _pupilCoordsFromRaDec(truth_data['raJ2000'],
                                                       truth_data['decJ2000'],
                                                       obs_metadata=obs_md)

        for i_obj, gs_obj in enumerate(gs_object_arr):
            self.assertEqual(truth_data['uniqueId'][i_obj], gs_obj.uniqueId)
            dd = np.sqrt((x_pup_test[i_obj]-gs_obj.xPupilRadians)**2 +
                         (y_pup_test[i_obj]-gs_obj.yPupilRadians)**2)
            dd = arcsecFromRadians(dd)
            self.assertLess(dd, 0.0005)

        ######## test that fluxes are correctly calculated

        bp_dict = BandpassDict.loadTotalBandpassesFromFiles()
        imsim_bp = Bandpass()
        imsim_bp.imsimBandpass()
        phot_params = PhotometricParameters(nexp=1, exptime=30.0)

        for i_obj, gs_obj in enumerate(gs_object_arr):
            sed = Sed()
            full_sed_name = os.path.join(os.environ['SIMS_SED_LIBRARY_DIR'],
                                         truth_data['sedFilename'][i_obj])
            sed.readSED_flambda(full_sed_name)
            fnorm = sed.calcFluxNorm(truth_data['magNorm'][i_obj], imsim_bp)
            sed.multiplyFluxNorm(fnorm)

            a_x, b_x = sed.setupCCMab()
            sed.addCCMDust(a_x, b_x, A_v=truth_data['internalAv'][i_obj],
                           R_v=truth_data['internalRv'][i_obj])

            sed.redshiftSED(truth_data['redshift'][i_obj], dimming=True)
            sed.resampleSED(wavelen_match=bp_dict.wavelenMatch)
            a_x, b_x = sed.setupCCMab()
            sed.addCCMDust(a_x, b_x, A_v=truth_data['galacticAv'][i_obj],
                           R_v=truth_data['galacticRv'][i_obj])

            for bp in ('u', 'g', 'r', 'i', 'z', 'y'):
                flux = sed.calcADU(bp_dict[bp], phot_params)*phot_params.gain
                self.assertAlmostEqual(flux/gs_obj.flux(bp), 1.0, 6)

        ######## test that objects are assigned to the right chip in
        ######## gs_object_dict

        unique_id_dict = {}
        for chip_name in gs_object_dict:
            local_unique_id_list = []
            for gs_object in gs_object_dict[chip_name]:
                local_unique_id_list.append(gs_object.uniqueId)
            local_unique_id_list = set(local_unique_id_list)
            unique_id_dict[chip_name] = local_unique_id_list

        valid = 0
        valid_chip_names = set()
        for unq, xpup, ypup in zip(truth_data['uniqueId'],
                                   truth_data['x_pupil'],
                                   truth_data['y_pupil']):

            chip_name = chipNameFromPupilCoordsLSST(xpup, ypup)[0]
            if chip_name is not None:
               self.assertIn(unq, unique_id_dict[chip_name])
               valid_chip_names.add(chip_name)
               valid += 1

        self.assertGreater(valid, 10)
        self.assertGreater(len(valid_chip_names), 5)


    def test_parsePhoSimInstanceFile_warning(self):
        "Test the warnings emitted by the instance catalog parser."
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(UserWarning,
                              desc.imsim.metadata_from_file,
                              self.extra_commands)

    def test_photometricParameters(self):
        "Test the photometricParameters function."
        commands = desc.imsim.metadata_from_file(self.phosim_file)

        phot_params = \
            desc.imsim.photometricParameters(commands)
        self.assertEqual(phot_params.gain, 1)
        self.assertEqual(phot_params.bandpass, 'r')
        self.assertEqual(phot_params.nexp, 2)
        self.assertAlmostEqual(phot_params.exptime, 15.)
        self.assertEqual(phot_params.readnoise, 0)
        self.assertEqual(phot_params.darkcurrent, 0)

    def test_validate_phosim_object_list(self):
        "Test the validation of the rows of the phoSimObjects DataFrame."
        cat_file = os.path.join(os.environ['IMSIM_DIR'], 'tests', 'tiny_instcat.txt')

        with warnings.catch_warnings(record=True) as wa:
            instcat_contents = desc.imsim.parsePhoSimInstanceFile(cat_file)
        self.assertGreater(len(wa), 1)

        # we must detect which warning is the warning we are actually
        # testing, because PALPY keeps raising ERFAWarnings over our
        # request for dates in the future
        desired_warning_dex = -1
        for i_ww, ww in enumerate(wa):
            if 'Omitted 5 suspicious objects' in ww.message.args[0]:
                desired_warning_dex = i_ww
                break

        if desired_warning_dex<0:
            raise RuntimeError("Expected warning about bad sources "
                               "not issued")

        message = wa[desired_warning_dex].message.args[0]

        # these are the objects that should be omitted
        bad_unique_ids = set([34307989098524, 811883374597,
                              811883374596, 956090392580,
                              34307989098523])

        self.assertIn('Omitted 5 suspicious objects', message)
        self.assertIn('4 had galactic_Av', message)
        self.assertIn('1 had mag_norm', message)
        self.assertIn('1 had semi_major_axis', message)

        self.assertEqual(len(instcat_contents.sources[0]), 16)
        for obj in instcat_contents.sources[0]:
            self.assertNotIn(obj.uniqueId, bad_unique_ids)

        for chip_name in instcat_contents.sources[1]:
            for obj in instcat_contents.sources[1][chip_name]:
                self.assertNotIn(obj.uniqueId, bad_unique_ids)


if __name__ == '__main__':
    unittest.main()
