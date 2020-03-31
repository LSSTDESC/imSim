"""
Unit tests for instance catalog parsing code.
"""
import os
import unittest
import warnings
import tempfile
import shutil
import numpy as np
import desc.imsim
from lsst.afw.cameraGeom import DetectorType
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.utils import _pupilCoordsFromRaDec
from lsst.sims.utils import altAzPaFromRaDec
from lsst.sims.utils import angularSeparation
from lsst.sims.utils import ObservationMetaData
from lsst.sims.utils import arcsecFromRadians
from lsst.sims.photUtils import Sed, BandpassDict
from lsst.sims.photUtils import Bandpass, PhotometricParameters
from lsst.sims.coordUtils import chipNameFromPupilCoordsLSST
from lsst.sims.GalSimInterface import LSSTCameraWrapper

def sources_from_list(lines, obs_md, phot_params, file_name):
    """Return a two-item tuple containing
       * a list of GalSimCelestialObjects for each object entry in `lines`
       * a dictionary of these objects disaggregated by chip_name.
    """
    target_chips = [det.getName() for det in lsst_camera()]
    gs_object_dict = dict()
    out_obj_dict = dict()
    for chip_name in target_chips:
        out_obj_dict[chip_name] \
            = [_ for _ in desc.imsim.GsObjectList(lines, obs_md, phot_params,
                                                  file_name, chip_name)]
        for gsobj in out_obj_dict[chip_name]:
            gs_object_dict[gsobj.uniqueId] = gsobj
    return list(gs_object_dict.values()), out_obj_dict

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
                for line in input_file.readlines()[:22]:
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
            results = desc.imsim.parsePhoSimInstanceFile(dummy_catalog, ())
        self.assertIn("Required commands", ee.exception.args[0])
        if os.path.isfile(dummy_catalog):
            os.remove(dummy_catalog)

    def test_metadata_from_file(self):
        """
        Test methods that get ObservationMetaData
        from InstanceCatalogs.
        """
        metadata = desc.imsim.metadata_from_file(self.phosim_file)
        self.assertAlmostEqual(metadata['rightascension'], 53.00913847303155535, 16)
        self.assertAlmostEqual(metadata['declination'], -27.43894880881512321, 16)
        self.assertAlmostEqual(metadata['mjd'], 59580.13974597222113516, 16)
        self.assertAlmostEqual(metadata['altitude'], 66.34657337061349835, 16)
        self.assertAlmostEqual(metadata['azimuth'], 270.27655488919378968, 16)
        self.assertEqual(metadata['filter'], 2)
        self.assertIsInstance(metadata['filter'], int)
        self.assertEqual(metadata['bandpass'], 'r')
        self.assertAlmostEqual(metadata['rotskypos'], 256.7507532, 7)
        self.assertAlmostEqual(metadata['dist2moon'], 124.2838277, 7)
        self.assertAlmostEqual(metadata['moonalt'], -36.1323801, 7)
        self.assertAlmostEqual(metadata['moondec'], -23.4960252, 7)
        self.assertAlmostEqual(metadata['moonphase'], 3.8193650, 7)
        self.assertAlmostEqual(metadata['moonra'], 256.4036553, 7)
        self.assertEqual(metadata['nsnap'], 2)
        self.assertIsInstance(metadata['nsnap'], int)
        self.assertEqual(metadata['obshistid'], 230)
        self.assertIsInstance(metadata['obshistid'], int)
        self.assertAlmostEqual(metadata['rottelpos'], 0.0000000, 7)
        self.assertEqual(metadata['seed'], 230)
        self.assertIsInstance(metadata['seed'], int)
        self.assertAlmostEqual(metadata['seeing'], 0.8662850, 7)
        self.assertAlmostEqual(metadata['sunalt'], -32.7358290, 7)
        self.assertAlmostEqual(metadata['vistime'], 33.0000000, 7)

        self.assertEqual(len(metadata), 20)  # 19 lines plus 'bandpass'

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
        with desc.imsim.fopen(self.phosim_file, mode='rt') as input_:
            lines = [x for x in input_ if x.startswith('object')]

        truth_dtype = np.dtype([('uniqueId', str, 200), ('x_pupil', float), ('y_pupil', float),
                                ('sedFilename', str, 200), ('magNorm', float),
                                ('raJ2000', float), ('decJ2000', float),
                                ('pmRA', float), ('pmDec', float),
                                ('parallax', float), ('v_rad', float),
                                ('Av', float), ('Rv', float)])

        truth_data = np.genfromtxt(os.path.join(self.data_dir, 'truth_stars.txt'),
                                   dtype=truth_dtype, delimiter=';')

        truth_data.sort()

        gs_object_arr, gs_object_dict \
            = sources_from_list(lines, obs_md, phot_params, self.phosim_file)

        id_arr = [None]*len(gs_object_arr)
        for i_obj in range(len(gs_object_arr)):
            id_arr[i_obj] = gs_object_arr[i_obj].uniqueId
        id_arr = sorted(id_arr)
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

        for gs_obj in gs_object_arr:
            i_obj = np.where(truth_data['uniqueId'] == gs_obj.uniqueId)[0][0]
            dd = np.sqrt((x_pup_test[i_obj]-gs_obj.xPupilRadians)**2 +
                         (y_pup_test[i_obj]-gs_obj.yPupilRadians)**2)
            dd = arcsecFromRadians(dd)
            self.assertLess(dd, 0.0005)

        ######## test that fluxes are correctly calculated

        bp_dict = BandpassDict.loadTotalBandpassesFromFiles()
        imsim_bp = Bandpass()
        imsim_bp.imsimBandpass()
        phot_params = PhotometricParameters(nexp=1, exptime=30.0)

        for gs_obj in gs_object_arr:
            i_obj = np.where(truth_data['uniqueId'] == gs_obj.uniqueId)[0][0]
            sed = Sed()
            full_sed_name = os.path.join(os.environ['SIMS_SED_LIBRARY_DIR'],
                                         truth_data['sedFilename'][i_obj])
            sed.readSED_flambda(full_sed_name)
            fnorm = sed.calcFluxNorm(truth_data['magNorm'][i_obj], imsim_bp)
            sed.multiplyFluxNorm(fnorm)
            sed.resampleSED(wavelen_match=bp_dict.wavelenMatch)
            a_x, b_x = sed.setupCCM_ab()
            sed.addDust(a_x, b_x, A_v=truth_data['Av'][i_obj],
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

            chip_name = chipNameFromPupilCoordsLSST(xpup, ypup)
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
        # Read in test_imsim_configs since default ones may change.
        desc.imsim.read_config(os.path.join(self.data_dir, 'test_imsim_configs'))
        galaxy_phosim_file = os.path.join(self.data_dir, 'phosim_galaxies.txt')
        commands = desc.imsim.metadata_from_file(galaxy_phosim_file)
        obs_md = desc.imsim.phosim_obs_metadata(commands)
        phot_params = desc.imsim.photometricParameters(commands)
        with desc.imsim.fopen(galaxy_phosim_file, mode='rt') as input_:
            lines = [x for x in input_ if x.startswith('object')]

        truth_dtype = np.dtype([('uniqueId', str, 200), ('x_pupil', float), ('y_pupil', float),
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

        truth_data.sort()

        gs_object_arr, gs_object_dict \
            = sources_from_list(lines, obs_md, phot_params, galaxy_phosim_file)

        id_arr = [None]*len(gs_object_arr)
        for i_obj in range(len(gs_object_arr)):
            id_arr[i_obj] = gs_object_arr[i_obj].uniqueId
        id_arr = sorted(id_arr)
        np.testing.assert_array_equal(truth_data['uniqueId'], id_arr)

        ######## test that galaxy parameters are correctly read in

        g1 = truth_data['gamma1']/(1.0-truth_data['kappa'])
        g2 = truth_data['gamma2']/(1.0-truth_data['kappa'])
        mu = 1.0/((1.0-truth_data['kappa'])**2 - (truth_data['gamma1']**2 + truth_data['gamma2']**2))
        for gs_obj in gs_object_arr:
            i_obj = np.where(truth_data['uniqueId'] == gs_obj.uniqueId)[0][0]
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
            self.assertAlmostEqual(2.*np.pi - gs_obj.positionAngleRadians,
                                   truth_data['positionAngle'][i_obj], 7)
            self.assertAlmostEqual(gs_obj.sindex,
                                   truth_data['sindex'][i_obj], 10)

        ######## test that pupil coordinates are correct to within
        ######## half a milliarcsecond

        x_pup_test, y_pup_test = _pupilCoordsFromRaDec(truth_data['raJ2000'],
                                                       truth_data['decJ2000'],
                                                       obs_metadata=obs_md)

        for gs_obj in gs_object_arr:
            i_obj = np.where(truth_data['uniqueId'] == gs_obj.uniqueId)[0][0]
            dd = np.sqrt((x_pup_test[i_obj]-gs_obj.xPupilRadians)**2 +
                         (y_pup_test[i_obj]-gs_obj.yPupilRadians)**2)
            dd = arcsecFromRadians(dd)
            self.assertLess(dd, 0.0005)

        ######## test that fluxes are correctly calculated

        bp_dict = BandpassDict.loadTotalBandpassesFromFiles()
        imsim_bp = Bandpass()
        imsim_bp.imsimBandpass()
        phot_params = PhotometricParameters(nexp=1, exptime=30.0)

        for gs_obj in gs_object_arr:
            i_obj = np.where(truth_data['uniqueId'] == gs_obj.uniqueId)[0][0]
            sed = Sed()
            full_sed_name = os.path.join(os.environ['SIMS_SED_LIBRARY_DIR'],
                                         truth_data['sedFilename'][i_obj])
            sed.readSED_flambda(full_sed_name)
            fnorm = sed.calcFluxNorm(truth_data['magNorm'][i_obj], imsim_bp)
            sed.multiplyFluxNorm(fnorm)

            a_x, b_x = sed.setupCCM_ab()
            sed.addDust(a_x, b_x, A_v=truth_data['internalAv'][i_obj],
                        R_v=truth_data['internalRv'][i_obj])

            sed.redshiftSED(truth_data['redshift'][i_obj], dimming=True)
            sed.resampleSED(wavelen_match=bp_dict.wavelenMatch)
            a_x, b_x = sed.setupCCM_ab()
            sed.addDust(a_x, b_x, A_v=truth_data['galacticAv'][i_obj],
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

            chip_name = chipNameFromPupilCoordsLSST(xpup, ypup)
            if chip_name is not None:
               self.assertIn(unq, unique_id_dict[chip_name])
               valid_chip_names.add(chip_name)
               valid += 1

        self.assertGreater(valid, 10)
        self.assertGreater(len(valid_chip_names), 5)

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

        camera = LSSTCameraWrapper().camera
        sensors = [det.getName() for det in camera
                   if det.getType() not in (DetectorType.WAVEFRONT, DetectorType.GUIDER)]
        with warnings.catch_warnings(record=True) as wa:
            instcat_contents \
                = desc.imsim.parsePhoSimInstanceFile(cat_file, sensors)
            my_objs = set()
            for sensor in sensors:
                [my_objs.add(x) for x in instcat_contents.sources[1][sensor]]

        # these are the objects that should be omitted
        bad_unique_ids = set([str(x) for x in
                              [34307989098524, 811883374597,
                               811883374596, 956090392580,
                               34307989098523, 34304522113056]])

        self.assertEqual(len(my_objs), 18)
        for obj in instcat_contents.sources[0]:
            self.assertNotIn(obj.uniqueId, bad_unique_ids)

        for chip_name in instcat_contents.sources[1]:
            for obj in instcat_contents.sources[1][chip_name]:
                self.assertNotIn(obj.uniqueId, bad_unique_ids)


if __name__ == '__main__':
    unittest.main()
