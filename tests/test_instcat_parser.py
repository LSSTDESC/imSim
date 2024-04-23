"""
Unit tests for instance catalog parsing code.
"""
import os
from pathlib import Path
import unittest
import tempfile
import shutil
import numpy as np
import galsim
import astropy.time
import imsim
from lsst.afw.cameraGeom import DetectorType

from test_batoid_wcs import sphere_dist

DATA_DIR = Path(__file__).parent / 'data'

def sources_from_list(lines, opsim_data, phot_params, file_name):
    """Return a two-item tuple containing
       * a list of GalSimCelestialObjects for each object entry in `lines`
       * a dictionary of these objects disaggregated by chip_name.
    """
    target_chips = [det.getName() for det in imsim.get_camera()]
    gs_object_dict = dict()
    out_obj_dict = dict()
    for chip_name in target_chips:
        out_obj_dict[chip_name] \
            = [_ for _ in imsim.GsObjectList(lines, opsim_data, phot_params,
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
        cls.data_dir = DATA_DIR
        cls.scratch_dir = tempfile.mkdtemp(prefix=str(cls.data_dir))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.scratch_dir):
            for file_name in os.listdir(cls.scratch_dir):
                os.unlink(os.path.join(cls.scratch_dir, file_name))
            shutil.rmtree(cls.scratch_dir)

    def setUp(self):
        self.phosim_file = str(self.data_dir / 'phosim_stars.txt')

    def make_wcs(self, instcat_file=None, sensors=None):
        # Make a wcs to use for this instance catalog.
        if instcat_file is None:
            instcat_file = self.phosim_file

        opsim_data = imsim.OpsimDataLoader(instcat_file)
        boresight = galsim.CelestialCoord(ra=opsim_data['rightascension'] * galsim.degrees,
                                          dec=opsim_data['declination'] * galsim.degrees)
        rotTelPos = opsim_data['rottelpos'] * galsim.degrees
        rotTelPos += 180*galsim.degrees  # We used to simulate the camera upside down...
        obstime = astropy.time.Time(opsim_data['mjd'], format='mjd', scale='tai')
        band = opsim_data['band']
        builder = imsim.BatoidWCSBuilder()

        if sensors is None:
            camera = imsim.get_camera()
            sensors = [det.getName() for det in camera
                       if det.getType() not in (DetectorType.WAVEFRONT, DetectorType.GUIDER)]

        all_wcs = {}
        for det_name in sensors:
            if det_name not in all_wcs:
                telescope = imsim.load_telescope(f"LSST_{band}.yaml", rotTelPos=rotTelPos)
                factory = builder.makeWCSFactory(boresight, obstime, telescope, bandpass=band)
                wcs = factory.getWCS(builder.camera[det_name])
                all_wcs[det_name] = wcs

        return all_wcs

    def test_metadata_from_opsim_db(self):
        """
        Test reading of the visit metadata from an opsim db file.
        """
        opsim_db_file = os.path.join(imsim.meta_data.data_dir, 'small_opsim.db')
        visit = 22184
        md = imsim.OpsimDataLoader(opsim_db_file, visit=visit, snap=0)
        self.assertAlmostEqual(md['observationId'], visit)
        self.assertAlmostEqual(md['fieldRA'], 65.00821243449612)
        self.assertAlmostEqual(md['fieldDec'], -33.20121826915378)
        self.assertAlmostEqual(md['rawSeeing'], 0.5833528497734382)
        self.assertAlmostEqual(md['FWHMeff'], 0.7790170013788277)
        self.assertAlmostEqual(md['FWHMgeom'], 0.6923519751333964)
        self.assertAlmostEqual(md['mjd'], 60248.33830784654 + 7.5/86400)
        self.assertAlmostEqual(md['airmass'], 1.0763250938907971)
        self.assertAlmostEqual(md['band'], 'z')
        self.assertAlmostEqual(md['exptime'], 15)

        # KeyError if key is invalid, unless using get with default (just like a dict)
        with np.testing.assert_raises(KeyError):
            md['exp_time']
        with np.testing.assert_raises(KeyError):
            md.get('exp_time')
        assert md.get('exp_time', 15) == 15

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

        with self.assertRaises(ValueError) as ee:
            instcat = imsim.OpsimDataLoader(dummy_catalog)
        self.assertIn("Required commands", ee.exception.args[0])
        if os.path.isfile(dummy_catalog):
            os.remove(dummy_catalog)

    def test_metadata_from_file(self):
        """
        Test methods that get ObservationMetaData
        from InstanceCatalogs.
        """
        metadata = imsim.OpsimDataLoader(self.phosim_file)
        self.assertAlmostEqual(metadata['fieldRA'], 53.00913847303155535, 16)
        self.assertAlmostEqual(metadata['fieldDec'], -27.43894880881512321, 16)
        self.assertAlmostEqual(metadata['mjd'], 59580.13974597222113516, 16)
        self.assertAlmostEqual(metadata['altitude'], 66.34657337061349835, 16)
        self.assertAlmostEqual(metadata['azimuth'], 270.27655488919378968, 16)
        self.assertEqual(metadata['filter'], 2)
        self.assertIsInstance(metadata['filter'], int)
        self.assertEqual(metadata['band'], 'r')
        self.assertAlmostEqual(metadata['rotSkyPos'], 256.7507532, 7)
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
        self.assertAlmostEqual(metadata['rawSeeing'], 0.8662850, 7)
        self.assertAlmostEqual(metadata['sunalt'], -32.7358290, 7)
        self.assertAlmostEqual(metadata['vistime'], 33.0000000, 7)

    def test_object_extraction_stars(self):
        """
        Test that method to get GalSimCelestialObjects from
        InstanceCatalogs works
        """
        md = imsim.OpsimDataLoader(self.phosim_file)

        truth_dtype = np.dtype([('uniqueId', str, 200), ('x_pupil', float), ('y_pupil', float),
                                ('sedFilename', str, 200), ('magNorm', float),
                                ('raJ2000', float), ('decJ2000', float),
                                ('pmRA', float), ('pmDec', float),
                                ('parallax', float), ('v_rad', float),
                                ('Av', float), ('Rv', float)])

        truth_data = np.genfromtxt(os.path.join(self.data_dir, 'truth_stars.txt'),
                                   dtype=truth_dtype, delimiter=';')
        truth_data.sort()

        all_wcs = self.make_wcs()
        all_cats = {}
        sed_dir = os.path.join(self.data_dir, 'test_sed_library')
        for det_name in all_wcs:
            cat = all_cats[det_name] = imsim.InstCatalog(self.phosim_file, all_wcs[det_name],
                                                         sed_dir=sed_dir, edge_pix=0)

        id_arr = np.concatenate([cat.id for cat in all_cats.values()])
        print('diff1 = ',set(truth_data['uniqueId'])-set(id_arr))
        print('diff2 = ',set(id_arr)-set(truth_data['uniqueId']))
        print('diff3 = ',set(id_arr)^set(truth_data['uniqueId']))
        # XXX: id 1704203146244 is in the InstCatalog, but not the truth catalog.
        #      It has a y value of 4093, which is inside the imsim bounds of [0,4096],
        #      but not the actual ITL sensor size of [0,4072].  Moreover, as we'll see
        #      below, the Batoid WCS is up to 11 arcsec different from the LSST WCS, so
        #      that may also contribute to the discrepancy.  (There are many more mismatches
        #      in the galaxy test below.)
        assert len(set(id_arr)^set(truth_data['uniqueId'])) <= 1
        index = np.argsort(id_arr)
        index = index[np.where(np.in1d(id_arr[index], truth_data['uniqueId']))]
        np.testing.assert_array_equal(truth_data['uniqueId'], id_arr[index])

        ######## test that pupil coordinates are correct to within
        ######## half a milliarcsecond
        if 0:
            # XXX: This test required lsst.sims.  Not sure if we still need it?

            x_pup_test, y_pup_test = _pupilCoordsFromRaDec(truth_data['raJ2000'],
                                                           truth_data['decJ2000'],
                                                           pm_ra=truth_data['pmRA'],
                                                           pm_dec=truth_data['pmDec'],
                                                           v_rad=truth_data['v_rad'],
                                                           parallax=truth_data['parallax'],
                                                           obs_metadata=opsim_data)

            for gs_obj in gs_object_arr:
                i_obj = np.where(truth_data['uniqueId'] == gs_obj.uniqueId)[0][0]
                dd = np.sqrt((x_pup_test[i_obj]-gs_obj.xPupilRadians)**2 +
                             (y_pup_test[i_obj]-gs_obj.yPupilRadians)**2)
                dd = arcsecFromRadians(dd)
                self.assertLess(dd, 0.0005)

        ######## test that positions are consistent

        for det_name in all_wcs:
            wcs = all_wcs[det_name]
            cat = all_cats[det_name]
            for i in range(cat.nobjects):
                image_pos = cat.image_pos[i]
                world_pos = cat.world_pos[i]
                self.assertLess(world_pos.distanceTo(wcs.toWorld(image_pos)), 0.0005*galsim.arcsec)

        ra_arr = np.array([pos.ra.rad for cat in all_cats.values() for pos in cat.world_pos])
        dec_arr = np.array([pos.dec.rad for cat in all_cats.values() for pos in cat.world_pos])
        # XXX: These are only within 10 arcsec, which is kind of a lot.
        #      I (MJ) think this is probalby related to differences in convention from how we
        #      used to do the WCS, so it's probably fine.  But someone who knows better might
        #      want to update how this test is done.
        #      cf. Issue #262
        print("ra diff = ",ra_arr[index]-truth_data['raJ2000'])
        print("dec diff = ",dec_arr[index]-truth_data['decJ2000'])
        dist = sphere_dist(ra_arr[index], dec_arr[index],
                           truth_data['raJ2000'], truth_data['decJ2000'])
        print("sphere dist = ",dist)
        print('max dist = ',np.max(dist))
        print('max dist (arcsec) = ',np.max(dist) * 180/np.pi * 3600)
        np.testing.assert_array_less(dist * 180/np.pi * 3600, 10.)  # largest is 9.97 arcscec.
        np.testing.assert_allclose(truth_data['raJ2000'], ra_arr[index], rtol=1.e-4)
        np.testing.assert_allclose(truth_data['decJ2000'], dec_arr[index], rtol=1.e-4)

        ######## test that fluxes are correctly calculated

        bp = imsim.RubinBandpass(md['band'])

        for det_name in all_wcs:
            wcs = all_wcs[det_name]
            cat = all_cats[det_name]
            for i in range(cat.nobjects):
                obj = cat.getObj(i)
                if 0:
                    # XXX: The old test used the sims Sed class.  Circumventing this now,
                    #      but leaving the old code in case there is a way to use it eventually.
                    sed = Sed()
                    i_obj = np.where(truth_data['uniqueId'] == cat.id[i])[0][0]
                    full_sed_name = os.path.join(sed_dir, truth_data['sedFilename'][i_obj])
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

                # Instead, this basically recapitulates the calculation in the InstCatalog class.
                magnorm = cat.getMagNorm(i)
                flux = np.exp(-0.9210340371976184 * magnorm)
                rubin_area = 0.25 * np.pi * 649**2 # cm^2
                exptime = 30
                fAt = flux * rubin_area * exptime
                sed = cat.getSED(i)
                flux = sed.calculateFlux(bp) * fAt
                self.assertAlmostEqual(flux, obj.calculateFlux(bp))

        ######## test that objects are assigned to the right chip in
        ######## gs_object_dict

        if 0:
            # XXX: This doesn't seem relevant anymore.  But leaving this here in case we want
            #      to reenable it somehow.
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
        galaxy_phosim_file = os.path.join(self.data_dir, 'phosim_galaxies.txt')
        md = imsim.OpsimDataLoader(galaxy_phosim_file)

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

        all_wcs = self.make_wcs()
        all_cats = {}
        sed_dir = os.path.join(self.data_dir, 'test_sed_library')
        for det_name in all_wcs:
            # Note: the truth catalog apparently didn't flip the g2 values, so use flip_g2=False.
            cat = all_cats[det_name] = imsim.InstCatalog(galaxy_phosim_file, all_wcs[det_name],
                                                         sed_dir=sed_dir, edge_pix=0, flip_g2=False)
            approx_nobj = cat.getApproxNObjects()  # This is only different from getNObjects
                                                   # if it is called first.
            nobj = cat.getNObjects()
            print(det_name, nobj, approx_nobj)
            assert approx_nobj > nobj

            # After loading, the two values are equal.
            assert cat.getApproxNObjects() == cat.getNObjects()

        id_arr = np.concatenate([cat.id for cat in all_cats.values()])
        print('diff1 = ',set(truth_data['uniqueId'])-set(id_arr))
        print('diff2 = ',set(id_arr)-set(truth_data['uniqueId']))
        print('diff3 = ',set(id_arr)^set(truth_data['uniqueId']))
        # XXX: There are more differences here.  I think mostly because of the WCS mismatch.
        #      We should probably figure this out to make sure the Batoid WCS isn't missing some
        #      bit of physics that the LSST WCS included...
        #      cf. Issue #262
        assert len(set(id_arr)^set(truth_data['uniqueId'])) <= 10
        index = np.argsort(id_arr)
        index1 = np.where(np.in1d(truth_data['uniqueId'], id_arr[index]))
        index2 = index[np.where(np.in1d(id_arr[index], truth_data['uniqueId']))]
        np.testing.assert_array_equal(truth_data['uniqueId'][index1], id_arr[index2])

        ######## test that galaxy parameters are correctly read in

        true_g1 = truth_data['gamma1']/(1.0-truth_data['kappa'])
        true_g2 = truth_data['gamma2']/(1.0-truth_data['kappa'])
        true_mu = 1.0/((1.0-truth_data['kappa'])**2 - (truth_data['gamma1']**2 + truth_data['gamma2']**2))
        for det_name in all_wcs:
            wcs = all_wcs[det_name]
            cat = all_cats[det_name]
            for i in range(cat.nobjects):
                obj_g1, obj_g2, obj_mu = cat.getLens(i)
                i_obj = np.where(truth_data['uniqueId'] == cat.id[i])[0]
                if len(i_obj) == 0: continue
                i_obj = i_obj[0]
                self.assertAlmostEqual(obj_mu/true_mu[i_obj], 1.0, 6)
                self.assertAlmostEqual(obj_g1/true_g1[i_obj], 1.0, 6)
                self.assertAlmostEqual(obj_g2/true_g2[i_obj], 1.0, 6)
                self.assertGreater(np.abs(obj_mu), 0.0)
                self.assertGreater(np.abs(obj_g1), 0.0)
                self.assertGreater(np.abs(obj_g2), 0.0)

                # We no longer give the galaxy parameters names, but they are available
                # in the objinfo array.
                arcsec = galsim.arcsec / galsim.radians
                self.assertAlmostEqual(float(cat.objinfo[i][1]) * arcsec,
                                       truth_data['majorAxis'][i_obj], 13)
                self.assertAlmostEqual(float(cat.objinfo[i][2]) * arcsec,
                                       truth_data['minorAxis'][i_obj], 13)
                self.assertAlmostEqual(float(cat.objinfo[i][3]) * np.pi/180,
                                       truth_data['positionAngle'][i_obj], 7)
                self.assertAlmostEqual(float(cat.objinfo[i][4]),
                                       truth_data['sindex'][i_obj], 10)

        ######## test that positions are consistent

        for det_name in all_wcs:
            wcs = all_wcs[det_name]
            cat = all_cats[det_name]
            for i in range(cat.nobjects):
                image_pos = cat.image_pos[i]
                world_pos = cat.world_pos[i]
                self.assertLess(world_pos.distanceTo(wcs.toWorld(image_pos)), 0.0005*galsim.arcsec)

        ra_arr = np.array([pos.ra.rad for cat in all_cats.values() for pos in cat.world_pos])
        dec_arr = np.array([pos.dec.rad for cat in all_cats.values() for pos in cat.world_pos])
        # XXX: These are slightly better than the stars actually.  But still max out at a few
        #      arcsec separation differences, which seems like a lot.
        #      cf. Issue #262
        dist = sphere_dist(ra_arr[index2], dec_arr[index2],
                           truth_data['raJ2000'][index1], truth_data['decJ2000'][index1])
        print("sphere dist = ",dist)
        print('max dist = ',np.max(dist))
        print('max dist (arcsec) = ',np.max(dist) * 180/np.pi * 3600)
        np.testing.assert_array_less(dist * 180/np.pi * 3600, 5.)  # largest is 3.3 arcsec
        np.testing.assert_allclose(truth_data['raJ2000'][index1], ra_arr[index2], rtol=1.e-4)
        np.testing.assert_allclose(truth_data['decJ2000'][index1], dec_arr[index2], rtol=1.e-4)

        ######## test that fluxes are correctly calculated

        bp = imsim.RubinBandpass(md['band'])

        for det_name in all_wcs:
            wcs = all_wcs[det_name]
            cat = all_cats[det_name]
            for i in range(cat.nobjects):
                obj = cat.getObj(i)
                i_obj = np.where(truth_data['uniqueId'] == cat.id[i])[0]
                if len(i_obj) == 0: continue
                i_obj = i_obj[0]
                if 0:
                    # XXX: The old test using the sims Sed class.
                    #      Saved in case it becomes reasonable to use it again.
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

                # Instead, this basically recapitulates the calculation in the InstCatalog class.
                magnorm = cat.getMagNorm(i)
                flux = np.exp(-0.9210340371976184 * magnorm)
                rubin_area = 0.25 * np.pi * 649**2 # cm^2
                exptime = 30
                fAt = flux * rubin_area * exptime
                sed = cat.getSED(i) # This applies the redshift internally.
                # TODO: We aren't applying dust terms currently.
                flux = sed.calculateFlux(bp) * fAt
                self.assertAlmostEqual(flux, obj.calculateFlux(bp))

        ######## test that objects are assigned to the right chip in
        ######## gs_object_dict

        if 0:
            # XXX: Skipping this again.
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
        meta = imsim.OpsimDataLoader(self.phosim_file)

        self.assertEqual(meta['gain'], 1)
        self.assertEqual(meta['band'], 'r')
        self.assertEqual(meta['nsnap'], 2)
        self.assertAlmostEqual(meta['exptime'], 30.)
        self.assertEqual(meta['readnoise'], 0)
        self.assertEqual(meta['darkcurrent'], 0)

    def test_validate_phosim_object_list(self):
        "Test the validation of the rows of the phoSimObjects DataFrame."
        cat_file = str(DATA_DIR / 'tiny_instcat.txt')

        camera = imsim.get_camera()
        sensors = [det.getName() for det in camera
                   if det.getType() not in (DetectorType.WAVEFRONT, DetectorType.GUIDER)]

        all_wcs = self.make_wcs(cat_file)
        all_cats = {}
        sed_dir = os.path.join(self.data_dir, 'test_sed_library')
        for det_name in all_wcs:
            cat = all_cats[det_name] = imsim.InstCatalog(cat_file, all_wcs[det_name],
                                                         sed_dir=sed_dir, edge_pix=50)
        id_arr = np.concatenate([cat.id for cat in all_cats.values()])

        # these are the objects that should be omitted
        bad_unique_ids = set([str(x) for x in
                              [34307989098524, 811883374597, 34304522113056]])

        # Note: these used to be skipped for having dust=none in one or both components, but
        # we now allow them and treat them as Av=0, Rv=3.1:
        #    811883374596, 956090392580, 34307989098523,
        # cf. Issue #213

        print('id_arr = ',id_arr)
        print('bad ids = ',bad_unique_ids)
        self.assertEqual(len(id_arr), 21)
        for obj_id in id_arr:
            self.assertNotIn(obj_id, bad_unique_ids)

    def radec_limits_contain(self, min_ra, max_ra, min_dec, max_dec, tests):
        from imsim.instcat import clarify_radec_limits
        min_ra, max_ra, min_dec, max_dec, ref_ra = clarify_radec_limits(
            min_ra*galsim.degrees, max_ra*galsim.degrees,
            min_dec*galsim.degrees, max_dec*galsim.degrees
        )
        for ra, dec, expected in tests:
            self.assertEqual(
                min_ra <= (ra*galsim.degrees).wrap(ref_ra) <= max_ra
                and min_dec <= dec*galsim.degrees <= max_dec,
                expected
            )

    def test_radec_clarification(self):
        # Normal case where min and max are on the same side of zero.
        for min_ra in [0.9, 0.9+360, 0.9-360]:
            for max_ra in [1.1, 1.1+360, 1.1-360]:
                self.radec_limits_contain(
                    min_ra, max_ra, # ra lim
                    0.9, 1.1, # dec lim
                    [
                        (1.0, 1.0, True),  # test ra/dec/iscontained tuples
                        (1.0, 0.0, False),
                        (0.0, 1.0, False),
                        (1.0+360, 1.0, True),
                        (1.0-360, 1.0, True),
                    ]
                )

        # Now with min and max on opposite sides of zero.
        for min_ra in [-0.1, -0.1+360, -0.1-360]:
            for max_ra in [0.1, 0.1+360, 0.1-360]:
                self.radec_limits_contain(
                    min_ra, max_ra, # ra lim
                    -0.1, 0.1, # dec lim
                    [
                        (0.0, 0.0, True),  # test ra/dec/iscontained tuples
                        (1.0, 0.0, False),
                        (0.0, 1.0, False),
                        (-0.001+360, 0.001, True),
                        (-0.001-360, 0.001, True),
                    ]
                )

        # And if we're close to the pole, then just accept any ra/dec.
        self.radec_limits_contain(
            -0.1, 0.1, # ra lim
            -89.7, -89.0, # dec lim
            [
                (0.0, -89.5, True),  # inside limits
                (100.0, -89.5, True),  # nominally outside ralim, but passes
                (0.0, -90.0, True),  # nominally outside declim, but passes
                (100.0, -90.0, True),  # nominally outside both, but passes
                (0.0, -88.0, False),   # too far north
                (100.0, -88.0, False),   # still too far north
            ]
        )


if __name__ == '__main__':
    unittest.main()
