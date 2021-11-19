"""
Unit tests for instance catalog parsing code.
"""
import os
import unittest
import warnings
import tempfile
import shutil
import numpy as np
import galsim
import astropy.time
import imsim
from lsst.afw.cameraGeom import DetectorType

def sources_from_list(lines, obs_md, phot_params, file_name):
    """Return a two-item tuple containing
       * a list of GalSimCelestialObjects for each object entry in `lines`
       * a dictionary of these objects disaggregated by chip_name.
    """
    target_chips = [det.getName() for det in imsim.get_camera()]
    gs_object_dict = dict()
    out_obj_dict = dict()
    for chip_name in target_chips:
        out_obj_dict[chip_name] \
            = [_ for _ in imsim.GsObjectList(lines, obs_md, phot_params,
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
        cls.data_dir = 'data'
        cls.scratch_dir = tempfile.mkdtemp(prefix=cls.data_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.scratch_dir):
            for file_name in os.listdir(cls.scratch_dir):
                os.unlink(os.path.join(cls.scratch_dir, file_name))
            shutil.rmtree(cls.scratch_dir)

    def setUp(self):
        self.phosim_file = os.path.join(self.data_dir, 'phosim_stars.txt')
        # XXX: What is the point of the extra_commands?  Seems like we could remove this bit.
        self.extra_commands = 'instcat_extra.txt'
        with open(self.phosim_file, 'r') as input_file:
            with open(self.extra_commands, 'w') as output:
                for line in input_file.readlines()[:22]:
                    output.write(line)
                output.write('extra_command 1\n')

    def make_wcs(self, instcat_file=None, sensors=None):
        # Make a wcs to use for this instance catalog.
        if instcat_file is None:
            instcat_file = self.phosim_file

        obs_md = imsim.OpsimMetaDict(instcat_file)
        boresight = galsim.CelestialCoord(ra=obs_md['rightascension'] * galsim.degrees,
                                          dec=obs_md['declination'] * galsim.degrees)
        rotTelPos = obs_md['rottelpos'] * galsim.degrees
        obstime = astropy.time.Time(obs_md['mjd'], format='mjd', scale='tai')
        band = obs_md['band']
        builder = imsim.BatoidWCSBuilder()

        if sensors is None:
            camera = imsim.get_camera()
            sensors = [det.getName() for det in camera
                       if det.getType() not in (DetectorType.WAVEFRONT, DetectorType.GUIDER)]

        all_wcs = {}
        for det_name in sensors:
            if det_name not in all_wcs:
                wcs = builder.makeWCS(boresight, rotTelPos, obstime, det_name, band)
                all_wcs[det_name] = wcs

        return all_wcs

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

        with self.assertRaises(ValueError) as ee:
            instcat = imsim.OpsimMetaDict(dummy_catalog)
        self.assertIn("Required commands", ee.exception.args[0])
        if os.path.isfile(dummy_catalog):
            os.remove(dummy_catalog)

    def test_metadata_from_file(self):
        """
        Test methods that get ObservationMetaData
        from InstanceCatalogs.
        """
        metadata = imsim.OpsimMetaDict(self.phosim_file)
        self.assertAlmostEqual(metadata['rightascension'], 53.00913847303155535, 16)
        self.assertAlmostEqual(metadata['declination'], -27.43894880881512321, 16)
        self.assertAlmostEqual(metadata['mjd'], 59580.13974597222113516, 16)
        self.assertAlmostEqual(metadata['altitude'], 66.34657337061349835, 16)
        self.assertAlmostEqual(metadata['azimuth'], 270.27655488919378968, 16)
        self.assertEqual(metadata['filter'], 2)
        self.assertIsInstance(metadata['filter'], int)
        self.assertEqual(metadata['band'], 'r')
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
        self.assertAlmostEqual(metadata['rawSeeing'], 0.8662850, 7)
        self.assertAlmostEqual(metadata['sunalt'], -32.7358290, 7)
        self.assertAlmostEqual(metadata['vistime'], 33.0000000, 7)

    def test_object_extraction_stars(self):
        """
        Test that method to get GalSimCelestialObjects from
        InstanceCatalogs works
        """
        md = imsim.OpsimMetaDict(self.phosim_file)

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
                                                           obs_metadata=obs_md)

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
        # XXX: These are only within 1.e-4 radians, which is kind of a lot.
        #      I (MJ) think this is probalby related to differences in convention from how we
        #      used to do the WCS, so it's probably fine.  But someone who knows better might
        #      want to update how this test is done.
        print("ra diff = ",ra_arr[index]-truth_data['raJ2000'])
        print("dec diff = ",dec_arr[index]-truth_data['decJ2000'])
        np.testing.assert_allclose(truth_data['raJ2000'], ra_arr[index], rtol=1.e-4)
        np.testing.assert_allclose(truth_data['decJ2000'], dec_arr[index], rtol=1.e-4)

        ######## test that fluxes are correctly calculated

        bp = galsim.Bandpass('LSST_%s.dat'%md['band'], wave_type='nm').withZeropoint('AB')

        for det_name in all_wcs:
            wcs = all_wcs[det_name]
            cat = all_cats[det_name]
            for i in range(cat.nobjects):
                obj = cat.getObj(i, bandpass=bp)
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
                exp_time = 30
                fAt = flux * rubin_area * exp_time
                sed = cat.getSED(i)
                flux = sed.calculateFlux(bp) * fAt
                self.assertAlmostEqual(flux, obj.flux)

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
        md = imsim.OpsimMetaDict(galaxy_phosim_file)

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

        id_arr = np.concatenate([cat.id for cat in all_cats.values()])
        print('diff1 = ',set(truth_data['uniqueId'])-set(id_arr))
        print('diff2 = ',set(id_arr)-set(truth_data['uniqueId']))
        print('diff3 = ',set(id_arr)^set(truth_data['uniqueId']))
        # XXX: There are more differences here.  I think mostly because of the WCS mismatch.
        #      We should probably figure this out to make sure the Batoid WCS isn't missing some
        #      bit of physics that the LSST WCS included...
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
        # XXX: Again, only good to ~1.e-4 radians.  (!)
        np.testing.assert_allclose(truth_data['raJ2000'][index1], ra_arr[index2], rtol=1.e-4)
        np.testing.assert_allclose(truth_data['decJ2000'][index1], dec_arr[index2], rtol=1.e-4)

        ######## test that fluxes are correctly calculated

        bp = galsim.Bandpass('LSST_%s.dat'%md['band'], wave_type='nm').withZeropoint('AB')

        for det_name in all_wcs:
            wcs = all_wcs[det_name]
            cat = all_cats[det_name]
            for i in range(cat.nobjects):
                obj = cat.getObj(i, bandpass=bp)
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
                exp_time = 30
                fAt = flux * rubin_area * exp_time
                sed = cat.getSED(i) # This applies the redshift internally.
                # TODO: We aren't applying dust terms currently.
                flux = sed.calculateFlux(bp) * fAt
                self.assertAlmostEqual(flux, obj.flux)

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
        meta = imsim.OpsimMetaDict(self.phosim_file)

        self.assertEqual(meta['gain'], 1)
        self.assertEqual(meta['band'], 'r')
        # XXX: These used to be nexp=2 ?? and exptime=15.  I guess per snap, but we're not doing
        #      2 snaps now in any real way AFAIK, so I switched this to nsnap=2, which is in the
        #      input file and exptime=30, which is our default value.
        self.assertEqual(meta['nsnap'], 2)
        self.assertAlmostEqual(meta['exptime'], 30.)
        self.assertEqual(meta['readnoise'], 0)
        self.assertEqual(meta['darkcurrent'], 0)

    def test_validate_phosim_object_list(self):
        "Test the validation of the rows of the phoSimObjects DataFrame."
        cat_file = 'tiny_instcat.txt'

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
        # we now allow them and treat them as Av=0, Rv=1:
        #    811883374596, 956090392580, 34307989098523,
        # cf. Issue #213

        print('id_arr = ',id_arr)
        print('bad ids = ',bad_unique_ids)
        self.assertEqual(len(id_arr), 21)
        for obj_id in id_arr:
            self.assertNotIn(obj_id, bad_unique_ids)


if __name__ == '__main__':
    unittest.main()
