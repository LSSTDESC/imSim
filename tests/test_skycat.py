import unittest
import astropy.time
import numpy as np
import galsim
import imsim
import pandas as pd

np.random.seed(42)

class SkyCatalogInterfaceTestCase(unittest.TestCase):
    """
    TestCase class for skyCatalogs interface code.
    """
    @classmethod
    def setUpClass(cls):
        """Read in the sky catalog data used by the various tests."""
        opsim_db_file = 'data/small_opsim_9683.db'
        visit = 449053
        det_name = 'R22_S11'   # detector 94

        # Make the WCS object.
        obs_md = imsim.OpsimMetaDict(opsim_db_file, visit=visit)
        boresight = galsim.CelestialCoord(ra=obs_md['fieldRA']*galsim.degrees,
                                          dec=obs_md['fieldDec']*galsim.degrees)
        rottelpos = obs_md['rotTelPos']*galsim.degrees
        obstime = astropy.time.Time(obs_md['mjd'], format='mjd', scale='tai')
        band = obs_md['band']
        cls.bandpass = galsim.Bandpass(f'LSST_{band}.dat',
                                       wave_type='nm').withZeropoint('AB')
        wcs_builder = imsim.BatoidWCSBuilder()
        wcs = wcs_builder.makeWCS(boresight, rottelpos, obstime, det_name, band)

        # Create the sky catalog interface object.
        skycat_file = 'data/sky_cat_9683.yaml'
        cls.skycat = imsim.SkyCatalogInterface(skycat_file, wcs, band,
                                               obj_types=['galaxy'])

        # Read in the data from the parquet file directly for
        # comparison to the outputs from the sky catalog interface.
        cls.df = pd.read_parquet('data/galaxy_9683_449053_det94.parquet')

    def setUp(self):
        """Select some objects to test."""
        self.indexes = np.random.choice(range(self.skycat.getNObjects()),
                                        size=10)

    def test_getWorldPos(self):
        """Check that the sky positions match the entries in the parquet file."""
        for index in self.indexes:
            obj = self.skycat.objects[index]
            galaxy_id = obj.get_native_attribute('galaxy_id')
            row = self.df.query(f'galaxy_id == {galaxy_id}').iloc[0]
            pos = self.skycat.getWorldPos(index)
            self.assertEqual(pos.ra.deg, row.ra)
            self.assertEqual(pos.dec.deg, row.dec)

    def test_getLens(self):
        """Check that the computed lensing parameters match the entries in the parquet file."""
        for index in self.indexes:
            obj = self.skycat.objects[index]
            galaxy_id = obj.get_native_attribute('galaxy_id')
            row = self.df.query(f'galaxy_id == {galaxy_id}').iloc[0]
            g1, g2, mu = self.skycat.getLens(obj)
            gamma1 = row['shear_1']
            gamma2 = row['shear_2']
            kappa = row['convergence']
            self.assertAlmostEqual(g1, gamma1/(1. - kappa))
            self.assertAlmostEqual(g2, gamma2/(1. - kappa))
            self.assertAlmostEqual(mu, 1./((1. - kappa)**2 - (gamma1**2 + gamma2**2)))

    def test_getDust(self):
        """Check that the extinction parameters match the entries in the parquet file."""
        for index in self.indexes:
            obj = self.skycat.objects[index]
            galaxy_id = obj.get_native_attribute('galaxy_id')
            row = self.df.query(f'galaxy_id == {galaxy_id}').iloc[0]
            iAv, iRv, gAv, gRv = self.skycat.getDust(obj)
            # For galaxies, we use the SED values that have internal
            # extinction included, so should have iAv=0, iRv=1.
            self.assertEqual(iAv, 0)
            self.assertEqual(iRv, 1)
            colname = f'MW_av_lsst_{self.skycat.band}'
            self.assertEqual(gAv, row[colname])
            self.assertEqual(gRv, row['MW_rv'])

    def test_get_gsobject(self):
        """Check some properties of the objects returned by the .get_gsobject function."""
        for index in self.indexes:
            obj = self.skycat.objects[index]
            galaxy_id = obj.get_native_attribute('galaxy_id')
            row = self.df.query(f'galaxy_id == {galaxy_id}').iloc[0]
            for component in obj.subcomponents:
                gs_obj_info = self.skycat.get_gsobject(
                    obj, component, self.bandpass, None, None, 30.)
                if gs_obj_info is None:
                    sed, magnorm = self.skycat.getSED_info(obj, component)
                    if sed is not None:
                        # We are skipping objects with magnorm > 50.
                        self.assertGreater(magnorm, 50)
                    continue
                gs_obj = gs_obj_info.gs_obj
                if component in 'disk bulge':
                    # Check sersic index
                    self.assertEqual(gs_obj.original.n,
                                     row[f'sersic_{component}'])
                elif component == 'knots':
                    # Check number of knots
                    self.assertEqual(gs_obj.original.npoints, row['n_knots'])
                # Check redshift
                sed = gs_obj_info.sed
                self.assertEqual(sed.redshift, row['redshift'])


if __name__ == '__main__':
    unittest.main()
