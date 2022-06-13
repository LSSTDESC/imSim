from pathlib import Path
import unittest
import astropy.time
import numpy as np
import pandas as pd
import yaml
import galsim
import imsim

DATA_DIR = Path(__file__).parent / 'data'


class SkyCatalogInterfaceTestCase(unittest.TestCase):
    """
    TestCase class for skyCatalogs interface code.
    """
    @classmethod
    def setUpClass(cls):
        """Read in the sky catalog data used by the various tests."""
        opsim_db_file = str(DATA_DIR / 'small_opsim_9683.db')
        visit = 449053
        det_name = 'R22_S11'   # detector 94

        # Make the WCS object.
        obs_md = imsim.OpsimMetaDict(opsim_db_file, visit=visit)
        boresight = galsim.CelestialCoord(ra=obs_md['fieldRA']*galsim.degrees,
                                          dec=obs_md['fieldDec']*galsim.degrees)
        rottelpos = obs_md['rotTelPos']*galsim.degrees
        obstime = astropy.time.Time(obs_md['mjd'], format='mjd', scale='tai')
        cls.band = obs_md['band']
        cls.bandpass = galsim.Bandpass(f'LSST_{cls.band}.dat',
                                       wave_type='nm').withZeropoint('AB')
        wcs_builder = imsim.BatoidWCSBuilder()
        wcs = wcs_builder.makeWCS(boresight, rottelpos, obstime, det_name,
                                  cls.band)

        # Create the sky catalog interface object.
        skycat_file = str(DATA_DIR / 'sky_cat_9683.yaml')

        cls.skycat = imsim.SkyCatalogInterface(skycat_file, wcs, cls.bandpass,
                                               obj_types=['galaxy'])

        # Read in the data from the parquet file directly for
        # comparison to the outputs from the sky catalog interface.
        cls.df = pd.read_parquet(str(DATA_DIR / 'galaxy_9683_449053_det94.parquet'))

    def setUp(self):
        """Select some objects to test."""
        np.random.seed(42)
        self.indexes = np.random.choice(range(self.skycat.getNObjects()),
                                        size=100)

    def test_getWorldPos(self):
        """
        Check that the sky positions match the entries in the parquet file.
        """
        for index in self.indexes:
            obj = self.skycat.objects[index]
            galaxy_id = obj.get_native_attribute('galaxy_id')
            row = self.df.query(f'galaxy_id == {galaxy_id}').iloc[0]
            pos = self.skycat.getWorldPos(index)
            self.assertEqual(pos.ra.deg, row.ra)
            self.assertEqual(pos.dec.deg, row.dec)

    def test_get_wl_params(self):
        """
        Check that the computed lensing parameters match the entries
        in the parquet file.
        """
        for index in self.indexes:
            obj = self.skycat.objects[index]
            galaxy_id = obj.get_native_attribute('galaxy_id')
            row = self.df.query(f'galaxy_id == {galaxy_id}').iloc[0]
            g1, g2, mu = obj.get_wl_params()
            gamma1 = row['shear_1']
            gamma2 = row['shear_2']
            kappa = row['convergence']
            self.assertAlmostEqual(g1, gamma1/(1. - kappa))
            self.assertAlmostEqual(g2, gamma2/(1. - kappa))
            self.assertAlmostEqual(mu, 1./((1. - kappa)**2 - (gamma1**2 + gamma2**2)))

    def test_get_dust(self):
        """
        Check that the extinction parameters match the entries in
        the parquet file.
        """
        for index in self.indexes:
            obj = self.skycat.objects[index]
            galaxy_id = obj.get_native_attribute('galaxy_id')
            row = self.df.query(f'galaxy_id == {galaxy_id}').iloc[0]
            iAv, iRv, gAv, gRv = obj.get_dust()
            # For galaxies, we use the SED values that have internal
            # extinction included, so should have iAv=0, iRv=1.
            self.assertEqual(iAv, 0)
            self.assertEqual(iRv, 1)
            self.assertEqual(gAv, row['MW_av'])
            self.assertEqual(gRv, row['MW_rv'])

    def test_get_gsobject_components(self):
        """Check some properties of the objects returned by the
           .get_gsobject_components function."""
        for index in self.indexes:
            obj = self.skycat.objects[index]
            galaxy_id = obj.get_native_attribute('galaxy_id')
            row = self.df.query(f'galaxy_id == {galaxy_id}').iloc[0]
            gs_objs = obj.get_gsobject_components(None, None)
            for component, gs_obj in gs_objs.items():
                if component in 'disk bulge':
                    # Check sersic index
                    self.assertEqual(gs_obj.original.n,
                                     row[f'sersic_{component}'])
                elif component == 'knots':
                    # Check number of knots
                    self.assertEqual(gs_obj.original.npoints,
                                     row['n_knots'])

    def test_get_sed_components(self):
        """Check sed components."""
        for index in self.indexes:
            obj = self.skycat.objects[index]
            galaxy_id = obj.get_native_attribute('galaxy_id')
            row = self.df.query(f'galaxy_id == {galaxy_id}').iloc[0]
            seds = obj.get_observer_sed_components()
            for component, sed in seds.items():
                if sed is not None:
                    self.assertEqual(sed.redshift, row['redshift'])

if __name__ == '__main__':
    unittest.main()
