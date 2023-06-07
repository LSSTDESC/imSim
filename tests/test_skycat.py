from pathlib import Path
import unittest
import astropy.time
import numpy as np
import pandas as pd
import galsim
import imsim

DATA_DIR = Path(__file__).parent / 'data'

def load_test_skycat(apply_dc2_dilation=False):
    """Read in the sky catalog data used by the various tests."""
    opsim_db_file = str(DATA_DIR / "small_opsim_9683.db")
    visit = 449053
    det_name = "R22_S11"  # detector 94

    # Make the WCS object.
    opsim_data = imsim.OpsimDataLoader(opsim_db_file, visit=visit)
    boresight = galsim.CelestialCoord(
        ra=opsim_data["fieldRA"] * galsim.degrees,
        dec=opsim_data["fieldDec"] * galsim.degrees,
    )
    rottelpos = opsim_data["rotTelPos"] * galsim.degrees
    obstime = astropy.time.Time(opsim_data["mjd"], format="mjd", scale="tai")
    band = opsim_data["band"]
    bandpass = galsim.Bandpass(
        f"LSST_{band}.dat", wave_type="nm"
    ).withZeropoint("AB")
    wcs_builder = imsim.BatoidWCSBuilder()
    telescope = imsim.load_telescope(f"LSST_{band}.yaml", rotTelPos=rottelpos)
    factory = wcs_builder.makeWCSFactory(boresight, obstime, telescope, bandpass=band)
    wcs = factory.getWCS(wcs_builder.camera[det_name])

    # Create the sky catalog interface object.
    skycat_file = str(DATA_DIR / "sky_cat_9683.yaml")

    return imsim.SkyCatalogInterface(
        skycat_file, wcs, band, obj_types=["galaxy"],
        apply_dc2_dilation=apply_dc2_dilation
    )


class SkyCatalogInterfaceTestCase(unittest.TestCase):
    """
    TestCase class for skyCatalogs interface code.
    """
    @classmethod
    def setUpClass(cls):
        """Prepare a test skycat and read in a parquet file into a pandas dataframe."""
        cls.skycat = load_test_skycat()

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

class EmptySkyCatalogInterfaceTestCase(unittest.TestCase):
    """TestCase for empty catalogs."""

    @classmethod
    def setUpClass(cls):
        """Setup an empty catalog"""
        cls.skycat = load_test_skycat()
        cls.skycat._objects = []

    def test_empty_catalog_raises_on_get_obj(self):
        with self.assertRaises(RuntimeError):
            self.skycat.getObj(0)


class DC2DilationTestCase(unittest.TestCase):
    """TestCase for DC2 dilation applied to galaxy components."""

    @classmethod
    def setUpClass(cls):
        """Setup an empty catalog"""
        cls.skycat = load_test_skycat()
        cls.skycat_dc2 = load_test_skycat(apply_dc2_dilation=True)

    def test_dc2_dilation(self):
        """Test DC2 dilation for a random sample of skyCatalogs galaxies"""
        nsamp = 10
        indexes = np.random.choice(range(len(self.skycat.objects)), nsamp)
        for index in indexes:
            skycat_obj = self.skycat_dc2.objects[index]
            # Compute the DC2 scale factors from the semi-major and
            # semi-minor axes of each component.
            scales = []
            for component in skycat_obj.get_gsobject_components():
                if component == 'knots':
                    component = 'disk'
                a = skycat_obj.get_native_attribute(f'size_{component}_true')
                b = skycat_obj.get_native_attribute(f'size_minor_{component}_true')
                scales.append(np.sqrt(a/b))
            # Check that the scalings were applied to each component
            # by comparing to the ratio of Jacobians.
            obj = self.skycat.getObj(index)
            dc2_obj = self.skycat_dc2.getObj(index)
            for i, scale in enumerate(scales):
                jac_ratio = (dc2_obj.obj_list[i].original.jac
                             /obj.obj_list[i].original.jac)[0, 0]
                np.testing.assert_approx_equal(scale, jac_ratio)


if __name__ == '__main__':
    unittest.main()
