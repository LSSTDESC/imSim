"""
Unit tests for PSF-related code.
"""
import os
import glob
import unittest
import desc.imsim


def are_psfs_equal(psf1, psf2):
    """
    Test that two PSFs are equal.  For PSFs implemented in
    sims_GalSimInterface, it is sufficient to compare the
    ._cached_psf attributes.  For the AtmosphericPSF, only certain
    attributes are meaningful to compare. See issue #117.
    """
    if type(psf1) != type(psf2):
        return False
    if not isinstance(psf1, desc.imsim.atmPSF.AtmosphericPSF):
        # Compare cached galsim objects.
        return psf1._cached_psf == psf2._cached_psf
    # See issue #117 for an explanation of these comparisons:
    return (psf1.atm[:-1] == psf2.atm[:-1]) and (psf1.aper == psf2.aper)


class PsfTestCase(unittest.TestCase):
    """
    TestCase class for PSF-related functions.
    """
    def setUp(self):
        self.test_dir = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                                     'psf_tests_dir')
        os.makedirs(self.test_dir)

    def tearDown(self):
        for item in glob.glob(os.path.join(self.test_dir, '*')):
            os.remove(item)
        os.rmdir(self.test_dir)

    def test_save_and_load_psf(self):
        """
        Test that the different imSim PSFs are saved and retrieved
        correctly.
        """
        instcat = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                               'tiny_instcat.txt')
        obs_md, _, _ = desc.imsim.parsePhoSimInstanceFile(instcat)
        for psf_name in ("DoubleGaussian", "Kolmogorov", "Atmospheric"):
            psf = desc.imsim.make_psf(psf_name, obs_md, screen_scale=0.4)
            psf_file = os.path.join(self.test_dir, '{}.pkl'.format(psf_name))
            desc.imsim.save_psf(psf, psf_file)
            psf_retrieved = desc.imsim.load_psf(psf_file)
            self.assertTrue(are_psfs_equal(psf, psf_retrieved))


if __name__ == '__main__':
    unittest.main()
