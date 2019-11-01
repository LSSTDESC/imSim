"""
Unit tests for PSF-related code.
"""
import os
import glob
import unittest
import desc.imsim


class PsfTestCase(unittest.TestCase):
    """
    TestCase class for PSF-related functions.
    """
    def setUp(self):
        self.test_dir = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                                     'psf_tests_dir')
        instcat = os.path.join(os.environ['IMSIM_DIR'], 'tests',
                               'tiny_instcat.txt')
        self.obs_md, _, _ = desc.imsim.parsePhoSimInstanceFile(instcat, ())
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
        for psf_name in ("DoubleGaussian", "Kolmogorov", "Atmospheric"):
            if psf_name == 'Atmospheric':
                screen_file = os.path.join(self.test_dir, "screens.pkl")
            else:
                screen_file = None
            psf = desc.imsim.make_psf(psf_name, self.obs_md, screen_scale=6.4)
            psf_file = os.path.join(self.test_dir, '{}.pkl'.format(psf_name))
            desc.imsim.save_psf(psf, psf_file, screen_file)
            psf_retrieved = desc.imsim.load_psf(psf_file, screen_file)
            self.assertEqual(psf, psf_retrieved)


    def test_atm_psf_config(self):
        """
        Test that the psf delivered by make_psf correctly applies the
        config file value for gaussianFWHM as input to the AtmosphericPSF.
        """
        config = desc.imsim.get_config()
        psf_name = 'Atmospheric'
        screen_scale = 6.4
        # PSF with gaussianFWHM explicitly set to zero.
        psf_0 = desc.imsim.make_psf(psf_name, self.obs_md,
                                    gaussianFWHM=0.,
                                    screen_scale=screen_scale)

        # PSF with gaussianFWHM explicitly set to config file value.
        psf_c = desc.imsim.make_psf(psf_name, self.obs_md,
                                    gaussianFWHM=config['psf']['gaussianFWHM'],
                                    screen_scale=screen_scale)

        # PSF using the config file value implicitly.
        psf = desc.imsim.make_psf(psf_name, self.obs_md,
                                  screen_scale=screen_scale)

        self.assertEqual(psf, psf_c)
        self.assertNotEqual(psf, psf_0)


if __name__ == '__main__':
    unittest.main()
