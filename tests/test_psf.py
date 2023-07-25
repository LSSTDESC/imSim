"""
Unit tests for PSF-related code.
"""
import os
import numpy as np
from pathlib import Path
import glob
import unittest
import time
import imsim
import galsim

DATA_DIR = Path(__file__).parent / 'data'

from imsim_test_helpers import CaptureLog


class PsfTestCase(unittest.TestCase):
    """
    TestCase class for PSF-related functions.
    """
    def setUp(self):
        self.test_dir = 'psf_tests_dir'
        instcat = DATA_DIR / 'tiny_instcat.txt'
        self.opsim_data = imsim.OpsimDataLoader(str(instcat))
        if not os.path.exists(self.test_dir):
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
        config = {
            'DoubleGaussian': {
                'type': 'DoubleGaussianPSF',
                'fwhm': self.opsim_data['FWHMgeom'],
                'pixel_scale': 0.2,
            },
            'Kolmogorov': {
                'type': 'KolmogorovPSF',
                'airmass': self.opsim_data['airmass'],
                'rawSeeing': self.opsim_data['rawSeeing'],
                'band': self.opsim_data['band'],
            },
            'Atmospheric': {
                'type': 'Convolve',
                'items': [
                    { 'type': 'AtmosphericPSF' },
                    { 'type': 'Gaussian', 'fwhm': 0.3 },
                ],
            },
            'input': {
                'atm_psf': {
                    'airmass': self.opsim_data['airmass'],
                    'rawSeeing': self.opsim_data['rawSeeing'],
                    'band':  self.opsim_data['band'],
                    'screen_scale': 6.4,
                    'boresight': {
                        'type': 'RADec',
                        'ra': { 'type': 'Degrees', 'theta': self.opsim_data['rightascension'], },
                        'dec': { 'type': 'Degrees', 'theta': self.opsim_data['declination'], }
                    }
                }
            },
            'image_pos': galsim.PositionD(0,0),  # This would get set appropriately during
                                                 # normal config processing.
            'image' : {
                'random_seed': 1234,
                'wcs': {
                    'type' : 'Tan',
                    'dudx' : 0.2,
                    'dudy' : 0.,
                    'dvdx' : 0.,
                    'dvdy' : 0.2,
                    'ra' : '@input.atm_psf.boresight.ra',
                    'dec' : '@input.atm_psf.boresight.dec',
                }
            }
        }
        galsim.config.ProcessInput(config)
        config['wcs'] = galsim.config.BuildWCS(config['image'], 'wcs', config)
        for psf_name in ("DoubleGaussian", "Kolmogorov", "Atmospheric"):
            psf = galsim.config.BuildGSObject(config, psf_name)
            psf_file = os.path.join(self.test_dir, '{}.pkl'.format(psf_name))
            imsim.save_psf(psf, psf_file)
            psf_retrieved = imsim.load_psf(psf_file)
            self.assertEqual(psf, psf_retrieved)

    def test_atm_psf_save_file(self):
        """Test using save_file in  AtmosphericPSF
        """
        psf_file = os.path.join(self.test_dir, 'save_atm_psf.pkl')
        config = {
            'psf': {
                'type': 'AtmosphericPSF'
            },
            'input': {
                'atm_psf': {
                    'airmass': self.opsim_data['airmass'],
                    'rawSeeing': self.opsim_data['rawSeeing'],
                    'band':  self.opsim_data['band'],
                    'screen_scale': 6.4,
                    'boresight': {
                        'type': 'RADec',
                        'ra': { 'type': 'Degrees', 'theta': self.opsim_data['rightascension'], },
                        'dec': { 'type': 'Degrees', 'theta': self.opsim_data['declination'], }
                    },
                    'save_file': psf_file
                }
            },
            'image_pos': galsim.PositionD(0,0),  # This would get set appropriately during
                                                 # normal config processing.
            'image' : {
                'random_seed': 1234,
                'wcs': {
                    'type' : 'Tan',
                    'dudx' : 0.2,
                    'dudy' : 0.,
                    'dvdx' : 0.,
                    'dvdy' : 0.2,
                    'ra' : '@input.atm_psf.boresight.ra',
                    'dec' : '@input.atm_psf.boresight.dec',
                }
            }
        }

        if os.path.isfile(psf_file):
            os.remove(psf_file)

        config['wcs'] = galsim.config.BuildWCS(config['image'], 'wcs', config)
        config1 = galsim.config.CopyConfig(config)
        config2 = galsim.config.CopyConfig(config)

        # The first time, it will build the psf from scratch and save the screens.
        t0 = time.time()
        galsim.config.ProcessInput(config1)
        t1 = time.time()

        assert os.path.isfile(psf_file)

        # The second time, it will be faster, since it loads the screens from the file.
        t2 = time.time()
        galsim.config.ProcessInput(config2)
        t3 = time.time()

        print('Times = ',t1-t0,t3-t2)
        assert t1-t0 > t3-t2

        # Both input objects will make the same PSF at the same location:
        psf1 = galsim.config.BuildGSObject(config1, 'psf')[0]
        psf2 = galsim.config.BuildGSObject(config2, 'psf')[0]
        assert psf1 == psf2


    def test_atm_psf_config(self):
        """
        Test that the psf delivered by make_psf correctly applies the
        config file value for gaussianFWHM as input to the AtmosphericPSF.
        """
        config = {
            'psf': {
                'type': 'Convolve',
                'items': [
                    { 'type': 'AtmosphericPSF' },
                    { 'type': 'Gaussian', 'fwhm': 0.3 },
                ],
            },
            'input': {
                'atm_psf': {
                    'airmass': self.opsim_data['airmass'],
                    'rawSeeing': self.opsim_data['rawSeeing'],
                    'band':  self.opsim_data['band'],
                    'screen_scale': 6.4,
                    'boresight': {
                        'type': 'RADec',
                        'ra': { 'type': 'Degrees', 'theta': self.opsim_data['rightascension'], },
                        'dec': { 'type': 'Degrees', 'theta': self.opsim_data['declination'], }
                    }
                }
            },
            'image_pos': galsim.PositionD(0,0),  # This would get set appropriately during
                                                 # normal config processing.
            'image' : {
                'random_seed': 1234,
                'wcs': {
                    'type' : 'Tan',
                    'dudx' : 0.2,
                    'dudy' : 0.,
                    'dvdx' : 0.,
                    'dvdy' : 0.2,
                    'ra' : '@input.atm_psf.boresight.ra',
                    'dec' : '@input.atm_psf.boresight.dec',
                }
            }
        }
        config['wcs'] = galsim.config.BuildWCS(config['image'], 'wcs', config)
        config2 = galsim.config.CopyConfig(config)

        galsim.config.ProcessInput(config)

        # PSF without gaussian
        config1 = galsim.config.CopyConfig(config)
        del config1['psf']['items'][1]
        psf_0 = galsim.config.BuildGSObject(config1, 'psf')[0]

        # PSF with gaussian
        psf_c = galsim.config.BuildGSObject(config, 'psf')[0]

        # PSF made manually convolving Atm with Gaussian.
        psf_a = galsim.config.BuildGSObject(config['psf']['items'], 0, config)[0]
        psf = galsim.Convolve(psf_a, galsim.Gaussian(fwhm=0.3))

        self.assertEqual(psf, psf_c)
        self.assertNotEqual(psf, psf_0)

        # random_seed is required
        del config2['image']['random_seed']
        with np.testing.assert_raises(RuntimeError):
            galsim.config.ProcessInput(config2)

        # random_seed can be a list
        # And the first item can be something that needs to be evaluated.
        config2['image']['random_seed'] = [
            '$run_id',
            {'type': 'Sequence', 'first': '$run_id', 'repeat': 189}
        ]
        config2['eval_variables'] = {'irun_id': 1234}
        galsim.config.ProcessInput(config2)
        psf_d = galsim.config.BuildGSObject(config, 'psf')[0]
        assert psf == psf_d


    def test_r0_500(self):
        """Test that inversion of the Tokovinin fitting formula for r0_500 works."""
        np.random.seed(57721)
        for _ in range(10):
            airmass = np.random.uniform(1.001, 1.5)
            rawSeeing = np.random.uniform(0.5, 1.5)
            band = 'ugrizy'[np.random.randint(6)]
            boresight = galsim.CelestialCoord(0 * galsim.radians, 0 * galsim.radians)
            rng = galsim.BaseDeviate(np.random.randint(2**32))
            atmPSF = imsim.AtmosphericPSF(airmass, rawSeeing, band, boresight, rng, screen_size=6.4)

            wlen = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
            targetFWHM = rawSeeing * airmass**0.6 * (wlen/500)**(-0.3)

            r0_500 = atmPSF.atm.r0_500_effective
            L0 = atmPSF.atm[0].L0
            vkFWHM = imsim.AtmosphericPSF._vkSeeing(r0_500, wlen, L0)

            np.testing.assert_allclose(targetFWHM, vkFWHM, atol=1e-3, rtol=0)


    def test_chromatic_psf(self):
        import copy

        # Make some monochromatic SEDs for testing.
        sed400 = galsim.SED(
            galsim.LookupTable(
                [200,399,400,401,1100], [0,0,1,0,0], interpolant='linear'
            ),
            'nm',
            'fphotons'
        )
        sed900 = galsim.SED(
            galsim.LookupTable(
                [200,899,900,901,1100], [0,0,1,0,0], interpolant='linear'
            ),
            'nm',
            'fphotons'
        )

        bandpass = galsim.Bandpass(
            galsim.LookupTable(
                [200,201,1099,1100],
                [0,1,1,0],
                interpolant='linear'
            ),
            wave_type='nm'
        )

        template = {
            'psf': {
                'type': 'AtmosphericPSF'
            },
            'input': {
                'atm_psf': {
                    'airmass': 1.0,
                    'rawSeeing': 1.0,
                    'band':  'r',
                    'screen_size': 51.2,
                    'boresight': {
                        'type': 'RADec',
                        'ra': { 'type': 'Degrees', 'theta': 0.0},
                        'dec': { 'type': 'Degrees', 'theta': 0.0}
                    },
                    'exponent': None,
                    'exptime': 600.0,  # Lots of mixing.
                    'doOpt': False,
                    '_no2k': True,  # turn off second-kick since it's achromatic
                }
            },
            'image_pos': galsim.PositionD(0,0),  # This would get set appropriately during
                                                 # normal config processing.
            'image' : {
                'random_seed': 1234,
                'draw_method' : 'phot',
                'wcs': {
                    'type' : 'Tan',
                    'dudx' : 0.05,
                    'dudy' : 0.,
                    'dvdx' : 0.,
                    'dvdy' : 0.05,
                    'ra' : '@input.atm_psf.boresight.ra',
                    'dec' : '@input.atm_psf.boresight.dec',
                },
                'bandpass': bandpass,
                'size': 127
            },
            'gal': {
                'type': 'DeltaFunction',
                'flux': 1e5,
                'sed': None
            }
        }

        for exponent in [-0.2, -0.3]:
            config400 = copy.deepcopy(template)
            config400['input']['atm_psf']['exponent'] = exponent
            config400['gal']['sed'] = sed400
            img400 = galsim.config.BuildImage(config400)
            sigma400 = galsim.hsm.FindAdaptiveMom(img400).moments_sigma

            config900 = copy.deepcopy(template)
            config900['input']['atm_psf']['exponent'] = exponent
            config900['gal']['sed'] = sed900
            img900 = galsim.config.BuildImage(config900)
            sigma900 = galsim.hsm.FindAdaptiveMom(img900).moments_sigma

            np.testing.assert_allclose(
                sigma400/sigma900,
                (400/900)**exponent,
                rtol=0.001  # 0.1% error on value of around ~1.2 to 1.3
            )

    def test_atm_psf_fft(self):
        """
        Test using an atmospheric PSF with a star bright enough to switch over to FFT.
        """
        config = {
            'psf': {
                'type': 'AtmosphericPSF'
            },
            'gal': {
                'type': 'DeltaFunction',
                'flux': 1e5,
                'sed': {
                    'file_name': 'vega.txt',
                    'wave_type': 'nm',
                    'flux_type': 'fnu',
                },
            },
            'input': {
                'atm_psf': {
                    'airmass': self.opsim_data['airmass'],
                    'rawSeeing': self.opsim_data['rawSeeing'],
                    'band':  self.opsim_data['band'],
                    'screen_size': 409.6,
                    'boresight': {
                        'type': 'RADec',
                        'ra': { 'type': 'Degrees', 'theta': self.opsim_data['rightascension'], },
                        'dec': { 'type': 'Degrees', 'theta': self.opsim_data['declination'], }
                    }
                }
            },
            'stamp': {
                'type': 'LSST_Silicon',

                'fft_sb_thresh': 2.e5,   # When to switch to fft and a simpler PSF and skip silicon
                'max_flux_simple': 100,  # When to switch to simple SED

                'airmass': self.opsim_data['airmass'],
                'rawSeeing': self.opsim_data['rawSeeing'],
                'band':  self.opsim_data['band'],

                'diffraction_psf': {
                    'enabled': False,
                    'exptime': 30,
                    'azimuth': "0 deg",
                    'altitude': "60 deg",
                    'rotTelPos': "0 deg",
                },
                'det_name': 'R22_S11',
                'world_pos':  {
                    'type': 'RADec',
                    'ra' : '@input.atm_psf.boresight.ra',
                    'dec' : '@input.atm_psf.boresight.dec',
                },
            },
            'image' : {
                'size': 64,
                'random_seed': 1234,
                'wcs': {
                    'type' : 'Tan',
                    'dudx' : 0.2,
                    'dudy' : 0.,
                    'dvdx' : 0.,
                    'dvdy' : 0.2,
                    'ra' : '@input.atm_psf.boresight.ra',
                    'dec' : '@input.atm_psf.boresight.dec',
                },
                'bandpass': {
                    'file_name': 'LSST_r.dat',
                    'wave_type': 'nm',
                    'thin': 1.e-4,
                },
                'noise': {'type': 'Poisson'},
            }
        }

        # First make a reference image, using photon shooting
        config1 = galsim.config.CopyConfig(config)
        ref_img = galsim.config.BuildImage(config1)

        # Repeat with an object bright enough to switch to FFT
        config['gal']['flux'] = 1.e8
        with CaptureLog() as cl:
            img = galsim.config.BuildImage(config, logger=cl.logger)
        #print(cl.output)
        assert 'Check if we should switch to FFT' in cl.output
        assert 'Yes. Use FFT for this object.' in cl.output

        print('Peak of reference PSF (flux=1.e5): ',ref_img.array.max())
        print('Peak of FFT PSF (flux=1.e8): ',img.array.max())
        print('FWHM of reference PSF: ',ref_img.view(scale=0.2).calculateFWHM())
        print('FWHM of FFT PSF: ',img.view(scale=0.2).calculateFWHM())
        print('Rmom of reference PSF: ',ref_img.view(scale=0.2).calculateMomentRadius())
        print('Rmom of FFT PSF: ',img.view(scale=0.2).calculateMomentRadius())

        # The FFT image is about 10^3 x brighter than the reference image.
        # Scale it down to make it easier to compare to the reference image.
        img /= 1.e3

        # Peaks should now be similar
        np.testing.assert_allclose(ref_img.array.max(), img.array.max(), rtol=0.05)

        # The sizes should also be pretty close
        np.testing.assert_allclose(ref_img.view(scale=0.2).calculateFWHM(),
                                   img.view(scale=0.2).calculateFWHM(), rtol=0.05)
        np.testing.assert_allclose(ref_img.view(scale=0.2).calculateMomentRadius(),
                                   img.view(scale=0.2).calculateMomentRadius(), rtol=0.1)

        # Inded the whole image should be similar, but this is pretty noisy,
        # so we need some loose tolerances for this one.
        np.testing.assert_allclose(ref_img.array, img.array, rtol=0.15, atol=50)


if __name__ == '__main__':
    unittest.main()
