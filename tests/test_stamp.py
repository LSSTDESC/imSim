from textwrap import dedent
from unittest import mock
from pathlib import Path
import os
import yaml
import galsim
import imsim
from test_photon_ops import create_test_icrf_to_field

METHOD_PHOT = "phot"
METHOD_FFT = "fft"

DATA_DIR = Path(__file__).parent / 'data'

def create_test_config():
    wcs = galsim.PixelScale(0.2)
    config = {
        "input": {
            "telescope": {
                "file_name": "LSST_r.yaml",
            }
        },
        # Note: This _icrf_to_field thing means that all the RubinOptics, RubinDiffraction,
        #       and RubinDiffractionOptics photon ops implicitly *require* the WCS be
        #       a BatoidWCS.  This doesn't seem great, but I'm not fixing it now.
        "_icrf_to_field": create_test_icrf_to_field(
            galsim.CelestialCoord(
                1.1047934165124105 * galsim.radians, -0.5261230452954583 * galsim.radians
            ),
            "R22_S11",
        ),
        "image_pos": galsim.PositionD(20,20),
        "sky_pos": galsim.CelestialCoord(
            1.1056660811384078 * galsim.radians, -0.5253441048502933 * galsim.radians
        ),
        "stamp": {
            'type': 'LSST_Silicon',
            'photon_ops': [{
                "type": "RubinDiffractionOptics",
                "altitude": 80 * galsim.degrees,
                "azimuth": 0 * galsim.degrees,
                "latitude": -30.24463 * galsim.degrees,
                "boresight": galsim.CelestialCoord(0*galsim.degrees, 0*galsim.degrees),
                "camera": 'LsstCam',
                "det_name": 'R22_S11',
            }],
            'fft_photon_ops': [{
                "type": "RubinDiffraction",
                "altitude": 80 * galsim.degrees,
                "azimuth": 0 * galsim.degrees,
                "latitude": -30.24463 * galsim.degrees,
            }],
        },
        "bandpass": galsim.Bandpass('LSST_r.dat', wave_type='nm'),
        "wcs": wcs,
        "current_image": galsim.Image(1024, 1024, wcs=wcs)
    }
    galsim.config.ProcessInput(config)
    galsim.config.SetupInputsForImage(config, None)
    return config


def create_test_lsst_silicon(faint: bool):
    lsst_silicon = imsim.LSST_SiliconBuilder()
    lsst_silicon.nominal_flux = 100 if not faint else 10
    lsst_silicon.phot_flux = lsst_silicon.nominal_flux
    lsst_silicon.fft_flux = lsst_silicon.nominal_flux
    lsst_silicon.rng = galsim.BaseDeviate(1234)
    lsst_silicon.diffraction_fft = None
    return lsst_silicon


def test_lsst_silicon_builder_passes_correct_photon_ops_to_drawImage() -> None:
    """LSST_SiliconBuilder.draw passes the correct list of photon_ops
    to prof.drawImage."""
    lsst_silicon = create_test_lsst_silicon(faint=False)
    image = galsim.Image(ncol=256, nrow=256)
    offset = galsim.PositionD(0,0)
    config = create_test_config()
    prof = galsim.Gaussian(sigma=2) * galsim.SED('vega.txt', 'nm', 'flambda')
    logger = galsim.config.LoggerWrapper(None)

    expected_phot_args = {
        "rng": lsst_silicon.rng,
        "maxN": int(1e6),
        "n_photons": lsst_silicon.phot_flux,
        "image": image,
        "sensor": None,
        "add_to_image": True,
        "poisson_flux": False,
    }
    expected_fft_args = {
        "maxN": int(1e6),
        "rng": lsst_silicon.rng,
        "n_subsample": 1,
        "image": mock.ANY,  # For fft, the image gets modified from the original
    }
    if galsim.__version_info__ < (2,5):
        phot_type = 'galsim.ChromaticTransformation'
    else:
        # In GalSim 2.5+, this is now a SimpleChromaticTransformation
        phot_type = 'galsim.SimpleChromaticTransformation'
    for method, expected_specific_args, prof_type, op_type in (
        ("phot", expected_phot_args, phot_type, imsim.RubinDiffractionOptics),
        ("fft", expected_fft_args, 'galsim.ChromaticConvolution', imsim.RubinDiffraction),
    ):
        # mock.patch basically wraps these functions so we can access how they were called
        # and what their return values were.
        image.added_flux = lsst_silicon.phot_flux
        with mock.patch(prof_type+'.drawImage', return_value=image) as mock_drawImage:
            lsst_silicon.draw(
                prof,
                image,
                method,
                offset,
                config=config["stamp"],
                base=config,
                logger=logger,
            )
            mock_drawImage.assert_called_once_with(
                config['bandpass'],
                method=method,
                offset=offset,
                photon_ops=mock.ANY,  # Check the items in this below.
                **expected_specific_args
            )
            called_photon_ops = mock_drawImage.call_args.kwargs['photon_ops']
            assert len(called_photon_ops) == 1
            assert type(called_photon_ops[0]) == op_type

def test_stamp_builder_works_without_photon_ops_or_faint() -> None:
    """Here, we test that if LSST_SiliconBuilder.drawImage passes empty photon_ops,
    when faint is True or photon ops for the used method are empty.
    """

    image = galsim.Image(ncol=256, nrow=256)
    offset = galsim.PositionD(0,0)
    config = create_test_config()
    prof = galsim.Gaussian(sigma=2) * galsim.SED('vega.txt', 'nm', 'flambda')
    logger = galsim.config.LoggerWrapper(None)

    expected_phot_args = {
        "rng": mock.ANY,
        "maxN": int(1e6),
        "n_photons": None,  # Rewrite this below.
        "image": image,
        "sensor": None,
        "photon_ops": [],
        "add_to_image": True,
        "poisson_flux": False,
    }
    expected_fft_args = {
        "image": mock.ANY
    }
    if galsim.__version_info__ < (2,5):
        phot_type = 'galsim.ChromaticTransformation'
    else:
        phot_type = 'galsim.SimpleChromaticTransformation'
    for method, expected_specific_args, photon_ops_key, prof_type in (
        ("phot", expected_phot_args, 'photon_ops', phot_type),
        ("fft", expected_fft_args, 'fft_photon_ops', 'galsim.ChromaticConvolution'),
    ):
        for faint in [True, False]:
            use_config = galsim.config.CopyConfig(config)
            lsst_silicon = create_test_lsst_silicon(faint=faint)
            if not faint:
                # When not faint, check that photon_ops ends up empty when none are listed in
                # the config (and there are no PSFs).
                del use_config['stamp'][photon_ops_key]
            expected_phot_args['n_photons'] = lsst_silicon.phot_flux

            image.added_flux = lsst_silicon.phot_flux
            with mock.patch(prof_type+'.drawImage', return_value=image) as mock_drawImage:
                lsst_silicon.draw(
                    prof,
                    image,
                    method,
                    offset,
                    config=use_config["stamp"],
                    base=use_config,
                    logger=logger,
                )
                mock_drawImage.assert_called_once_with(
                    config['bandpass'],
                    method=method,
                    offset=offset,
                    **expected_specific_args
                )

def test_stamp_sizes():
    """Test that the stamp sizes come out reasonably for a range of object types.
    """
    # This is basically imsim-config-skycat, but a few adjustments for speed.
    config = yaml.safe_load(dedent("""
        modules:
            - imsim
        template: imsim-config-skycat
        input.atm_psf.screen_size: 40.96
        input.checkpoint: ""
        input.sky_catalog.file_name: data/sky_cat_9683.yaml
        input.opsim_data.file_name: data/small_opsim_9683.db
        input.opsim_data.visit: 449053
        input.tree_rings.only_dets: [R22_S11]
        image.random_seed: 42
        output.det_num.first: 94
        eval_variables.sdet_name: R22_S11
        image.sensor: ""
        psf.items.0:
            type: Kolmogorov
            fwhm: '@stamp.rawSeeing'
        stamp.diffraction_fft: ""
        stamp.photon_ops: ""
        """))
    # If tests aren't run from test directory, need this:
    config['input.sky_catalog.file_name'] = str(DATA_DIR / "sky_cat_9683.yaml")
    config['input.opsim_data.file_name'] = str(DATA_DIR / "small_opsim_9683.db")
    os.environ['SIMS_SED_LIBRARY_DIR'] = str(DATA_DIR / "test_sed_library")

    galsim.config.ProcessAllTemplates(config)

    # Hotfix indeterminism in skyCatalogs 1.6.0. 
    # cf. https://github.com/LSSTDESC/skyCatalogs/pull/62
    # Remove this bit once we are dependent on a version that includes the above PR.
    orig_toplevel_only = imsim.skycat.skyCatalogs.SkyCatalog.toplevel_only
    def new_toplevel_only(self, object_types):
        return sorted(orig_toplevel_only(self, object_types))
    imsim.skycat.skyCatalogs.SkyCatalog.toplevel_only = new_toplevel_only

    # Run through some setup things that BuildImage normally does for us.
    logger = galsim.config.LoggerWrapper(None)
    builder = galsim.config.valid_image_types['LSST_Image']

    # Note: the safe_only=True call is only required with GalSim 2.4.
    # It's a workaround for a bug that we fixed in 2.5.
    if galsim.__version_info__ < (2,5):
        galsim.config.ProcessInput(config, logger, safe_only=True)
    galsim.config.ProcessInput(config, logger)
    galsim.config.SetupConfigImageNum(config, 0, 0, logger)
    xsize, ysize = builder.setup(config['image'], config, 0, 0, galsim.config.image_ignore, logger)
    galsim.config.SetupConfigImageSize(config, xsize, ysize, logger)
    galsim.config.SetupInputsForImage(config, logger)
    galsim.config.SetupExtraOutputsForImage(config, logger)
    config['bandpass'] = builder.buildBandpass(config['image'], config, 0, 0, logger)
    config['sensor'] = builder.buildSensor(config['image'], config, 0, 0, logger)
    if galsim.__version_info__ < (2,5):
        config['current_image'] = galsim.Image(config['image_xsize'], config['image_ysize'],
                                               wcs=config['wcs'])

    nobj = builder.getNObj(config['image'], config, 0)
    print('nobj = ',nobj)

    # Test a few different kinds of objects to make sure they return a reasonable stamp size.
    # This is written as a regression test, given the current state of the code, but the
    # point is that these don't become massively different from any code updates.
    # Small changes in the exact stamp size are probably acceptable if something changes in
    # the stamp building code or skycatalog source processing.
    # We pick out specific obj_nums with known properties in this opsim db.

    # 1. Faint star
    # There aren't any very faint stars in this db file.  The faintest is around flux=2000.
    # But that's easily small enough to hit the minimum stamp size.
    obj_num = 2619
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (40,40)
    assert 2000 < image.array.sum() < 2300  # 2173

    # 2. 10x brighter star.  Still minimum stamp size.
    obj_num = 2699
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (40,40)
    assert 24000 < image.array.sum() < 27000  # 25593

    # 3. 10x brighter star.  Needs bigger stamp.
    obj_num = 2746
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (106,106)
    assert 250_000 < image.array.sum() < 280_000  # 264459

    # 4. 10x brighter star.  (Uses photon shooting, but checks the max sb.)
    obj_num = 2611
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (350,350)
    assert 2_900_000 < image.array.sum() < 3_200_000  # 3086402

    # 5. Extremely bright star.  Maxes out size at _Nmax.  (And uses fft.)
    obj_num = 2697
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (4096,4096)
    assert 470_000_000 < image.array.sum() < 500_000_000  # 481466430

    # 6. Faint galaxy.
    # Again, this db doesn't have extremely faint objects.  The minimum seems to be 47.7.
    # However, this particular galaxy has a bulge with half-light-radius = 1.3,
    # which means in needs a pretty big stamp.
    obj_num = 2538
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (328,328)
    assert 20 < image.array.sum() < 50  # 38

    # None of the objects trigger the tiny flux option, but for extremely faint things (flux<10),
    # we use a fixed size (32,32) stamp.  Test this by mocking the tiny_flux value.
    with mock.patch('imsim.LSST_SiliconBuilder._tiny_flux', 60):
        image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
        print(obj_num, image.center, image.array.shape, image.array.sum())
        assert image.array.shape == (32,32)
        assert 10 < image.array.sum() < 40  # 29

    # 7. Small, faint galaxy.
    # This one is also a bulge + disk + knots galaxy, but it has a very small half-light-radius.
    # hlr = 0.07.  So it ends up being the smallest stamp size.
    obj_num = 860
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (26,26)
    assert 1100 < image.array.sum() < 1400  # 1252

    # 8. Bright, small galaxy.
    # For bright galaxies, we check if we might need to scale back the size.
    # This one triggers the check, but not a reduction in size.
    obj_num = 12
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (80,80)
    assert 690_000 < image.array.sum() < 720_000  # 700510

    # 9. Bright, big galaxy
    # None of the galaxies are big enough to trigger the reduction.  This is the largest.
    obj_num = 75
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (1978,1978)
    assert 450_000 < image.array.sum() < 480_000  # 460282

    # We can trigger the reduction by mocking the _Nmax value to a lower value.
    # This triggers a recalculation with a different calculation, but that ends up less than
    # _Nmax, so it doesn't end up maxing at _Nmax.
    with mock.patch('imsim.LSST_SiliconBuilder._Nmax', 1024):
        image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
        print(obj_num, image.center, image.array.shape, image.array.sum())
        assert image.array.shape == (104,104)
        assert 270_000 < image.array.sum() < 300_000  # 280812

    # With even smaller Nmax, it triggers a second recalculation, which again ends up
    # less than Nmax without pinning at Nmax.
    with mock.patch('imsim.LSST_SiliconBuilder._Nmax', 100):
        image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
        print(obj_num, image.center, image.array.shape, image.array.sum())
        assert image.array.shape == (69,69)
        assert 210_000 < image.array.sum() < 240_000  # 227970

    # Finally, with an even smaller Nmax, it will max out at Nmax.
    with mock.patch('imsim.LSST_SiliconBuilder._Nmax', 60):
        image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
        print(obj_num, image.center, image.array.shape, image.array.sum())
        assert image.array.shape == (60,60)
        assert 200_000 < image.array.sum() < 230_000  # 210266

    # 10. Force stamp size in config.
    # There are two ways for the user to force the size of the stamp.
    # First, set stamp.size in the config file.
    config['stamp']['size'] = 64
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (64,64)
    assert 200_000 < image.array.sum() < 230_000  # 218505

    # There is also a code path where the xsize,ysize is dictated by the calling routine.
    del config['stamp']['size']
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False,
                                        xsize=128, ysize=128)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (128,128)
    assert 290_000 < image.array.sum() < 320_000  # 306675

def test_faint_high_redshift_stamp():
    """Test the stamp size calculation in u-band for a faint cosmoDC2
    galaxy with redshift > 2.71 so that the SED is zero-valued at
    bandpass.effective_wavelength.
    """
    # This is basically imsim-config-skycat, but a few adjustments for speed.
    config = yaml.safe_load(dedent("""
        modules:
            - imsim
        template: imsim-config-skycat
        input.atm_psf.screen_size: 40.96
        input.checkpoint: ""
        input.sky_catalog.file_name: data/sky_cat_9683.yaml
        input.sky_catalog.obj_types: [galaxy]
        input.sky_catalog.band: u
        input.opsim_data.file_name: data/small_opsim_9683.db
        input.opsim_data.visit: 449053
        input.tree_rings.only_dets: [R22_S11]
        image.random_seed: 42
        output.det_num.first: 94
        eval_variables.sdet_name: R22_S11
        eval_variables.sband: u
        image.sensor: ""
        psf.items.0:
            type: Kolmogorov
            fwhm: '@stamp.rawSeeing'
        stamp.diffraction_fft: ""
        stamp.photon_ops: ""
        """))
    # If tests aren't run from test directory, need this:
    config['input.sky_catalog.file_name'] = str(DATA_DIR / "sky_cat_9683.yaml")
    config['input.opsim_data.file_name'] = str(DATA_DIR / "small_opsim_9683.db")
    os.environ['SIMS_SED_LIBRARY_DIR'] = str(DATA_DIR / "test_sed_library")

    galsim.config.ProcessAllTemplates(config)

    # Hotfix indeterminism in skyCatalogs 1.6.0.
    # cf. https://github.com/LSSTDESC/skyCatalogs/pull/62
    # Remove this bit once we are dependent on a version that includes the above PR.
    orig_toplevel_only = imsim.skycat.skyCatalogs.SkyCatalog.toplevel_only
    def new_toplevel_only(self, object_types):
        return sorted(orig_toplevel_only(self, object_types))
    imsim.skycat.skyCatalogs.SkyCatalog.toplevel_only = new_toplevel_only

    # Run through some setup things that BuildImage normally does for us.
    logger = galsim.config.LoggerWrapper(None)
    builder = galsim.config.valid_image_types['LSST_Image']

    # Note: the safe_only=True call is only required with GalSim 2.4.
    # It's a workaround for a bug that we fixed in 2.5.
    if galsim.__version_info__ < (2,5):
        galsim.config.ProcessInput(config, logger, safe_only=True)
    galsim.config.ProcessInput(config, logger)
    galsim.config.SetupConfigImageNum(config, 0, 0, logger)
    xsize, ysize = builder.setup(config['image'], config, 0, 0, galsim.config.image_ignore, logger)
    galsim.config.SetupConfigImageSize(config, xsize, ysize, logger)
    galsim.config.SetupInputsForImage(config, logger)
    galsim.config.SetupExtraOutputsForImage(config, logger)
    config['bandpass'] = builder.buildBandpass(config['image'], config, 0, 0, logger)
    config['sensor'] = builder.buildSensor(config['image'], config, 0, 0, logger)
    if galsim.__version_info__ < (2,5):
        config['current_image'] = galsim.Image(config['image_xsize'], config['image_ysize'],
                                               wcs=config['wcs'])

    nobj = builder.getNObj(config['image'], config, 0)
    print('nobj = ',nobj)
    obj_num = 2561
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (32, 32)
    assert 5 < image.array.sum() < 10  # 8


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
