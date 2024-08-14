import numpy as np
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
    lsst_silicon.do_reweight = False
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
        input.sky_catalog.pupil_area:
            type: Eval
            str: "0.25 * np.pi * 649**2"
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
    assert 2300 < image.array.sum() < 2600  # 2443

    # 2. 10x brighter star.  Still minimum stamp size.
    obj_num = 2699
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (40,40)
    assert 27000 < image.array.sum() < 30000  # 28124

    # 3. 10x brighter star.  Needs bigger stamp.
    obj_num = 2746
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (106,106)
    assert 280_000 < image.array.sum() < 310_000  # 292627

    # 4. 10x brighter star.  (Uses photon shooting, but checks the max sb.)
    obj_num = 2611
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (350,350)
    assert 3_250_000 < image.array.sum() < 3_550_000  # 3_402_779

    # 5. Extremely bright star.  Maxes out size at _Nmax.  (And uses fft.)
    obj_num = 2697
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (4096,4096)
    assert 500_000_000 < image.array.sum() < 580_000_000  # 531_711_520

    # 6. Faint galaxy.
    # Again, this db doesn't have extremely faint objects.  The minimum seems to be 47.7.
    # However, this particular galaxy has a bulge with half-light-radius = 1.3,
    # which means in needs a pretty big stamp.
    obj_num = 2538
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (328,328)
    assert 20 < image.array.sum() < 50  # 42

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
    assert 1300 < image.array.sum() < 1600  # 1449

    # 8. Bright, small galaxy.
    # For bright galaxies, we check if we might need to scale back the size.
    # This one triggers the check, but not a reduction in size.
    obj_num = 12
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (73,73)
    assert 740_000 < image.array.sum() < 800_000  # 768_701

    # 9. Bright, big galaxy
    # None of the galaxies are big enough to trigger the reduction.  This is the largest.
    obj_num = 75
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (1978,1978)
    assert 490_000 < image.array.sum() < 530_000  # 507_192

    # We can trigger the reduction by mocking the _Nmax value to a lower value.
    # This triggers a recalculation with a different calculation, but that ends up less than
    # _Nmax, so it doesn't end up maxing at _Nmax.
    with mock.patch('imsim.LSST_SiliconBuilder._Nmax', 1024):
        image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
        print(obj_num, image.center, image.array.shape, image.array.sum())
        assert image.array.shape == (104,104)
        assert 300_000 < image.array.sum() < 330_000  # 309_862

    # With even smaller Nmax, it triggers a second recalculation, which again ends up
    # less than Nmax without pinning at Nmax.
    with mock.patch('imsim.LSST_SiliconBuilder._Nmax', 100):
        image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
        print(obj_num, image.center, image.array.shape, image.array.sum())
        assert image.array.shape == (69,69)
        # assert 230_000 < image.array.sum() < 270_000  # 251_679

    # Finally, with an even smaller Nmax, it will max out at Nmax.
    with mock.patch('imsim.LSST_SiliconBuilder._Nmax', 60):
        image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
        print(obj_num, image.center, image.array.shape, image.array.sum())
        assert image.array.shape == (60,60)
        assert 210_000 < image.array.sum() < 250_000  # 232_194

    # 10. Force stamp size in config.
    # There are two ways for the user to force the size of the stamp.
    # First, set stamp.size in the config file.
    config['stamp']['size'] = 64
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (64,64)
    assert 230_000 < image.array.sum() < 260_000  # 241_136

    # There is also a code path where the xsize,ysize is dictated by the calling routine.
    del config['stamp']['size']
    image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False,
                                        xsize=128, ysize=128)
    print(obj_num, image.center, image.array.shape, image.array.sum())
    assert image.array.shape == (128,128)
    assert 320_000 < image.array.sum() < 350_000  # 338_265

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
    assert 7 < image.array.sum() < 13  # 10


def test_stamp_bandpass_airmass():
    """Test that LSST_SiliconBuilder correctly uses the specified airmass,
    rather than the airmass in the precomputed flux.
    """

    config_str = dedent(f"""
        modules:
            - imsim
        template: imsim-config-skycat
        input.atm_psf.screen_size: 40.96
        input.checkpoint: ""
        input.sky_catalog.file_name: data/sky_cat_9683.yaml
        input.sky_catalog.pupil_area:
            type: Eval
            str: "0.25 * np.pi * 649**2"
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
        """)

    def get_fluxes(airmass, camera, det_name):
        print(f"{airmass = }")
        print(f"{camera = }")
        print(f"{det_name = }")
        this_config_str = "" + config_str  # Make a copy
        if airmass is not None:
            this_config_str += f"image.bandpass.airmass: {airmass}\n"
        if camera is not None:
            this_config_str += f"image.bandpass.camera: {camera}\n"
            this_config_str += f"image.bandpass.det_name: {det_name}\n"

        config = yaml.safe_load(this_config_str)
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

        galsim.config.ProcessInput(config, logger)
        galsim.config.SetupConfigImageNum(config, 0, 0, logger)
        xsize, ysize = builder.setup(
            config['image'], config, 0, 0, galsim.config.image_ignore, logger
        )
        galsim.config.SetupConfigImageSize(config, xsize, ysize, logger)
        galsim.config.SetupInputsForImage(config, logger)
        galsim.config.SetupExtraOutputsForImage(config, logger)
        config['bandpass'] = builder.buildBandpass(config['image'], config, 0, 0, logger)
        config['sensor'] = builder.buildSensor(config['image'], config, 0, 0, logger)

        nobj = builder.getNObj(config['image'], config, 0)

        ref_fluxes = []
        realized_fluxes = []
        for obj_num in [75, 2697, 2619, 2699, 2538]:  # bright, big galaxy, extremely bright star
            print(f"{obj_num = }")
            image, _ = galsim.config.BuildStamp(config, obj_num, logger=logger, do_noise=False)
            realized_flux = config['realized_flux']
            print(f"{realized_flux = }")
            gal = galsim.config.BuildGSObject(config, 'gal', logger=logger)[0]
            ref_flux = gal.sed.calculateFlux(config['bandpass'])
            print(f"{ref_flux = }")
            print()

            ref_fluxes.append(ref_flux)
            realized_fluxes.append(realized_flux)

        print("\n"*3)
        return np.array(ref_fluxes), np.array(realized_fluxes)

    ref_QE, realized_QE = get_fluxes(None, 'LsstCam', 'R22_S11')
    ref_X_None, realized_X_None = get_fluxes(None, None, None)
    ref_X10, realized_X10 = get_fluxes(1.0, None, None)
    ref_X12, realized_X12 = get_fluxes(1.2, None, None)
    ref_X20, realized_X20 = get_fluxes(2.0, None, None)
    ref_X35, realized_X35 = get_fluxes(3.5, None, None)

    # Reference bandpass is close to (but not exactly equal to) the X=1.2 bandpass.
    np.testing.assert_allclose(ref_X_None, ref_X12, rtol=1e-2, atol=0)

    # Predict realized_X from realized_X_None and the ratio of ref_X_None to ref_X.
    for ref_X, realized_X, rtol in zip(
        [ref_X10, ref_X12, ref_X20, ref_X35, ref_QE],
        [realized_X10, realized_X12, realized_X20, realized_X35, realized_QE],
        [1e-3, 1e-3, 3e-3, 1e-2, 3e-2]
    ):
        with np.printoptions(formatter={'float': '{: 15.3f}'.format}, linewidth=100):
            predict = realized_X_None * (ref_X/ref_X_None)
            print("delivered:     ", realized_X)
            print("expected:      ", predict)
            print("diff:          ", realized_X - predict)
            print("diff/expected: ", (realized_X - predict)/predict)
            # ought to be within the Poisson level.
            err = np.sqrt(ref_X_None + ref_X)
            print("Poisson err:   ", err)
            print()
            print()
            # But we actually deliver much better than the Poisson level since the
            # rngs align.
            np.testing.assert_allclose(realized_X, predict, rtol=rtol)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
