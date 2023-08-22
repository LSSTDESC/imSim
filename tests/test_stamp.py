from unittest import mock
import galsim
import imsim
from test_photon_ops import create_test_icrf_to_field

METHOD_PHOT = "phot"
METHOD_FFT = "fft"


def create_test_config():
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
        "wcs": galsim.PixelScale(0.2),
    }
    galsim.config.ProcessInput(config)
    galsim.config.SetupInputsForImage(config, None)
    return config


def create_test_lsst_silicon(faint: bool):
    lsst_silicon = imsim.LSST_SiliconBuilder()
    lsst_silicon.realized_flux = 100 if not faint else 99
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
        "n_photons": lsst_silicon.realized_flux,
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
    for method, expected_specific_args, prof_type, op_type in (
        ("phot", expected_phot_args, 'galsim.ChromaticTransformation', imsim.RubinDiffractionOptics),
        ("fft", expected_fft_args, 'galsim.ChromaticConvolution', imsim.RubinDiffraction),
    ):
        # mock.patch basically wraps these functions so we can access how they were called
        # and what their return values were.
        with mock.patch(prof_type+'.drawImage') as mock_drawImage:
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
                wcs=config['wcs'],
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
    for method, expected_specific_args, photon_ops_key, prof_type in (
        ("phot", expected_phot_args, 'photon_ops', 'galsim.ChromaticTransformation'),
        ("fft", expected_fft_args, 'fft_photon_ops', 'galsim.ChromaticConvolution'),
    ):
        for faint in [True, False]:
            use_config = galsim.config.CopyConfig(config)
            lsst_silicon = create_test_lsst_silicon(faint=faint)
            if not faint:
                # When not faint, check that photon_ops ends up empty when none are listed in
                # the config (and there are no PSFs).
                del use_config['stamp'][photon_ops_key]
            expected_phot_args['n_photons'] = lsst_silicon.realized_flux

            with mock.patch(prof_type+'.drawImage') as mock_drawImage:
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
                    wcs=config['wcs'],
                    **expected_specific_args
                )
