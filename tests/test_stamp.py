from unittest import mock, TestCase
import galsim
from imsim import stamp
from imsim.opsim_data import OpsimDataLoader

METHOD_PHOT = "phot"
METHOD_FFT = "fft"


def create_test_config(methods):
    config = {
        "input": {
            "telescope": {
                "file_name": "LSST_r.yaml",
            }
        },
        "_input_objs": {
            "opsim_data": [
                OpsimDataLoader.from_dict({"altitude": 43.0, "azimuth": 0.0})
            ]
        },
        "sky_pos": {
            "type": "RADec",
            "ra": "1.1056660811384078 radians",
            "dec": "-0.5253441048502933 radians",
        },
        "stamp": {},
        "bandpass": mock.Mock(),
        "wcs": mock.Mock(),
    }
    if METHOD_PHOT in methods:
        config["stamp"]["photon_ops"] = [
            {
                "type": "lsst_diffraction",
                "latitude": -30.24463,
            }
        ]
    if METHOD_FFT in methods:
        config["stamp"]["fft_photon_ops"] = [
            {
                "type": "lsst_diffraction",
                "latitude": -30.24463,
            }
        ]

    galsim.config.ProcessInput(config)
    return config


def create_test_lsst_silicon(faint: bool):
    lsst_silicon = stamp.LSST_SiliconBuilder()
    lsst_silicon.realized_flux = 100 if not faint else 99
    lsst_silicon.rng = mock.Mock()
    lsst_silicon.diffraction_fft = mock.Mock()
    return lsst_silicon


def create_test_image():
    image_copy = mock.MagicMock(
        __iadd__=mock.Mock(), array=mock.MagicMock(__lt__=mock.Mock())
    )
    image = mock.Mock(copy=mock.Mock(return_value=image_copy))
    return image


def create_test_prof():
    prof = mock.MagicMock(withFlux=mock.Mock())
    prof.evaluateAtWavelength = mock.Mock(return_value=prof)
    prof.__mul__ = mock.Mock(return_value=prof)
    return prof


def test_lsst_silicon_builder_passes_correct_photon_ops_to_drawImage() -> None:
    """LSST_SiliconBuilder.draw passes the correct list of photon_ops
    to prof.drawImage."""
    lsst_silicon = create_test_lsst_silicon(faint=False)
    image = create_test_image()
    image_copy = image.copy.return_value
    offset = mock.Mock()
    logger = mock.Mock(info=mock.Mock())
    built_photon_ops = mock.Mock()
    expected_phot_args = {
        "maxN": mock.ANY,
        "n_photons": mock.ANY,
        "add_to_image": True,
        "poisson_flux": False,
        "image": image,
        "sensor": None,
    }
    expected_fft_args = {"n_subsample": 1, "image": image_copy, "maxN": mock.ANY}
    for method, expected_specific_args in (
        ("phot", expected_phot_args),
        ("fft", expected_fft_args),
    ):
        config = create_test_config(methods={method})
        mock_build_photon_ops = mock.Mock(return_value=built_photon_ops)
        prof = create_test_prof()
        with TestCase().subTest(method=method):
            with mock.patch(
                "galsim.config.BuildPhotonOps", mock_build_photon_ops
            ), mock.patch("galsim.PoissonNoise"):
                lsst_silicon.draw(
                    prof,
                    image,
                    method,
                    offset,
                    config=config["stamp"],
                    base=config,
                    logger=logger,
                )
            modified_prof = prof.withFlux.return_value
            modified_prof.drawImage.assert_called_once_with(
                mock.ANY,
                method=method,
                offset=offset,
                wcs=mock.ANY,
                photon_ops=built_photon_ops,
                rng=lsst_silicon.rng,
                **expected_specific_args
            )
            expected_photon_ops_config_field = {
                "phot": "photon_ops",
                "fft": "fft_photon_ops",
            }
            mock_build_photon_ops.assert_called_once_with(
                config["stamp"],
                expected_photon_ops_config_field[method],
                config,
                logger,
            )


def test_stamp_builder_works_without_photon_ops_or_faint() -> None:
    """Here, we test that if LSST_SiliconBuilder.drawImage passes empty photon_ops,
    when faint is True or photon ops for the used method are empty."""

    image = create_test_image()
    image_copy = image.copy.return_value
    offset = mock.Mock()
    logger = mock.Mock(info=mock.Mock())
    expected_phot_args = {
        "maxN": mock.ANY,
        "n_photons": mock.ANY,
        "add_to_image": True,
        "poisson_flux": False,
        "image": image,
        "sensor": None,
        "photon_ops": [],
        "rng": mock.ANY,
    }
    expected_fft_args = {"image": image_copy}
    for method, expected_specific_args in (
        ("phot", expected_phot_args),
        ("fft", expected_fft_args),
    ):
        for provided_photon_op_configs in (
            {METHOD_PHOT, METHOD_FFT},
            {METHOD_PHOT, METHOD_FFT} - {method},
            (),
        ):
            config = create_test_config(methods=provided_photon_op_configs)
            for faint in (True, False):
                if not faint and method in provided_photon_op_configs:
                    # This is neither faint nor is the photon_ops list for the used method empty.
                    # => Dont want to test this case here.
                    continue
                lsst_silicon = create_test_lsst_silicon(faint=faint)
                prof = create_test_prof()
                with TestCase().subTest(
                    method=method,
                    provided_photon_op_configs=provided_photon_op_configs,
                    faint=faint,
                ):
                    with mock.patch("galsim.PoissonNoise"):
                        lsst_silicon.draw(
                            prof,
                            image,
                            method,
                            offset,
                            config=config["stamp"],
                            base=config,
                            logger=logger,
                        )
                    prof.withFlux.return_value.drawImage.assert_called_once_with(
                        mock.ANY,
                        method=method,
                        offset=offset,
                        wcs=mock.ANY,
                        **expected_specific_args
                    )
