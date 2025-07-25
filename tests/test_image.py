"""Integration tests for image and stamp."""
from pathlib import Path
import logging

import imsim
import galsim
from astropy.time import Time
import numpy as np

from imsim_test_helpers import assert_no_error_logs


DATA_DIR = Path(__file__).parent / 'data'
INSTCAT = DATA_DIR / 'tiny_instcat.txt'
SED_DIR = DATA_DIR / 'test_sed_library'
STAMP_SIZE = 1000

def assert_objects_at_positions(image, expected_positions, expected_brightness_values, pixel_radius=10, rtol=0.1):
    """Sum the brightness values of squares of side length `2*pixel_radius` centered at `expected_positions` and compare against `expected_brightness_values`."""
    brightness_values = np.empty_like(expected_brightness_values)
    for i, (col, row) in enumerate(expected_positions):
        neighbourhood = image[row-pixel_radius:row+pixel_radius, col-pixel_radius:col+pixel_radius]
        brightness_values[i] = np.sum(neighbourhood)
        print("Object: ", i, expected_brightness_values[i], brightness_values[i])
    np.testing.assert_allclose(brightness_values, expected_brightness_values, rtol=rtol)


def create_test_config(
    image_type="LSST_PhotonPoolingImageBuilder",
    stamp_type="LSST_Photons",
    exptime: float = 30.0,
    enable_diffraction: bool = True,
    band="r",
    rottelpos=20.0 * galsim.degrees,
    fft_sb_thresh: float=10000.0
):
    bandpass = galsim.Bandpass(f"LSST_{band}.dat", wave_type="nm").withZeropoint("AB")
    opsim_data = imsim.OpsimDataLoader(str(INSTCAT))
    boresight = galsim.CelestialCoord(
        opsim_data['rightascension'] * galsim.degrees, opsim_data['declination'] * galsim.degrees
    )
    telescope = imsim.load_telescope(f"LSST_{band}.yaml", rotTelPos=rottelpos)
    wcs_factory = imsim.BatoidWCSFactory(
        boresight,
        obstime=Time.strptime("2022-08-06 06:50:59.337600", "%Y-%m-%d %H:%M:%S.%f"),
        telescope=telescope,
        wavelength=622.3195217611445,  # nm
        camera=imsim.get_camera(),
        temperature=280.0,
        pressure=72.7,
        H2O_pressure=1.0,
    )
    det_name = "R22_S11"
    camera = imsim.get_camera()[det_name]
    wcs = wcs_factory.getWCS(det=camera)

    alt_az = {
        "altitude": 88.0 * galsim.degrees,
        "azimuth": 73.7707957 * galsim.degrees,
    }
    if enable_diffraction:
        optics_args = {
            "type": "RubinDiffractionOptics",
            "det_name": "R22_S11",
            "disable_field_rotation": exptime == 0.0,
            **alt_az,
        }
    else:
        optics_args = {"type": "RubinOptics",
                       "det_name": "R22_S11",
                       }
    if stamp_type == "LSST_Photons":
        output_config = {
            "camera": "LsstCam",
            "photon_pooling_truth": {
                "dir": "output",
                "file_name": "test-truth.txt",
                "columns": {
                    "object_id": "obj_num",
                    "x": "$image_pos.x",
                    "y": "$image_pos.y",
                    "nominal_flux": "@nominal_flux",
                    "phot_flux": "@phot_flux",
                    "fft_flux": "@fft_flux",
                    "incident_flux": "@incident_flux",
                },
            },
        }
    else:
        output_config = {"camera": "LsstCam"}
    config = {
        "input": {
            "telescope": {
                "file_name": f"LSST_{band}.yaml",
                "rotTelPos": rottelpos,
            },
            "instance_catalog": {
                "file_name": str(INSTCAT),
                "sed_dir": str(SED_DIR),
            }
        },
        "det_name": det_name,
        "output": {**output_config},
        "_icrf_to_field": wcs_factory.get_icrf_to_field(camera),
        "gal": {
            "type": "DeltaFunction",
            "sed": {
                "file_name": "vega.txt",
                "wave_type": "nm",
                "flux_type": "flambda",
                "norm_wavelength": 500,
                # 10 FFT objects, 10 photon shooting objects (including 1 faint object)
                "norm_flux_density": [5.0e4] * 10 + [5.0e3] * 9 + [1.0],
            }
        },
        "bandpass": bandpass,
        "wcs": wcs,
        "image": {
            "type": image_type,
            "det_name": "R22_S11",
            "wcs": wcs,
            "random_seed": 12345,
            "bandpass": bandpass,
            "nobjects": 20,
            "nbatch": 10,
            "nbatch_fft": 5
        },
        "psf": {"type": "Convolve", "items": [{"type": "Gaussian", "fwhm": 0.3}]},
        "stamp": {
            "type": stamp_type,
            "fft_sb_thresh": fft_sb_thresh,
            "max_flux_simple": 100,
            "world_pos": {"type": "InstCatWorldPos"},
            "size": STAMP_SIZE,
            "diffraction_fft": {
                **alt_az,
                "exptime": exptime,
                "rotTelPos": rottelpos,
                "enabled": enable_diffraction,
                "brightness_threshold": 1.0e5,
            },
            "det_name": "R22_S11",
            "photon_ops": [
                {"type": "TimeSampler", "t0": 0.0, "exptime": exptime},
                {"type": "PupilAnnulusSampler", "R_outer": 4.18, "R_inner": 2.55},
                {
                    **optics_args,
                    "boresight": boresight,
                    "camera": "LsstCam",
                },
            ]
        },
    }
    galsim.config.ProcessInput(config)
    return config


def run_lsst_image(image_type, stamp_type):
    """Create an image using stamps of the given type and check that objects are batched as expected and stars at the correct positions."""

    config = create_test_config(image_type, stamp_type)

    n_images = 1
    n_expected_objects = 20
    all_obj_indices = frozenset(range(20))
    expected_fft_obj_indices = frozenset(range(10))
    with assert_no_error_logs(logger_level=logging.INFO) as logger:
        [image] = galsim.config.BuildImages(n_images, config, logger=logger)
    image.write("/tmp/tiny_instcat.fits")
    fft_obj_indices = {n for n in range(n_expected_objects) if any(f"Use FFT for object {n}." in msg for msg in logger.messages)}
    phot_obj_indices = {n for n in range(n_expected_objects) if any(f"Use photon shooting for object {n}." in msg for msg in logger.messages)}
    assert fft_obj_indices == expected_fft_obj_indices
    assert phot_obj_indices == all_obj_indices - expected_fft_obj_indices

    # Objects in the same order as in the catalog. Positions of [0, 0] mean
    # That the objects are outside of the image.
    expected_positions = np.array([
        [2046, 2000],
        [3770, 1128],
        [0, 0],
        [3777, 3880],
        [3817, 3934],
        [3727, 1123],
        [0, 0],
        [3722, 1119],
        [3721, 1094],
        [3752, 1124],
        [3702, 1081],
        [3762, 1137], # Obj #11
        [3715, 1114],
        [3762, 1135], # Almost indistinguishable from #11
        [3789, 1099],
        [3802, 3273],
        [3898, 3275],
        [3969, 3292],
        [3482, 3339],
        [3452, 3751],
    ])
    expected_brightness_values = np.array([
        1.974843e+06,
        4.325211e+06,
        0.0,
        1.978054e+06,
        1.973144e+06,
        1.981440e+06,
        0.0,
        3.131857e+06,
        1.980739e+06,
        1.977764e+06,
        1.983800e+05,
        4.045435e+06,
        2.195470e+06,
        4.045435e+06,
        1.974780e+05,
        1.966890e+05,
        1.968660e+05,
        1.970680e+05,
        1.965050e+05,
        4.000000e+01,
    ])
    assert_objects_at_positions(image.array, expected_positions, expected_brightness_values)

    # Return the config and image for further tests if needed.
    return config, image


def test_lsst_image_original_pipeline():
    """Check that LSSTImage batches objects as expected and renders objects at the correct positions."""
    run_lsst_image("LSST_Image", "LSST_Silicon")

def test_lsst_image_photon_pooling_pipeline():
    """Check that LSST_PhotonPoolingImage batches objects as expected and renders objects at the correct positions."""
    config, image = run_lsst_image("LSST_PhotonPoolingImage", "LSST_Photons")
    # We also check that we get reasonable truth outputs when using the photon
    # pooling truth. So, write to file, read from file, then compare the
    # recorded nominal flux and incident flux. Assume otherwise that the GalSim
    # tests cover the base truth output.
    galsim.config.extra.WriteExtraOutputs(config, image)
    nominal_flux, incident_flux = np.loadtxt('output/test-truth.txt', usecols=(3,6), unpack=True)
    np.testing.assert_allclose(nominal_flux, incident_flux, rtol=0.1)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
