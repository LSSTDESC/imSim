import os
import logging
from enum import Enum
from contextlib import contextmanager
import numpy as np
import scipy.stats
import galsim
from astropy.time import Time
from lsst.obs.lsst.translators.lsst import SIMONYI_LOCATION as RUBIN_LOC

from imsim import stamp, lsst_image, BatoidWCSFactory, diffraction_fft
from imsim.camera import get_camera
from imsim.telescope_loader import load_telescope
import imsim.instcat

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "fft-diffraction")


def create_test_wcs_factory(boresight, telescope):
    camera = get_camera()
    return BatoidWCSFactory(
        boresight,
        obstime=Time.strptime("2022-08-06 06:50:59.337600", "%Y-%m-%d %H:%M:%S.%f"),
        telescope=telescope,
        wavelength=622.3195217611445,  # nm
        camera=camera,
        temperature=280.0,
        pressure=72.7,
        H2O_pressure=1.0,
    )


class Mode(Enum):
    FFT = 0
    RAYTRACING = 1


def create_test_config(
    xsize: int,
    ysize: int,
    stamp_size: int,
    mode=Mode.FFT,
    exptime: float = 30.0,
    enable_diffraction: bool = True,
    rottelpos=20.0 * galsim.degrees,
):
    band = "r"
    bandpass = galsim.Bandpass(f"LSST_{band}.dat", wave_type="nm").withZeropoint("AB")
    boresight = galsim.CelestialCoord(
        1.1047934165124105 * galsim.radians, -0.5261230452954583 * galsim.radians
    )
    telescope = load_telescope(f"LSST_{band}.yaml", rotTelPos=rottelpos)
    wcs_factory = create_test_wcs_factory(boresight, telescope)
    det_name = "R22_S11"
    camera = get_camera()[det_name]
    wcs = wcs_factory.getWCS(det=camera)

    raytracing_config = {
        "input": {
            "telescope": {
                "file_name": f"LSST_{band}.yaml",
                "rotTelPos": rottelpos,
            }
        },
        "det_name": det_name,
        "output": {"camera": "LsstCam"},
        "_icrf_to_field": wcs_factory.get_icrf_to_field(camera),
    }
    alt_az = {
        "altitude": 88.0 * galsim.degrees,
        "azimuth": 73.7707957 * galsim.degrees,
    }
    if enable_diffraction:
        optics_args = {
            "type": "RubinDiffractionOptics",
            "disable_field_rotation": exptime == 0.0,
            **alt_az,
        }
    else:
        optics_args = {"type": "RubinOptics"}
    if mode == Mode.RAYTRACING:
        stamp_args = {"photon_ops": [
        {"type": "TimeSampler", "t0": 0.0, "exptime": exptime},
        {"type": "PupilAnnulusSampler", "R_outer": 4.18, "R_inner": 2.55},
        {
            **optics_args,
            "boresight": boresight,
            "camera": "LsstCam",
        },
    ]
                      }
    else:
        stamp_args = {}
    config = {
        **raytracing_config,
        "gal": {
            "type": "DeltaFunction",
            "sed": {
                "file_name": "vega.txt",
                "wave_type": "nm",
                "flux_type": "flambda",
                # "norm_flux_density": 10e5,
                "norm_flux_density": 10e4,
                "norm_wavelength": 500,
            },
        },
        "bandpass": bandpass,
        "wcs": wcs,
        "image": {
            "type": "LSST_Image",
            "xsize": xsize,
            "ysize": ysize,
            "wcs": wcs,
            "random_seed": 12345,
            "nobjects": 1,
            "bandpass": bandpass,
        },
        "psf": {"type": "Convolve", "items": [{"type": "Gaussian", "fwhm": 0.3}]},
        "stamp": {
            "type": "LSST_Silicon",
            # "fft_sb_thresh": 60000.0 if mode == Mode.FFT else 750000,
            "fft_sb_thresh": 6000.0 if mode == Mode.FFT else 800000,
            "max_flux_simple": 100,
            "world_pos": galsim.CelestialCoord(
                1.1056660811384078 * galsim.radians,
                -0.5253441048502933 * galsim.radians,
            ),
            "size": stamp_size,
            "diffraction_psf": {
                **alt_az,
                "exptime": exptime,
                "rottelpos": rottelpos,
                "enabled": enable_diffraction,
                "brightness_threshold": 1.0e5,
            },
            **stamp_args
        },
    }
    galsim.config.ProcessInput(config)
    return config


XSIZE = YSIZE = 3500
STAMP_SIZE = 1000


def test_add_image_works_for_overlay_contained_in_img():
    img_a = np.zeros((5, 6), dtype=int)
    img_b = np.ones((3, 3), dtype=int)
    diffraction_fft.add_image(img_a, img_b, 1, 2)
    expected_img = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(img_a, expected_img)


def test_add_image_works_for_overlay_outside_img():
    img_a = np.zeros((5, 6), dtype=int)
    img_b = np.ones((3, 3), dtype=int)
    diffraction_fft.add_image(img_a, img_b, -3, -3)
    expected_img = np.zeros((5, 6), dtype=int)
    np.testing.assert_array_equal(img_a, expected_img)


def test_add_image_works_for_overlay_overlapping_img_lu():
    img_a = np.zeros((5, 6), dtype=int)
    img_b = np.ones((3, 3), dtype=int)
    diffraction_fft.add_image(img_a, img_b, -1, -2)
    expected_img = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(img_a, expected_img)


def test_add_image_works_for_overlay_overlapping_img_rb():
    img_a = np.zeros((5, 6), dtype=int)
    img_b = np.ones((3, 3), dtype=int)
    diffraction_fft.add_image(img_a, img_b, 4, 4)
    expected_img = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(img_a, expected_img)


def test_add_image_works_for_covering_overlay():
    img_a = np.zeros((5, 6), dtype=int)
    img_b = np.ones((7, 7), dtype=int)
    diffraction_fft.add_image(img_a, img_b, -1, 0)
    expected_img = np.ones((5, 6), dtype=int)
    np.testing.assert_array_equal(img_a, expected_img)


def test_saturated_clusters_works_for_no_saturated_pixels():
    img = np.zeros((5, 6), dtype=int)
    regions = diffraction_fft.saturated_clusters(img, 0.5)
    assert not regions


def test_saturated_clusters_works_for_1_saturated_pixel():
    img = np.zeros((5, 6), dtype=int)
    img[2, 3] = 1.0
    regions = diffraction_fft.saturated_clusters(img, 0.5)
    assert regions == [(slice(2, 3), slice(3, 4))]


def test_saturated_clusters_finds_saturated_clusters():
    img = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1],
        ]
    )
    regions = diffraction_fft.saturated_clusters(img, 0.5)
    expected_regions = [
        (slice(0, 1), slice(1, 2)),
        (slice(1, 2), slice(4, 6)),
        (slice(2, 5), slice(0, 3)),
        (slice(3, 5), slice(5, 6)),
    ]
    assert sorted(regions) == sorted(
        expected_regions
    ), f"{sorted(regions)} != {sorted(expected_regions)}"


def test_fft_diffraction_is_similar_to_raytracing_for_0_exptime():
    """Check, that raytracing diffraction and FFT diffraction are "similar"."""
    config = create_test_config(XSIZE, YSIZE, STAMP_SIZE, exptime=0.0)
    # To compare with the raytracing method, instead use:
    # config = create_test_config(XSIZE, YSIZE, STAMP_SIZE, exptime=0.0, mode=Mode.RAYTRACING)

    with assert_no_error_logs() as logger:
        image = galsim.config.BuildImage(config, logger=logger)
    # To save the image produced here, use:
    # image.write("/tmp/spikes.fits")
    brightness = image.array
    # Center of star:
    [c_x, c_y] = center_of_brighness(brightness)
    # 2 Pixel tolerance:
    np.testing.assert_allclose(
        np.array([c_x, c_y]), np.array([1923.0, 3161.0]), atol=2.0, rtol=0.0
    )
    angle, angle_stddev = folded_spike_angle(brightness, c_x, c_y, r_min=10.0)
    expected_angle = (
        45.0 * galsim.degrees - config["stamp"]["diffraction_psf"]["rottelpos"]
    ).deg
    # 1° tolerance:
    np.testing.assert_allclose(np.rad2deg(angle), expected_angle, atol=1.0, rtol=0.0)
    expected_max_spike_width = 5.0
    np.testing.assert_array_less(np.rad2deg(angle_stddev), expected_max_spike_width)
    slope, intercept, slope_stderr, intercept_stderr = brightness_log_profile(
        brightness, c_x, c_y
    )
    np.testing.assert_allclose(slope, -2.0, atol=0.6, rtol=0.0)
    np.testing.assert_allclose(intercept, 12.5, atol=3.0, rtol=0.0)
    # Assert some rough evidence, that brightness decays as
    # brightness ~ exp(intercept) * r**slope:
    np.testing.assert_array_less(slope_stderr, 0.2)
    np.testing.assert_array_less(intercept_stderr, 0.8)


def test_fft_diffraction_is_similar_to_raytracing_for_field_rotation():
    """Check, that raytracing diffraction and FFT diffraction are "similar"."""
    config = create_test_config(XSIZE, YSIZE, STAMP_SIZE, exptime=300.0)
    # To compare with the raytracing method, instead use:
    # config = create_test_config(XSIZE, YSIZE, STAMP_SIZE, exptime=300.0, mode=Mode.RAYTRACING)

    with assert_no_error_logs() as logger:
        image = galsim.config.BuildImage(config, logger=logger)
    # To save the image produced here, use:
    # image.write("/tmp/spikes.fits")
    brightness = image.array
    [c_x, c_y] = center_of_brighness(brightness)
    # 2 Pixel tolerance:
    np.testing.assert_allclose(
        np.array([c_x, c_y]), np.array([1923.0, 3161.0]), atol=2.0, rtol=0.0
    )
    angle, angle_stddev = folded_spike_angle(brightness, c_x, c_y)
    expected_spike_width = 14.4
    expected_angle = (
        45.0 * galsim.degrees - config["stamp"]["diffraction_psf"]["rottelpos"]
         - expected_spike_width / 2.0 * galsim.degrees
    ).deg
    # 2.5° tolerance (we neglect the time dependence of the rotation rate):
    np.testing.assert_allclose(np.rad2deg(angle), expected_angle, atol=2.5, rtol=0.0)
    np.testing.assert_allclose(
        np.rad2deg(angle_stddev), expected_spike_width / 2., atol=1.0, rtol=0.0
    )
    slope, intercept, slope_stderr, intercept_stderr = brightness_log_profile(
        brightness, c_x, c_y
    )
    # Brightness should decay as ~1/r**2:
    np.testing.assert_allclose(slope, -2.0, atol=0.6, rtol=0.0)
    np.testing.assert_allclose(intercept, 12.6, atol=3.0, rtol=0.0)
    # Assert some rough evidence, that brightness decays as
    # brightness ~ exp(intercept) * r**slope:
    np.testing.assert_array_less(slope_stderr, 0.2)
    np.testing.assert_array_less(intercept_stderr, 0.9)


@contextmanager
def assert_no_error_logs():
    """Context manager, which provides a InMemoryLogger instance and checks,
    that no errors have been logged on exit.
    """
    logger = InMemoryLogger()
    yield logger
    assert not logger.errors


class InMemoryLogger(logging.Logger):
    """Logger which buffers errors in memory."""

    def __init__(self):
        super().__init__("TestLogger")
        self.setLevel(logging.ERROR)
        self.errors = []

    def error(self, msg, *args, **kwargs):
        self.errors.append(msg)


R_MIN = 5.0


def center_of_brighness(image):
    """Calculate the (pixel) center of brightness in the given image."""
    return np.sum(
        np.array(
            [
                image * np.arange(image.shape[0])[:, None],
                image * np.arange(image.shape[1]),
            ]
        ),
        axis=(1, 2),
    ) / np.sum(image)


def folded_spike_angle(image, x_center, y_center, r_min=R_MIN):
    """Standard deviation of the angles of the pixels wrt (x_center, y_center)
    modulo 90°, weighted by brightness."""
    alpha, brightness = folded_angle(image, x_center, y_center, r_min=r_min)
    # Make angular interval [0°, 90°) periodic, by stretching it to [0°, 360°)
    # and finally divide by 4 again
    # (catches cases where the mean angle is near 0° / 90°).
    alpha_mean, alpha_stddev = angular_mean(alpha * 4, brightness)
    return alpha_mean / 4, alpha_stddev / 4


def folded_angle(image, x_center, y_center, r_min=R_MIN):
    """For each pixel of the image, determine its angle relative to
    (x_center, y_center).
    Returns an array of the pixel angles together with the brightness values
    at the same pixel."""
    x, y = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    r = np.hypot(y - y_center, x - x_center)
    alpha = np.arctan2(
        y[r > r_min] - y_center,
        x[r > r_min] - x_center,
    ) % (np.pi / 2.0)
    return alpha, image[r > r_min]


def angular_mean(angles, weights):
    """Similar to scipy.stats.circmean, scipy.stats.circstd, with weighting support."""
    w_norm = weights / np.sum(weights)
    x_mean = np.sum(np.cos(angles) * w_norm)
    y_mean = np.sum(np.sin(angles) * w_norm)
    R = np.sqrt(x_mean**2 + y_mean**2)
    stddev = np.sqrt(-2 * np.log(R))
    return np.arctan2(y_mean, x_mean), stddev


def brightness_log_profile(image, x_center, y_center, r_min=R_MIN):
    """Fits a curve brightness = a*r**b to the radial distribution of brighness
    of image around a center `x_center`, `y_center`.
    Returns `log(a)`, `b` and stderr of `log(a)` and `b`.
    """

    def radius(x, y, image):
        r = np.hypot(y - y_center, x - x_center)
        # Determine max radius where there are non 0 pixels:
        r_max = np.max(r[image > 0.0])
        brightness = image[(r_min <= r) & (r <= r_max)]
        r = r[(r_min <= r) & (r <= r_max)]
        return r, brightness

    r_m, brightness_dist = brightness_distribution(image, radius, scale=np.geomspace)
    reg = scipy.stats.linregress(np.log(r_m), np.log(brightness_dist))
    return reg.slope, reg.intercept, reg.stderr, reg.intercept_stderr


def brightness_distribution(
    image: np.ndarray,
    quantity: "Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]",
    scale=np.linspace,
    num_bins: int = 25,
):
    """Computes the distribution of a quantity weighted with the brightness."""
    x, y = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    q, brightness = quantity(x, y, image)
    # Determine max q where there are non 0 pixels:
    q_min = np.min(q)
    q_max = np.max(q)
    q_bins = scale(q_min, q_max, num=num_bins)
    dist, _ = np.histogram(q, bins=q_bins, weights=brightness)
    dist /= np.diff(q_bins)
    q_m = (q_bins[1:] + q_bins[:-1]) / 2.0
    return q_m, dist


def save_pic(filename: str, exptime: float, mode: Mode, enable_diffraction: bool):
    """Save a fits image of a single star."""
    config = create_test_config(
        XSIZE,
        YSIZE,
        STAMP_SIZE,
        exptime=exptime,
        mode=mode,
        enable_diffraction=enable_diffraction,
    )
    image = galsim.config.BuildImage(config, logger=None)
    image.write(file_name)


def save_pics():
    """Use this to save images for FFT spikes, raytracing spikes with and without field rotation."""
    for mode, prefix in [(Mode.FFT, "fft"), (Mode.RAYTRACING, "raytracing")]:
        save_pic(
            f"{prefix}-no-diffraction.fits",
            exptime=0.0,
            mode=mode,
            enable_diffraction=False,
        )
        save_pic(
            f"{prefix}-0exptime.fits", exptime=0.0, mode=mode, enable_diffraction=True
        )
        save_pic(
            f"{prefix}-300exptime.fits",
            exptime=300.0,
            mode=mode,
            enable_diffraction=True,
        )


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "--pics":
        save_pics()
        sys.exit(0)
    if sys.argv[1] == "-k":
        test_prefix = sys.argv[2]
    else:
        test_prefix = "test_"
    testfns = [v for k, v in vars().items() if k.startswith(test_prefix) and callable(v)]
    for testfn in testfns:
        testfn()
