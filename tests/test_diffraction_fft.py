import os
from enum import Enum
import numpy as np
import scipy.stats
import galsim
from astropy.time import Time

from imsim import BatoidWCSFactory, diffraction_fft
from imsim.camera import get_camera
from imsim.telescope_loader import load_telescope

from imsim_test_helpers import assert_no_error_logs


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
    band="r",
    rottelpos=20.0 * galsim.degrees,
):
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
            "det_name": "R22_S11",
            "disable_field_rotation": exptime == 0.0,
            **alt_az,
        }
    else:
        optics_args = {"type": "RubinOptics"}
    if mode == Mode.RAYTRACING:
        stamp_args = {
            "photon_ops": [
                {"type": "TimeSampler", "t0": 0.0, "exptime": exptime},
                {"type": "PupilAnnulusSampler", "R_outer": 4.18, "R_inner": 2.55},
                {"type": "Shift"},
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
            "det_name": "R22_S11",
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
            "diffraction_fft": {
                **alt_az,
                "exptime": exptime,
                "rotTelPos": rottelpos,
                "enabled": enable_diffraction,
                "brightness_threshold": 1.0e5,
            },
            "det_name": "R22_S11",
            **stamp_args,
        },
    }
    galsim.config.ProcessInput(config)
    return config


def test_convolve_region_works():
    image = np.array(
        [
            [0, 0, 0, 0, 0, -1],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    row_range = slice(1, 4)
    col_range = slice(2, 4)
    stencil = np.array([
        [0, 0, 1, 0, 0],
        [1, 1, 2, 1, 1],
        [0, 0, 1, 0, 0]
    ])
    diffraction_fft.convolve_region(image, row_range, col_range, stencil)
    expected_image = np.array(
        [
            [0, 0, 1, 1, 0, -1],
            [1, 2, 4, 4, 2, 1],
            [1, 2, 5, 5, 2, 1],
            [1, 2, 4, 4, 2, 1],
            [0, 0, 1, 1, 0, 0],
        ]
    )
    np.testing.assert_array_equal(image, expected_image)


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
    img_b = np.full((9, 9), 1)
    img_b[4, 4] = 2
    diffraction_fft.add_image(img_a, img_b, -2, -3)
    expected_img = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(img_a, expected_img)


def test_saturated_region_works_for_no_saturated_pixels():
    img = np.zeros((5, 6), dtype=int)
    region = diffraction_fft.saturated_region(img, 0.5)
    assert region is None


def test_saturated_region_works_for_1_saturated_pixel():
    img = np.zeros((5, 6), dtype=int)
    img[2, 3] = 1.0
    region = diffraction_fft.saturated_region(img, 0.5)
    expected_region = (slice(2, 3), slice(3, 4))
    assert region == expected_region, f"{region} != {expected_region}"


def test_saturated_region_works_for_multiple_saturated_pixel():
    img = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
        ]
    )
    region = diffraction_fft.saturated_region(img, 0.5)
    expected_region = (slice(1, 5), slice(1, 6))
    assert region == expected_region, f"{region} != {expected_region}"


def generate_reference_data_from_raytracing(parameters):
    config = create_test_config(**parameters, mode=Mode.RAYTRACING)
    brightness = galsim.config.BuildImage(config, logger=None).array
    [c_x, c_y] = center_of_brightness(brightness)
    angle, angle_stddev = folded_spike_angle(brightness, c_x, c_y, r_min=10.0)
    slope, intercept, slope_stderr, intercept_stderr = radial_brightness_asymptotics(
        brightness, c_x, c_y
    )
    return {
        "c": np.array([c_x, c_y]),
        "angle": angle,
        "angle_stddev": angle_stddev,
        "slope": slope,
        "intercept": intercept,
        "slope_stderr": slope_stderr,
        "intercept_stderr": intercept_stderr,
    }


XSIZE = YSIZE = 3500
STAMP_SIZE = 1000


def test_photon_and_pixel_distributions_match():
    """Check that computing the radial brightness distribution based on
    photon positions and based on pixel brightness yield a similar results."""
    config = create_test_config(
        xsize=XSIZE,
        ysize=YSIZE,
        stamp_size=STAMP_SIZE,
        exptime=0.0,
        mode=Mode.RAYTRACING,
    )
    sensor = config["sensor"] = PhotonCapturingSensor()
    brightness = galsim.config.BuildImage(config, logger=None).array
    (
        phot_slope,
        phot_intercept,
        phot_slope_stderr,
        phot_intercept_stderr,
    ) = radial_photon_asymptotics(sensor.detected_photons())
    [c_x, c_y] = center_of_brightness(brightness)
    slope, intercept, slope_stderr, intercept_stderr = radial_brightness_asymptotics(
        brightness, c_x, c_y
    )
    # Brightness should decay as ~1/r**2:
    np.testing.assert_allclose(slope, -2.0, atol=0.2, rtol=0.0)
    np.testing.assert_allclose(phot_slope, -2.0, atol=0.2, rtol=0.0)

    np.testing.assert_allclose(intercept, phot_intercept, atol=0.5, rtol=0.0)

    np.testing.assert_array_less(slope_stderr, 0.2)
    np.testing.assert_array_less(phot_slope_stderr, 0.2)
    np.testing.assert_array_less(intercept_stderr, 0.8)
    np.testing.assert_array_less(phot_intercept_stderr, 0.8)


def test_spike_profile_has_correct_radial_brightness_distribution():
    """Check that the spikes resulting from 1 pixel have the correct radial
    brightness distribution."""
    for d_alpha in (0.0, np.pi / 6.0):
        spikes = diffraction_fft.prepare_psf_field_rotation(
            w=1000,
            h=1000,
            wavelength=diffraction_fft.WAVELENGTH,
            alpha=0.0,
            d_alpha=d_alpha,
        )
        [c_x, c_y] = center_of_brightness(spikes)
        (
            slope,
            intercept,
            slope_stderr,
            intercept_stderr,
        ) = radial_brightness_asymptotics(spikes, c_x, c_y)
        # Brightness should decay as ~1/r**2:
        np.testing.assert_allclose(slope, -2.0, atol=0.2, rtol=0.0)
        np.testing.assert_allclose(
            intercept, np.log(diffraction_fft.A), atol=0.25, rtol=0.0
        )
        np.testing.assert_array_less(slope_stderr, 0.2)
        np.testing.assert_array_less(intercept_stderr, 0.8)


def test_fft_diffraction_is_similar_to_raytracing_for_0_exptime():
    """Check, that raytracing diffraction and FFT diffraction are "similar"."""

    parameters = {
        "xsize": XSIZE,
        "ysize": YSIZE,
        "stamp_size": STAMP_SIZE,
        "exptime": 0.0,
    }
    file_name = os.path.join(DATA_DIR, "raytrace_diffraction_values_0_exptime.npz")
    # Generate reference data if not existing:
    if not os.path.exists(file_name):
        np.savez(file_name, **generate_reference_data_from_raytracing(parameters))
    raytrace_data = np.load(file_name)

    config = create_test_config(**parameters)

    with assert_no_error_logs() as logger:
        image = galsim.config.BuildImage(config, logger=logger)
    # To save the image produced here, use:
    # image.write("/tmp/spikes.fits")
    brightness = image.array
    # Center of star:
    [c_x, c_y] = center_of_brightness(brightness)
    # 2 Pixel tolerance:
    np.testing.assert_allclose(
        np.array([c_x, c_y]), raytrace_data["c"], atol=2.0, rtol=0.0
    )

    angle, angle_stddev = folded_spike_angle(brightness, c_x, c_y, r_min=10.0)
    expected_angle = (
        45.0 * galsim.degrees - config["stamp"]["diffraction_fft"]["rotTelPos"]
    )

    # 1° tolerance:
    np.testing.assert_allclose(
        np.rad2deg(angle), expected_angle.deg, atol=1.0, rtol=0.0
    )
    np.testing.assert_allclose(
        np.rad2deg(raytrace_data["angle"]), expected_angle.deg, atol=1.0, rtol=0.0
    )
    np.testing.assert_allclose(
        np.rad2deg(angle_stddev),
        np.rad2deg(raytrace_data["angle_stddev"]),
        atol=2.0,
        rtol=0.0,
    )
    slope, intercept, slope_stderr, intercept_stderr = radial_brightness_asymptotics(
        brightness, c_x, c_y
    )
    # Brightness should decay as ~1/r**2:
    np.testing.assert_allclose(slope, -2.0, atol=0.6, rtol=0.0)
    np.testing.assert_allclose(raytrace_data["slope"], -2.0, atol=0.6, rtol=0.0)

    # rho(r) ~ a*r^-2
    s = diffraction_fft.WAVELENGTH / config["bandpass"].effective_wavelength
    # \int rho(s*r) s dr = 1
    # => rho(s*r)*s ~ a/s * r^-2
    expected_intercept = raytrace_data["intercept"] - np.log(s)
    np.testing.assert_allclose(intercept, expected_intercept, atol=0.5, rtol=0.0)
    # Here we dont compare against raytracing, but only make sure that both FFT and ratracing values are bounded:
    np.testing.assert_array_less(slope_stderr, 0.2)
    np.testing.assert_array_less(raytrace_data["slope_stderr"], 0.2)
    np.testing.assert_array_less(intercept_stderr, 0.8)
    np.testing.assert_array_less(raytrace_data["intercept_stderr"], 0.8)


def test_fft_diffraction_is_similar_to_raytracing_for_field_rotation():
    """Check, that raytracing diffraction and FFT diffraction are "similar"."""

    parameters = {
        "xsize": XSIZE,
        "ysize": YSIZE,
        "stamp_size": STAMP_SIZE,
        "exptime": 300.0,
    }
    file_name = os.path.join(DATA_DIR, "raytrace_diffraction_values_300_exptime.npz")
    # Generate reference data if not existing:
    if not os.path.exists(file_name):
        np.savez(file_name, **generate_reference_data_from_raytracing(parameters))
    raytrace_data = np.load(file_name)

    config = create_test_config(**parameters)

    with assert_no_error_logs() as logger:
        image = galsim.config.BuildImage(config, logger=logger)
    # To save the image produced here, use:
    # image.write("/tmp/spikes.fits")
    brightness = image.array
    [c_x, c_y] = center_of_brightness(brightness)
    # 2 Pixel tolerance:
    np.testing.assert_allclose(
        np.array([c_x, c_y]), raytrace_data["c"], atol=2.0, rtol=0.0
    )
    angle, angle_stddev = folded_spike_angle(brightness, c_x, c_y)
    # 2.5° tolerance (we neglect the time dependence of the rotation rate):
    np.testing.assert_allclose(
        np.rad2deg(angle), np.rad2deg(raytrace_data["angle"]), atol=2.5, rtol=0.0
    )
    np.testing.assert_allclose(
        np.rad2deg(angle_stddev),
        np.rad2deg(raytrace_data["angle_stddev"]),
        atol=1.5,
        rtol=0.0,
    )

    slope, intercept, slope_stderr, intercept_stderr = radial_brightness_asymptotics(
        brightness, c_x, c_y
    )
    # Brightness should decay as ~1/r**2:
    np.testing.assert_allclose(slope, -2.0, atol=0.6, rtol=0.0)
    np.testing.assert_allclose(raytrace_data["slope"], -2.0, atol=0.6, rtol=0.0)

    # rho(r) ~ a*r^-2
    s = diffraction_fft.WAVELENGTH / config["bandpass"].effective_wavelength
    # \int rho(s*r) s dr = 1
    # => rho(s*r)*s ~ a/s * r^-2
    expected_intercept = raytrace_data["intercept"] - np.log(s)
    np.testing.assert_allclose(intercept, expected_intercept, atol=0.5, rtol=0.0)
    # Here we dont compare against raytracing, but only make sure that both FFT and ratracing values are bounded:
    np.testing.assert_array_less(slope_stderr, 0.2)
    np.testing.assert_array_less(raytrace_data["slope_stderr"], 0.2)
    np.testing.assert_array_less(intercept_stderr, 0.9)
    np.testing.assert_array_less(raytrace_data["intercept_stderr"], 0.9)


def test_apply_diffraction_psf_for_no_saturated_pixels():
    """Large, bright objects, such as nearby galaxies, can have
    sufficiently high fluxes that they trigger FFT rendering, but they
    may still have zero saturated pixels, so check that
    apply_diffraction_psf does not raise a TypeError when
    saturated_region(...) returns None.
    """
    wavelength = 870.
    rottelpos = -4.53
    exptime = 15.
    latitude = -0.528
    azimuth = 2.08
    altitude = 0.947
    brightness_threshold = 100000.0
    spike_length_cutoff = 4000

    image = np.zeros((100, 100))

    diffraction_fft.apply_diffraction_psf(image, wavelength, rottelpos,
                                          exptime, latitude, azimuth,
                                          altitude, brightness_threshold,
                                          spike_length_cutoff)

    np.testing.assert_allclose(image, 0.)


R_MIN = 5.0


def center_of_brightness(image):
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


def radial_brightness_asymptotics(image, x_center, y_center, r_min=R_MIN):
    """Fits a curve brightness = a*r**b to the radial distribution of brighness
    of image around a center `x_center`, `y_center`.
    Returns `log(a)`, `b` and stderr of `log(a)` and `b`.
    """

    x, y = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    r = np.hypot(y - y_center, x - x_center)
    # Determine max radius where there are non 0 pixels:
    r_max = np.max(r[image > 0.0])
    brightness = image[r <= r_max]
    r = r[r <= r_max]

    r_m, brightness_dist = brightness_distribution(
        r, brightness, q_min=r_min, scale=np.geomspace
    )
    return linear_regression(np.log(r_m), np.log(brightness_dist))


def linear_regression(
    x: np.ndarray, y: np.ndarray
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Wrapper around scipy.stats.linregress which directly returns slope, intercept
    with stderr."""
    reg = scipy.stats.linregress(x, y)
    return reg.slope, reg.intercept, reg.stderr, reg.intercept_stderr


def brightness_distribution(
    quantity: np.ndarray,
    brightness: np.ndarray,
    q_min: float,
    scale=np.linspace,
    num_bins: int = 25,
) -> "tuple[np.ndarray, np.ndarray]":
    """Computes the distribution of a quantity weighted with brightness."""
    # Determine max q where there are non 0 pixels:
    q_max = np.max(quantity)
    q_bins = scale(q_min, q_max, num=num_bins)
    dist, _ = np.histogram(quantity, bins=q_bins, weights=brightness)
    dist /= np.diff(q_bins) * np.sum(brightness)
    q_m = (q_bins[1:] + q_bins[:-1]) / 2.0
    return q_m, dist


def radial_photon_asymptotics(photons, r_min=R_MIN):
    """Fits a curve brightness = a*r**b to the radial distribution of photon coordinates
    of detection. The center of the coordinates is determined automatically.
    Returns `log(a)`, `b` and stderr of `log(a)` and `b`.
    """

    c = np.nanmean(photons, axis=1)
    r = np.hypot(*(photons - c[:, None]))
    r = r[~np.isnan(r)]

    r_m, radial_dist = photon_distribution(r, q_min=r_min, scale=np.geomspace)
    log_mask = radial_dist > 1.0e-30
    return linear_regression(np.log(r_m[log_mask]), np.log(radial_dist[log_mask]))


def photon_distribution(
    quantity: np.ndarray,
    q_min: float,
    q_quantile: float = 0.99,
    scale=np.linspace,
    num_bins: int = 50,
) -> "tuple[np.ndarray, np.ndarray]":
    """Photon version of brightness_distribution: Instead of brightness pixel data
    and quantity, process a np.ndarray of that quantity.
    """
    q_sorted = np.sort(quantity[quantity > q_min])
    # Cut off a queue of lonely photons (beyond quantile(p=q_quantile):
    i_quantile = int(q_sorted.size * q_quantile)
    q_max = q_sorted[i_quantile]
    q_bins = scale(q_min, q_max, num=num_bins)
    dist, _ = np.histogram(quantity, bins=q_bins)
    dist = dist / (np.diff(q_bins) * quantity.size)
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
    if mode == Mode.RAYTRACING:
        sensor = config["sensor"] = PhotonCapturingSensor()
    image = galsim.config.BuildImage(config, logger=None)
    image.write(filename)
    if mode == Mode.RAYTRACING:
        np.save(filename.removesuffix("fits") + "npy", sensor.detected_photons())


class PhotonCapturingSensor(galsim.Sensor):
    """Sensor which captures the exact photon coordinates."""

    def __init__(self):
        super().__init__()
        self._photons = []

    def accumulate(self, photons, image, orig_center=None, resume=False):
        self._photons.append(photons)
        return super().accumulate(photons, image, orig_center, resume)

    def detected_photons(self) -> np.ndarray:
        return np.array(
            [
                np.concatenate([p.x for p in self._photons]),
                np.concatenate([p.y for p in self._photons]),
            ]
        )


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
    import argparse
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pics",
        action="store_true",
        help="Save fits images for all test cases and exit without testing.",
    )
    parser.add_argument(
        "-k",
        dest="test_prefix",
        help="Similar to -k of pytest / unittest: restrict tests to tests starting with the specified prefix.",
        default="test_",
    )
    args = parser.parse_args()
    if args.pics:
        save_pics()
        sys.exit(0)

    testfns = [
        v for k, v in vars().items() if k.startswith(args.test_prefix) and callable(v)
    ]
    for testfn in testfns:
        testfn()
