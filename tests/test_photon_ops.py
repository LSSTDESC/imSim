import numpy as np
import galsim
import batoid
from astropy.time import Time
from astropy import units
from coord import degrees

from imsim import photon_ops, BatoidWCSFactory, get_camera, diffraction
from imsim.telescope_loader import load_telescope


def create_test_telescope(rottelpos=np.pi / 3 * galsim.radians):
    return load_telescope("LSST_r.yaml", rotTelPos=rottelpos)


def create_test_icrf_to_field(boresight, det_name):
    camera = get_camera()
    telescope = create_test_telescope()
    factory = BatoidWCSFactory(
        boresight,
        obstime=Time("J2020") + 0.5 * units.year,
        telescope=telescope,
        wavelength=620.0,  # nm
        camera=camera,
        temperature=290.0,
        pressure=70.0,
        H2O_pressure=1.1,
    )
    return factory.get_icrf_to_field(camera[det_name])


def create_test_wcs():
    return galsim.AffineTransform(
        0.168,
        0.108,
        -0.108,
        0.168,
        origin=galsim.PositionD(x=-0.349, y=-0.352),
        world_origin=galsim.PositionD(x=0.0, y=0.0),
    )


def create_test_photon_array(t=0.0, n_photons=10000):
    """This corresponds to a single star."""
    # u, v: r: 2.5 - 4.2
    rng = np.random.default_rng(seed=42)
    r_uv = rng.uniform(2.5, 4.2, n_photons)
    phi_uv = rng.uniform(0.0, 2.0 * np.pi, n_photons)
    u = r_uv * np.cos(phi_uv)
    v = r_uv * np.sin(phi_uv)
    r_xy = rng.uniform(0.0, 5.0, n_photons)
    phi_xy = rng.uniform(0.0, 2.0 * np.pi, n_photons)
    x = r_xy * np.cos(phi_xy)
    y = r_xy * np.sin(phi_xy)
    return galsim.PhotonArray(
        n_photons,
        x=x,
        y=y,
        wavelength=np.full(n_photons, 577.6),
        flux=np.ones(n_photons),
        pupil_u=u,
        pupil_v=v,
        time=np.full(n_photons, t),
    )


def create_test_rubin_optics(**kwargs):
    return photon_ops.RubinOptics(**create_test_rubin_optics_kwargs(**kwargs))


def create_test_rubin_optics_kwargs(
    boresight=galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians),
    icrf_to_field=None,
    sky_pos=galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians),
    image_pos=galsim.PositionD(809.6510740536025, 3432.6477953336625),
    rottelpos=np.pi / 3 * galsim.radians,
):
    det_name = "R22_S11"
    if icrf_to_field is None:
        icrf_to_field = create_test_icrf_to_field(boresight, det_name=det_name)
    return dict(
        telescope=create_test_telescope(rottelpos),
        boresight=boresight,
        sky_pos=sky_pos,
        image_pos=image_pos,
        icrf_to_field=icrf_to_field,
        det_name=det_name,
        camera=get_camera(),
    )


def create_test_rubin_diffraction(
    latitude=-30.24463 * degrees,
    azimuth=45.0 * degrees,
    altitude=89.9 * degrees,
    sky_pos=galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians),
    icrf_to_field=None,
    **kwargs
):
    if icrf_to_field is None:
        boresight = galsim.CelestialCoord(
            0.543 * galsim.radians, -0.174 * galsim.radians
        )
        icrf_to_field = create_test_icrf_to_field(boresight, det_name="R22_S11")

    return photon_ops.RubinDiffraction(
        telescope=create_test_telescope(),
        sky_pos=sky_pos,
        icrf_to_field=icrf_to_field,
        latitude=latitude,
        azimuth=azimuth,
        altitude=altitude,
        **kwargs,
    )


def create_test_rubin_diffraction_optics(
    latitude=-30.24463 * degrees,
    azimuth=45.0 * degrees,
    altitude=89.9 * degrees,
    boresight=galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians),
    icrf_to_field=None,
    sky_pos=galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians),
    image_pos=galsim.PositionD(809.6510740536025, 3432.6477953336625),
    rottelpos=np.pi / 3 * galsim.radians,
    **kwargs
):
    rubin_diffraction = create_test_rubin_diffraction(
        latitude=latitude,
        azimuth=azimuth,
        altitude=altitude,
        sky_pos=sky_pos,
        icrf_to_field=icrf_to_field,
        **kwargs,
    )
    return photon_ops.RubinDiffractionOptics(
        **create_test_rubin_optics_kwargs(
            boresight,
            icrf_to_field,
            image_pos=image_pos,
            sky_pos=sky_pos,
            rottelpos=rottelpos,
        ),
        rubin_diffraction=rubin_diffraction,
    )


def create_test_rng():
    return galsim.random.BaseDeviate(seed=42)


def test_rubin_optics() -> None:
    """Check that the image of a star is contained in a disc."""

    rubin_optics = create_test_rubin_optics(rottelpos=0.0 * galsim.radians)
    photon_array = create_test_photon_array()
    local_wcs = create_test_wcs()
    rubin_optics.applyTo(photon_array, local_wcs=local_wcs, rng=create_test_rng())
    expected_x_pic_center = 564.5
    expected_y_pic_center = -1431.4
    expected_r_pic_center = 20.0
    np.testing.assert_array_less(
        np.hypot(
            photon_array.x - expected_x_pic_center,
            photon_array.y - expected_y_pic_center,
        ),
        expected_r_pic_center,
    )


def test_rubin_diffraction_produces_spikes() -> None:
    """Checks that we have spike photons and that the spkies form a cross."""
    rubin_diffraction_optics = create_test_rubin_diffraction_optics(
        rottelpos=0.0 * galsim.radians
    )
    photon_array = create_test_photon_array(n_photons=1000000)
    local_wcs = create_test_wcs()
    rubin_diffraction_optics.applyTo(
        photon_array, local_wcs=local_wcs, rng=create_test_rng()
    )

    # The expected image is contained in a disc + spikes outside the disc:
    spike_angles = extract_spike_angles(
        photon_array,
        x_center=564.5,
        y_center=-1431.4,
        r=20.0,
    )

    # Find the angle, the cross is rotated relative to the axis cross:
    cross_rot_angle = np.mean(spike_angles % (np.pi / 2.0))

    # Define a tolerance for the spike width in rad:
    spike_angle_tolerance = np.pi / 6.0

    delta_angles = spike_angles - cross_rot_angle
    delta_angles[delta_angles < 0.0] += 2.0 * np.pi

    # Compute a histogram with non uniform bins
    # (spike regions and in-between spike regions):
    h, _ = np.histogram(
        delta_angles,
        bins=sum(
            (
                (
                    -spike_angle_tolerance / 2 + i * np.pi / 2.0,
                    spike_angle_tolerance / 2 + i * np.pi / 2.0,
                )
                for i in range(5)
            ),
            (),
        ),
    )
    # Merge last and first bin:
    h[0] += h[-1]
    h = h[:-1]

    # Check that there less than 0.5% of the spike photons outside of the spike regions:
    np.testing.assert_array_less(h[1::2], spike_angles.size // 200)
    # Check that there are photons in all spike regions:
    np.testing.assert_array_less(0, h[0::2])


def test_rubin_diffraction_optics_is_same_as_diffraction_and_optics() -> None:
    """Checks that the result of applying RubinDiffraction and then RubinOptics
    is the same as applying the combined photon op RubinDiffractionOptics."""
    photon_array_combined = create_test_photon_array(n_photons=100000)
    local_wcs = create_test_wcs()
    rubin_diffraction_optics = create_test_rubin_diffraction_optics()
    rubin_diffraction_optics.applyTo(
        photon_array_combined, local_wcs=local_wcs, rng=create_test_rng()
    )
    rubin_diffraction = create_test_rubin_diffraction()
    rubin_optics = create_test_rubin_optics()
    photon_array_modular = create_test_photon_array(n_photons=100000)
    rubin_diffraction.applyTo(
        photon_array_modular, local_wcs=local_wcs, rng=create_test_rng()
    )
    rubin_optics.applyTo(
        photon_array_modular, local_wcs=local_wcs, rng=create_test_rng()
    )
    np.testing.assert_array_almost_equal(
        photon_array_combined.x, photon_array_modular.x
    )
    np.testing.assert_array_almost_equal(
        photon_array_combined.y, photon_array_modular.y
    )


def extract_spike_angles(photon_array, x_center, y_center, r):
    """Filters out a disc centered at (x_center, y_center) with radius r.
    The remaining photons will be considered as spike photons.
    Returns the angles of the spike photons wrt (x_center, y_center).
    """
    spike_photons = (
        np.hypot(
            photon_array.x - x_center,
            photon_array.y - y_center,
        )
        > r
    )
    return np.arctan2(
        photon_array.y[spike_photons] - y_center,
        photon_array.x[spike_photons] - x_center,
    )


def test_rubin_diffraction_shows_field_rotation() -> None:
    """Checks that the spikes rotate."""
    latitude = -30.24463 * degrees
    azimuth = 45.0 * degrees
    altitude = 89.9 * degrees
    rubin_diffraction_optics = create_test_rubin_diffraction_optics(
        latitude, azimuth, altitude, rottelpos=0.0 * galsim.radians
    )
    dt = 1.0
    photon_array_0 = create_test_photon_array(t=0.0, n_photons=1000000)
    photon_array_1 = create_test_photon_array(t=dt, n_photons=1000000)
    local_wcs = create_test_wcs()
    rubin_diffraction_optics.applyTo(
        photon_array_0, local_wcs=local_wcs, rng=create_test_rng()
    )
    rubin_diffraction_optics.applyTo(
        photon_array_1, local_wcs=local_wcs, rng=create_test_rng()
    )

    # The expected image is contained in a disc + spikes outside the disc:
    spike_angles_0 = extract_spike_angles(
        photon_array_0,
        x_center=564.5,
        y_center=-1431.4,
        r=20.0,
    )
    spike_angles_1 = extract_spike_angles(
        photon_array_1,
        x_center=564.5,
        y_center=-1431.4,
        r=20.0,
    )

    # Find the angle, the cross is rotated relative to the axis cross:
    cross_rot_angle_0 = np.mean(spike_angles_0 % (np.pi / 2.0))
    cross_rot_angle_1 = np.mean(spike_angles_1 % (np.pi / 2.0))

    # Check that the angle of the crosses are rotated relative to each other:
    expected_angle_difference = field_rotation_angle(latitude, altitude, azimuth, dt)

    np.testing.assert_allclose(
        cross_rot_angle_1 - cross_rot_angle_0, expected_angle_difference, rtol=0.03
    )


def test_rubin_diffraction_does_not_show_field_rotation_when_deactivated() -> None:
    """Checks that the spikes dont rotate if disable_field_rotation is set."""
    latitude = -30.24463 * degrees
    azimuth = 45.0 * degrees
    altitude = 89.9 * degrees
    rubin_diffraction_optics = create_test_rubin_diffraction_optics(
        latitude, azimuth, altitude, disable_field_rotation=True
    )
    dt = 1.0
    photon_array_0 = create_test_photon_array(t=0.0, n_photons=100000)
    photon_array_1 = create_test_photon_array(t=dt, n_photons=100000)
    local_wcs = create_test_wcs()
    rubin_diffraction_optics.applyTo(
        photon_array_0, local_wcs=local_wcs, rng=create_test_rng()
    )
    rubin_diffraction_optics.applyTo(
        photon_array_1, local_wcs=local_wcs, rng=create_test_rng()
    )

    np.testing.assert_array_almost_equal(photon_array_0.x, photon_array_1.x)
    np.testing.assert_array_almost_equal(photon_array_0.y, photon_array_1.y)


def field_rotation_angle(
    latitude: float, altitude: float, azimuth: float, t: float
) -> float:
    """For given latitude and az/alt position of a star, compute the field rotation angle around this star after time t."""
    e_focal = diffraction.e_equatorial(
        latitude=latitude, altitude=altitude, azimuth=azimuth
    )
    e_z_0, e_z = diffraction.prepare_e_z(latitude)
    rot = diffraction.field_rotation_matrix(e_z_0, e_z, e_focal, np.array([t]))
    return np.arctan2(rot[0, 0, 1], rot[0, 0, 0])


def test_xy_to_v_inverse():
    """Tests if the transform photon_ops.XyToV and its inverse combine to
    the identity operation."""
    boresight = galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians)
    local_wcs = galsim.AffineTransform(
        0.168,
        0.108,
        -0.108,
        0.168,
        origin=galsim.PositionD(x=-0.349, y=-0.352),
        world_origin=galsim.PositionD(x=0.0, y=0.0),
    )
    icrf_to_field = create_test_icrf_to_field(
        boresight=boresight,
        det_name="R22_S11",
    )
    sky_pos = galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians)
    xy_to_v = photon_ops.XyToV(local_wcs, icrf_to_field, sky_pos)

    x, y = np.array(
        np.meshgrid(np.linspace(-10.0, 10, 20), np.linspace(-10.0, 10, 20))
    ).reshape((2, -1))
    v = xy_to_v(x, y)
    x_after, y_after = xy_to_v.inverse(v)
    np.testing.assert_array_almost_equal(x, x_after)
    np.testing.assert_array_almost_equal(y, y_after)


def test_xy_to_v():
    """Tests if the transform photon_ops.XyToV."""
    boresight = galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians)
    local_wcs = galsim.AffineTransform(
        0.168,
        0.108,
        -0.108,
        0.168,
        origin=galsim.PositionD(x=-0.349, y=-0.352),
        world_origin=galsim.PositionD(x=0.0, y=0.0),
    )
    icrf_to_field = create_test_icrf_to_field(
        boresight=boresight,
        det_name="R22_S11",
    )
    sky_pos = galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians)
    xy_to_v = photon_ops.XyToV(local_wcs, icrf_to_field, sky_pos)

    x, y = np.array(
        np.meshgrid(np.linspace(-10.0, 10, 20), np.linspace(-10.0, 10, 20))
    ).reshape((2, -1))
    v = xy_to_v(x, y)
    x_after, y_after = xy_to_v.inverse(v)
    np.testing.assert_array_almost_equal(x, x_after)
    np.testing.assert_array_almost_equal(y, y_after)


def assert_photon_ops_act_equal(photon_op, photon_op_reference):
    photon_array = apply_to_photon_array(photon_op)
    photon_array_reference = apply_to_photon_array(photon_op_reference)
    np.testing.assert_array_almost_equal(
        np.c_[photon_array.x, photon_array.y],
        np.c_[photon_array_reference.x, photon_array_reference.y],
    )


def apply_to_photon_array(photon_op):
    photon_array = create_test_photon_array(n_photons=10000)
    photon_op.applyTo(photon_array, local_wcs=create_test_wcs(), rng=create_test_rng())
    return photon_array


TEST_BASE_CONFIG = {
    "input": {
        "telescope": {
            "file_name": "LSST_r.yaml",
            "rotTelPos": np.pi / 3 * galsim.radians,
        }
    },
    "det_name": "R22_S11",
    "output": {"camera": "LsstCam"},
    "_icrf_to_field": create_test_icrf_to_field(
        galsim.CelestialCoord(
            1.1047934165124105 * galsim.radians, -0.5261230452954583 * galsim.radians
        ),
        "R22_S11",
    ),
    "sky_pos": galsim.CelestialCoord(
        1.1056660811384078 * galsim.radians, -0.5253441048502933 * galsim.radians
    ),
}
TEST_ALT_AZ_CONFIG = {"altitude": "43.0 degrees", "azimuth": "0.0 degrees"}


def test_config_rubin_diffraction():
    """Check the config interface to RubinDiffraction."""

    config = {
        **TEST_BASE_CONFIG,
        "stamp": {
            "photon_ops": [
                {
                    "type": "RubinDiffraction",
                    "latitude": "-30.24463 degrees",
                    **TEST_ALT_AZ_CONFIG
                }
            ]
        },
    }
    galsim.config.ProcessInput(config)
    galsim.config.input.SetupInputsForImage(config, None)
    [photon_op] = galsim.config.BuildPhotonOps(config["stamp"], "photon_ops", config)
    reference_op = create_test_rubin_diffraction(
        altitude=43.0 * degrees,
        azimuth=0.0 * degrees,
        sky_pos=TEST_BASE_CONFIG["sky_pos"],
        icrf_to_field=TEST_BASE_CONFIG["_icrf_to_field"],
    )
    assert_photon_ops_act_equal(photon_op, reference_op)


def test_config_rubin_diffraction_without_field_rotation():
    """Check the config interface to RubinDiffraction."""

    config = {
        **TEST_BASE_CONFIG,
        "stamp": {
            "photon_ops": [
                {
                    "type": "RubinDiffraction",
                    "latitude": "-30.24463 degrees",
                    "disable_field_rotation": True,
                    **TEST_ALT_AZ_CONFIG
                }
            ]
        },
    }
    galsim.config.ProcessInput(config)
    galsim.config.input.SetupInputsForImage(config, None)
    [photon_op] = galsim.config.BuildPhotonOps(config["stamp"], "photon_ops", config)
    reference_op = create_test_rubin_diffraction(
        altitude=43.0 * degrees,
        azimuth=0.0 * degrees,
        sky_pos=TEST_BASE_CONFIG["sky_pos"],
        disable_field_rotation=True,
        icrf_to_field=TEST_BASE_CONFIG["_icrf_to_field"],
    )
    assert_photon_ops_act_equal(photon_op, reference_op)


def test_config_rubin_diffraction_optics():
    """Check the config interface to RubinDiffractionOptics."""

    image_pos = galsim.PositionD(3076.4462608524213, 1566.4896702703757)
    config = {
        **TEST_BASE_CONFIG,
        "image_pos": image_pos,  # This would get set appropriately during normal config processing.
        "stamp": {
            "photon_ops": [
                {
                    "type": "RubinDiffractionOptics",
                    "camera": "LsstCam",
                    "boresight": {
                        "type": "RADec",
                        "ra": "1.1047934165124105 radians",
                        "dec": "-0.5261230452954583 radians",
                    },
                    "latitude": "-30.24463 degrees",
                    **TEST_ALT_AZ_CONFIG
                }
            ]
        },
    }
    galsim.config.ProcessInput(config)
    galsim.config.input.SetupInputsForImage(config, None)
    [photon_op] = galsim.config.BuildPhotonOps(config["stamp"], "photon_ops", config)
    reference_op = create_test_rubin_diffraction_optics(
        altitude=43.0 * degrees,
        azimuth=0.0 * degrees,
        sky_pos=TEST_BASE_CONFIG["sky_pos"],
        image_pos=image_pos,
        icrf_to_field=TEST_BASE_CONFIG["_icrf_to_field"],
        boresight=photon_op.boresight,
    )
    assert_photon_ops_act_equal(photon_op, reference_op)


def test_config_rubin_diffraction_optics_without_field_rotation():
    """Check the config interface to RubinDiffractionOptics."""

    image_pos = galsim.PositionD(3076.4462608524213, 1566.4896702703757)
    config = {
        **TEST_BASE_CONFIG,
        "image_pos": image_pos,  # This would get set appropriately during normal config processing.
        "stamp": {
            "photon_ops": [
                {
                    "type": "RubinDiffractionOptics",
                    "camera": "LsstCam",
                    "boresight": {
                        "type": "RADec",
                        "ra": "1.1047934165124105 radians",
                        "dec": "-0.5261230452954583 radians",
                    },
                    "disable_field_rotation": True,
                    **TEST_ALT_AZ_CONFIG
                }
            ]
        },
    }
    galsim.config.ProcessInput(config)
    galsim.config.input.SetupInputsForImage(config, None)
    [photon_op] = galsim.config.BuildPhotonOps(config["stamp"], "photon_ops", config)
    reference_op = create_test_rubin_diffraction_optics(
        altitude=43.0 * degrees,
        azimuth=0.0 * degrees,
        sky_pos=TEST_BASE_CONFIG["sky_pos"],
        image_pos=image_pos,
        icrf_to_field=TEST_BASE_CONFIG["_icrf_to_field"],
        boresight=photon_op.boresight,
        disable_field_rotation=True,
    )
    assert_photon_ops_act_equal(photon_op, reference_op)


def test_config_rubin_optics():
    """Check the config interface to RubinOptics."""

    image_pos = galsim.PositionD(3076.4462608524213, 1566.4896702703757)
    config = {
        **TEST_BASE_CONFIG,
        "image_pos": image_pos,  # This would get set appropriately during normal config processing.
        "stamp": {
            "photon_ops": [
                {
                    "type": "RubinOptics",
                    "camera": "LsstCam",
                    "boresight": {
                        "type": "RADec",
                        "ra": "1.1047934165124105 radians",
                        "dec": "-0.5261230452954583 radians",
                    },
                },
            ]
        },
    }
    galsim.config.ProcessInput(config)
    galsim.config.input.SetupInputsForImage(config, None)
    [photon_op] = galsim.config.BuildPhotonOps(config["stamp"], "photon_ops", config)
    reference_op = create_test_rubin_optics(
        sky_pos=TEST_BASE_CONFIG["sky_pos"],
        image_pos=image_pos,
        icrf_to_field=TEST_BASE_CONFIG["_icrf_to_field"],
        boresight=photon_op.boresight,
    )
    assert_photon_ops_act_equal(photon_op, reference_op)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k.startswith("test_") and callable(v)]
    for testfn in testfns:
        testfn()
