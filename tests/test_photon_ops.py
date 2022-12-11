import numpy as np
import galsim
import batoid
from astropy.time import Time
from astropy import units

from imsim import photon_ops, BatoidWCSFactory, get_camera, diffraction
from imsim.opsim_meta import OpsimMetaDict
from imsim.telescope_loader import load_telescope

def create_test_icrf_to_field(boresight, det_name):
    camera = get_camera()
    telescope = load_telescope("LSST_r.yaml", rotTelPos=np.pi/3*galsim.radians)
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


def create_test_lsst_optics():
    boresight = galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians)
    telescope = load_telescope("LSST_r.yaml")

    det_name = "R22_S11"
    return photon_ops.LsstOptics(
        telescope=telescope,
        boresight=boresight,
        sky_pos=galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians),
        image_pos=galsim.PositionD(809.6510740536025, 3432.6477953336625),
        icrf_to_field=create_test_icrf_to_field(boresight, det_name),
        det_name=det_name,
        camera=get_camera(),
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


def create_test_lsst_diffraction():
    boresight = galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians)
    det_name = "R22_S11"

    return photon_ops.LsstDiffraction(
        telescope=load_telescope("LSST_r.yaml"),
        sky_pos=galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians),
        icrf_to_field=create_test_icrf_to_field(boresight, det_name),
        latitude=-30.24463,
        azimuth=45.0,
        altitude=89.9,
    )


def create_test_rng():
    return galsim.random.BaseDeviate(seed=42)


def test_lsst_optics() -> None:
    """Check that the image of a star is contained in a disc."""

    lsst_optics = create_test_lsst_optics()
    photon_array = create_test_photon_array()
    local_wcs = create_test_wcs()
    lsst_optics.applyTo(photon_array, local_wcs=local_wcs, rng=create_test_rng())
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


def test_lsst_diffraction_produces_spikes() -> None:
    """Checks that we have spike photons and that the spkies form a cross."""
    lsst_diffraction = create_test_lsst_diffraction()
    photon_array = create_test_photon_array(n_photons=1000000)
    local_wcs = create_test_wcs()
    lsst_diffraction.applyTo(photon_array, local_wcs=local_wcs, rng=create_test_rng())
    lsst_optics = create_test_lsst_optics()
    lsst_optics.applyTo(photon_array, local_wcs=local_wcs, rng=create_test_rng())

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


def extract_spike_angles(photon_array, x_center, y_center, r):
    """Filters out a disc centered at (x_center, y_center) with radius r.
    The reminding photons will be considered as spike photons.
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


def test_lsst_diffraction_shows_field_rotation() -> None:
    """Checks that the spikes rotate."""
    lsst_diffraction = create_test_lsst_diffraction()
    dt = 1.0
    photon_array_0 = create_test_photon_array(t=0.0, n_photons=1000000)
    photon_array_1 = create_test_photon_array(t=dt, n_photons=1000000)
    local_wcs = create_test_wcs()
    lsst_optics = create_test_lsst_optics()
    lsst_diffraction.applyTo(photon_array_0, local_wcs=local_wcs, rng=create_test_rng())
    lsst_optics.applyTo(photon_array_0, local_wcs=local_wcs, rng=create_test_rng())
    lsst_diffraction.applyTo(photon_array_1, local_wcs=local_wcs, rng=create_test_rng())
    lsst_optics.applyTo(photon_array_1, local_wcs=local_wcs, rng=create_test_rng())

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
    expected_angle_difference = field_rotation_angle(
        lsst_diffraction.latitude,
        lsst_diffraction.altitude,
        lsst_diffraction.azimuth,
        dt,
    )

    np.testing.assert_allclose(
        cross_rot_angle_1 - cross_rot_angle_0, expected_angle_difference, rtol=0.03
    )


def field_rotation_angle(
    latitude: float, altitude: float, azimuth: float, t: float
) -> float:
    """For given latitude and az/alt position of a star, compute the field rotation angle around this star after time t."""
    e_star = diffraction.star_trace(
        latitude=latitude, altitude=altitude, azimuth=azimuth, t=np.array([t])
    )
    rot = diffraction.field_rotation_matrix(latitude, e_star, np.array([t]))
    (alpha,) = np.arctan2(rot[:, 0, 1], rot[:, 0, 0])
    return alpha


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


def test_config_lsst_diffraction():
    """Check the config interface to BatoidPhotonOps."""

    boresight = galsim.CelestialCoord(
        1.1047934165124105 * galsim.radians, -0.5261230452954583 * galsim.radians
    )
    config = {
        "input": {
            "telescope": {
                "file_name":"LSST_r.yaml",
            }
        },
        "_input_objs": {
            "opsim_meta_dict": [
                OpsimMetaDict.from_dict({"altitude": 43.0, "azimuth": 0.0})
            ]
        },
        "_icrf_to_field": create_test_icrf_to_field(boresight, "R22_S11"),
        "sky_pos": {
            "type": "RADec",
            "ra": "1.1056660811384078 radians",
            "dec": "-0.5253441048502933 radians",
        },
        "stamp": {
            "photon_ops": [
                {
                    "type": "lsst_diffraction",
                    "latitude": -30.24463,
                }
            ]
        },
    }
    galsim.config.ProcessInput(config)
    galsim.config.BuildPhotonOps(config["stamp"], "photon_ops", config)


def test_config_lsst_optics():
    """Check the config interface to BatoidPhotonOps."""

    boresight = galsim.CelestialCoord(
        1.1047934165124105 * galsim.radians, -0.5261230452954583 * galsim.radians
    )
    config = {
        "input": {
            "telescope": {
                "file_name":"LSST_r.yaml",
            }
        },
        "sky_pos": {
            "type": "RADec",
            "ra": "1.1056660811384078 radians",
            "dec": "-0.5253441048502933 radians",
        },
        "det_name": "R22_S11",
        "image_pos": galsim.PositionD(
            3076.4462608524213, 1566.4896702703757
        ),  # This would get set appropriately during normal config processing.
        "_icrf_to_field": create_test_icrf_to_field(boresight, "R22_S11"),
        "stamp": {
            "photon_ops": [
                {
                    "type": "lsst_optics",
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
    galsim.config.BuildPhotonOps(config["stamp"], "photon_ops", config)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
