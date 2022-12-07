import numpy as np

from imsim import diffraction

T_SIDERIAL = 2 * np.pi / diffraction.OMEGA_EARTH  # One siderial rotation period


def test_directed_dist() -> None:
    geometry = diffraction.Geometry(
        thick_lines=np.array([[0.0, 1.0, 0.25, 0.125], [1.0, 0.0, 0.0, 0.0]]),
        circles=np.array([[0.0, 0.0, 1.0]]),
    )
    points = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
    d, n = diffraction.directed_dist(geometry, points)
    np.testing.assert_array_almost_equal(d, np.array([0.0, np.sqrt(2.0) - 1.0, 0.125]))
    np.testing.assert_array_almost_equal(
        n,
        np.array([[1.0, 0.0], [-1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)], [0.0, 1.0]]),
    )


def test_dist_thick_line() -> None:
    thick_line = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.5])
    point = np.array([0.5, 0.5])
    np.testing.assert_array_almost_equal(
        diffraction.dist_thick_line(thick_line, point), 0.5
    )

    thick_line = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.5])
    point = np.array([0.0, 1.0])
    np.testing.assert_array_almost_equal(
        diffraction.dist_thick_line(thick_line, point), 1.0 / np.sqrt(2.0) - 0.5
    )

    thick_line = np.array([[0.0, 1.0, 0.25, 0.125], [1.0, 0.0, 0.0, 0.0]])
    point = np.array([2.0, 2.0])
    np.testing.assert_array_almost_equal(
        diffraction.dist_thick_line(thick_line, point),
        np.array([1.625, 2.0]),
    )

    thick_line = np.array([0.0, 1.0, 0.25, 0.125])
    point = np.array([[2.0, 2.0], [-2.0, 2.0], [2.0, -2.0]])
    np.testing.assert_array_almost_equal(
        diffraction.dist_thick_line(thick_line, point),
        np.array([1.625, 1.625, 2.125]),
    )

    thick_line = np.array([[0.0, 1.0, 0.25, 0.125], [1.0, 0.0, 0.0, 0.0]])
    point = np.array([[2.0, 2.0], [-2.0, 2.0], [2.0, -2.0]])
    np.testing.assert_array_almost_equal(
        diffraction.dist_thick_line(thick_line, point),
        np.array([[1.625, 1.625, 2.125], [2.0, 2.0, 2.0]]),
    )


def test_dist_circle() -> None:
    circle = np.array([-1.0, 1.0, 1.0])
    point = np.array([0.5, 3.0])
    np.testing.assert_array_almost_equal(diffraction.dist_circle(circle, point), 1.5)

    circle = np.array([[0.0, 1.0, 1.0], [-1.0, 10.0, 3.0]])
    point = np.array([4.0, -2.0])
    np.testing.assert_array_almost_equal(
        diffraction.dist_circle(circle, point),
        np.array([4.0, 10.0]),
    )

    circle = np.array([-1.0, 1.0, 1.0])
    points = np.array([[0.5, 3.0], [-2.5, -1.0]])
    np.testing.assert_array_almost_equal(
        diffraction.dist_circle(circle, points),
        [1.5, 1.5],
    )

    circle = np.array([[0.0, 1.0, 1.0], [-1.0, 10.0, 3.0]])
    points = np.array([[4.0, -2.0], [4.0, -2.0], [-4.0, 4.0]])
    np.testing.assert_array_almost_equal(
        diffraction.dist_circle(circle, points),
        [[4.0, 4.0, 4.0], [10.0, 10.0, 3.0 * (np.sqrt(5.0) - 1.0)]],
    )


def test_decompose_e_z() -> None:
    lat = 45.0
    e_z_ax, e_z_perp_x, e_z_perp_y = diffraction.decompose_e_z(lat)
    np.testing.assert_array_almost_equal(e_z_ax.dot(e_z_perp_x), 0.0)
    np.testing.assert_array_almost_equal(e_z_ax.dot(e_z_perp_y), 0.0)
    np.testing.assert_array_almost_equal(e_z_perp_x.dot(e_z_perp_y), 0.0)
    np.testing.assert_array_almost_equal(e_z_ax + e_z_perp_x, diffraction.E_Z)
    np.testing.assert_array_almost_equal(np.linalg.norm(e_z_ax + e_z_perp_y), 1.0)
    np.testing.assert_array_almost_equal(
        np.cross(e_z_ax, np.array([0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])),
        np.zeros(3),
    )


def rot_2x2(alpha: np.array) -> np.array:
    rot = np.empty(np.shape(alpha) + (2, 2))
    rot[..., 0, 0] = rot[..., 1, 1] = np.cos(alpha)
    rot[..., 0, 1] = np.sin(alpha)
    rot[..., 1, 0] = -rot[..., 0, 1]
    return rot


def test_field_rotation_matrix_is_correct_at_ncp() -> None:
    """A telescope centered at the north celestial pole (NCP)
    will observe a uniform rotation of the stars around NCP."""
    lat = 40.0
    t = 3600.0 * np.linspace(-2.0, 2.0, num=10)
    e_star = np.array(
        [[0.0, np.cos(lat / 180.0 * np.pi), np.sin(lat / 180.0 * np.pi)]] * t.size
    )  # NCP
    rot = diffraction.field_rotation_matrix(lat, e_star, t)

    # Around NCP, the field rotation angle should agree with earth's rotation angle
    # around its axis:
    np.testing.assert_array_almost_equal(rot, rot_2x2(diffraction.OMEGA_EARTH * t))


def test_field_rotation_matrix_is_correct_at_horizon_east_and_west() -> None:
    """A star moving along the celestial equatorial will rise and set at east and west at the horizon."""
    lat = 30.0
    t = np.array([-T_SIDERIAL / 4.0, T_SIDERIAL / 4.0])
    e_star = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # West and East at horizon
    rot = diffraction.field_rotation_matrix(lat, e_star, t)

    # At horizon, the field rotation angle should be exactly +/- (90°-lat):
    np.testing.assert_array_almost_equal(
        rot,
        np.array(
            (
                rot_2x2(-(90.0 - lat) / 180.0 * np.pi),
                rot_2x2((90.0 - lat) / 180.0 * np.pi),
            )
        ),
    )


def test_field_rotation_matrix_is_correct_near_zenith() -> None:
    """Test if a star near zenith has a field rotation angle similar
    to the angle predicted by the field rotation rate formula."""

    alt = 89.9
    az = 45.0
    lat = -30.24463
    dt = 1.0
    t = np.linspace(0.0, dt, 100)
    e_star = diffraction.star_trace(latitude=lat, altitude=alt, azimuth=az, t=t)
    rot = diffraction.field_rotation_matrix(lat, e_star[-1:], np.array([t[-1]]))

    alt_t = np.arctan2(e_star[:, 2], np.hypot(e_star[:, 0], e_star[:, 1]))
    az_t = np.arctan2(e_star[:, 0], e_star[:, 1])
    rate = (
        diffraction.OMEGA_EARTH
        * np.cos(lat / 180.0 * np.pi)
        * np.cos(az_t)
        / np.cos(alt_t)
    )
    # Expected field rotation angle is the integral over the rate:
    expected_angle = np.trapz(rate, t)

    alpha = np.arctan2(rot[0, 0, 1], rot[0, 0, 0])
    np.testing.assert_allclose(alpha, expected_angle, rtol=1.0e-7)


def test_star_trace_is_correct_at_equator() -> None:
    """Observing a star in zenith from equator."""
    t = np.array([-T_SIDERIAL / 4.0, 0.0, T_SIDERIAL / 4.0])
    e_star = diffraction.star_trace(0.0, 90.0, 0.0, t)
    # Star raises E, passes zenith and sets W:
    np.testing.assert_array_almost_equal(
        e_star, np.array([[1.0, 0.0, 0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    )


def test_star_trace_is_correct_at_scp() -> None:
    """Observing a star in the South Celestial Pole."""
    lat = -45.0
    alt = 90.0 - np.abs(lat)
    az = 180.0
    t = np.array([-T_SIDERIAL / 4.0, 0.0, T_SIDERIAL / 4.0, T_SIDERIAL / 2.0])
    e_star = diffraction.star_trace(lat, alt, az, t)
    scp = np.array([0.0, -1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    # Star raises E, passes zenith and sets W:
    np.testing.assert_array_almost_equal(e_star, np.array([scp] * 4))


def test_star_trace_is_correct_at_zenith() -> None:
    """Observing a star at zenith from a latitude of 45°."""
    lat = 45.0
    alt = 90.0
    az = 0.0
    t = np.array([-T_SIDERIAL / 4.0, 0.0, T_SIDERIAL / 4.0, T_SIDERIAL / 2.0])
    e_star = diffraction.star_trace(lat, alt, az, t)
    s = 1.0 / np.sqrt(2.0)
    # Star visible NE, zenith, NW and north:
    np.testing.assert_array_almost_equal(
        e_star,
        np.array([[s, 0.5, 0.5], [0.0, 0.0, 1.0], [-s, 0.5, 0.5], [0.0, 1.0, 0.0]]),
    )


def test_star_trace_yields_back_alt_az_for_t_eq_0() -> None:
    """For t=0 we should easily get back alt/az."""
    lat = 42.0
    alt = 13.0
    az = 55.0
    t = np.array([0.0])
    e_star = diffraction.star_trace(lat, alt, az, t).squeeze()
    alt2 = np.arctan2(e_star[2], np.hypot(e_star[0], e_star[1])) / np.pi * 180.0
    az2 = np.arctan2(e_star[0], e_star[1]) / np.pi * 180.0

    np.testing.assert_array_almost_equal(alt, alt2)
    np.testing.assert_array_almost_equal(az, az2)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
