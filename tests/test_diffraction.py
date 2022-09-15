import numpy as np

from imsim import diffraction


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
