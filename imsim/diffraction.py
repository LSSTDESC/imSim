"""Routines to implement diffraction due to the LSST spider.

The core idea is taken from https://ntrs.nasa.gov/citations/19990094899
For more details, see doc/diffraction.md (Statistical Aproach)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np


@dataclass
class Geometry:
    """Defining a 2D geometry consisting of circles and lines with a thickness.

    Parameters
    ----------
    thick_lines : numpy array of shape (n,4) where each row has the form
        [nx, ny, d, thickness]. (nx, ny) is the normal to the line and d is
        such that d*(nx, ny) lies on the line.
    circles : numpy array of shape (n,3) where each row has the form
        [x, y, r]. (x,y) is the center of the circle, and r the radius.
    """

    thick_lines: np.ndarray
    circles: np.ndarray


LSST_SPIDER_GEOMETRY = Geometry(
    thick_lines=np.array(
        [
            [1 / np.sqrt(2.0), 1 / np.sqrt(2.0), -0.4, 0.025],
            [-1 / np.sqrt(2.0), 1 / np.sqrt(2.0), -0.4, 0.025],
            [1 / np.sqrt(2.0), 1 / np.sqrt(2.0), 0.4, 0.025],
            [-1 / np.sqrt(2.0), 1 / np.sqrt(2.0), 0.4, 0.025],
        ]
    ),
    circles=np.array([[0.0, 0.0, 2.558], [0.0, 0.0, 4.18]]),
)


def diffraction_kick(
    pos: np.ndarray,
    v_z: np.ndarray,
    wavelength: np.ndarray,
    geometry: Geometry,
    distribution: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """For a ray of photons with positions pos, wavelengths wavelength,
    randomly generate diffraction angles and compute the resulting change of the
    directions of the photons within the pupil plane.

    Parameters
    ----------
    pos : 2d positions of the intersections of the rays with the pupil plane.
    v_z : Component of the direction of the photons perpendicular to the pupil plane.
    wavelength : Wavelength of the photons.
    geometry : Geometry representing the 2d projection of the spider of a telescope
        into the pupil plane.
    distribution : Random number generator representing the distribution of the diffraction
        angles (depending on the wavelength and distance to the geometry).
        It should be a callable accepting an array for phi_star and returning the
        random values for phi
    """

    d, n = directed_dist(geometry, pos)
    d_tan_phi = distribution(phi_star(d, wavelength))
    return d_tan_phi[:, None] * v_z[:, None] * n


def phi_star(delta: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """Standard deviation in radians for a deflection.

    According to https://ntrs.nasa.gov/citations/19990094899
    Equation (4.9)
    """
    k = 2.0 * np.pi / wavelength
    return np.arctan(1.0 / (2.0 * k * delta))


def directed_dist(
    geometry: Geometry, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Determine the minimal distance to the objects defined by geometry,
    together with the directions to the minimizing points relative to points.

    Parameters
    ----------
    geometry: Geometry containing cirlces and thick lines.
    points: Numpy array of shape (2,n)

    Returns:
        Numpy array of shape (3,n), where each line is of the form [d, nx, ny],
        d being the distance and [nx, ny] the direction to the minimizing point.
    """
    dist_lines = dist_thick_line(geometry.thick_lines, points)
    dist_circles = dist_circle(geometry.circles, points)
    n_points = points.shape[0]
    col_idx = np.arange(n_points, dtype=np.intp)
    min_line_idx = np.argmin(dist_lines, axis=0)
    min_circle_idx = np.argmin(dist_circles, axis=0)
    min_dist_lines = dist_lines[min_line_idx, col_idx]
    min_dist_circles = dist_circles[min_circle_idx, col_idx]
    dist = np.empty((n_points,))
    n = np.empty((n_points, 2))
    # mask for points which are closer to some line than to any circle:
    line_mask = min_dist_lines < min_dist_circles
    dist[line_mask] = min_dist_lines[line_mask]
    n[line_mask] = geometry.thick_lines[min_line_idx[line_mask]][..., :2]
    dist[~line_mask] = min_dist_circles[~line_mask]
    d = geometry.circles[min_circle_idx[~line_mask]][..., :2] - points[~line_mask]
    n[~line_mask] = d / np.linalg.norm(d)
    return dist, n


def dist_thick_line(thick_line: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Determines the distance of one or multiple thick lines to one or multiple points.

    Parameters
    ----------
    thick_line : Numpy array of shape (4,) or (n,4). See doc of Geometry on how the
        rows of thick_line should look like.
    point: Numpy array of shape (2,) or (m,2).

    Returns:
        Numpy array of shape (), (n,), (m,) or (n,m) depending on the shapes of
        thick_lines and point.
    """
    n = thick_line[..., :2]
    d = thick_line[..., 2]
    thickness = thick_line[..., 3]
    out = np.empty(d.shape + point.shape[:-1])
    out_v = np.atleast_1d(out)

    point_extra_shape = (len(point.shape) - 1) * (1,)
    d_ext = d.reshape(d.shape + point_extra_shape)
    thickness_ext = thickness.reshape(thickness.shape + point_extra_shape)

    # The extra dim for point is to align dimensions for np.dot
    out_v[:] = np.abs(np.abs(n.dot(point[..., None])[..., 0] - d_ext) - thickness_ext)
    return out


def dist_circle(circle: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Determines the distance of one or circles lines to one or multiple points.

    Parameters
    ----------
    circle : Numpy array of shape (3,) or (n,3). See doc of Geometry on how the
        rows of circle should look like.
    point: Numpy array of shape (2,) or (m,2).

    Returns:
        Numpy array of shape (), (n,), (m,) or (n,m) depending on the shapes of
        circles and point.
    """
    c = circle[..., :2]
    r = circle[..., 2]
    out = np.empty(r.shape + point.shape[:-1])
    out_v = np.atleast_1d(out)
    point_extra_shape = (len(point.shape) - 1) * (1,)
    c_ext = c.reshape(c.shape[:-1] + point_extra_shape + (c.shape[-1],))
    r_ext = r.reshape(r.shape + point_extra_shape)
    out_v[:] = np.abs(np.linalg.norm(point - c_ext, axis=-1) - r_ext)
    return out
