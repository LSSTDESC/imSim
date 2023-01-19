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


def apply_diffraction_delta(
    pos: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    wavelength: np.ndarray,
    lat: float,
    az: float,
    alt: float,
    geometry: Geometry,
    distribution: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Statistically diffract photons.

    For a ray of photons with positions pos, wavelengths wavelength and pupil plane
    coordinates pos,
    randomly generate diffraction angles and change the directions of the photons
    entering the pupil plane.

    Parameters
    ----------
    pos : 2d positions of the intersections of the rays with the pupil plane (shape (n, 2)).
    v : Direction of the photons (shape (n, 3)).
    t : Time [sec] when photons enter the pupil plane (shape (n,)).
    wavelength : Wavelength of the photons (shape (n,)).
    lat : Latitude of the telescope.
    az : Azimuth of the focal point of the telescope at t=0.
    alt : Altitude of the focal point of the telescope at t=0.
    geometry : Geometry representing the 2d projection of the spider of a telescope
        into the pupil plane.
    distribution : Random number generator representing the distribution of the diffraction
        angles (depending on the wavelength and distance to the geometry).
        It should be a callable accepting an array for phi_star and returning the
        random values for phi
    """
    e_focal = star_trace(latitude=lat, azimuth=az, altitude=alt, t=t)
    R = field_rotation_matrix(lat, e_focal, t)

    def rot(w):
        return np.einsum("kij,kj->ki", R, w)

    def rot_inv(w):
        return np.einsum("kji,kj->ki", R, w)

    # Rotate position in the inverse direction of the field rotation, then rotate back
    # the shift:
    shift = rot(
        diffraction_delta(rot_inv(pos), -v[:, 2], wavelength, geometry, distribution)
    )
    v_before = np.linalg.norm(v, axis=1)
    v[:, :2] += shift
    # renormalize (leave norm invariant)
    v_after = np.linalg.norm(v, axis=1)
    f_scale = v_before / v_after
    v *= f_scale[:, None]
    return v


def diffraction_delta(
    pos: np.ndarray,
    v_z: np.ndarray,
    wavelength: np.ndarray,
    geometry: Geometry,
    distribution: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """For a ray of photons with positions pos, wavelengths wavelength,
    randomly generate diffraction angles and compute the resulting change of the
    directions of the photons entering the pupil plane.

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


# Earth rotation, (T: Sidereal Rotation Period, 2 pi/T) [1/sec]:
OMEGA_EARTH = 7.292115826090781e-05
E_Z = np.array([0.0, 0.0, 1.0])


def decompose_e_z(lat: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a decomposition of e_z (direction to zenith) in 3 orthogonal
    components along
    * ax (earth axis)
    * a_perp (projection of e_z into the plane perpendicular to ax)
    * a_perp x ax.

    Will fail at lat = +/- 90°"""
    ax = np.array([0.0, np.cos(lat / 180.0 * np.pi), np.sin(lat / 180.0 * np.pi)])
    e_z_ax = ax.dot(E_Z) * ax
    return e_z_ax, E_Z - e_z_ax, np.cross(ax, E_Z)


def field_rotation_matrix(lat: float, e_star: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Computes the field rotation matrix for a given latitude lat, time t and a star
    on the celestial sphere in cartesian coordinates relative to the observer.

    Parameters
    ----------
    lat : Latitude of the observer
    e_star : Numpy array of shape (n, 3) containing points on the celestial sphere
        in cartesian coordinates (x,y,z).
        x: east, y: north, z direction: direction to zenith.
    t : Numpy array of shape (n,) containing the times of observation of the coordinates in e_star

    Returns:
        Numpy array of shape (n, 2, 2) containing a rotation matrix for each point in e_star.
        The matrices live in the tangent spaces of the points in e_star, with the
        y axis pointing to the zenith of the oberver.
    """
    e_z_ax, e_z_perp_x, e_z_perp_y = decompose_e_z(lat)
    e_z_rot = (
        e_z_ax
        + np.cos(-OMEGA_EARTH * t[:, None]) * e_z_perp_x
        + np.sin(-OMEGA_EARTH * t[:, None]) * e_z_perp_y
    )
    e_h_0 = np.cross(e_star, e_z_rot)
    e_h = np.cross(e_star, E_Z)
    rot = np.empty(np.shape(t) + (2, 2))
    nrm = np.linalg.norm(e_h, axis=-1) * np.linalg.norm(e_h_0, axis=-1)
    rot[:, 1, 1] = rot[:, 0, 0] = np.einsum("ij,ij->i", e_h, e_h_0) / nrm
    rot[:, 0, 1] = np.einsum("ij,ij->i", e_star, np.cross(e_h_0, e_h)) / nrm
    rot[:, 1, 0] = -rot[:, 0, 1]
    return rot


def star_trace(
    latitude: float, altitude: float, azimuth: float, t: np.ndarray
) -> np.ndarray:
    """Computes the trace of a star at given times t,
    observed from a given latitude,
    which passes altitude/azimuth at time t=0.

    The output is an array of shape (t.size, 3) in an earth fixed
    cartesian coordinate system (x: East, y: North, z: zenith).
    """

    lat = latitude / 180.0 * np.pi
    alt = altitude / 180.0 * np.pi
    az = azimuth / 180.0 * np.pi

    e_r_0 = np.array(
        [
            np.cos(lat),
            0.0,
            np.sin(lat),
        ]
    )
    e_east_0 = np.array([0.0, 1.0, 0.0])
    e_north_0 = np.array([-e_r_0[2], 0.0, e_r_0[0]])
    # Transform star at t=0 from earth fixed system to celestial system:
    e_star = (
        e_east_0 * np.cos(alt) * np.sin(az)
        + e_north_0 * np.cos(alt) * np.cos(az)
        + e_r_0 * np.sin(alt)
    )

    # Transform celestial system -> earth fixed system
    e_enr = np.column_stack(
        [
            # e_east
            -np.sin(OMEGA_EARTH * t),
            np.cos(OMEGA_EARTH * t),
            np.zeros(t.size),
            # e_north
            -np.sin(lat) * np.cos(OMEGA_EARTH * t),
            -np.sin(lat) * np.sin(OMEGA_EARTH * t),
            np.full(t.size, np.cos(lat)),
            # e_r
            np.cos(lat) * np.cos(OMEGA_EARTH * t),
            np.cos(lat) * np.sin(OMEGA_EARTH * t),
            np.full(t.size, np.sin(lat)),
        ]
    ).reshape((-1, 3, 3))

    return e_enr.dot(e_star)
