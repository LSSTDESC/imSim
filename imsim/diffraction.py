"""Routines to implement diffraction due to the Rubin spider.

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


RUBIN_SPIDER_GEOMETRY = Geometry(
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


def apply_delta_v(v: np.ndarray, delta_v: np.ndarray) -> np.ndarray:
    """Applies a change in the x-y plane to a vector v, preserving the norm of the
    original vector v.

    Parameters
    ----------
    v : Direction of the photons (shape (n, 3)).
    delta_v : Change to apply (shape (n, 2).
    """
    v_before = np.linalg.norm(v, axis=1)
    v[:, :2] += delta_v
    # renormalize (leave norm invariant)
    v_after = np.linalg.norm(v, axis=1)
    f_scale = v_before / v_after
    v *= f_scale[:, None]
    return v


def apply_diffraction_delta_field_rot(
    pos: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    wavelength: np.ndarray,
    field_rot_matrix: Callable[[np.ndarray], np.ndarray],
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
    field_rot_matrix: A function returning the field rotation matrix for given times t
        (as returned by prepare_field_rotation_matrix).
    geometry : Geometry representing the 2d projection of the spider of a telescope
        into the pupil plane.
    distribution : Random number generator representing the distribution of the diffraction
        angles (depending on the wavelength and distance to the geometry).
        It should be a callable accepting an array for phi_star and returning the
        random values for phi
    """
    shift = diffraction_delta_field_rot(
        pos, -v[:, 2], t, wavelength, field_rot_matrix, geometry, distribution
    )
    return apply_delta_v(v, shift)


def apply_diffraction_delta(
    pos: np.ndarray,
    v: np.ndarray,
    wavelength: np.ndarray,
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
    wavelength : Wavelength of the photons (shape (n,)).
    geometry : Geometry representing the 2d projection of the spider of a telescope
        into the pupil plane.
    distribution : Random number generator representing the distribution of the diffraction
        angles (depending on the wavelength and distance to the geometry).
        It should be a callable accepting an array for phi_star and returning the
        random values for phi
    """

    shift = diffraction_delta(pos, -v[:, 2], wavelength, geometry, distribution)
    return apply_delta_v(v, shift)


def diffraction_delta_field_rot(
    pos: np.ndarray,
    v_z: np.ndarray,
    t: np.ndarray,
    wavelength: np.ndarray,
    field_rot_matrix: Callable[[np.ndarray], np.ndarray],
    geometry: Geometry,
    distribution: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    R = field_rot_matrix(t)

    def rot(w):
        return np.einsum("kij,kj->ki", R, w, out=w)

    def rot_inv(w):
        return np.einsum("kji,kj->ki", R, w)

    # Rotate position in the inverse direction of the field rotation, then rotate back
    # the shift:
    return rot(diffraction_delta(rot_inv(pos), v_z, wavelength, geometry, distribution))


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
    geometry: Geometry containing circles and thick lines.
    points: Numpy array of shape (2,n)

    Returns:
    dist: Numpy array of shape (n,) with the minimal distance to the closest
     of the objects defined in the geometry.
    n: Numpy array of shape (3,n), where each line is of the form [d, nx, ny],
     d being the distance and [nx, ny] the direction to the minimizing point.
    """
    nlines = geometry.thick_lines.shape[0]
    ncircles = geometry.circles.shape[0]
    min_dist = np.full(points.shape[0], np.inf)
    min_idx = np.full(points.shape[0], -1, dtype=np.intp)
    # Loop through all structures in geometry to determine which is closest to
    # each point.
    for idx in range(nlines + ncircles):
        if idx < nlines:
            # Structure is a thick line.
            thick_line = geometry.thick_lines[idx, :]
            distance = dist_thick_line(thick_line[:], points)
        else:
            # Structure is a circle.
            circle = geometry.circles[idx-nlines, :]
            distance = dist_circle(circle, points)
        # Update minimum distances and structure IDs.
        dist_mask = distance < min_dist
        min_dist[dist_mask] = distance[dist_mask]
        min_idx[dist_mask] = idx
    # At this point, we know which structure is closest to each point, and what
    # the minimum distance between them is. We need the directions to those structures.
    n = np.empty((points.shape[0], 2))
    # For points which are closer to some line than to any circle, the directions to
    # those lines are their normals.
    line_mask = min_idx < nlines
    n[line_mask] = geometry.thick_lines[min_idx[line_mask]][..., :2]
    # For points closer to a circle than to a line, compute vector from point to
    # circle center and normalize it.
    d = geometry.circles[min_idx[~line_mask]-nlines][..., :2] - points[~line_mask]
    n[~line_mask] = d / np.linalg.norm(d)
    return min_dist, n


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


def prepare_e_z(lat: float) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Prepares a function computing the direction
    to zenith of an observer at latitude lat (in radian), given in cartesian coordinates
    in an equatorial system:
      z: earth axis
      x: Projection of observers position to the equatorial plane
      y: Orhtonormal complement to x and z.

    Will fail at lat = +/- 90°"""
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)

    def _e_z(t: np.ndarray) -> np.ndarray:
        out = np.empty((np.size(t), 3))
        np.cos(OMEGA_EARTH * t, out=out[:, 0])
        np.sin(OMEGA_EARTH * t, out=out[:, 1])
        out[:, 2] = sin_lat
        out[:, :2] *= cos_lat
        return out

    return np.array([cos_lat, 0.0, sin_lat]), _e_z


def prepare_field_rotation_matrix(
    latitude: float, azimuth: float, altitude: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Prepares a callable returning the field rotation matrix for given times t.
    The field rotation matrix will be for a given latitude, focussing azimuth / altitude.
    """
    e_z_0, e_z = prepare_e_z(latitude)
    e_focal = e_equatorial(latitude=latitude, azimuth=azimuth, altitude=altitude)
    return lambda t: field_rotation_matrix(e_z_0, e_z, e_focal, t)


def field_rotation_sin_cos(
    e_z_0: np.ndarray,
    e_z: Callable[[np.ndarray], np.ndarray],
    e_focal: np.ndarray,
    t: np.ndarray,
    out: np.ndarray
):
    """Computes `sin(theta), cos(theta)` (`theta`: field rotation angle)
    for a given latitude lat,times t and a focal point given by cartesian coordinates
    in an equatorial system.

    Parameters
    ----------
    e_z_0 : returning the direction to zenith relative to the observer at t=0
        (as obtained by prepare_e_z).
    e_z : A function returning the direction to zenith relative to the observer,
        in an equatorial system (as obtained by prepare_e_z).
    e_focal : Numpy array of shape (3,) containing the focal point
        in cartesian coordinates (x,y,z).
        x: Projection of the observers position to the equatorial plane,
        z direction: earth axis.
    t : Numpy array of shape (n,) containing the times of observation.

    Returns:
        Numpy array of shape (n, 2, 2) containing a rotation matrix for each time in t.
        The matrices live in the tangent spaces of the points in e_star, with the
        y axis pointing to the zenith of the oberver.
    """
    e_z_rot = e_z(t)
    e_h = np.cross(e_focal, e_z_rot)
    e_h_0 = np.cross(e_focal, e_z_0)[None, :]
    nrm = np.linalg.norm(e_h, axis=-1) * np.linalg.norm(e_h_0, axis=-1)
    out[..., 0] = np.einsum("ij,ij->i", e_h, e_h_0) / nrm
    out[..., 1] = np.einsum("ij,ij->i", e_z_rot, e_h_0) / nrm


def field_rotation_matrix(
    e_z_0: np.ndarray,
    e_z: Callable[[np.ndarray], np.ndarray],
    e_focal: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Computes the field rotation matrix for a given latitude lat, times t and a
    focal point given by cartesian coordinates in an equatorial system.

    Parameters
    ----------
    e_z_0 : returning the direction to zenith relative to the observer at t=0
        (as obtained by prepare_e_z).
    e_z : A function returning the direction to zenith relative to the observer,
        in an equatorial system (as obtained by prepare_e_z).
    e_focal : Numpy array of shape (3,) containing the focal point
        in cartesian coordinates (x,y,z).
        x: Projection of the observers position to the equatorial plane,
        z direction: earth axis.
    t : Numpy array of shape (n,) containing the times of observation.

    Returns:
        Numpy array of shape (n, 2, 2) containing a rotation matrix for each time in t.
        The matrices live in the tangent spaces of the points in e_star, with the
        y axis pointing to the zenith of the oberver.
    """
    rot = np.empty(np.shape(t) + (2, 2))
    field_rotation_sin_cos(e_z_0, e_z, e_focal, t, rot[:, 0, :])
    rot[:, 1, 1] = rot[:, 0, 0]
    rot[:, 1, 0] = -rot[:, 0, 1]
    return rot


def e_equatorial(latitude: float, altitude: float, azimuth: float) -> np.ndarray:
    """Transforms a position on the celestial sphere at altitude/azimuth, observed
    from a given latitude to cartesian coordinates in an equatorial system.

    The output is an array of shape (t.size, 3) in an equatorial
    cartesian coordinate system (x: Projection of the observers position to the equatorial plane,
    y: x rotated 90° around z, z: earth axis).

    Parameters
    ----------
    latitude: Latitude of the observer [rad]
    altitude: Altitude of the observed object [rad]
    azimuth: Azimuth of the observed object [rad]
    """
    e_zenith = np.array(
        [
            np.cos(latitude),
            0.0,
            np.sin(latitude),
        ]
    )
    e_east = np.array([0.0, 1.0, 0.0])
    e_north = np.array([-e_zenith[2], 0.0, e_zenith[0]])
    # Transform point from celestial sphere at earth fixed system to equatorial system:
    return (
        e_east * np.cos(altitude) * np.sin(azimuth)
        + e_north * np.cos(altitude) * np.cos(azimuth)
        + e_zenith * np.sin(altitude)
    )


def star_trace(
    latitude: float, altitude: float, azimuth: float, t: np.ndarray
) -> np.ndarray:
    """Computes the trace of a star at given times t,
    observed from a given latitude,
    which passes altitude/azimuth at time t=0.

    The output is an array of shape (t.size, 3) in an earth fixed
    cartesian coordinate system (x: East, y: North, z: zenith).
    """

    # Transform star at t=0 from earth fixed system to celestial system:
    e_star = e_equatorial(latitude, altitude, azimuth)

    # Transform celestial system -> earth fixed system
    e_enr = np.column_stack(
        [
            # e_east
            -np.sin(OMEGA_EARTH * t),
            np.cos(OMEGA_EARTH * t),
            np.zeros(t.size),
            # e_north
            -np.sin(latitude) * np.cos(OMEGA_EARTH * t),
            -np.sin(latitude) * np.sin(OMEGA_EARTH * t),
            np.full(t.size, np.cos(latitude)),
            # e_r
            np.cos(latitude) * np.cos(OMEGA_EARTH * t),
            np.cos(latitude) * np.sin(OMEGA_EARTH * t),
            np.full(t.size, np.sin(latitude)),
        ]
    ).reshape((-1, 3, 3))

    return e_enr.dot(e_star)
