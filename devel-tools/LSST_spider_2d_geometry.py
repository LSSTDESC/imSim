#!/bin/env python3

"""A small tool to convert yaml files from batoid/data to 2D geometries
which can be used in the imsim.diffraction module."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from collections.abc import Sequence, Set

from numpy.typing import ArrayLike
import numpy as np

import batoid
from batoid import Optic, Baffle, ObscUnion, ObscRectangle, ObscAnnulus, ObscNegation

from imsim.diffraction import Geometry


def load_spider_geometry(
    path: Path,
    max_strut_thickness: Optional[float] = None,
    names: Optional[Set[str]] = None,
) -> Geometry:
    """Load geometry from a batoid yaml

    Currently, coordinate systems are ignored.
    """
    optical_system = Optic.fromYaml(path)

    def unpack_union(obscuration):
        if isinstance(obscuration, ObscUnion):
            return obscuration.items
        return [obscuration]

    def unpack_negation(obscuration):
        if isinstance(obscuration, ObscNegation):
            return obscuration.original
        return obscuration

    obscurations = sum(
        (
            unpack_union(item.obscuration)
            for item in optical_system.items
            if hasattr(item, "obscuration") and (names is None or item.name in names)
        ),
        [],
    )
    obscurations = [unpack_negation(obs) for obs in obscurations]
    rectangles = [
        component for component in obscurations if isinstance(component, ObscRectangle)
    ]
    rectangles = [
        rect
        for rect in rectangles
        if max_strut_thickness is None
        or rect.width < max_strut_thickness
        or rect.width < max_strut_thickness
    ]

    annuli = [
        component for component in obscurations if isinstance(component, ObscAnnulus)
    ]
    circles = [np.array([circ.x, circ.y, circ.inner]) for circ in annuli] + [
        np.array([circ.x, circ.y, circ.outer]) for circ in annuli
    ]
    return Geometry(
        thick_lines=into_thick_lines(rectangles),
        circles=np.array(circles),
    )


def into_thick_line(rect: ObscRectangle, out: Optional[ArrayLike] = None) -> ArrayLike:
    if out is None:
        out = np.empty((4,))
    e = np.array([np.cos(rect.theta), np.sin(rect.theta)])
    if rect.width > rect.height:
        n = np.array([-e[1], e[0]])
        thickness = rect.height / 2.0
    else:
        n = e
        thickness = rect.width / 2.0
    center = np.array([rect.x, rect.y])
    d = n.dot(center)
    out[:2] = n
    out[2] = d
    out[3] = thickness
    return out


def into_thick_lines(rects: Sequence[ObscRectangle]) -> ArrayLike:
    out = np.empty((len(rects), 4))
    for row, rect in zip(out, rects):
        into_thick_line(rect, out=row)
    return out


def polygonize(rect: ObscRectangle) -> ArrayLike:
    e1 = np.array([np.cos(rect.theta), np.sin(rect.theta)])
    e2 = np.array([-e1[1], e1[0]])
    center = np.array([rect.x, rect.y])
    return (
        np.array(
            [
                center,
                center + e1 * rect.width,
                center + e1 * rect.width + e2 * rect.height,
                center + e2 * rect.height,
            ]
        )
        - e1 * rect.width / 2.0
        - e2 * rect.height / 2.0
    )


def polygonize_thick_line(thick_line: ArrayLike, width: float) -> ArrayLike:
    n = thick_line[:2]
    d = thick_line[2]
    thickness = thick_line[3]
    t = np.array([-n[1], n[0]])
    center = d * n
    return (
        np.array(
            [
                center,
                center + t * width,
                center + t * width + n * thickness,
                center + n * thickness,
            ]
        )
        - t * width / 2.0
        - n * thickness / 2.0
    )


if __name__ == "__main__":
    from matplotlib.patches import Circle, Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    _geometry = load_spider_geometry(
        Path(batoid.__path__[0]) / "data/LSST/LSST_r_baffles.yaml",
        names={"M1", "CenterPlateAndSpiderTop"},
        max_strut_thickness=0.1,
    )
    print(_geometry.thick_lines)
    print(_geometry.circles)

    _fig, _ax = plt.subplots()

    _patches = [
        Polygon(polygonize_thick_line(l, width=10.0), closed=True)
        for l in _geometry.thick_lines
    ] + [Circle(circ[:2], circ[2]) for circ in _geometry.circles]

    _ax.add_collection(PatchCollection(_patches, alpha=0.4))
    _ax.set_xbound([-5.0, 5.0])
    _ax.set_ybound([-5.0, 5.0])

    plt.show()
