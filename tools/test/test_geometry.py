import numpy as np

from batoid import ObscRectangle, ObscAnnulus

import LSST_spider_2d_geometry


def test_into_thick_line() -> None:
    rect = ObscRectangle(x=1.0, y=2.0, width=0.5, height=0.25, theta=0.0)
    thick_line = LSST_spider_2d_geometry.into_thick_line(rect)
    np.testing.assert_array_almost_equal(
        thick_line,
        np.array([0.0, 1.0, 2.0, 0.125]),
    )

    rect = ObscRectangle(x=1.0, y=2.0, width=0.5, height=0.25, theta=np.pi / 6)
    LSST_spider_2d_geometry.into_thick_line(rect, out=thick_line)
    np.testing.assert_array_almost_equal(
        thick_line,
        np.array([-0.5, np.sqrt(3.0) / 2.0, -0.5 + np.sqrt(3.0), 0.125]),
    )
