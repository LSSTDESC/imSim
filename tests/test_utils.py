import numpy as np
from imsim import utils, get_camera


def test_focal_to_pixel() -> None:
    det = get_camera()[0]
    x = np.linspace(start=-190.615, stop=-188.615, num=4)
    y = np.linspace(start=-317.255, stop=-315.255, num=4)
    x_t, y_t = utils.focal_to_pixel(np.tile(x, len(y)), np.repeat(y, len(x)), det)

    x_expected = y_expected = np.linspace(start=-100.0, stop=100.0, num=4)
    np.testing.assert_almost_equal(x_t, np.tile(x_expected, len(y)))
    np.testing.assert_almost_equal(y_t, np.repeat(y_expected, len(x)))


def test_pixel_to_focal() -> None:
    det = get_camera()[0]
    x = y = np.linspace(start=-100.0, stop=100.0, num=4)
    x_t, y_t = utils.pixel_to_focal(np.tile(x, len(y)), np.repeat(y, len(x)), det)

    x_expected = np.linspace(start=-190.615, stop=-188.615, num=4)
    y_expected = np.linspace(start=-317.255, stop=-315.255, num=4)
    np.testing.assert_almost_equal(x_t, np.tile(x_expected, len(y)))
    np.testing.assert_almost_equal(y_t, np.repeat(y_expected, len(x)))
