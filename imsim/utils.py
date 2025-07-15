"""Shared functionality"""

import warnings
import requests
from functools import wraps
from astropy.utils import iers
import erfa

# These need conda (via stackvana).  Not pip-installable
from lsst.afw import cameraGeom
from lsst.geom import Point2D

import numpy as np

# Check IERS download availablity.  If not available, disable auto
# downloads of IERS correction and leap second data and set
# auto_age_max to None.  See
# https://docs.astropy.org/en/stable/utils/iers.html#configuration-parameters
try:
    response = requests.get(iers.conf.iers_auto_url, timeout=5)
    iers_accessible = (response.status_code == 200)
except requests.exceptions.RequestException:
    iers_accessible = False
if not iers_accessible:
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = 'ignore'
    iers.conf.auto_max_age = None
    warnings.warn("IERS downloads are not available.")

RUBIN_AREA = np.pi * (418.**2 - 255.**2)  # cm^2


def ignore_erfa_warnings(func):
    @wraps(func)
    def call_func(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'ERFA', erfa.ErfaWarning)
            return func(*args, **kwargs)
    return call_func


def focal_to_pixel(fpx, fpy, det):
    """
    Parameters
    ----------
    fpx, fpy : array
        Focal plane position in millimeters in DVCS
        See https://lse-349.lsst.io/
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    x, y : array
        Pixel coordinates.
    """
    tx = det.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
    x, y = tx.getMapping().applyForward(np.vstack((fpx, fpy)))
    return x.ravel(), y.ravel()


def jac_focal_to_pixel(fpx, fpy, det):
    """
    Parameters
    ----------
    fpx, fpy : float
        Focal plane position in millimeters in DVCS
        See https://lse-349.lsst.io/
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    jac : array
        Jacobian of the transformation `focal_to_pixel`.
    """
    fp_to_pix = det.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
    return fp_to_pix.getJacobian(Point2D(fpx, fpy))


def pixel_to_focal(x, y, det):
    """
    Parameters
    ----------
    x, y : array
        Pixel coordinates.
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    fpx, fpy : array
        Focal plane position in millimeters in DVCS
        See https://lse-349.lsst.io/
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()
