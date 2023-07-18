import os
import json
import numpy as np
import scipy
from galsim.config import InputLoader, RegisterInputType
from lsst.afw import cameraGeom
import lsst.geom
from .meta_data import data_dir


class Vignetting:
    """
    Compute vignetting using a 1D spline to model the radial profile
    of the vignetting function.
    """
    _req_params = {'file_name': str}

    def __init__(self, file_name):
        """
        Parameters
        ----------
        file_name : str
            File containing the 1D spline model.
        """
        if not os.path.isfile(file_name):
            # Check if spline data file is in data_dir.
            data_file = os.path.join(data_dir, file_name)
            if not os.path.isfile(data_file):
                raise OSError(f"Vignetting data file {file_name} not found.")
        else:
            data_file = file_name
        with open(data_file) as fobj:
            spline_data = json.load(fobj)

        self.spline_model = scipy.interpolate.BSpline(*spline_data)

        # Use the value at the center of the focal plane to
        # normalize the vignetting profile.
        self.value_at_zero = self.spline_model(0)

    @staticmethod
    def get_pixel_radii(det):
        """
        Return an array of radial distance from the focal plane center for
        each pixel in a CCD.

        Parameters
        ----------
        det : lsst.afw.cameraGeom.Detector
            CCD in the focal plane.
        """
        # Compute the distance from the center of the focal plane to
        # each pixel in the CCD, by generating the grid of pixel x-y
        # locations in the focal plane.
        bbox = det.getBBox()
        nx = bbox.getWidth()
        ny = bbox.getHeight()

        # Pixel size in mm
        pixel_size = det.getPixelSize()

        # CCD center in focal plane coordinates (mm)
        center = det.getCenter(cameraGeom.FOCAL_PLANE)

        # Pixel center x-y offsets from the CCD center in mm
        dx = pixel_size.x*(np.arange(-nx/2, nx/2, dtype=float) + 0.5)
        dy = pixel_size.y*(np.arange(-ny/2, ny/2, dtype=float) + 0.5)

        # Corner raft CCDs have integral numbers of 90 degree
        # rotations about the CCD center, so account for these when
        # computing the x-y pixel locations.
        n_rot = det.getOrientation().getNQuarter() % 4
        if n_rot == 0:
            xarr, yarr = np.meshgrid(center.x + dx, center.y + dy)
        elif n_rot == 1:
            yarr, xarr = np.meshgrid(center.y + dx, center.x - dy)
        elif n_rot == 2:
            xarr, yarr = np.meshgrid(center.x - dx, center.y - dy)
        else:
            yarr, xarr = np.meshgrid(center.y - dx, center.x + dy)

        pixel_radii = np.sqrt(xarr**2 + yarr**2)
        return pixel_radii

    def apply_to_radii(self, radii):
        return self.spline_model(radii)/self.value_at_zero

    def __call__(self, det):
        return self.apply_to_radii(self.get_pixel_radii(det))

    def at_sky_coord(self, sky_coord, wcs, det):
        """
        Vignetting function value at the specified sky coordinates
        for a particular CCD.

        Parameters
        ----------
        sky_coord : galsim.CelestialCoord
            Sky coordinate at which the vignetting should be computed.
        wcs : galsim.Wcs
            The WCS for the CCD being considered.  This is used to find
            the location on the focal plane of focused light from the
            sky position.
        det : lsst.afw.cameraGeom.Detector
            The Detector object for the CCD being simulated.

        Returns
        -------
        float : The scale factor to multiply a source's flux by to account
            for vignetting.
        """
        # Compute pixel coordinates of the sky position on the CCD,
        # then convert to focal plane coordinates to compute the
        # radial distance from the focal plane center.  This will be
        # used with the spline model to obtain the vignetting scale
        # factor.
        pos = wcs.toImage(sky_coord)
        pix_to_fp = det.getTransform(cameraGeom.PIXELS,
                                     cameraGeom.FOCAL_PLANE)
        fp_pos = pix_to_fp.applyForward(lsst.geom.Point2D(pos.x, pos.y))
        r = np.sqrt(fp_pos.x**2 + fp_pos.y**2)

        return self.spline_model(r)/self.value_at_zero


RegisterInputType('vignetting', InputLoader(Vignetting, takes_logger=False))
