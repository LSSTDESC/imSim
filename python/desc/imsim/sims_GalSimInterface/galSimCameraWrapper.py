"""
This module provides wrappers for afwCameraGeom camera objects.
This is necessary because of a 90-degree rotation between how
the LSST Data Management software team defines coordinate
axes on the focal plane and how the LSST Camera team defines
coorindate axes on the focal plane.  Specifically

Camera +y = DM +x
Camera +x = DM -y

Because we want ImSim images to have the same WCS conventions
as PhoSim e-images, we need to apply this rotation to the
mappings between RA, Dec and pixel coordinates.  We may not
wish to do that for arbitrary cameras, so we will give
users the ability to apply a no-op wrapper to their cameras.

The class LSSTCameraWrapper applies this transformation.
In cases where users do not wish to apply any transformation
to their pixel coordinate system, the class GalSimCameraWrapper
provides the same API as LSSTCamerWrapper, but treats the
software-based pixel coordinates as truth.

In order to implement your own camera wrapper, create a python
class that inherits from GalSimCameraWrapper.  This class will
need:

- a property self.camera that is an afwCamerGeom camera object

- a method getBBox() that returns the bounding box in pixel space
  of a detector, given that detector's name

- a method getCenterPixel() that returns the central pixel of a
  detector, given that detector's name

- a method getCenterPupil() that returns the pupil coordinates
  (or field angle) in radians of the center of a detector given
  that detector's name

- a method getCornerPupilList that returns the pupil coordinates
  (or field angles) in radians of the corners of a detector given
  that detector's name

- a method getTanPixelBounds() that returns the minimum and maximum
  x and y pixel values of a detector, ignoring radial distortions,
  given that detector's name

- wrappers to the corresponding methods in lsst.sims.coordUtils that
  use the self.camera property and apply the necessary transformations
  to pixel space coordinates:
      pixelCoordsFromPupilCoords()
      pupilCoordsFromPixelCoords()
      pixelCoordsFromRaDec()
      _pixelCoordsFromRaDec()
      raDecFromPixelCoords()
      _raDecFromPixelCoords()
"""

import numpy as np
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS, TAN_PIXELS
from lsst.afw.cameraGeom import FIELD_ANGLE
import lsst.sims.coordUtils as coordUtils
import lsst.sims.utils as simsUtils

__all__ = ["GalSimCameraWrapper", "LSSTCameraWrapper"]

class GalSimCameraWrapper:
    """
    This is a no-op camera wrapper.
    """

    def __init__(self, camera):
        """
        Parameters
        ----------
        camera is an instantiation of an afwCameraGeom camera
        """
        self._camera = camera
        self._focal_to_field = None
        self._center_pixel_cache = {}
        self._corner_pupil_cache = {}
        self._center_pupil_cache = {}
        self._tan_pixel_bounds_cache = {}

    @property
    def camera(self):
        return self._camera

    @property
    def focal_to_field(self):
        """
        Transformation to go from FOCAL_PLANE to FIELD_ANGLE
        """
        if self._focal_to_field is None:
            self._focal_to_field \
                = self.camera.getTransformMap().getTransform(FOCAL_PLANE,
                                                             FIELD_ANGLE)
        return self._focal_to_field

    def getBBox(self, detector_name):
        """
        Return the bounding box for the detector named by detector_name
        """
        return self._camera[detector_name].getBBox()

    def getCenterPixel(self, detector_name):
        """
        Return the central pixel for the detector named by detector_name
        """
        if detector_name not in self._center_pixel_cache:
            centerPoint = self._camera[detector_name].getCenter(FOCAL_PLANE)
            centerPixel = self._camera[detector_name]\
                              .getTransform(FOCAL_PLANE, PIXELS)\
                              .applyForward(centerPoint)
            self._center_pixel_cache[detector_name] = centerPixel

        return self._center_pixel_cache[detector_name]

    def getCenterPupil(self, detector_name):
        """
        Return the pupil coordinates of the center of the named detector
        as an LsstGeom.Point2D
        """
        if detector_name not in self._center_pupil_cache:
            dd = self._camera[detector_name]
            centerPoint = dd.getCenter(FOCAL_PLANE)
            pupilPoint = self.focal_to_field.applyForward(centerPoint)
            self._center_pupil_cache[detector_name] = pupilPoint

        return self._center_pupil_cache[detector_name]

    def getCornerPupilList(self, detector_name):
        """
        Return a list of the pupil coordinates of the corners of the named
        detector as a list of LsstGeom.Point2D objects
        """
        if detector_name not in self._corner_pupil_cache:
            dd = self._camera[detector_name]
            cornerPointList = dd.getCorners(FOCAL_PLANE)
            pupil_point_list = self.focal_to_field.applyForward(cornerPointList)
            self._corner_pupil_cache[detector_name] = pupil_point_list

        return self._corner_pupil_cache[detector_name]

    def getTanPixelBounds(self, detector_name):
        """
        Return the min and max pixel values of a detector, assuming
        all radial distortions are set to zero (i.e. using the afwCameraGeom
        TAN_PIXELS coordinate system)

        Parameters
        ----------
        detector_name is a string denoting the name of the detector

        Returns
        -------
        xmin, xmax, ymin, ymax pixel values
        """
        if detector_name not in self._tan_pixel_bounds_cache:
            afwDetector = self._camera[detector_name]
            focal_to_tan_pix = afwDetector.getTransform(FOCAL_PLANE, TAN_PIXELS)
            xPixMin = None
            xPixMax = None
            yPixMin = None
            yPixMax = None
            cornerPointList = focal_to_tan_pix.applyForward(
                afwDetector.getCorners(FOCAL_PLANE))
            for cornerPoint in cornerPointList:
                xx = cornerPoint.getX()
                yy = cornerPoint.getY()
                if xPixMin is None or xx < xPixMin:
                    xPixMin = xx
                if xPixMax is None or xx > xPixMax:
                    xPixMax = xx
                if yPixMin is None or yy < yPixMin:
                    yPixMin = yy
                if yPixMax is None or yy > yPixMax:
                    yPixMax = yy

            self._tan_pixel_bounds_cache[detector_name] \
                = (xPixMin, xPixMax, yPixMin, yPixMax)

        return self._tan_pixel_bounds_cache[detector_name]

    def pixelCoordsFromPupilCoords(self, xPupil, yPupil, chipName, obs_metadata,
                                   includeDistortion=True):
        """
        Get the pixel positions (or nan if not on a chip) for objects based
        on their pupil coordinates.

        Parameters
        ---------
        xPupil is the x pupil coordinates in radians. Can be either a float
        or a numpy array.

        yPupil is the y pupil coordinates in radians. Can be either a float
        or a numpy array.

        chipName designates the names of the chips on which the pixel
        coordinates will be reckoned.  Can be either single value, an
        array, or None.  If an array, there must be as many chipNames
        as there are (RA, Dec) pairs.  If a single value, all of the
        pixel coordinates will be reckoned on the same chip.  If None,
        this method will calculate which chip each(RA, Dec) pair
        actually falls on, and return pixel coordinates for each (RA,
        Dec) pair on the appropriate chip.

        obs_metadata is an ObservationMetaData characterizing the telescope
        pointing.

        includeDistortion is a boolean.  If True (default), then this
        method will return the true pixel coordinates with optical
        distortion included.  If False, this method will return
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the x pixel coordinate
        and the second row is the y pixel coordinate
        """
        if obs_metadata is None:
            raise RuntimeError("Must pass obs_metdata to "
                               "cameraWrapper.pixelCoordsFromPupilCoords")

        return coordUtils.pixelCoordsFromPupilCoords(
            xPupil, yPupil, chipName=chipName,
            camera=self._camera, includeDistortion=includeDistortion)

    def pupilCoordsFromPixelCoords(self, xPix, yPix, chipName,
                                   includeDistortion=True):

        """
        Convert pixel coordinates into pupil coordinates

        Parameters
        ----------
        xPix is the x pixel coordinate of the point.
        Can be either a float or a numpy array.

        yPix is the y pixel coordinate of the point.
        Can be either a float or a numpy array.

        chipName is the name of the chip(s) on which the pixel
        coordinates are defined.  This can be a list (in which case
        there should be one chip name for each (xPix, yPix) coordinate
        pair), or a single value (in which case, all of the (xPix,
        yPix) points will be reckoned on that chip).

        includeDistortion is a boolean.  If True (default), then this
        method will expect the true pixel coordinates with optical
        distortion included.  If False, this method will expect
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the x pupil coordinate
        and the second row is the y pupil coordinate (both in radians)
        """
        return coordUtils.pupilCoordsFromPixelCoords(
            xPix, yPix, chipName, camera=self._camera,
            includeDistortion=includeDistortion)

    def _raDecFromPixelCoords(self, xPix, yPix, chipName, obs_metadata,
                              epoch=2000.0, includeDistortion=True):
        """
        Convert pixel coordinates into RA, Dec

        Parameters
        ----------
        xPix is the x pixel coordinate.  It can be either
        a float or a numpy array.

        yPix is the y pixel coordinate.  It can be either
        a float or a numpy array.

        chipName is the name of the chip(s) on which the pixel
        coordinates are defined.  This can be a list (in which case
        there should be one chip name for each (xPix, yPix) coordinate
        pair), or a single value (in which case, all of the (xPix,
        yPix) points will be reckoned on that chip).

        obs_metadata is an ObservationMetaData defining the pointing

        epoch is the mean epoch in years of the celestial coordinate system.
        Default is 2000.

        includeDistortion is a boolean.  If True (default), then this
        method will expect the true pixel coordinates with optical
        distortion included.  If False, this method will expect
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the RA coordinate
        and the second row is the Dec coordinate (both in radians; in the
        International Celestial Reference System)

        WARNING: This method does not account for apparent motion due
        to parallax.  This method is only useful for mapping positions
        on a theoretical focal plane to positions on the celestial
        sphere.
        """

        return coordUtils._raDecFromPixelCoords(
            xPix, yPix, chipName, camera=self._camera,
            obs_metadata=obs_metadata, epoch=epoch,
            includeDistortion=includeDistortion)

    def raDecFromPixelCoords(self, xPix, yPix, chipName, obs_metadata,
                             epoch=2000.0, includeDistortion=True):

        """
        Convert pixel coordinates into RA, Dec

        Parameters
        ----------
        xPix is the x pixel coordinate.  It can be either
        a float or a numpy array.

        yPix is the y pixel coordinate.  It can be either
        a float or a numpy array.

        chipName is the name of the chip(s) on which the pixel
        coordinates are defined.  This can be a list (in which case
        there should be one chip name for each (xPix, yPix) coordinate
        pair), or a single value (in which case, all of the (xPix,
        yPix) points will be reckoned on that chip).

        obs_metadata is an ObservationMetaData defining the pointing

        epoch is the mean epoch in years of the celestial coordinate system.
        Default is 2000.

        includeDistortion is a boolean.  If True (default), then this
        method will expect the true pixel coordinates with optical
        distortion included.  If False, this method will expect
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the RA coordinate
        and the second row is the Dec coordinate (both in degrees; in the
        International Celestial Reference System)

        WARNING: This method does not account for apparent motion due
        to parallax.  This method is only useful for mapping positions
        on a theoretical focal plane to positions on the celestial
        sphere.
        """
        return coordUtils.raDecFromPixelCoords(
            xPix, yPix, chipName, camera=self._camera,
            obs_metadata=obs_metadata,
            epoch=2000.0, includeDistortion=True)

    def _pixelCoordsFromRaDec(self, ra, dec, pm_ra=None, pm_dec=None,
                              parallax=None, v_rad=None,
                              obs_metadata=None,
                              chipName=None,
                              epoch=2000.0, includeDistortion=True):
        """Get the pixel positions (or nan if not on a chip) for objects based
        on their RA, and Dec (in radians)

        Parameters
        ----------
        ra is in radians in the International Celestial Reference System.
        Can be either a float or a numpy array.

        dec is in radians in the International Celestial Reference System.
        Can be either a float or a numpy array.

        pm_ra is proper motion in RA multiplied by cos(Dec) (radians/yr)
        Can be a numpy array or a number or None (default=None).

        pm_dec is proper motion in dec (radians/yr)
        Can be a numpy array or a number or None (default=None).

        parallax is parallax in radians
        Can be a numpy array or a number or None (default=None).

        v_rad is radial velocity (km/s)
        Can be a numpy array or a number or None (default=None).

        obs_metadata is an ObservationMetaData characterizing the telescope
        pointing.

        epoch is the epoch in Julian years of the equinox against which
        RA is measured.  Default is 2000.

        chipName designates the names of the chips on which the pixel
        coordinates will be reckoned.  Can be either single value, an
        array, or None.  If an array, there must be as many chipNames
        as there are (RA, Dec) pairs.  If a single value, all of the
        pixel coordinates will be reckoned on the same chip.  If None,
        this method will calculate which chip each(RA, Dec) pair
        actually falls on, and return pixel coordinates for each (RA,
        Dec) pair on the appropriate chip.  Default is None.

        includeDistortion is a boolean.  If True (default), then this
        method will return the true pixel coordinates with optical
        distortion included.  If False, this method will return
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the x pixel coordinate
        and the second row is the y pixel coordinate

        """

        return coordUtils._pixelCoordsFromRaDec(
            ra, dec, pm_ra=pm_ra, pm_dec=pm_dec,
            parallax=parallax, v_rad=v_rad,
            obs_metadata=obs_metadata,
            chipName=chipName, camera=self._camera,
            epoch=epoch, includeDistortion=includeDistortion)

    def pixelCoordsFromRaDec(self, ra, dec, pm_ra=None, pm_dec=None,
                             parallax=None, v_rad=None,
                             obs_metadata=None,
                             chipName=None, camera=None,
                             epoch=2000.0, includeDistortion=True):
        """
        Get the pixel positions (or nan if not on a chip) for objects based
        on their RA, and Dec (in degrees)

        Parameters
        ----------
        ra is in degrees in the International Celestial Reference System.
        Can be either a float or a numpy array.

        dec is in degrees in the International Celestial Reference System.
        Can be either a float or a numpy array.

        pm_ra is proper motion in RA multiplied by cos(Dec) (arcsec/yr)
        Can be a numpy array or a number or None (default=None).

        pm_dec is proper motion in dec (arcsec/yr)
        Can be a numpy array or a number or None (default=None).

        parallax is parallax in arcsec
        Can be a numpy array or a number or None (default=None).

        v_rad is radial velocity (km/s)
        Can be a numpy array or a number or None (default=None).

        obs_metadata is an ObservationMetaData characterizing the telescope
        pointing.

        epoch is the epoch in Julian years of the equinox against which
        RA is measured.  Default is 2000.

        chipName designates the names of the chips on which the pixel
        coordinates will be reckoned.  Can be either single value, an
        array, or None.  If an array, there must be as many chipNames
        as there are (RA, Dec) pairs.  If a single value, all of the
        pixel coordinates will be reckoned on the same chip.  If None,
        this method will calculate which chip each(RA, Dec) pair
        actually falls on, and return pixel coordinates for each (RA,
        Dec) pair on the appropriate chip.  Default is None.

        includeDistortion is a boolean.  If True (default), then this
        method will return the true pixel coordinates with optical
        distortion included.  If False, this method will return
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the x pixel coordinate
        and the second row is the y pixel coordinate
        """
        return coordUtils.pixelCoordsFromRaDec(
            ra, dec, pm_ra=pm_ra, pm_dec=pm_dec,
            parallax=parallax, v_rad=v_rad,
            obs_metadata=obs_metadata,
            chipName=chipName, camera=self._camera,
            epoch=epoch, includeDistortion=includeDistortion)


class LSSTCameraWrapper(coordUtils.DMtoCameraPixelTransformer,
                        GalSimCameraWrapper):
    def __init__(self):
        super(LSSTCameraWrapper, self).__init__()
        GalSimCameraWrapper.__init__(self, coordUtils.lsst_camera())

    def getTanPixelBounds(self, detector_name):
        """
        Return the min and max pixel values of a detector, assuming
        all radial distortions are set to zero (i.e. using the afwCameraGeom
        TAN_PIXELS coordinate system)

        Parameters
        ----------
        detector_name is a string denoting the name of the detector

        Returns
        -------
        xmin, xmax, ymin, ymax pixel values
        """
        if not hasattr(self, '_tan_pixel_bounds_cache'):
            self._tan_pixel_bounds_cache = {}

        if detector_name not in self._tan_pixel_bounds_cache:
            dm_xmin, dm_xmax, dm_ymin, dm_ymax \
                = GalSimCameraWrapper.getTanPixelBounds(self, detector_name)
            self._tan_pixel_bounds_cache[detector_name] \
                = (dm_ymin, dm_ymax, dm_xmin, dm_xmax)

        return self._tan_pixel_bounds_cache[detector_name]

    def pixelCoordsFromPupilCoords(self, xPupil, yPupil, chipName, obs_metadata,
                                   includeDistortion=True):
        """
        Get the pixel positions (or nan if not on a chip) for objects based
        on their pupil coordinates.

        Paramters
        ---------
        xPupil is the x pupil coordinates in radians. Can be either a float
        or a numpy array.

        yPupil is the y pupil coordinates in radians. Can be either a float
        or a numpy array.

        chipName designates the names of the chips on which the pixel
        coordinates will be reckoned.  Can be either single value, an
        array, or None.  If an array, there must be as many chipNames
        as there are (RA, Dec) pairs.  If a single value, all of the
        pixel coordinates will be reckoned on the same chip.  If None,
        this method will calculate which chip each(RA, Dec) pair
        actually falls on, and return pixel coordinates for each (RA,
        Dec) pair on the appropriate chip.

        obs_metadata is an ObservationMetaData characterizing the telescope
        pointing.

        includeDistortion is a boolean.  If True (default), then this
        method will return the true pixel coordinates with optical
        distortion included.  If False, this method will return
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the x pixel coordinate
        and the second row is the y pixel coordinate.  These pixel coordinates
        are defined in the Camera team system, rather than the DM system.
        """
        dm_x_pix, dm_y_pix = coordUtils.pixelCoordsFromPupilCoordsLSST(
            xPupil, yPupil, chipName=chipName,
            band=obs_metadata.bandpass, includeDistortion=includeDistortion)

        cam_y_pix = dm_x_pix
        if isinstance(chipName, list) or isinstance(chipName, np.ndarray):
            center_pix_dict = {}
            cam_x_pix = np.zeros(len(dm_y_pix))
            for ix, (det_name, yy) in enumerate(zip(chipName, dm_y_pix)):
                if det_name not in center_pix_dict:
                    center_pix = self.getCenterPixel(det_name)
                    center_pix_dict[det_name] = center_pix
                else:
                    center_pix = center_pix_dict[det_name]
                cam_x_pix[ix] = 2.0*center_pix[0]-yy
        else:
            center_pix = self.getCenterPixel(chipName)
            cam_x_pix = 2.0*center_pix[0] - dm_y_pix

        return cam_x_pix, cam_y_pix

    def pupilCoordsFromPixelCoords(self, xPix, yPix, chipName, obs_metadata,
                                   includeDistortion=True):
        """
        Convert pixel coordinates into pupil coordinates

        Parameters
        ----------
        xPix is the x pixel coordinate of the point.
        Can be either a float or a numpy array.
        Defined in the Camera team system (not the DM system).

        yPix is the y pixel coordinate of the point.
        Can be either a float or a numpy array.
        Defined in the Camera team system (not the DM system).

        chipName is the name of the chip(s) on which the pixel
        coordinates are defined.  This can be a list (in which case
        there should be one chip name for each (xPix, yPix) coordinate
        pair), or a single value (in which case, all of the (xPix,
        yPix) points will be reckoned on that chip).

        obs_metadata is an ObservationMetaData characterizing the telescope
        pointing.

        includeDistortion is a boolean.  If True (default), then this
        method will expect the true pixel coordinates with optical
        distortion included.  If False, this method will expect
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the x pupil coordinate
        and the second row is the y pupil coordinate (both in radians)

        """
        dm_xPix = yPix
        if isinstance(chipName, list) or isinstance(chipName, np.ndarray):
            dm_yPix = np.zeros(len(xPix))
            for ix, (det_name, _) in enumerate(zip(chipName, xPix)):
                cam_center_pix = self.getCenterPixel(det_name)
                dm_yPix[ix] = 2.0*cam_center_pix.getX()-xPix[ix]
        else:
            cam_center_pix = self.getCenterPixel(chipName)
            dm_yPix = 2.0*cam_center_pix.getX()-xPix
        return coordUtils.pupilCoordsFromPixelCoordsLSST(
            dm_xPix, dm_yPix, chipName,
            band=obs_metadata.bandpass,
            includeDistortion=includeDistortion)

    def _raDecFromPixelCoords(self, xPix, yPix, chipName, obs_metadata,
                              epoch=2000.0, includeDistortion=True):
        """
        Convert pixel coordinates into RA, Dec

        Parameters
        ----------
        xPix is the x pixel coordinate.  It can be either
        a float or a numpy array.  Defined in the Camera
        team system (not the DM system).

        yPix is the y pixel coordinate.  It can be either
        a float or a numpy array.  Defined in the Camera
        team system (not the DM system).

        chipName is the name of the chip(s) on which the pixel
        coordinates are defined.  This can be a list (in which case
        there should be one chip name for each (xPix, yPix) coordinate
        pair), or a single value (in which case, all of the (xPix,
        yPix) points will be reckoned on that chip).

        obs_metadata is an ObservationMetaData defining the pointing

        epoch is the mean epoch in years of the celestial coordinate system.
        Default is 2000.

        includeDistortion is a boolean.  If True (default), then this
        method will expect the true pixel coordinates with optical
        distortion included.  If False, this method will expect
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the RA coordinate
        and the second row is the Dec coordinate (both in radians; in the
        International Celestial Reference System)

        WARNING: This method does not account for apparent motion due
        to parallax.  This method is only useful for mapping positions
        on a theoretical focal plane to positions on the celestial
        sphere.
        """
        if isinstance(chipName, list) or isinstance(chipName, np.ndarray):
            dm_xPix = yPix
            dm_yPix = np.zeros(len(xPix))
            for ix, (det_name, xx) in enumerate(zip(chipName, xPix)):
                cam_center_pix = self.getCenterPixel(det_name)
                dm_yPix[ix] = 2.0*cam_center_pix.getX() - xx
        else:
            dm_xPix = yPix
            cam_center_pix = self.getCenterPixel(chipName)
            dm_yPix = 2.0*cam_center_pix.getX() - xPix

        return coordUtils._raDecFromPixelCoordsLSST(
            dm_xPix, dm_yPix, chipName,
            obs_metadata=obs_metadata,
            band=obs_metadata.bandpass,
            epoch=epoch, includeDistortion=includeDistortion)

    def raDecFromPixelCoords(self, xPix, yPix, chipName, obs_metadata,
                             epoch=2000.0, includeDistortion=True):
        """
        Convert pixel coordinates into RA, Dec

        Parameters
        ----------
        xPix is the x pixel coordinate.  It can be either
        a float or a numpy array.  Defined in the Camera
        team system (not the DM system).

        yPix is the y pixel coordinate.  It can be either
        a float or a numpy array.  Defined in the Camera
        team system (not the DM system).

        chipName is the name of the chip(s) on which the pixel
        coordinates are defined.  This can be a list (in which case
        there should be one chip name for each (xPix, yPix) coordinate
        pair), or a single value (in which case, all of the (xPix,
        yPix) points will be reckoned on that chip).

        obs_metadata is an ObservationMetaData defining the pointing

        epoch is the mean epoch in years of the celestial coordinate system.
        Default is 2000.

        includeDistortion is a boolean.  If True (default), then this
        method will expect the true pixel coordinates with optical
        distortion included.  If False, this method will expect
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the RA coordinate
        and the second row is the Dec coordinate (both in degrees; in the
        International Celestial Reference System)

        WARNING: This method does not account for apparent motion due
        to parallax.  This method is only useful for mapping positions
        on a theoretical focal plane to positions on the celestial
        sphere.
        """
        _ra, _dec = self._raDecFromPixelCoords(
            xPix, yPix, chipName,
            obs_metadata=obs_metadata,
            epoch=2000.0, includeDistortion=True)

        return np.degrees(_ra), np.degrees(_dec)

    def _pixelCoordsFromRaDec(self, ra, dec, pm_ra=None, pm_dec=None,
                              parallax=None, v_rad=None,
                              obs_metadata=None,
                              chipName=None,
                              epoch=2000.0, includeDistortion=True):
        """
        Get the pixel positions (or nan if not on a chip) for objects based
        on their RA, and Dec (in radians)

        Parameters
        ----------
        ra is in radians in the International Celestial Reference System.
        Can be either a float or a numpy array.

        dec is in radians in the International Celestial Reference System.
        Can be either a float or a numpy array.

        pm_ra is proper motion in RA multiplied by cos(Dec) (radians/yr)
        Can be a numpy array or a number or None (default=None).

        pm_dec is proper motion in dec (radians/yr)
        Can be a numpy array or a number or None (default=None).

        parallax is parallax in radians
        Can be a numpy array or a number or None (default=None).

        v_rad is radial velocity (km/s)
        Can be a numpy array or a number or None (default=None).

        obs_metadata is an ObservationMetaData characterizing the telescope
        pointing.

        epoch is the epoch in Julian years of the equinox against which
        RA is measured.  Default is 2000.

        chipName designates the names of the chips on which the pixel
        coordinates will be reckoned.  Can be either single value, an
        array, or None.  If an array, there must be as many chipNames
        as there are (RA, Dec) pairs.  If a single value, all of the
        pixel coordinates will be reckoned on the same chip.  If None,
        this method will calculate which chip each(RA, Dec) pair
        actually falls on, and return pixel coordinates for each (RA,
        Dec) pair on the appropriate chip.  Default is None.

        includeDistortion is a boolean.  If True (default), then this
        method will return the true pixel coordinates with optical
        distortion included.  If False, this method will return
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the x pixel coordinate
        and the second row is the y pixel coordinate.  These pixel coordinates
        are defined in the Camera team system, rather than the DM system.
        """

        dm_xPix, dm_yPix = coordUtils._pixelCoordsFromRaDecLSST(
            ra, dec,
            pm_ra=pm_ra, pm_dec=pm_dec,
            parallax=parallax, v_rad=v_rad,
            obs_metadata=obs_metadata,
            chipName=chipName,
            band=obs_metadata.bandpass,
            epoch=epoch,
            includeDistortion=includeDistortion)

        return self.cameraPixFromDMPix(dm_xPix, dm_yPix, chipName)

    def pixelCoordsFromRaDec(self, ra, dec, pm_ra=None, pm_dec=None,
                             parallax=None, v_rad=None,
                             obs_metadata=None, chipName=None,
                             epoch=2000.0, includeDistortion=True):
        """
        Get the pixel positions (or nan if not on a chip) for objects based
        on their RA, and Dec (in degrees)

        Parameters
        ----------
        ra is in degrees in the International Celestial Reference System.
        Can be either a float or a numpy array.

        dec is in degrees in the International Celestial Reference System.
        Can be either a float or a numpy array.

        pm_ra is proper motion in RA multiplied by cos(Dec) (arcsec/yr)
        Can be a numpy array or a number or None (default=None).

        pm_dec is proper motion in dec (arcsec/yr)
        Can be a numpy array or a number or None (default=None).

        parallax is parallax in arcsec
        Can be a numpy array or a number or None (default=None).

        v_rad is radial velocity (km/s)
        Can be a numpy array or a number or None (default=None).

        obs_metadata is an ObservationMetaData characterizing the telescope
        pointing.

        epoch is the epoch in Julian years of the equinox against which
        RA is measured.  Default is 2000.

        chipName designates the names of the chips on which the pixel
        coordinates will be reckoned.  Can be either single value, an
        array, or None.  If an array, there must be as many chipNames
        as there are (RA, Dec) pairs.  If a single value, all of the
        pixel coordinates will be reckoned on the same chip.  If None,
        this method will calculate which chip each(RA, Dec) pair
        actually falls on, and return pixel coordinates for each (RA,
        Dec) pair on the appropriate chip.  Default is None.

        includeDistortion is a boolean.  If True (default), then this
        method will return the true pixel coordinates with optical
        distortion included.  If False, this method will return
        TAN_PIXEL coordinates, which are the pixel coordinates with
        estimated optical distortion removed.  See the documentation
        in afw.cameraGeom for more details.

        Returns
        -------
        a 2-D numpy array in which the first row is the x pixel coordinate
        and the second row is the y pixel coordinate.  These pixel coordinates
        are defined in the Camera team system, rather than the DM system.
        """
        if pm_ra is not None:
            pm_ra_out = simsUtils.radiansFromArcsec(pm_ra)
        else:
            pm_ra_out = None

        if pm_dec is not None:
            pm_dec_out = simsUtils.radiansFromArcsec(pm_dec)
        else:
            pm_dec_out = None

        if parallax is not None:
            parallax_out = simsUtils.radiansFromArcsec(parallax)
        else:
            parallax_out = None

        return self._pixelCoordsFromRaDec(
            np.radians(ra), np.radians(dec),
            pm_ra=pm_ra_out, pm_dec=pm_dec_out,
            parallax=parallax_out, v_rad=v_rad,
            chipName=chipName, obs_metadata=obs_metadata,
            epoch=2000.0, includeDistortion=includeDistortion)
