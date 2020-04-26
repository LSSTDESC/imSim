from builtins import zip
from builtins import object
import re
import copy
from collections import namedtuple
import warnings
import astropy.time
import astropy.coordinates
from astropy._erfa import ErfaWarning
import galsim
import numpy as np
import lsst.geom as LsstGeom
from lsst.obs.lsstSim import LsstSimMapper
from lsst.sims.utils import arcsecFromRadians
from . import GalSimCameraWrapper
from .wcsUtils import tanSipWcsFromDetector
from lsst.sims.photUtils import PhotometricParameters

__all__ = ["GalSimDetector", "make_galsim_detector", "LsstObservatory"]


class GalSim_afw_TanSipWCS(galsim.wcs.CelestialWCS):
    """
    This class uses methods from lsst.geom and meas_astrom to
    fit a TAN-SIP WCS to an afw.cameraGeom.Detector and then wrap
    that WCS into something that GalSim can parse.

    For documentation on the TAN-SIP WCS see

    Shupe and Hook (2008)
    http://fits.gsfc.nasa.gov/registry/sip/SIP_distortion_v1_0.pdf
    """

    def __init__(self, detectorName, cameraWrapper, obs_metadata, epoch,
                 photParams=None, wcs=None):
        """
        @param [in] detectorName is the name of the detector as stored
        by afw

        @param [in] cameraWrapper is an instantionat of a GalSimCameraWrapper

        @param [in] obs_metadata is an instantiation of ObservationMetaData
        characterizing the telescope pointing

        @param [in] epoch is the epoch in Julian years of the equinox against
        which RA and Dec are measured

        @param [in] photParams is an instantiation of PhotometricParameters
        (it will contain information about gain, exposure time, etc.)

        @param [in] wcs is a kwarg that is used by the method _newOrigin().
        The wcs kwarg in this constructor method should not be used by users.
        """

        if not isinstance(cameraWrapper, GalSimCameraWrapper):
            raise RuntimeError("You must pass GalSim_afw_TanSipWCS "
                               "an instantiation "
                               "of GalSimCameraWrapper or one of its daughter "
                               "classes")

        if wcs is None:
            self._tanSipWcs = tanSipWcsFromDetector(
                detectorName, cameraWrapper, obs_metadata, epoch)
        else:
            self._tanSipWcs = wcs

        self.detectorName = detectorName
        self.cameraWrapper = cameraWrapper
        self.obs_metadata = obs_metadata
        self.photParams = photParams
        self.epoch = epoch

        # this is needed to match the GalSim v1.5 API
        self._color = None

        self.fitsHeader = self._tanSipWcs.getFitsMetadata()
        self.fitsHeader.set("EXTTYPE", "IMAGE")

        if self.obs_metadata.bandpass is not None:
            if (not isinstance(self.obs_metadata.bandpass, list) and not
                isinstance(self.obs_metadata.bandpass, np.ndarray)):
                self.fitsHeader.set("FILTER", self.obs_metadata.bandpass)

        if self.obs_metadata.mjd is not None:
            self.fitsHeader.set("MJD-OBS", self.obs_metadata.mjd.TAI)
            mjd_obs = astropy.time.Time(self.obs_metadata.mjd.TAI, format='mjd')
            self.fitsHeader.set('DATE-OBS', mjd_obs.isot)

        if self.photParams is not None:
            exptime = self.photParams.nexp*self.photParams.exptime
            self.fitsHeader.set("EXPTIME", exptime)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'ERFA function', ErfaWarning)
                mjd_end = mjd_obs + astropy.time.TimeDelta(exptime, format='sec')
                self.fitsHeader.set('DATE-END', mjd_end.isot)

        # Add pointing information to FITS header.
        if self.obs_metadata.pointingRA is not None:
            self.fitsHeader.set('RATEL', obs_metadata.pointingRA)
        if self.obs_metadata.pointingDec is not None:
            self.fitsHeader.set('DECTEL', obs_metadata.pointingDec)
        if self.obs_metadata.rotSkyPos is not None:
            self.fitsHeader.set('ROTANGLE', obs_metadata.rotSkyPos)

        # Add airmass, needed by jointcal.
        if self.obs_metadata.OpsimMetaData is not None:
            try:
                airmass = self.obs_metadata.OpsimMetaData['airmass']
            except KeyError:
                pass
            else:
                self.fitsHeader.set('AIRMASS', airmass)

        # Add boilerplate keywords requested by DM.
        self.fitsHeader.set('TELESCOP', 'LSST')
        self.fitsHeader.set('INSTRUME', 'CAMERA')
        self.fitsHeader.set('SIMULATE', True)
        self.fitsHeader.set('ORIGIN', 'IMSIM')
        observatory = LsstObservatory()
        self.fitsHeader.set('OBS-LONG', observatory.getLongitude().asDegrees())
        self.fitsHeader.set('OBS-LAT', observatory.getLatitude().asDegrees())
        self.fitsHeader.set('OBS-ELEV', observatory.getElevation())
        obs_location = observatory.getLocation()
        self.fitsHeader.set('OBSGEO-X', obs_location.geocentric[0].value)
        self.fitsHeader.set('OBSGEO-Y', obs_location.geocentric[1].value)
        self.fitsHeader.set('OBSGEO-Z', obs_location.geocentric[2].value)

        self.crpix1 = self.fitsHeader.getScalar("CRPIX1")
        self.crpix2 = self.fitsHeader.getScalar("CRPIX2")

        self.afw_crpix1 = self.crpix1
        self.afw_crpix2 = self.crpix2

        self.crval1 = self.fitsHeader.getScalar("CRVAL1")
        self.crval2 = self.fitsHeader.getScalar("CRVAL2")

        self.origin = galsim.PositionD(x=self.crpix1, y=self.crpix2)
        self._color = None

    def _radec(self, x, y, color=None):
        """
        This is a method required by the GalSim WCS API

        Convert pixel coordinates into ra, dec coordinates.
        x and y already have crpix1 and crpix2 subtracted from them.
        Return ra, dec in radians.

        Note: the color arg is ignored.  It is only there to
        match the GalSim v1.5 API
        """

        chipNameList = [self.detectorName]

        if type(x) is np.ndarray:
            chipNameList = chipNameList * len(x)

        ra, dec = self.cameraWrapper._raDecFromPixelCoords(
            x + self.afw_crpix1, y + self.afw_crpix2, chipNameList,
            obs_metadata=self.obs_metadata, epoch=self.epoch)

        if type(x) is np.ndarray:
            return (ra, dec)
        else:
            return (ra[0], dec[0])

    def _xy(self, ra, dec):
        """
        This is a method required by the GalSim WCS API

        Convert ra, dec in radians into x, y in pixel space with crpix
        subtracted.
        """

        chipNameList = [self.detectorName]

        if type(ra) is np.ndarray:
            chipNameList = chipNameList * len(ra)

        xx, yy = self.cameraWrapper._pixelCoordsFromRaDec(
            ra=ra, dec=dec, chipName=chipNameList,
            obs_metadata=self.obs_metadata, epoch=self.epoch)

        if type(ra) is np.ndarray:
            return (xx-self.crpix1, yy-self.crpix2)
        else:
            return (xx[0]-self.crpix1, yy-self.crpix2)

    def _newOrigin(self, origin):
        """
        This is a method required by the GalSim WCS API.  It returns a
        copy of self, but with the pixel-space origin translated to a
        new position.

        @param [in] origin is an instantiation of a galsim.PositionD
        representing the a point in pixel space to which you want to
        move the origin of the WCS

        @param [out] _newWcs is a WCS identical to self, but with the origin
        in pixel space moved to the specified origin
        """
        _newWcs = GalSim_afw_TanSipWCS.__new__(GalSim_afw_TanSipWCS)
        _newWcs.__dict__.update(self.__dict__)
        _newWcs.crpix1 = origin.x
        _newWcs.crpix2 = origin.y
        _newWcs.fitsHeader = copy.deepcopy(self.fitsHeader)
        _newWcs.fitsHeader.set('CRPIX1', origin.x)
        _newWcs.fitsHeader.set('CRPIX2', origin.y)
        return _newWcs

    def _writeHeader(self, header, bounds):
        for key in self.fitsHeader.getOrderedNames():
            header[key] = self.fitsHeader.getScalar(key)

        return header

TreeRingInfo = namedtuple('TreeRingInfo', ['center', 'func'])

class GalSimDetector(object):
    """
    This class stores information about individual detectors for use
    by the GalSimInterpreter
    """

    def __init__(self, detectorName, cameraWrapper, obs_metadata, epoch,
                 photParams=None):
        """
        @param [in] detectorName is the name of the detector as stored
        by afw

        @param [in] cameraWrapper is an instantionat of a GalSimCameraWrapper

        @param [in] photParams is an instantiation of the
        PhotometricParameters class that carries details about the
        photometric response of the telescope.

        This class will generate its own internal variable
        self.fileName which is the name of the detector as it will
        appear in the output FITS files
        """

        if not isinstance(cameraWrapper, GalSimCameraWrapper):
            raise RuntimeError("You must pass GalSimDetector an instantiation "
                               "of GalSimCameraWrapper or one of its daughter "
                               "classes")

        if detectorName not in cameraWrapper.camera:
            raise RuntimeError("detectorName needs to be in the camera "
                               " wrapped by cameraWrapper when instantiating "
                               "a GalSimDetector\n"
                               "%s is not in your cameraWrapper.camera"
                               % detectorName)

        if photParams is None:
            raise RuntimeError("You need to specify an instantiation "
                               "of PhotometricParameters "
                               "when constructing a GalSimDetector")

        self._wcs = None  # this will be created when it is actually called for
        self._name = detectorName
        self._cameraWrapper = cameraWrapper
        self._obs_metadata = obs_metadata
        self._epoch = epoch
        self._detector_type = self._cameraWrapper.camera[self._name].getType()

        # Default Tree Ring properties, i.e., no tree rings:
        self._tree_rings = TreeRingInfo(galsim.PositionD(0, 0), None)

        # We are transposing the coordinates because of the difference
        # between how DM defines pixel coordinates and how the
        # Camera team defines pixel coordinates
        bbox = self._cameraWrapper.getBBox(self._name)
        self._xMinPix = bbox.getMinX()
        self._xMaxPix = bbox.getMaxX()
        self._yMinPix = bbox.getMinY()
        self._yMaxPix = bbox.getMaxY()

        self._bbox = LsstGeom.Box2D(bbox)

        centerPupil = self._cameraWrapper.getCenterPupil(self._name)
        self._xCenterArcsec = arcsecFromRadians(centerPupil.getX())
        self._yCenterArcsec = arcsecFromRadians(centerPupil.getY())

        centerPixel = self._cameraWrapper.getCenterPixel(self._name)
        self._xCenterPix = centerPixel.getX()
        self._yCenterPix = centerPixel.getY()

        self._xMinArcsec = None
        self._yMinArcsec = None
        self._xMaxArcsec = None
        self._yMaxArcsec = None

        for cameraPointPupil in \
            self._cameraWrapper.getCornerPupilList(self._name):

            xx = arcsecFromRadians(cameraPointPupil.getX())
            yy = arcsecFromRadians(cameraPointPupil.getY())
            if self._xMinArcsec is None or xx < self._xMinArcsec:
                self._xMinArcsec = xx
            if self._xMaxArcsec is None or xx > self._xMaxArcsec:
                self._xMaxArcsec = xx
            if self._yMinArcsec is None or yy < self._yMinArcsec:
                self._yMinArcsec = yy
            if self._yMaxArcsec is None or yy > self._yMaxArcsec:
                self._yMaxArcsec = yy

        self._photParams = photParams
        self._fileName = self._getFileName()

    def _getFileName(self):
        """
        Format the name of the detector to add to the name of the FITS file
        """
        detectorName = self.name
        detectorName = detectorName.replace(',', '')
        detectorName = detectorName.replace(':', '')
        detectorName = detectorName.replace(' ', '_')
        return detectorName

    def pixelCoordinatesFromRaDec(self, ra, dec):
        """
        Convert RA, Dec into pixel coordinates on this detector

        @param [in] ra is a numpy array or a float indicating RA in radians

        @param [in] dec is a numpy array or a float indicating Dec in radians

        @param [out] xPix is a numpy array indicating the x pixel coordinate

        @param [out] yPix is a numpy array indicating the y pixel coordinate
        """

        nameList = [self.name]
        if type(ra) is np.ndarray:
            nameList = nameList*len(ra)
            raLocal = ra
            decLocal = dec
        else:
            raLocal = np.array([ra])
            decLocal = np.array([dec])

        xPix, yPix = self._cameraWrapper._pixelCoordsFromRaDec(raLocal, decLocal, chipName=nameList,
                                                               obs_metadata=self._obs_metadata,
                                                               epoch=self._epoch)

        return xPix, yPix

    def pixelCoordinatesFromPupilCoordinates(self, xPupil, yPupil):
        """
        Convert pupil coordinates into pixel coordinates on this detector

        @param [in] xPupil is a numpy array or a float indicating x
        pupil coordinates in radians

        @param [in] yPupil a numpy array or a float indicating y pupil
        coordinates in radians

        @param [out] xPix is a numpy array indicating the x pixel coordinate

        @param [out] yPix is a numpy array indicating the y pixel coordinate
        """
        nameList = [self._name]
        if type(xPupil) is np.ndarray:
            nameList = nameList*len(xPupil)
            xp = xPupil
            yp = yPupil
        else:
            xp = np.array([xPupil])
            yp = np.array([yPupil])

        xPix, yPix = self._cameraWrapper.pixelCoordsFromPupilCoords(
            xp, yp, nameList, self.obs_metadata)

        return xPix, yPix

    def containsRaDec(self, ra, dec):
        """
        Does a given RA, Dec fall on this detector?

        @param [in] ra is a numpy array or a float indicating RA in radians

        @param [in] dec is a numpy array or a float indicating Dec in radians

        @param [out] answer is an array of booleans indicating whether or not
        the corresponding RA, Dec pair falls on this detector
        """

        xPix, yPix = self.pixelCoordinatesFromRaDec(ra, dec)
        points = [LsstGeom.Point2D(xx, yy) for xx, yy in zip(xPix, yPix)]
        answer = [self._bbox.contains(pp) for pp in points]
        return answer

    def containsPupilCoordinates(self, xPupil, yPupil):
        """
        Does a given set of pupil coordinates fall on this detector?

        @param [in] xPupil is a numpy array or a float indicating x
        pupil coordinates in radians

        @param [in] yPupuil is a numpy array or a float indicating y
        pupil coordinates in radians

        @param [out] answer is an array of booleans indicating whether
        or not the corresponding RA, Dec pair falls on this detector
        """
        xPix, yPix = self.pixelCoordinatesFromPupilCoordinates(xPupil, yPupil)
        points = [LsstGeom.Point2D(xx, yy) for xx, yy in zip(xPix, yPix)]
        answer = [self._bbox.contains(pp) for pp in points]
        return answer

    @property
    def xMinPix(self):
        """Minimum x pixel coordinate of the detector"""
        return self._xMinPix

    @xMinPix.setter
    def xMinPix(self, value):
        raise RuntimeError("You should not be setting xMinPix on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def xMaxPix(self):
        """Maximum x pixel coordinate of the detector"""
        return self._xMaxPix

    @xMaxPix.setter
    def xMaxPix(self, value):
        raise RuntimeError("You should not be setting xMaxPix on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def yMinPix(self):
        """Minimum y pixel coordinate of the detector"""
        return self._yMinPix

    @yMinPix.setter
    def yMinPix(self, value):
        raise RuntimeError("You should not be setting yMinPix on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def yMaxPix(self):
        """Maximum y pixel coordinate of the detector"""
        return self._yMaxPix

    @yMaxPix.setter
    def yMaxPix(self, value):
        raise RuntimeError("You should not be setting yMaxPix on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def xCenterPix(self):
        """Center x pixel coordinate of the detector"""
        return self._xCenterPix

    @xCenterPix.setter
    def xCenterPix(self, value):
        raise RuntimeError("You should not be setting xCenterPix on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def yCenterPix(self):
        """Center y pixel coordinate of the detector"""
        return self._yCenterPix

    @yCenterPix.setter
    def yCenterPix(self, value):
        raise RuntimeError("You should not be setting yCenterPix on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def xMaxArcsec(self):
        """Maximum x pupil coordinate of the detector in arcseconds"""
        return self._xMaxArcsec

    @xMaxArcsec.setter
    def xMaxArcsec(self, value):
        raise RuntimeError("You should not be setting xMaxArcsec on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def xMinArcsec(self):
        """Minimum x pupil coordinate of the detector in arcseconds"""
        return self._xMinArcsec

    @xMinArcsec.setter
    def xMinArcsec(self, value):
        raise RuntimeError("You should not be setting xMinArcsec on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def yMaxArcsec(self):
        """Maximum y pupil coordinate of the detector in arcseconds"""
        return self._yMaxArcsec

    @yMaxArcsec.setter
    def yMaxArcsec(self, value):
        raise RuntimeError("You should not be setting yMaxArcsec on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def yMinArcsec(self):
        """Minimum y pupil coordinate of the detector in arcseconds"""
        return self._yMinArcsec

    @yMinArcsec.setter
    def yMinArcsec(self, value):
        raise RuntimeError("You should not be setting yMinArcsec on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def xCenterArcsec(self):
        """Center x pupil coordinate of the detector in arcseconds"""
        return self._xCenterArcsec

    @xCenterArcsec.setter
    def xCenterArcsec(self, value):
        raise RuntimeError("You should not be setting xCenterArcsec on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def yCenterArcsec(self):
        """Center y pupil coordinate of the detector in arcseconds"""
        return self._yCenterArcsec

    @yCenterArcsec.setter
    def yCenterArcsec(self, value):
        raise RuntimeError("You should not be setting yCenterArcsec on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def epoch(self):
        """Epoch of the equinox against which RA and Dec are measured in Julian years"""
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        raise RuntimeError("You should not be setting epoch on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def obs_metadata(self):
        """ObservationMetaData instantiation describing the telescope pointing"""
        return self._obs_metadata

    @obs_metadata.setter
    def obs_metadata(self, value):
        raise RuntimeError("You should not be setting obs_metadata on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def name(self):
        """Name of the detector"""
        return self._name

    @name.setter
    def name(self, value):
        raise RuntimeError("You should not be setting name on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def camera_wrapper(self):
        return self._cameraWrapper

    @camera_wrapper.setter
    def camera_wrapper(self, value):
        raise RuntimeError("You should not be setting the camera_wrapper on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def photParams(self):
        """PhotometricParameters instantiation characterizing the detector"""
        return self._photParams

    @photParams.setter
    def photParams(self, value):
        raise RuntimeError("You should not be setting photParams on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def fileName(self):
        """Name of the FITS file corresponding to this detector"""
        return self._fileName

    @fileName.setter
    def fileName(self, value):
        raise RuntimeError("You should not be setting fileName on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def wcs(self):
        """WCS corresponding to this detector"""
        if self._wcs is None:
            self._wcs = GalSim_afw_TanSipWCS(self._name, self._cameraWrapper,
                                             self.obs_metadata, self.epoch,
                                             photParams=self.photParams)

            if re.match('R[0-9][0-9]_S[0-9][0-9]', self.fileName) is not None:
                # This is an LSST camera; format the FITS header to
                # feed through DM code

                wcsName = self.fileName

                self._wcs.fitsHeader.set("CHIPID", wcsName)

                obshistid = 9999

                if self.obs_metadata.OpsimMetaData is not None:
                    if 'obshistID' in self.obs_metadata.OpsimMetaData:
                        self._wcs.fitsHeader.set(
                            "OBSID",
                            self.obs_metadata.OpsimMetaData['obshistID'])
                        obshistid = self.obs_metadata.OpsimMetaData['obshistID']

                bp = self.obs_metadata.bandpass
                if not isinstance(bp, list) and not isinstance(bp, np.ndarray):
                    filt_num = {'u': 0, 'g': 1, 'r': 2, 'i': 3,
                                'z': 4, 'y': 5}[bp]
                else:
                    filt_num = 2

                out_name = 'lsst_e_%d_f%d_%s_E000' % (obshistid, filt_num,
                                                      wcsName)
                self._wcs.fitsHeader.set("OUTFILE", out_name)

        return self._wcs

    @wcs.setter
    def wcs(self, value):
        raise RuntimeError("You should not be setting wcs on the fly; "
                           "just instantiate a new GalSimDetector")

    @property
    def tree_rings(self):
        return self._tree_rings

    @tree_rings.setter
    def tree_rings(self, center_func_tuple):
        self._tree_rings = TreeRingInfo(*center_func_tuple)


def make_galsim_detector(camera_wrapper, detname, phot_params,
                         obs_metadata, epoch=2000.0):
    """
    Create a GalSimDetector object given the desired detector name.

    Parameters
    ----------
    camera_wrapper: lsst.sims.GalSimInterface.GalSimCameraWrapper
        An object representing the camera being simulated

    detname: str
        The name of the detector in the LSST focal plane to create,
        e.g., "R:2,2 S:1,1".

    phot_params: lsst.sims.photUtils.PhotometricParameters
        An object containing the physical parameters representing
        the photometric properties of the system

    obs_metadata: lsst.sims.utils.ObservationMetaData
        Characterizing the pointing of the telescope

    epoch: float
        Representing the Julian epoch against which RA, Dec are
        reckoned (default = 2000)

    Returns
    -------
    GalSimDetector
    """
    centerPupil = camera_wrapper.getCenterPupil(detname)
    centerPixel = camera_wrapper.getCenterPixel(detname)

    translationPupil = camera_wrapper.pupilCoordsFromPixelCoords(
        centerPixel.getX()+1, centerPixel.getY()+1,
        detname, obs_metadata)

    plateScale = (np.sqrt(np.power(translationPupil[0]-centerPupil.getX(), 2) +
                          np.power(translationPupil[1]-centerPupil.getY(), 2))
                  /np.sqrt(2.0))

    plateScale = 3600.0*np.degrees(plateScale)

    # make a detector-custom photParams that copies all of the quantities
    # in the catalog photParams, except the platescale, which is
    # calculated above
    params = PhotometricParameters(exptime=phot_params.exptime,
                                   nexp=phot_params.nexp,
                                   effarea=phot_params.effarea,
                                   gain=phot_params.gain,
                                   readnoise=phot_params.readnoise,
                                   darkcurrent=phot_params.darkcurrent,
                                   othernoise=phot_params.othernoise,
                                   platescale=plateScale)

    return GalSimDetector(detname, camera_wrapper,
                          obs_metadata=obs_metadata, epoch=epoch,
                          photParams=params)


class LsstObservatory:
    """
    Class to encapsulate an Observatory object and compute the
    observatory location information.
    """
    def __init__(self):
        self.observatory = LsstSimMapper().MakeRawVisitInfoClass().observatory

    def getLocation(self):
        """
        The LSST observatory location in geocentric coordinates.

        Returns
        -------
        astropy.coordinates.earth.EarthLocation
        """
        return astropy.coordinates.EarthLocation.from_geodetic(
            self.observatory.getLongitude().asDegrees(),
            self.observatory.getLatitude().asDegrees(),
            self.observatory.getElevation())

    def __getattr__(self, attr):
        if hasattr(self.observatory, attr):
            return getattr(self.observatory, attr)
