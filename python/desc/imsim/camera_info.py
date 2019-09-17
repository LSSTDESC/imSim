"""
Class to encapsulate info from lsst.obs.lsst.imsim.ImsimMapper().camera.
"""
import astropy.time
import lsst.geom as lsst_geom
from lsst.obs.lsst.imsim import ImsimMapper

__all__ = ['CameraInfo', 'getHourAngle']


class CameraInfo:
    """
    Class to encapsulate info from lsst.obs.lsst.imsim.ImsimMapper().camera.
    """
    def __init__(self):
        self.det_catalog = {det.getName(): det for det in ImsimMapper().camera}

    def get_amp_names(self, det_name):
        """
        Get the amplifier names of the form 'R22_S11_C00' for the desired
        detector.

        Parameters
        ----------
        det_name: str
            The detector name, e.g., 'R22_S11'

        Returns
        -------
        str
        """
        amp_names = []
        for amp_info in self.det_catalog[det_name].getAmpInfoCatalog():
            amp_names.append('_'.join((det_name, amp_info.getName())))
        return amp_names

    def get_amp_info(self, amp_name):
        """
        Get the AmpInfoRecord object for the desired amplifier.

        Parameters
        ----------
        amp_name: str
            The amplifier name, e.g., "R22_S11_C00".

        Returns
        -------
        lsst.afw.table.ampInfo.ampInfo.AmpInfoRecord
        """
        det_name = '_'.join(amp_name.split('_')[:2])
        channel_name = amp_name[-3:]
        for amp_info in self.det_catalog[det_name].getAmpInfoCatalog():
            if amp_info.getName() == channel_name:
                return amp_info

    @staticmethod
    def mosaic_section(amp_info):
        """
        The bounding box for the NOAO mosaic section to assemble the
        image section into a full CCD.

        Parameters
        ----------
        amp_info: lsst.afw.table.ampInfo.ampInfo.AmpInfoRecord

        Returns
        -------
        lsst.geom.Box2I
        """
        yseg, xseg = (int(x) for x in amp_info.getName()[-2:])
        width = amp_info.getBBox().getWidth()
        height = amp_info.getBBox().getHeight()
        xmin = xseg*width
        ymin = 0 if yseg == 1 else height
        return lsst_geom.Box2I(lsst_geom.Point2I(xmin, ymin),
                               lsst_geom.Extent2I(width, height))


def getHourAngle(observatory, mjd, ra):
    """
    Compute the local hour angle of an object for the specified
    MJD and RA.

    Parameters
    ----------
    observatory: lsst.sims.GalSimInterface.LsstObservatory
        Extension of lsst.afw.coord.Observatory object.
    mjd: float
        Modified Julian Date of the observation.
    ra: float
        Right Ascension (in degrees) of the object.

    Returns
    -------
    float: hour angle in degrees
    """
    time = astropy.time.Time(mjd, format='mjd',
                             location=observatory.getLocation())
    # Get the local apparent sidereal time.
    last = time.sidereal_time('apparent').degree
    ha = (last - ra) % 360.
    if ha > 180:
        ha -= 360.
    return ha
