"""
Class to encapsulate info from obs_lsstCam.LsstCamMapper().camera.
"""
import lsst.afw.geom as afw_geom
import lsst.obs.lsstCam as obs_lsstCam

__all__ = ['CameraInfo']


class CameraInfo:
    def __init__(self):
        self.det_catalog = {det.getName(): det for det in
                            obs_lsstCam.LsstCamMapper().camera}

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
        lsst.afw.geom.Box2I
        """
        yseg, xseg = (int(x) for x in amp_info.getName()[-2:])
        width = amp_info.getBBox().getWidth()
        height = amp_info.getBBox().getHeight()
        xmin = xseg*width
        ymin = 0 if yseg == 1 else height
        return afw_geom.Box2I(afw_geom.Point2I(xmin, ymin),
                              afw_geom.Extent2I(width, height))
