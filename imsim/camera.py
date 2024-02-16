import os
import json
from collections import defaultdict
import galsim
import lsst.utils
from .meta_data import data_dir


__all__ = ['get_camera', 'Camera']


def get_gs_bounds(bbox):
    """
    Return a galsim.BoundsI object created from an lsst.afw.Box2I object.
    """
    return galsim.BoundsI(xmin=bbox.getMinX() + 1, xmax=bbox.getMaxX() + 1,
                          ymin=bbox.getMinY() + 1, ymax=bbox.getMaxY() + 1)


class Amp:
    """
    Class to contain the pixel geometry and electronic readout properties
    of the amplifier segments in the Rubin Camera CCDs.
    """
    def __init__(self):
        self.bounds = None
        self.raw_flip_x = None
        self.raw_flip_y = None
        self.gain = None
        self.full_well = None
        self.raw_bounds = None
        self.raw_data_bounds = None
        self.read_noise = None
        self.bias_level = None
        self.lsst_amp = None

    def update(self, other):
        """
        Method to copy the properties of another Amp object.
        """
        self.__dict__.update(other.__dict__)

    @staticmethod
    def make_amp_from_lsst(lsst_amp, bias_level=1000.):
        """
        Static function to create an Amp object, extracting its properties
        from an lsst.afw.cameraGeom.Amplifier object.

        Parameters
        ----------
        lsst_amp : lsst.afw.cameraGeom.Amplifier
           The LSST Science Pipelines class representing an amplifier
           segment in a CCD.
        bias_level : float [1000.]
           The bias level (ADU) to use since the camerGeom.Amplifier
           object doesn't have a this value encapsulated.

        Returns
        -------
        Amp object
        """
        my_amp = Amp()
        my_amp.lsst_amp = lsst_amp
        my_amp.bias_level = bias_level
        my_amp.bounds = get_gs_bounds(lsst_amp.getBBox())
        my_amp.raw_flip_x = lsst_amp.getRawFlipX()
        my_amp.raw_flip_y = lsst_amp.getRawFlipY()
        my_amp.gain = lsst_amp.getGain()
        # Saturation values in obs_lsst are in ADU and include bias levels,
        # so subtract bias level and convert to electrons.
        my_amp.full_well = (lsst_amp.getSaturation() - bias_level)*my_amp.gain
        my_amp.raw_bounds = get_gs_bounds(lsst_amp.getRawBBox())
        my_amp.raw_data_bounds = get_gs_bounds(lsst_amp.getRawDataBBox())
        my_amp.read_noise = lsst_amp.getReadNoise()
        return my_amp

    def __getattr__(self, attr):
        """Provide access to the attributes of the underlying lsst_amp."""
        return getattr(self.lsst_amp, attr)

class CCD(dict):
    """
    A dict subclass to contain the Amp representations of a CCD's
    amplifier segments along with the pixel bounds of the CCD in focal
    plane coordinates, as well as other CCD-level information such as
    the crosstalk between amps.  Amp objects are keyed by LSST amplifier
    name, e.g., 'C10'.

    """
    def __init__(self):
        super().__init__()
        self.bounds = None
        self.xtalk = None
        self.lsst_ccd = None
        self.full_well = None

    def update(self, other):
        """
        Method to copy the properties of another CCD object.
        """
        self.__dict__.update(other.__dict__)
        for key, value in other.items():
            if not key in self:
                self[key] = Amp()
            self[key].update(value)

    @staticmethod
    def make_ccd_from_lsst(lsst_ccd, bias_level=1000.0, bias_levels_dict=None):
        """
        Static function to create a CCD object, extracting its properties
        from an lsst.afw.cameraGeom.Detector object, including CCD and
        amp-level bounding boxes, and intra-CCD crosstalk, if it's
        available.

        Parameters
        ----------
        lsst_ccd : lsst.afw.cameraGeom.Detector
           The LSST Science Pipelines class representing a CCD.
        bias_level : float [1000.0]
           Default bias level for all amps if bias_levels_dict is None.
        bias_levels_dict : dict [None]
           Python dictonary of bias levels in ADU, keyed by amp name.

        Returns
        -------
        CCD object
        """
        my_ccd = CCD()
        my_ccd.bounds = get_gs_bounds(lsst_ccd.getBBox())
        my_ccd.lsst_ccd = lsst_ccd
        for lsst_amp in lsst_ccd:
            amp_name = lsst_amp.getName()
            if bias_levels_dict is not None:
                bias_level = bias_levels_dict[amp_name]
            my_ccd[amp_name] = Amp.make_amp_from_lsst(lsst_amp,
                                                      bias_level=bias_level)
        # The code in imsim/bleed_trails.py cannot handle per-amp
        # full_well values, so set the CCD-wide value to the maximum
        # per-amp value.
        my_ccd.full_well = max(_.full_well for _ in my_ccd.values())
        if lsst_ccd.hasCrosstalk():
            my_ccd.xtalk = lsst_ccd.getCrosstalk()
        return my_ccd

    def __getattr__(self, attr):
        """Provide access to the attributes of the underlying lsst_ccd."""
        return getattr(self.lsst_ccd, attr)


_camera_cache = {}
def get_camera(camera='LsstCam'):
    """
    Return an lsst camera object.

    Parameters
    ----------
    camera : str
       The class name of the LSST camera object. Valid names
       are 'LsstCam', 'LsstCamImSim', 'LsstComCamSim'. [default: 'LsstCam']

    Returns
    -------
    lsst.afw.cameraGeom.Camera
    """
    valid_cameras = ('LsstCam', 'LsstCamImSim', 'LsstComCamSim')
    if camera not in valid_cameras:
        raise ValueError('Invalid camera: %s', camera)
    if camera not in _camera_cache:
        _camera_cache[camera] = lsst.utils.doImport('lsst.obs.lsst.' + camera)().getCamera()
    return _camera_cache[camera]


class Camera(dict):
    """
    Class to represent the LSST Camera as a dictionary of CCD objects,
    keyed by the CCD name in the focal plane, e.g., 'R01_S00'.
    """
    def __init__(self, camera_class='LsstCam', bias_levels_file=None,
                 bias_level=1000.0):
        """
        Initialize a Camera object from the lsst instrument class.
        """
        super().__init__()
        self.lsst_camera = get_camera(camera_class)
        if bias_levels_file is not None:
            if not os.path.isfile(bias_levels_file):
                bias_levels_file = os.path.join(data_dir, bias_levels_file)
                if not os.path.isfile(bias_levels_file):
                    raise FileNotFoundError(f"{bias_levels_file} not found.")
            with open(bias_levels_file) as fobj:
                bias_level_dicts = json.load(fobj)
        else:
            # Create a dict-of-dicts that returns the single
            # bias_level value for all amps in all CCDs.
            bias_level_dicts = defaultdict(
                lambda: defaultdict(lambda: bias_level))

        for lsst_ccd in self.lsst_camera:
            det_name = lsst_ccd.getName()
            self[det_name] = CCD.make_ccd_from_lsst(
                lsst_ccd, bias_levels_dict=bias_level_dicts[det_name])

    def update(self, other):
        """
        Method to copy the properties of the CCDs in this object from
        another Camera object.
        """
        self.__dict__.update(other.__dict__)
        for key, value in other.items():
            if not key in self:
                self[key] = CCD()
            self[key].update(value)

    def __getattr__(self, attr):
        """Provide access to the attributes of the underlying lsst_camera."""
        return getattr(self.lsst_camera, attr)
