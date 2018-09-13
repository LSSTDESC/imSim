"""
Base module for the imSim package.
"""
from __future__ import absolute_import, print_function, division
import os
import sys
import warnings
from collections import namedtuple, defaultdict
import pickle
import logging
import gc
import copy
import galsim

# python_future no longer handles configparser as of 0.16.
# This is needed for PY2/3 compatibility.
try:
    import configparser
except ImportError:
    # python 2 backwards-compatibility
    import ConfigParser as configparser

import numpy as np
import lsst.log as lsstLog
import lsst.obs.lsstSim as obs_lsstSim
import lsst.utils as lsstUtils
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.photUtils import LSSTdefaults, PhotometricParameters
from lsst.sims.utils import ObservationMetaData, radiansFromArcsec
from lsst.sims.utils import applyProperMotion, ModifiedJulianDate
from lsst.sims.coordUtils import getCornerPixels
from lsst.sims.coordUtils import pixelCoordsFromPupilCoords
from lsst.sims.catUtils.mixins import PhoSimAstrometryBase
from lsst.sims.utils import _pupilCoordsFromObserved
from lsst.sims.utils import _observedFromAppGeo
from lsst.sims.utils import radiansFromArcsec
from lsst.sims.GalSimInterface import GalSimCelestialObject, SNRdocumentPSF,\
    Kolmogorov_and_Gaussian_PSF
from lsst.sims.photUtils import BandpassDict, Sed, getImsimFluxNorm
from lsst.sims.utils import defaultSpecMap
from .tree_rings import TreeRings
from .cosmic_rays import CosmicRays
from .sed_wrapper import SedWrapper
from .fopen import fopen
from .trim import InstCatTrimmer
from .sed_wrapper import SedWrapper
from .atmPSF import AtmosphericPSF

_POINT_SOURCE = 1
_SERSIC_2D = 2
_RANDOM_WALK = 3
_FITS_IMAGE = 4

__all__ = ['PhosimInstanceCatalogParseError',
           'photometricParameters', 'phosim_obs_metadata',
           'sources_from_list',
           'metadata_from_file',
           'read_config', 'get_config', 'get_logger', 'get_image_dirs',
           'get_obs_lsstSim_camera',
           'add_cosmic_rays',
           '_POINT_SOURCE', '_SERSIC_2D', '_RANDOM_WALK', '_FITS_IMAGE',
           'parsePhoSimInstanceFile',
           'add_treering_info', 'airmass', 'FWHMeff', 'FWHMgeom', 'make_psf',
           'save_psf', 'load_psf']


class PhosimInstanceCatalogParseError(RuntimeError):
    "Exception class for instance catalog parser."

PhoSimInstanceCatalogContents = namedtuple('PhoSimInstanceCatalogContents',
                                            ('obs_metadata',
                                             'phot_params',
                                             'sources'))

_required_commands = set("""rightascension
declination
mjd
altitude
azimuth
filter
rotskypos
rottelpos
dist2moon
moonalt
moondec
moonphase
moonra
nsnap
obshistid
seed
seeing
sunalt
vistime""".split())


def get_obs_lsstSim_camera(log_level=lsstLog.WARN):
    """
    Get the obs_lsstSim CameraMapper object, setting the default
    log-level at WARN in order to silence the INFO message about
    "Loading Posix exposure registry from .". Note that this only
    affects the 'CameraMapper' logging level.  The logging level set
    by any calling code (e.g., imsim.py) will still apply to other log
    messages made by imSim code.
    """
    lsstLog.setLevel('CameraMapper', log_level)
    return lsst_camera()


def metadata_from_file(file_name):
    """
    Read in the InstanceCatalog specified by file_name.
    Return a dict of the header values from that
    InstanceCatalog.
    """
    input_params = {}
    with fopen(file_name, mode='rt') as in_file:
        for line in in_file:
            if line[0] == '#':
                continue

            params = line.strip().split()

            if params[0] == 'object':
                break

            float_val = float(params[1])
            int_val = int(float_val)
            if np.abs(float_val-int_val)>1.0e-10:
                val = float_val
            else:
                val = int_val
            input_params[params[0]] = val

    command_set = set(input_params.keys())
    missing_commands = _required_commands - command_set
    if missing_commands:
        message = "\nRequired commands that are missing from the instance catalog %s:\n   " \
            % file_name + "\n   ".join(missing_commands)
        raise PhosimInstanceCatalogParseError(message)

    # Report on commands that are not part of the required set.
    extra_commands = command_set - _required_commands
    if extra_commands:
        message = "\nExtra commands in the instance catalog %s that are not in the required set:\n   " \
            % file_name + "\n   ".join(extra_commands)
        warnings.warn(message)

    commands = dict(((key, value) for key, value in input_params.items()))
    # Add bandpass for convenience
    commands['bandpass'] = 'ugrizy'[commands['filter']]

    return commands


def sources_from_list(object_lines, obs_md, phot_params, file_name):
    camera = get_obs_lsstSim_camera()
    config = get_config()

    num_objects = len(object_lines)

    # RA, Dec in the coordinate system expected by PhoSim
    ra_phosim = np.zeros(num_objects, dtype=float)
    dec_phosim = np.zeros(num_objects, dtype=float)

    sed_name = [None]*num_objects
    mag_norm = 55.0*np.ones(num_objects, dtype=float)
    gamma1 = np.zeros(num_objects, dtype=float)
    gamma2 = np.zeros(num_objects, dtype=float)
    gamma2_sign = config['wl_params']['gamma2_sign']
    kappa = np.zeros(num_objects, dtype=float)

    internal_av = np.zeros(num_objects, dtype=float)
    internal_rv = np.zeros(num_objects, dtype=float)
    galactic_av = np.zeros(num_objects, dtype=float)
    galactic_rv = np.zeros(num_objects, dtype=float)
    semi_major_arcsec = np.zeros(num_objects, dtype=float)
    semi_minor_arcsec = np.zeros(num_objects, dtype=float)
    position_angle_degrees = np.zeros(num_objects, dtype=float)
    sersic_index = np.zeros(num_objects, dtype=float)
    npoints = np.zeros(num_objects, dtype=int)
    redshift = np.zeros(num_objects, dtype=float)
    pixel_scale = np.zeros(num_objects, dtype=float)
    rotation_angle = np.zeros(num_objects, dtype=float)
    fits_image_file = dict()

    unique_id = [None]*num_objects
    object_type = np.zeros(num_objects, dtype=int)

    i_obj = -1
    for line in object_lines:
        params = line.strip().split()
        if params[0] != 'object':
            continue
        i_obj += 1
        unique_id[i_obj] = params[1]
        ra_phosim[i_obj] = float(params[2])
        dec_phosim[i_obj] = float(params[3])
        mag_norm[i_obj] = float(params[4])
        sed_name[i_obj] = params[5]
        redshift[i_obj] = float(params[6])
        gamma1[i_obj] = float(params[7])
        gamma2[i_obj] = gamma2_sign*float(params[8])
        kappa[i_obj] = float(params[9])
        if params[12].lower() == 'point':
            object_type[i_obj] = _POINT_SOURCE
            i_gal_dust_model = 14
            if params[13].lower() != 'none':
                i_gal_dust_model = 16
                internal_av[i_obj] = float(params[14])
                internal_rv[i_obj] =float(params[15])
            if params[i_gal_dust_model].lower() != 'none':
                galactic_av[i_obj] = float(params[i_gal_dust_model+1])
                galactic_rv[i_obj] = float(params[i_gal_dust_model+2])
        elif params[12].lower() == 'sersic2d':
            object_type[i_obj] = _SERSIC_2D
            semi_major_arcsec[i_obj] = float(params[13])
            semi_minor_arcsec[i_obj] = float(params[14])
            position_angle_degrees[i_obj] = float(params[15])
            sersic_index[i_obj] = float(params[16])
            i_gal_dust_model = 18
            if params[17].lower() != 'none':
                i_gal_dust_model = 20
                internal_av[i_obj] = float(params[18])
                internal_rv[i_obj] = float(params[19])
            if params[i_gal_dust_model].lower() != 'none':
                galactic_av[i_obj] = float(params[i_gal_dust_model+1])
                galactic_rv[i_obj] =float(params[i_gal_dust_model+2])
        elif params[12].lower() == 'knots':
            object_type[i_obj] = _RANDOM_WALK
            semi_major_arcsec[i_obj] = float(params[13])
            semi_minor_arcsec[i_obj] = float(params[14])
            position_angle_degrees[i_obj] = float(params[15])
            npoints[i_obj] = int(params[16])
            i_gal_dust_model = 18
            if params[17].lower() != 'none':
                i_gal_dust_model = 20
                internal_av[i_obj] = float(params[18])
                internal_rv[i_obj] = float(params[19])
            if params[i_gal_dust_model].lower() != 'none':
                galactic_av[i_obj] = float(params[i_gal_dust_model+1])
                galactic_rv[i_obj] = float(params[i_gal_dust_model+2])
        elif (params[12].endswith('.fits') or params[12].endswith('.fits.gz')):
            object_type[i_obj] = _FITS_IMAGE
            fits_image_file[i_obj] = find_file_path(params[12], get_image_dirs())
            pixel_scale[i_obj] = float(params[13])
            rotation_angle[i_obj] = float(params[14])
            i_gal_dust_model = 16
            if params[15].lower() != 'none':
                i_gal_dust_model = 18
                internal_av[i_obj] = float(params[16])
                internal_rv[i_obj] = float(params[17])
            if params[i_gal_dust_model].lower() != 'none':
                galactic_av[i_obj] = float(params[i_gal_dust_model+1])
                galactic_rv[i_obj] = float(params[i_gal_dust_model+2])
        else:
            raise RuntimeError("Do not know how to handle "
                               "object type: %s" % params[12])

    ra_appGeo, dec_appGeo \
        = PhoSimAstrometryBase._appGeoFromPhoSim(np.radians(ra_phosim),
                                                 np.radians(dec_phosim),
                                                 obs_md)

    ra_obs_rad, dec_obs_rad \
        = _observedFromAppGeo(ra_appGeo, dec_appGeo,
                              obs_metadata=obs_md,
                              includeRefraction=True)

    semi_major_radians = radiansFromArcsec(semi_major_arcsec)
    semi_minor_radians = radiansFromArcsec(semi_minor_arcsec)
    # Account for PA sign difference wrt phosim convention.
    position_angle_radians = np.radians(360. - position_angle_degrees)

    x_pupil, y_pupil = _pupilCoordsFromObserved(ra_obs_rad,
                                                dec_obs_rad,
                                                obs_md)

    bp_dict = BandpassDict.loadTotalBandpassesFromFiles()

    object_is_valid = np.array([True]*num_objects)

    invalid_objects = np.where(np.logical_or(np.logical_or(
                                    mag_norm>50.0,
                                    np.logical_and(galactic_av==0.0, galactic_rv==0.0)),
                               np.logical_or(
                                    np.logical_and(object_type==_SERSIC_2D,
                                                 semi_major_arcsec<semi_minor_arcsec),
                                    np.logical_and(object_type==_RANDOM_WALK,npoints<=0))))

    object_is_valid[invalid_objects] = False

    if len(invalid_objects[0]) > 0:
        message = "\nOmitted %d suspicious objects from " % len(invalid_objects[0])
        message += "the instance catalog:\n"
        n_bad_mag_norm = len(np.where(mag_norm>50.0)[0])
        message += "    %d had mag_norm > 50.0\n" % n_bad_mag_norm
        n_bad_av = len(np.where(np.logical_and(galactic_av==0.0, galactic_rv==0.0))[0])
        message += "    %d had galactic_Av == galactic_Rv == 0\n" % n_bad_av
        n_bad_axes = len(np.where(np.logical_and(object_type==_SERSIC_2D,
                                                 semi_major_arcsec<semi_minor_arcsec))[0])
        message += "    %d had semi_major_axis < semi_minor_axis\n" % n_bad_axes
        n_bad_knots = len(np.where(np.logical_and(object_type==_RANDOM_WALK,npoints<=0))[0])
        message += "    %d had n_points <= 0 \n" % n_bad_knots
        warnings.warn(message)

    wav_int = None
    wav_gal = None

    my_sed_dirs = sed_dirs(file_name)

    gs_object_arr = []
    for i_obj in range(num_objects):
        if not object_is_valid[i_obj]:
            continue

        fits_file = None
        if object_type[i_obj] == _POINT_SOURCE:
            gs_type = 'pointSource'
        elif object_type[i_obj] == _SERSIC_2D:
            gs_type = 'sersic'
        elif object_type[i_obj] == _RANDOM_WALK:
            gs_type = 'RandomWalk'
        elif object_type[i_obj] == _FITS_IMAGE:
            gs_type = 'FitsImage'
            fits_file = fits_image_file[i_obj]

        sed_obj = SedWrapper(find_file_path(sed_name[i_obj], my_sed_dirs),
                             mag_norm[i_obj], redshift[i_obj],
                             internal_av[i_obj], internal_rv[i_obj],
                             galactic_av[i_obj], galactic_rv[i_obj],
                             bp_dict)

        gs_object = GalSimCelestialObject(gs_type,
                                          x_pupil[i_obj],
                                          y_pupil[i_obj],
                                          semi_major_radians[i_obj],
                                          semi_minor_radians[i_obj],
                                          semi_major_radians[i_obj],
                                          position_angle_radians[i_obj],
                                          sersic_index[i_obj],
                                          sed_obj,
                                          bp_dict,
                                          phot_params,
                                          npoints[i_obj],
                                          fits_file,
                                          pixel_scale[i_obj],
                                          rotation_angle[i_obj],
                                          gamma1=gamma1[i_obj],
                                          gamma2=gamma2[i_obj],
                                          kappa=kappa[i_obj],
                                          uniqueId=unique_id[i_obj])

        gs_object_arr.append(gs_object)

    gs_object_arr = np.array(gs_object_arr)

    # how close to the edge of the detector a source has
    # to be before we will just simulate it anyway
    pix_tol = 50.0

    # any source brighter than this will be considered
    # so bright that it should be simulated for all
    # detectors, just in case light scatters onto them.
    max_mag = 16.0

    # down-select mag_norm, x_pupil, and y_pupil
    # to only contain those objects that were
    # deemed to be valid above
    valid = np.where(object_is_valid)
    mag_norm = mag_norm[valid]
    x_pupil = x_pupil[valid]
    y_pupil = y_pupil[valid]

    assert len(mag_norm) == len(gs_object_arr)
    assert len(x_pupil) == len(gs_object_arr)
    assert len(y_pupil) == len(gs_object_arr)

    out_obj_dict = {}
    for det in lsst_camera():
        chip_name = det.getName()
        pixel_corners = getCornerPixels(chip_name, lsst_camera())
        x_min = pixel_corners[0][0]
        x_max = pixel_corners[2][0]
        y_min = pixel_corners[0][1]
        y_max = pixel_corners[3][1]
        xpix, ypix = pixelCoordsFromPupilCoords(x_pupil, y_pupil,
                                                chipName=chip_name,
                                                camera=lsst_camera())

        on_chip = np.where(np.logical_or(mag_norm<max_mag,
                           np.logical_and(xpix>x_min-pix_tol,
                           np.logical_and(xpix<x_max+pix_tol,
                           np.logical_and(ypix>y_min-pix_tol,
                                          ypix<y_max+pix_tol)))))

        out_obj_dict[chip_name] = gs_object_arr[on_chip]

    return gs_object_arr, out_obj_dict

def get_image_dirs():
    """
    Return a list of possible directories for FITS images, making sure
    the list of image_dirs contains '.'.
    """
    image_dirs = os.environ.get('IMSIM_IMAGE_PATH', '.').split(':')
    if '.' not in image_dirs:
        # Follow the usual convention of searching '.' first.
        image_dirs.insert(0, '.')
    return image_dirs

def sed_dirs(instcat_file):
    """
    Return a list of SED directories to check for SED folders. This
    includes $SIMS_SED_LIBRARY_DIR and the directory containing
    the instance catalog.
    """
    return [lsstUtils.getPackageDir('sims_sed_library'),
            os.path.dirname(os.path.abspath(instcat_file))]

def find_file_path(file_name, path_directories):
    """
    Search the path_directories for the specific filename
    and return the first file found.
    """
    for path_dir in path_directories:
        my_path = os.path.join(path_dir, file_name)
        if os.path.isfile(my_path):
            return my_path
    return file_name

def photometricParameters(phosim_commands):
    """
    Factory function to create a PhotometricParameters object based on
    the instance catalog commands.

    Parameters
    ----------
    dict
        The phosim commands provided by parsePhoSimInstanceFile.

    Returns
    -------
    lsst.sims.photUtils.PhotometricParameters
        The object containing the photometric parameters.

    Notes
    -----
    The gain is set to unity so that the resulting eimage has units of
    electrons/pixel.  Read noise and dark current are set to zero.
    The effects from all three of those will be added by the
    electronics chain readout code.
    """
    config = get_config()
    nsnap = phosim_commands['nsnap']
    vistime = phosim_commands['vistime']
    readout_time = config['electronics_readout']['readout_time']
    exptime = (vistime - (nsnap-1)*readout_time)/float(nsnap)
    return PhotometricParameters(exptime=exptime,
                                 nexp=nsnap,
                                 gain=1,
                                 readnoise=0,
                                 darkcurrent=0,
                                 bandpass=phosim_commands['bandpass'])


def phosim_obs_metadata(phosim_commands):
    """
    Factory function to create an ObservationMetaData object based
    on the PhoSim commands extracted from an instance catalog.

    Parameters
    ----------
    phosim_commands : dict
        Dictionary of PhoSim physics commands.

    Returns
    -------
    lsst.sims.utils.ObservationMetaData
    """
    bandpass = phosim_commands['bandpass']
    fwhm_eff = FWHMeff(phosim_commands['seeing'], bandpass,
                       phosim_commands['altitude'])
    fwhm_geom = FWHMgeom(phosim_commands['seeing'], bandpass,
                         phosim_commands['altitude'])
    obs_md = ObservationMetaData(pointingRA=phosim_commands['rightascension'],
                                 pointingDec=phosim_commands['declination'],
                                 mjd=phosim_commands['mjd'],
                                 rotSkyPos=phosim_commands['rotskypos'],
                                 bandpassName=bandpass,
                                 m5=LSSTdefaults().m5(bandpass),
                                 seeing=fwhm_eff)
    # Set the OpsimMetaData attribute with the obshistID info.
    obs_md.OpsimMetaData = {'obshistID': phosim_commands['obshistid']}
    obs_md.OpsimMetaData['FWHMgeom'] = fwhm_geom
    obs_md.OpsimMetaData['FWHMeff'] =  fwhm_eff
    obs_md.OpsimMetaData['rawSeeing'] = phosim_commands['seeing']
    obs_md.OpsimMetaData['altitude'] = phosim_commands['altitude']
    obs_md.OpsimMetaData['seed'] = phosim_commands['seed']
    return obs_md


def parsePhoSimInstanceFile(fileName, sensor_list, numRows=None):
    """
    Read a PhoSim instance catalog into a Pandas dataFrame. Then use
    the information that was read-in to build and return a command
    dictionary and object dataFrame.

    Parameters
    ----------
    fileName : str
        The instance catalog filename.
    sensor_list: list
        List of sensors for which to extract object lists.
    numRows : int, optional
        The number of rows to read from the instance catalog.
        If None (the default), then all of the rows will be read in.

    Returns
    -------
    namedtuple
        This contains the PhoSim commands, the objects, and the
        original DataFrames containing the header lines and object
        lines
    """

    commands = metadata_from_file(fileName)
    obs_metadata = phosim_obs_metadata(commands)
    phot_params = photometricParameters(commands)
    instcats = InstCatTrimmer(fileName, sensor_list, numRows=numRows)
    gs_object_dict = {detname: GsObjectList(instcats[detname], instcats.obs_md,
                                            phot_params, instcats.instcat_file)
                      for detname in sensor_list}

    return PhoSimInstanceCatalogContents(obs_metadata,
                                         phot_params,
                                         ([], gs_object_dict))


class GsObjectList:
    """
    List-like class to provide access to lists of objects from an
    instance catalog, deferring creation of GalSimCelestialObjects
    until items in the list are accessed.
    """
    def __init__(self, object_lines, obs_md, phot_params, file_name,
                 chip_name=None):
        self.object_lines = object_lines
        self.obs_md = obs_md
        self.phot_params = phot_params
        self.file_name = file_name
        self.chip_name = chip_name
        self._gs_objects = None

    @property
    def gs_objects(self):
        if self._gs_objects is None:
            obj_arr, obj_dict \
                = sources_from_list(self.object_lines, self.obs_md,
                                    self.phot_params, self.file_name)
            if self.chip_name is not None:
                try:
                    self._gs_objects = obj_dict[self.chip_name]
                except KeyError:
                    self._gs_objects = []
            else:
                self._gs_objects = obj_arr
        return self._gs_objects

    def reset(self):
        """
        Reset the ._gs_objects attribute to None in order to recover
        memory devoted to the GalSimCelestialObject instances.
        """
        self._gs_objects = None

    def __len__(self):
        try:
            return len(self._gs_objects)
        except TypeError:
            return len(self.object_lines)

    def __iter__(self):
        for gs_obj in self.gs_objects:
            yield gs_obj

    def __getitem__(self, index):
        return self.gs_objects[index]


class ImSimConfiguration(object):
    """
    Configuration parameters for the simulation.  All parameters are
    set in a class-level dictionary to ensure that they are the same
    across all class instances.

    Individual parameter access is via section name:

    >>> config = get_config()
    >>> config['electronics_readout']['readout_time']
    3.
    """
    imsim_sections = defaultdict(dict)

    def __getitem__(self, section_name):
        return self.imsim_sections[section_name]

    def set_from_config(self, section_name, key, value):
        "Set the parameter value with the cast from a string applied."
        self[section_name][key] = self.cast(value)

    @staticmethod
    def cast(value):
        """
        Try to do sensible default casting of string representations
        of the parameters that are read from the config file.

        Parameters
        ----------
        value : str
            The string value returned, e.g., by ConfigParser.items(...).

        Returns
        -------
        None, int, float, bool, str
            Depending on the first workable cast, in that order.
        """
        # Remove any inline comments after a '#' delimiter.
        value = value.split('#')[0].strip()

        if value == 'None':
            return None
        try:
            if value.find('.') == -1 and value.find('e') == -1:
                return int(value)
            else:
                return float(value)
        except ValueError:
            # Check if it can be cast as a boolean.
            if value in 'True False'.split():
                return eval(value)
            # Return as the original string.
            return value


def get_config():
    """
    Get an ImSimConfiguration object with the current configuration.

    Returns
    -------
    ImSimConfiguration object
    """
    return ImSimConfiguration()


def read_config(config_file=None):
    """
    Read the configuration parameters for the simulation that are not
    given in the instance catalogs.

    Parameters
    ----------
    config_file : str, optional
        The file containing the configuration parameters.  If None
        (the default), then read the default parameters from
        data/default_imsim_configs.

    Returns
    -------
    dict ImSimConfiguration object
        An instance of ImSimConfiguration filled with the parameters from
        config_file.
    """
    my_config = ImSimConfiguration()
    cp = configparser.ConfigParser()
    cp.optionxform = str
    if config_file is None:
        config_file = os.path.join(lsstUtils.getPackageDir('imsim'),
                                   'data', 'default_imsim_configs')
    cp.read(config_file)
    for section in cp.sections():
        for key, value in cp.items(section):
            my_config.set_from_config(section, key, value)
    return my_config

read_config()


def get_logger(log_level, name=None):
    """
    Set up standard logging module and set lsst.log to the same log
    level.

    Parameters
    ----------
    log_level: str
        This is converted to logging.<log_level> and set in the logging
        config.
    name: str [None]
        The name to preprend to the log message to identify different
        logging contexts.  If None, then the root context is used.
    """
    # Setup logging output.
    logging.basicConfig(format="%(asctime)s %(name)s: %(message)s",
                        stream=sys.stdout)
    logger = logging.getLogger(name)
    logger.setLevel(eval('logging.' + log_level))

#    # Set similar logging level for Stack code.
#    if log_level == "CRITICAL":
#        log_level = "FATAL"
#    lsstLog.setLevel(lsstLog.getDefaultLoggerName(),
#                     eval('lsstLog.%s' % log_level))

    return logger

def add_cosmic_rays(gs_interpreter, phot_params):
    """
    Add cosmic rays draw from a catalog of CRs extracted from single
    sensor darks.

    Parameters
    ----------
    gs_interpreter: lsst.sims.GalSimInterface.GalSimInterpreter
        The object that is actually drawing the images

    phot_params: lsst.sims.photUtils.PhotometricParameters
        An object containing the physical parameters characterizing
        the photometric properties of the telescope/camera system.

    Returns
    -------
    None
        Will act on gs_interpreter, adding cosmic rays to its images.
    """
    config = get_config()
    ccd_rate = config['cosmic_rays']['ccd_rate']
    if ccd_rate == 0:
        return
    catalog = config['cosmic_rays']['catalog']
    if catalog == 'default':
        catalog = os.path.join(lsstUtils.getPackageDir('imsim'),
                               'data', 'cosmic_ray_catalog.fits.gz')
    crs = CosmicRays.read_catalog(catalog, ccd_rate=ccd_rate)

    # Retrieve the visit number for the random seeds.
    visit = gs_interpreter.obs_metadata.OpsimMetaData['obshistID']

    exptime = phot_params.nexp*phot_params.exptime
    for name, image in gs_interpreter.detectorImages.items():
        imarr = copy.deepcopy(image.array)
        # Set the random number seed for painting the CRs.
        crs.set_seed(CosmicRays.generate_seed(visit, name))
        gs_interpreter.detectorImages[name] = \
            galsim.Image(crs.paint(imarr, exptime=exptime), wcs=image.wcs)

    return None


def add_treering_info(detectors, tr_filename=None):
    """
    Adds tree ring info based on a model derived from measured sensors.

    Parameters
    ----------
    detectors: list (or other iterable)
        A list of GalSimDetector objects.
    tr_filename: str
        Filename of tree rings parameter file.

    Returns
    -------
    None
        Will add tree ring information to each of the detectors.
    """
    if tr_filename is None:
        tr_filename = os.path.join(lsstUtils.getPackageDir('imsim'),
                                   'data', 'tree_ring_data',
                                   'tree_ring_parameters_2018-04-26.txt')
    TR = TreeRings(tr_filename)
    for detector in detectors:
        [Rx, Ry, Sx, Sy] = [int(s) for s in list(detector.name) if s.isdigit()]
        (tr_center, tr_function) = TR.Read_DC2_Tree_Ring_Model(Rx, Ry, Sx, Sy)
        new_center = galsim.PositionD(tr_center.x + detector._xCenterPix, tr_center.y + detector._yCenterPix)
        detector.tree_rings = (new_center, tr_function)
    return None


def airmass(altitude):
    """
    Function to compute the airmass from altitude using equation 3
    of Krisciunas and Schaefer 1991.

    Parameters
    ----------
    altitude: float
        Altitude of pointing direction in degrees.

    Returns
    -------
    float: the airmass in units of sea-level airmass at the zenith.
    """
    altRad = np.radians(altitude)
    return 1.0/np.sqrt(1.0 - 0.96*(np.sin(0.5*np.pi - altRad))**2)


def FWHMeff(rawSeeing, band, altitude):
    """
    Compute the effective FWHM for a single Gaussian describing the PSF.

    Parameters
    ----------
    rawSeeing: float
        The "ideal" seeing in arcsec at zenith and at 500 nm.
        reference: LSST Document-20160
    band: str
        The LSST ugrizy band.
    altitude: float
        The altitude in degrees of the pointing.

    Returns
    -------
    float: Effective FWHM in arcsec.
    """
    X = airmass(altitude)

    # Find the effective wavelength for the band.
    wl = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]

    # Compute the atmospheric contribution.
    FWHMatm = rawSeeing*(wl/500)**(-0.3)*X**(0.6)

    # The worst case instrument contribution (see LSE-30).
    FWHMsys = 0.4*X**(0.6)

    # From LSST Document-20160, p. 8.
    return 1.16*np.sqrt(FWHMsys**2 + 1.04*FWHMatm**2)


def FWHMgeom(rawSeeing, band, altitude):
    """
    FWHM of the "combined PSF".  This is FWHMtot from
    LSST Document-20160, p. 8.

    Parameters
    ----------
    rawSeeing: float
        The "ideal" seeing in arcsec at zenith and at 500 nm.
        reference: LSST Document-20160
    band: str
        The LSST ugrizy band.
    altitude: float
        The altitude in degrees of the pointing.

    Returns
    -------
    float: FWHM of the combined PSF in arcsec.
    """
    return 0.822*FWHMeff(rawSeeing, band, altitude) + 0.052


def make_psf(psf_name, obs_md, log_level='WARN', rng=None, **kwds):
    """
    Make the requested PSF object.

    Parameters
    ----------
    psf_name: str
        Either "DoubleGaussian", "Kolmogorov", or "Atmospheric".
        The name is case-insensitive.
    obs_md: lsst.sims.utils.ObservationMetaData
        Metadata associated with the visit, e.g., pointing direction,
        observation time, seeing, etc..
    log_level: str ['WARN']
        Logging level ('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL').
    rng: galsim.BaseDeviate
        Instance of the galsim.baseDeviate random number generator.
    **kwds: **dict
        Additional keyword arguments to pass to the AtmosphericPSF,
        i.e., screen_size(=819.2) and screen_scale(=0.1).

    Returns
    -------
    lsst.sims.GalSimInterface.PSFbase: Instance of a subclass of PSFbase.
    """
    if psf_name.lower() == 'doublegaussian':
        return SNRdocumentPSF(obs_md.OpsimMetaData['FWHMgeom'])

    rawSeeing = obs_md.OpsimMetaData['rawSeeing']

    my_airmass = airmass(obs_md.OpsimMetaData['altitude'])

    if psf_name.lower() == 'kolmogorov':
        psf = Kolmogorov_and_Gaussian_PSF(my_airmass,
                                          rawSeeing=rawSeeing,
                                          band=obs_md.bandpass)
    elif psf_name.lower() == 'atmospheric':
        if rng is None:
            # Use the 'seed' value from the instance catalog for the rng
            # used by the atmospheric PSF.
            rng = galsim.UniformDeviate(obs_md.OpsimMetaData['seed'])
        logger = get_logger(log_level, 'psf')
        psf = AtmosphericPSF(airmass=my_airmass,
                             rawSeeing=rawSeeing,
                             band=obs_md.bandpass,
                             rng=rng,
                             logger=logger, **kwds)
    return psf

def save_psf(psf, outfile):
    """
    Save the psf as a pickle file.
    """
    # Set any logger attribute to None since loggers cannot be persisted.
    if hasattr(psf, 'logger'):
        psf.logger = None
    with open(outfile, 'wb') as output:
        pickle.dump(psf, output)

def load_psf(psf_file, log_level='INFO'):
    """
    Load a psf from a pickle file.
    """
    with open(psf_file, 'rb') as fd:
        psf = pickle.load(fd)

    # Since save_psf sets any logger attribute to None, restore
    # it here.
    if hasattr(psf, 'logger'):
        psf.logger = get_logger(log_level, 'psf')

    return psf
