"""
Base module for the imSim package.
"""
from __future__ import absolute_import, print_function, division
import os
import sys
import warnings
from collections import namedtuple, defaultdict
import logging
import gc

# python_future no longer handles configparser as of 0.16.
# This is needed for PY2/3 compatabiloty.
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
from lsst.sims.GalSimInterface import GalSimCelestialObject
from lsst.sims.photUtils import BandpassDict, Sed, getImsimFluxNorm
from lsst.sims.utils import defaultSpecMap

_POINT_SOURCE = 1
_SERSIC_2D = 2

__all__ = ['PhosimInstanceCatalogParseError',
           'photometricParameters', 'phosim_obs_metadata',
           'validate_phosim_object_list',
           'sources_from_file',
           'metadata_from_file',
           'read_config', 'get_config', 'get_logger',
           'get_obs_lsstSim_camera',
           '_POINT_SOURCE', '_SERSIC_2D']

class PhosimInstanceCatalogParseError(RuntimeError):
    "Exception class for instance catalog parser."


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
vistime
rawSeeing
FWHMeff
FWHMgeom""".split())


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
    with open(file_name, 'r') as in_file:
        for line in in_file:
            params = line.strip().split()
            if params[0] == 'object':
                continue
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
            % fileName + "\n   ".join(missing_commands)
        raise PhosimInstanceCatalogParseError(message)

    # Report on commands that are not part of the required set.
    extra_commands = command_set - _required_commands
    if extra_commands:
        message = "\nExtra commands in the instance catalog %s that are not in the required set:\n   " \
            % fileName + "\n   ".join(extra_commands)
        warnings.warn(message)

    commands = dict(((key, value) for key, value in input_params.items()))
    # Add bandpass for convenience
    commands['bandpass'] = 'ugrizy'[commands['filter']]

    return commands


def sources_from_file(file_name, obs_md, phot_params, numRows=None):
    """
    Read in an InstanceCatalog and extract all of the astrophysical
    sources from it

    Parameters
    ----------
    file_name: str
        The name of the InstanceCatalog

    obs_md: ObservationMetaData
        The ObservationMetaData characterizing the pointing

    phot_params: PhotometricParameters
        The PhotometricParameters characterizing this telescope

    numRows: int (optional)
        The number of rows of the InstanceCatalog to read in (including the
        header)

    Returns
    -------
    gs_obj_arr: numpy array
        Contains the GalSimCelestialObjects for all of the
        astrophysical sources in this InstanceCatalog

    out_obj_dict: dict
        Keyed on the names of the detectors in the LSST camera.
        The values are numpy arrays of GalSimCelestialObjects
        that should be simulated for that detector, including
        objects that are near the edge of the chip or
        just bright (in which case, they might still illuminate
        the detector).
    """

    camera = get_obs_lsstSim_camera()

    num_objects = 0
    ct_rows = 0
    with open(file_name, 'r') as input_file:
        for line in input_file:
            ct_rows += 1
            params = line.strip().split()
            if params[0] == 'object':
                num_objects += 1
            if numRows is not None and ct_rows>=numRows:
                break

    # RA, Dec in the coordinate system expected by PhoSim
    ra_phosim = np.zeros(num_objects, dtype=float)
    dec_phosim = np.zeros(num_objects, dtype=float)

    sed_name = [None]*num_objects
    mag_norm = 55.0*np.ones(num_objects, dtype=float)
    gamma1 = np.zeros(num_objects, dtype=float)
    gamma2 = np.zeros(num_objects, dtype=float)
    kappa = np.zeros(num_objects, dtype=float)

    internal_av = np.zeros(num_objects, dtype=float)
    internal_rv = np.zeros(num_objects, dtype=float)
    galactic_av = np.zeros(num_objects, dtype=float)
    galactic_rv = np.zeros(num_objects, dtype=float)
    semi_major_arcsec = np.zeros(num_objects, dtype=float)
    semi_minor_arcsec = np.zeros(num_objects, dtype=float)
    position_angle_degrees = np.zeros(num_objects, dtype=float)
    sersic_index = np.zeros(num_objects, dtype=float)
    redshift = np.zeros(num_objects, dtype=float)

    unique_id = np.zeros(num_objects, dtype=int)
    object_type = np.zeros(num_objects, dtype=int)

    i_obj = -1
    with open(file_name, 'r') as input_file:
        for line in input_file:
            params = line.strip().split()
            if params[0] != 'object':
                continue
            if numRows is not None and i_obj>=num_objects:
                break
            i_obj += 1
            unique_id[i_obj] = int(params[1])
            ra_phosim[i_obj] = float(params[2])
            dec_phosim[i_obj] = float(params[3])
            mag_norm[i_obj] = float(params[4])
            sed_name[i_obj] = params[5]
            redshift[i_obj] = float(params[6])
            gamma1[i_obj] = float(params[7])
            gamma2[i_obj] = float(params[8])
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
                    i_gal_dust_model = 19
                    internal_av[i_obj] = float(params[17])
                    internal_rv[i_obj] = float(params[18])
                if params[i_gal_dust_model].lower() != 'none':
                    galactic_av[i_obj] = float(params[i_gal_dust_model+1])
                    galactic_rv[i_obj] =float(params[i_gal_dust_model+2])

            else:
                raise RuntimeError("Do not know how to handle "
                                   "object type: %s" % params[12])

    ra_appGeo, dec_appGeo = PhoSimAstrometryBase._appGeoFromPhoSim(np.radians(ra_phosim),
                                                                   np.radians(dec_phosim),
                                                                   obs_md)

    (ra_obs_rad,
     dec_obs_rad) = _observedFromAppGeo(ra_appGeo, dec_appGeo,
                                        obs_metadata=obs_md,
                                        includeRefraction=True)

    semi_major_radians = radiansFromArcsec(semi_major_arcsec)
    semi_minor_radians = radiansFromArcsec(semi_minor_arcsec)
    position_angle_radians = np.radians(position_angle_degrees)

    x_pupil, y_pupil = _pupilCoordsFromObserved(ra_obs_rad,
                                                dec_obs_rad,
                                                obs_md)

    bp_dict = BandpassDict.loadTotalBandpassesFromFiles()

    sed_dir = lsstUtils.getPackageDir('sims_sed_library')

    gs_object_arr = []
    for i_obj in range(num_objects):
        if object_type[i_obj] == _POINT_SOURCE:
            gs_type = 'pointSource'
        elif object_type[i_obj] == _SERSIC_2D:
            gs_type = 'sersic'

        # load the SED
        sed_obj = Sed()
        sed_obj.readSED_flambda(os.path.join(sed_dir, sed_name[i_obj]))
        fnorm = getImsimFluxNorm(sed_obj, mag_norm[i_obj])
        sed_obj.multiplyFluxNorm(fnorm)
        if internal_av[i_obj] != 0.0:
            a_int, b_int= sed_obj.setupCCMab()
            sed_obj.addCCMDust(a_int, b_int,
                               A_v = internal_av[i_obj],
                               R_v = internal_rv[i_obj])

        if redshift[i_obj] != 0.0:
            sed_obj.redshiftSED(redshift[i_obj], dimming=True)

        if galactic_av[i_obj] != 0.0:
            a_g, b_g = sed_obj.setupCCMab()
            sed_obj.addCCMDust(a_g, b_g,
                               A_v = galactic_av[i_obj],
                               R_v = galactic_rv[i_obj])

        gs_object = GalSimCelestialObject(gs_type,
                                          x_pupil[i_obj],
                                          y_pupil[i_obj],
                                          semi_major_arcsec[i_obj],
                                          semi_minor_arcsec[i_obj],
                                          semi_major_arcsec[i_obj],
                                          position_angle_radians[i_obj],
                                          sersic_index[i_obj],
                                          sed_obj,
                                          bp_dict,
                                          phot_params,
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

        valid = np.where(np.logical_or(mag_norm<16.0,
                         np.logical_and(xpix>x_min-pix_tol,
                         np.logical_and(xpix<x_max+pix_tol,
                         np.logical_and(ypix>y_min-pix_tol,
                                        ypix<y_max+pix_tol)))))

        out_obj_dict[chip_name] = gs_object_arr[valid]

    return gs_object_arr, out_obj_dict


def validate_phosim_object_list(phoSimObjects):
    """
    Remove rows with column values that are known to cause problems with
    the sim_GalSimInterface code.

    Parameters
    ----------
    phoSimObjects : pandas.DataFrame
       DataFrame of parsed object lines from the instance catalog.

    Returns
    -------
    namedtuple
        A tuple of DataFrames containing the accepted and rejected objects.
    """
    bad_row_queries = ('(galSimType=="sersic" and majorAxis < minorAxis)',
                       '(magNorm > 50)',
                       '(galacticAv==0 and galacticRv==0)')

    rejected = dict((query, phoSimObjects.query(query))
                    for query in bad_row_queries)
    all_rejected = \
        pd.concat(rejected.values(), ignore_index=True).drop_duplicates()
    accepted = phoSimObjects.query('not (' + ' or '.join(bad_row_queries) + ')')
    if len(all_rejected) != 0:
        message = "\nOmitted %i suspicious objects from" % len(all_rejected)
        message += " the instance catalog satisfying:\n"
        for query, objs in rejected.items():
            message += "%i  %s\n" % (len(objs), query)
        message += "Some rows may satisfy more than one condition.\n"
        warnings.warn(message)
    checked_objects = namedtuple('checked_objects', ('accepted', 'rejected'))
    return checked_objects(accepted, all_rejected)


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

    Notes
    -----
    The seeing from the instance catalog is the value at 500nm at
    zenith.  Do we need to do a band-specific calculation?
    """
    bandpass = phosim_commands['bandpass']
    obs_md = ObservationMetaData(pointingRA=phosim_commands['rightascension'],
                                 pointingDec=phosim_commands['declination'],
                                 mjd=phosim_commands['mjd'],
                                 rotSkyPos=phosim_commands['rotskypos'],
                                 bandpassName=bandpass,
                                 m5=LSSTdefaults().m5(bandpass),
                                 seeing=phosim_commands['FWHMeff'])
    # Set the OpsimMetaData attribute with the obshistID info.
    obs_md.OpsimMetaData = {'obshistID': phosim_commands['obshistid']}
    obs_md.OpsimMetaData['FWHMgeom'] = phosim_commands['FWHMgeom']
    obs_md.OpsimMetaData['FWHMeff'] = phosim_commands['FWHMeff']
    obs_md.OpsimMetaData['rawSeeing'] = phosim_commands['rawSeeing']
    obs_md.OpsimMetaData['altitude'] = phosim_commands['altitude']
    return obs_md


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
        None, int, float, str
            Depending on the first workable cast, in that order.
        """
        if value == 'None':
            return None
        try:
            if value.find('.') == -1 and value.find('e') == -1:
                return int(value)
            else:
                return float(value)
        except ValueError:
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


def get_logger(log_level):
    """
    Set up standard logging module and set lsst.log to the same log
    level.

    Parameters
    ----------
    log_level : str
        This is converted to logging.<log_level> and set in the logging
        config.
    """
    # Setup logging output.
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(eval('logging.' + log_level))

    # Set similar logging level for Stack code.
    if log_level == "CRITICAL":
        log_level = "FATAL"
    lsstLog.setLevel(lsstLog.getDefaultLoggerName(),
                     eval('lsstLog.%s' % log_level))

    return logger

