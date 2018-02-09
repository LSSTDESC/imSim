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
import pandas as pd
import lsst.log as lsstLog
import lsst.obs.lsstSim as obs_lsstSim
import lsst.utils as lsstUtils
from lsst.sims.photUtils import LSSTdefaults, PhotometricParameters
from lsst.sims.utils import ObservationMetaData, radiansFromArcsec
from lsst.sims.utils import applyProperMotion, ModifiedJulianDate

__all__ = ['parsePhoSimInstanceFile', 'PhosimInstanceCatalogParseError',
           'photometricParameters', 'phosim_obs_metadata',
           'validate_phosim_object_list',
           'read_config', 'get_config', 'get_logger', 'get_obs_lsstSim_camera']

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


def parsePhoSimInstanceFile(fileName, numRows=None):
    """
    Read a PhoSim instance catalog into a Pandas dataFrame. Then use
    the information that was read-in to build and return a command
    dictionary and object dataFrame.

    Parameters
    ----------
    fileName : str
        The instance catalog filename.
    numRows : int, optional
        The number of rows to read from the instance catalog.
        If None (the default), then all of the rows will be read in.

    Returns
    -------
    dict
        contains the header metadata from the PhoSimInstanceCatalog.
    """

    # Read the text instance file into Pandas.  Note that the top of the file
    # has commands in it, followed by one line per object.
    #
    # Note: I have chosen to use pandas here (as opposed to straight numpy e.g.)
    # because Pandas gracefully handles missing values including at the end
    # of lines.  Not every line is the same length in the instance file since
    # different classes of objects have different numbers of parameters.  The
    # other table reading options do not handle this situation well.
    columnNames = ['STRING', 'VALUE', 'RA', 'DEC', 'MAG_NORM', 'SED_NAME',
                   'REDSHIFT', 'GAMMA1', 'GAMMA2', 'KAPPA',
                   'DELTA_RA', 'DELTA_DEC',
                   'SOURCE_TYPE',
                   'PAR1', 'PAR2', 'PAR3', 'PAR4',
                   'PAR5', 'PAR6', 'PAR7', 'PAR8', 'PAR9', 'PAR10']

    dataFrame = pd.read_csv(fileName, names=columnNames, nrows=numRows,
                            delim_whitespace=True, comment='#')

    # Any missing items from the end of the lines etc were turned into NaNs by
    # Pandas to represent that they were missing.  This causes problems later
    # with the checks in the SED calculations in the GalSim interface.  So,
    # convert them into 0.0 instead.
    dataFrame.fillna('0.0', inplace=True)

    # Split the dataFrame into commands and sources.
    phoSimHeaderCards = dataFrame.query("STRING != 'object'")
    phoSimSources = dataFrame.query("STRING == 'object'")

    # Check that the required commands are present in the instance catalog.
    command_set = set(phoSimHeaderCards['STRING'])
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

    # Turn the list of commands into a dictionary.
    commands = extract_commands(phoSimHeaderCards)

    return commands


def extract_commands(df):
    """
    Extract the phosim commands and repackage as a simple dictionary,
    applying appropriate casts.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the instance catalog command data.

    Returns
    -------
    dict
        A dictionary with the phosim command values.
    """
    my_dict = df[['STRING', 'VALUE']].set_index('STRING').T.to_dict('list')
    commands = dict(((key, value[0]) for key, value in my_dict.items()))
    commands['filter'] = int(commands['filter'])
    commands['nsnap'] = int(commands['nsnap'])
    commands['obshistid'] = int(commands['obshistid'])
    commands['seed'] = int(commands['seed'])
    commands['mjd'] = float(commands['mjd'])
    # Add bandpass for convenience
    commands['bandpass'] = 'ugrizy'[commands['filter']]
    return commands


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
    return obs_lsstSim.LsstSimMapper().camera
