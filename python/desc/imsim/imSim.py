"""
Base module for the imSim package.
"""
from __future__ import absolute_import, print_function, division
import os
import warnings
from collections import namedtuple
import ConfigParser
import numpy as np
import pandas as pd
import lsst.utils as lsstUtils
from lsst.sims.photUtils import PhotometricParameters
import lsst.sims.utils as sims_utils

__all__ = ['parsePhoSimInstanceFile', 'PhosimInstanceCatalogParseError',
           'photometricParameters', 'validate_phosim_object_list',
           'ImSimConfiguration', 'read_config']


class PhosimInstanceCatalogParseError(RuntimeError):
    "Exception class for instance catalog parser."

PhoSimInstanceCatalogContents = namedtuple('PhoSimInstanceCatalogContents',
                                           ('commands', 'objects'))

_expected_commands = set("""rightascension
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
    namedtuple
        This contains the PhoSim commands, the objects, and the
        original DataFrames containing the header lines and object
        lines which were parsed with pandas.read_csv.
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
                            delim_whitespace=True)

    # Any missing items from the end of the lines etc were turned into NaNs by
    # Pandas to represent that they were missing.  This causes problems later
    # with the checks in the SED calculations in the GalSim interface.  So,
    # convert them into 0.0 instead.
    dataFrame.fillna('0.0', inplace=True)

    # Split the dataFrame into commands and sources.
    phoSimHeaderCards = dataFrame.query("STRING != 'object'")
    phoSimSources = dataFrame.query("STRING == 'object'")

    # Check that the commands match the expected set.
    command_set = set(phoSimHeaderCards['STRING'])
    if command_set != _expected_commands:
        message = "Commands from the instance catalog %s do match the expected set: " % fileName + str(command_set - _expected_commands) + str(_expected_commands - command_set)
        raise PhosimInstanceCatalogParseError(message)

    # Turn the list of commands into a dictionary.
    commands = extract_commands(phoSimHeaderCards)

    # This dataFrame will contain all of the objects to return.
    phoSimObjectList = extract_objects(phoSimSources)
    return PhoSimInstanceCatalogContents(commands, phoSimObjectList)

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
    # Add bandpass for convenience
    commands['bandpass'] = 'ugrizy'[commands['filter']]
    return commands

def extract_objects(df):
    """
    Extract the object information needed by the sims code
    and pack into a new dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the instance catalog object data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the columns expected by the sims code.
    """
    # Check for unhandled source types and emit warning if any are present.
    valid_types = dict(point='pointSource',
                       sersic2d='sersic')
    invalid_types = set(df['SOURCE_TYPE']) - set(valid_types)
    if invalid_types:
        warnings.warn("Instance catalog contains unhandled source types:\n%s\nSkipping these." % '\n'.join(invalid_types))

    columns = ('objectID', 'galSimType',
               'magNorm', 'sedName', 'redShift',
               'ra', 'dec',
               'halfLightRadius',
               'halfLightSemiMinor',
               'halfLightSemiMajor',
               'positionAngle', 'sersicIndex',
               'internalAv', 'internalRv',
               'galacticAv', 'galacticRv')

    # Process stars and galaxies separately.
    source_type = 'point'
    stars = df.query("SOURCE_TYPE=='%s'" % source_type)
    phosim_stars = pd.DataFrame(np.zeros((len(stars), len(columns))),
                                columns=columns)
    phosim_stars['objectID'] = pd.to_numeric(stars['VALUE']).tolist()
    phosim_stars['galSimType'] = valid_types[source_type]
    phosim_stars['magNorm'] = pd.to_numeric(stars['MAG_NORM']).tolist()
    phosim_stars['sedName'] = stars['SED_NAME'].tolist()
    phosim_stars['redShift'] = pd.to_numeric(stars['REDSHIFT']).tolist()
    phosim_stars['ra'] = pd.to_numeric(stars['RA']).tolist()
    phosim_stars['dec'] = pd.to_numeric(stars['DEC']).tolist()
    phosim_stars['internalAv'] = pd.to_numeric(stars['PAR2']).tolist()
    phosim_stars['internalRv'] = pd.to_numeric(stars['PAR3']).tolist()
    phosim_stars['galacticAv'] = pd.to_numeric(stars['PAR5']).tolist()
    phosim_stars['galacticRv'] = pd.to_numeric(stars['PAR6']).tolist()

    source_type = 'sersic2d'
    galaxies = df.query("SOURCE_TYPE == '%s'" % source_type)
    phosim_galaxies = pd.DataFrame(np.zeros((len(galaxies), len(columns))),
                                   columns=columns)
    phosim_galaxies['objectID'] = pd.to_numeric(galaxies['VALUE']).tolist()
    phosim_galaxies['galSimType'] = valid_types[source_type]
    phosim_galaxies['magNorm'] = pd.to_numeric(galaxies['MAG_NORM']).tolist()
    phosim_galaxies['sedName'] = galaxies['SED_NAME'].tolist()
    phosim_galaxies['redShift'] = pd.to_numeric(galaxies['REDSHIFT']).tolist()
    phosim_galaxies['ra'] = pd.to_numeric(galaxies['RA']).tolist()
    phosim_galaxies['dec'] = pd.to_numeric(galaxies['DEC']).tolist()
    phosim_galaxies['halfLightSemiMajor'] = \
        sims_utils.radiansFromArcsec(pd.to_numeric(galaxies['PAR1'])).tolist()
    phosim_galaxies['halfLightSemiMinor'] = \
        sims_utils.radiansFromArcsec(pd.to_numeric(galaxies['PAR2'])).tolist()
    phosim_galaxies['halfLightRadius'] = phosim_galaxies['halfLightSemiMajor']
    phosim_galaxies['positionAngle'] = pd.to_numeric(galaxies['PAR3']).tolist()
    phosim_galaxies['sersicIndex'] = pd.to_numeric(galaxies['PAR4']).tolist()
    phosim_galaxies['internalAv'] = pd.to_numeric(galaxies['PAR6']).tolist()
    phosim_galaxies['internalRv'] = pd.to_numeric(galaxies['PAR7']).tolist()
    phosim_galaxies['galacticAv'] = pd.to_numeric(galaxies['PAR9']).tolist()
    phosim_galaxies['galacticRv'] = pd.to_numeric(galaxies['PAR10']).tolist()

    return pd.concat((phosim_stars, phosim_galaxies), ignore_index=True)


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
    bad_row_queries = ('(galSimType=="sersic" and halfLightSemiMajor < halfLightSemiMinor)',
                       '(magNorm > 50)')
    rejected = dict((query, phoSimObjects.query(query))
                    for query in bad_row_queries)
    all_rejected = pd.concat(rejected.values(), ignore_index=True)
    accepted = phoSimObjects.query('not (' + ' or '.join(bad_row_queries) + ')')
    message = "Omitted %i suspicious objects from" % len(all_rejected)
    message += " the instance catalog satisfying:\n"
    for query, objs in rejected.items():
        message += "%i  %s\n" % (len(objs), query)
    warnings.warn(message)
    checked_objects = namedtuple('checked_objects', ('accepted', 'rejected'))
    return checked_objects(accepted, all_rejected)

def photometricParameters(phosim_commands):
    """
    Factory method to create a PhotometricParameters object based on
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
    config = read_config()
    nsnap = phosim_commands['nsnap']
    vistime = phosim_commands['vistime']
    readout_time = config['readout_time']
    exptime = (vistime - (nsnap-1)*readout_time)/float(nsnap)
    return PhotometricParameters(exptime=exptime,
                                 nexp=nsnap,
                                 gain=1,
                                 readnoise=0,
                                 darkcurrent=0,
                                 bandpass=phosim_commands['bandpass'])


class ImSimConfiguration(object):
    """
    Configuration parameters for the simulation.  All parameters are
    set in a the class-level dictionary to ensure that they are the
    same across all class instances.
    """
    imsim_parameters = dict()

    def __getitem__(self, key):
        return self.imsim_parameters[key]

    def __setitem__(self, key, value):
        self.imsim_parameters[key] = self.cast(value)

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
    ImSimConfiguration object
        An instance of ImSimConfiguration filled with the parameters from
        config_file.
    """
    my_config = ImSimConfiguration()
    cp = ConfigParser.SafeConfigParser()
    if config_file is None:
        config_file = os.path.join(lsstUtils.getPackageDir('imsim'),
                                   'data', 'default_imsim_configs')
    cp.read(config_file)
    # Stuff all of the sections into the same dictionary for now.
    for section in cp.sections():
        for key, value in cp.items(section):
            my_config[key] = value
    return my_config
