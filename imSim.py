"""
Base module for the imSim package.
"""
from __future__ import absolute_import, print_function
import warnings
from collections import namedtuple
import numpy as np
import pandas as pd
import lsst.sims.utils as sims_utils

__all__ = ['parsePhoSimInstanceFile', 'PhosimInstanceCatalogParseError']


class PhosimInstanceCatalogParseError(RuntimeError):
    "Exception class for instance catalog parser."

PhoSimInstanceCatalogContents = namedtuple('PhoSimInstanceCatalogContents',
                                           ('commands', 'objects', 'header',
                                            'sources'))

_expected_commands = set("""rightascension
declination
mjd
altitude
azimuth
filter
rotskypos
dist2moon
moonalt
moondec
moonphase
moonra
nsnap
obshistid
rottelpos
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

    # Turn the list of commands into a dictionary.
    phoSimCommands = phoSimHeaderCards[['STRING', 'VALUE']]
    commandDictionary = phoSimCommands.set_index('STRING').T.to_dict('list')

    # Check that the commands match the expected set.
    if set(commandDictionary.keys()) != _expected_commands:
        raise PhosimInstanceCatalogParseError
        ("Commands from the instance catalog %s do match the expected set." % fileName)

    # This dataFrame will contain all of the objects to return.
    phoSimObjectList = extract_objects(phoSimSources)
    return PhoSimInstanceCatalogContents(commandDictionary, phoSimObjectList,
                                         phoSimHeaderCards, phoSimSources)

def extract_objects(df):
    """
    Extract the object information needed by the sims code
    and pack into a new dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the ray instance catalog object data.

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
    phosim_stars['objectID'] = stars['VALUE'].values
    phosim_stars['galSimType'] = valid_types[source_type]
    phosim_stars['magNorm'] = stars['MAG_NORM'].values
    phosim_stars['sedName'] = stars['SED_NAME'].values
    phosim_stars['redShift'] = stars['REDSHIFT'].values
    phosim_stars['ra'] = stars['RA'].values
    phosim_stars['dec'] = stars['DEC'].values
    phosim_stars['internalAv'] = stars['PAR2'].values
    phosim_stars['internalRv'] = stars['PAR3'].values
    phosim_stars['galacticAv'] = stars['PAR5'].values
    phosim_stars['galacticRv'] = stars['PAR6'].values

    source_type = 'sersic2d'
    galaxies = df.query("SOURCE_TYPE == '%s'" % source_type)
    phosim_galaxies = pd.DataFrame(np.zeros((len(galaxies), len(columns))),
                                   columns=columns)
    phosim_galaxies['objectID'] = galaxies['VALUE'].values
    phosim_galaxies['galSimType'] = valid_types[source_type]
    phosim_galaxies['magNorm'] = galaxies['MAG_NORM'].values
    phosim_galaxies['sedName'] = galaxies['SED_NAME'].values
    phosim_galaxies['redShift'] = galaxies['REDSHIFT'].values
    phosim_galaxies['ra'] = galaxies['RA'].values
    phosim_galaxies['dec'] = galaxies['DEC'].values
    print(galaxies['PAR1'])
    print(galaxies['PAR1'].values)
    print(pd.to_numeric(galaxies['PAR1']).values)
    print(type(pd.to_numeric(galaxies['PAR1']).values))
    print(type(pd.to_numeric(galaxies['PAR1']).tolist()))
    phosim_galaxies['halfLightSemiMajor'] = \
        sims_utils.radiansFromArcsec(pd.to_numeric(galaxies['PAR1'])).tolist()
    print(galaxies['PAR2'])
    phosim_galaxies['halfLightSemiMinor'] = \
        sims_utils.radiansFromArcsec(pd.to_numeric(galaxies['PAR2'])).tolist()
    phosim_galaxies['halfLightRadius'] = phosim_galaxies['halfLightSemiMajor']
    phosim_galaxies['positionAngle'] = galaxies['PAR3'].values
    phosim_galaxies['sersicIndex'] = pd.to_numeric(galaxies['PAR4']).tolist()
    phosim_galaxies['internalAv'] = galaxies['PAR6'].values
    phosim_galaxies['internalRv'] = galaxies['PAR7'].values
    phosim_galaxies['galacticAv'] = galaxies['PAR9'].values
    phosim_galaxies['galacticRv'] = galaxies['PAR10'].values

    phosim_galaxies = \
        phosim_galaxies.query("halfLightSemiMajor >= halfLightSemiMinor")
    return pd.concat((phosim_stars, phosim_galaxies), ignore_index=True)
