import sys
from collections import namedtuple
import pandas as pd

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

def parsePhoSimInstanceFile(fileName, numRows):
    """
    Read a PhoSim instance catalog into some Pandas dataFrames.  Then
    use the information that was read-in to build and return a
    GalSimCelestialObject.
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
        raise PhosimInstanceCatalogParseError("Commands from the instance catalog %s do match the expected set." % fileName)

    # This dataFrame will contain all of the objects to return.
    phoSimObjectList = pd.DataFrame(columns=('objectID', 'galSimType',
                                             'magNorm', 'sedName', 'redShift',
                                             'ra', 'dec',
                                             'halfLightRadius',
                                             'halfLightSemiMinor',
                                             'halfLightSemiMajor',
                                             'positionAngle', 'sersicIndex',
                                             'internalAv', 'internalRv',
                                             'galacticAv', 'galacticRv'))

    for row in phoSimSources.itertuples(name=None):
        (index, string, objectID, ra, dec, magNorm, sedName, redShift,
         gamma1, gamma2, kappa1, deltaRa, deltaDec, sourceType,
         par1, par2, par3, par4, par5, par6, par7, par8, par9, par10) = row

        # Not every variable will be filled on each entry since each class of
        # source has a different number of descriptive entries.
        # We need to set default empty values
        galSimType = 'notYetHandled'
        halfLightRadius = 0.0
        halfLightSemiMajor = 0.0
        halfLightSemiMinor = 0.0
        positionAngle = 0.0
        sersicIndex = 0.0
        dustRest = 0.0
        dustPar1A_v = 0.0
        dustPar1R_v = 0.0
        dustLab = 0.0
        dustPar2A_v = 0.0
        dustPar2R_v = 0.0

        # Currently not yet used.
        sigma = 0.0
        accRa = 0.0
        accDec = 0.0

        if sourceType == 'point':
            galSimType = 'pointSource'
            dustRest = par1
            dustPar1A_v = float(par2)
            dustPar1R_v = float(par3)
            dustLab = par4
            dustPar2A_v = float(par5)
            dustPar2R_v = float(par6)
        elif sourceType == 'gauss':
            galSimType = 'notYetHandled'
            sigma = par1
            dustRest = par2
            dustPar1A_v = par3
            dustPar1R_v = par4
            dustLab = par5
            dustPar2A_v = par6
            dustPar2R_v = par7
            print "I CAN'T HANDLE THIS QUITE YET!!!:", sourceType
            sys.exit()
        elif sourceType == "movingpoint":
            galSimType = 'notYetHandled'
            accRa = par1
            accDec = par2
            dustRest = par3
            dustPar1A_v = par4
            dustPar1R_v = par5
            dustLab = par6
            dustPar2A_v = par7
            dustPar2R_v = par8
            print "I CAN'T HANDLE THIS QUITE YET!!!:", sourceType
            sys.exit()
        elif sourceType == "sersic2d":
            galSimType = 'sersic'
            # We need to carefully look at what we are suppose to pass here
            halfLightSemiMajor = float(par1)*4.5e-6  # convert to radians
            halfLightSemiMinor = float(par2)*4.5e-6  # convert to radians

            # Special case handling.. Need to talk to Scott
            # Sometimes the minor axis is *larger* than the major axis.
            # Let's just bail in that case.
            if (halfLightSemiMinor - halfLightSemiMajor) > 0:
                print "---------------------------------"
                print "From parsePhoSimInstanceFile:"
                print "Sorry, in this galaxy the minor axis is > major axis!"
                print halfLightSemiMinor, halfLightSemiMajor
                print 'Difference (Minor - Major) ', halfLightSemiMinor - halfLightSemiMajor
                print "Skipping galaxy."
                print "---------------------------------"
                continue

            halfLightRadius = halfLightSemiMajor
            # positionAngle is in degrees (PhoSim API docs incorrectly say rad)
            positionAngle = float(par3)
            sersicIndex = float(par4)
            dustRest = par5
            dustPar1A_v = float(par6)
            dustPar1R_v = float(par7)
            dustLab = par8
            dustPar2A_v = float(par9)
            dustPar2R_v = float(par10)

        elif sourceType == "sersic":
            galSimType = 'notYetHandled'
            print "sersic"
            print "I CAN'T HANDLE THIS QUITE YET!!!:", sourceType
            sys.exit()
        else:
            print "I CAN'T HANDLE THIS!!!:", sourceType
            sys.exit()

        # Only used in debugging.
        if 0:
            print "par1", type(par1), par1
            print "par2", type(par2), par2
            print "par3", type(par3), par3
            print "par4", type(par4), par4
            print "par5", type(par5), par5

            print index, ":", objectID, ra, dec, sourceType, \
                halfLightSemiMinor, halfLightSemiMajor, positionAngle, \
                sersicIndex, \
                dustRest, dustPar1A_v, dustPar1R_v, \
                dustLab, dustPar2A_v, dustPar2R_v

        # Finally build a phoSimObject
        phoSimObject = (objectID, galSimType, magNorm, sedName, redShift,
                        ra, dec,
                        halfLightRadius, halfLightSemiMinor, halfLightSemiMajor,
                        positionAngle, sersicIndex,
                        dustPar1A_v, dustPar1R_v,  # Internal Dust
                        dustPar2A_v, dustPar2R_v)  # Galactic Dust

        # Append it to the dataFrame
        phoSimObjectList.loc[len(phoSimObjectList)] = phoSimObject

    return PhoSimInstanceCatalogContents(commandDictionary, phoSimObjectList,
                                         phoSimHeaderCards, phoSimSources)
