"""
This is the imSim program, used to drive GalSim to simulate the LSST.  Written
for the DESC collaboration and LSST project.  This version of the program can
read phoSim instance files as is. It leverages the LSST Sims GalSim interface
code found in sims_GalSimInterface.
"""
import os
import sys
import logging
import argparse
import pandas as pd

from lsst.obs.lsstSim import LsstSimMapper

from lsst.sims.photUtils import LSSTdefaults
from lsst.sims.utils import ObservationMetaData

from lsst.sims.GalSimInterface.galSimCatalogs import GalSimBase
from lsst.sims.GalSimInterface import GalSimStars
from lsst.sims.GalSimInterface import GalSimGalaxies

from monkeyPatchedGalSimBase import phoSimCalculateGalSimSeds
from monkeyPatchedGalSimBase import phoSimInitilizer
from monkeyPatchedGalSimBase import get_phoSimInstanceCatalog


def parsePhoSimInstanceFile(fileName, numRows):
    """
    Read a PhoSim instance catalog into a Pandas dataFrame.  Then use the
    information that was read-in to build and return a command dictionary and
    object dataFrame.
    """

    # I want to be able to look at the dataFrames in iPython after I run.
    global phoSimHeaderCards, phoSimSources, phoSimObjectList

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
            if ((halfLightSemiMinor - halfLightSemiMajor) > 0):
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

    return commandDictionary, phoSimObjectList


def main(argv):
    """
    Drive GalSim to simulate the LSST.
    """
    # Monkey Patch the GalSimBase class and replace the routines that
    # gets the objects and builds the catalog with my version that gets the
    # information from a phoSim instance catalog file.  This should be temporary
    # and is a hack.
    GalSimBase.__init__ = phoSimInitilizer
    GalSimBase._calculateGalSimSeds = phoSimCalculateGalSimSeds
    GalSimBase.get_fitsFiles = get_phoSimInstanceCatalog

    # Setup logging output.
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger()

    # Setup a parser to take command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.0')
    parser.add_argument('-f', '--file', help="Specify the instance file")
    parser.add_argument('-n', '--numrows', default=2000, type=int, help="read the first numrows of the file.")
    parser.add_argument('--outdir', type=str, default='fits',
                        help='output directory for eimage file')
    arguments = parser.parse_args()

    # Get the instance file name. A valid file must be specified.
    try:
        with open(arguments.file) as instanceFile:
            print 'Using', instanceFile.name, "as an instance file."
            pass
    except IOError as errorMessage:
        print errorMessage
        print "That instance file cannot be opened. Program will end.\n"
        sys.exit()

    # Get the number of rows to read from the instance file.  Use default if not
    # specified.
    numRows = arguments.numrows
    print "Reading", numRows, "rows from the instance file."

    # Start by reading a PhoSim instance file.
    logger.info('Reading PhoSim instance catalog')

    # The PhoSim instance file contains both pointing commands and objects.  The
    # parser will split them and return a both phosim command dictionary and a
    # dataframe of objects.
    commandDictionary, phoSimDataBase = parsePhoSimInstanceFile(instanceFile.name, numRows)

    # Now further sub-divide the source dataframe into stars and galaxies.
    starDataBase = phoSimObjectList.query("galSimType == 'pointSource'")
    galaxyDataBase = phoSimObjectList.query("galSimType == 'sersic'")

    # Access the relevant pointing commands from the dictionary.
    visitIDString = str(int(commandDictionary['obshistid'][0]))
    mjd = commandDictionary['mjd'][0]
    rightAscension = commandDictionary['rightascension'][0]
    declination = commandDictionary['declination'][0]
    rotSkyPosition = commandDictionary['rotskypos'][0]

    # Now build the ObservationMetaData with values taken from the PhoSim
    # commands at the top of the instance file.

    # We need to set the M5 limiting magnitudes etc.
    defaults = LSSTdefaults()

    # defaults.seeing('r') doesn't exist in my version of the stack yet.
    # Manually set to 0.7
    obs = ObservationMetaData(pointingRA=rightAscension, pointingDec=declination,
                              mjd=mjd, rotSkyPos=rotSkyPosition,
                              bandpassName=['r'],
                              m5=[defaults.m5('r')],
                              seeing=[0.7])

    # Simulate the objects in the Pandas Dataframes. I monkey patched the
    # abstract base class GalSimBase to take my dataFrame instead of using the
    # internal databases. This is a hack until we rethink the class design.
    #
    # Additionally, I have set the PSF and noise etc globally for all of the
    # derived classes in the __init__.  For simulation the LSST this should be
    # the same between all types of objects.

    # First simulate stars
    phoSimStarCatalog = GalSimStars(starDataBase, obs)
    phoSimStarCatalog.camera = LsstSimMapper().camera
    phoSimStarCatalog.get_fitsFiles()

    # Now galaxies
    phoSimGalaxyCatalog = GalSimGalaxies(galaxyDataBase, obs)
    phoSimGalaxyCatalog.copyGalSimInterpreter(phoSimStarCatalog)
    phoSimGalaxyCatalog.get_fitsFiles()

    # Write out the fits files
    output_dir = arguments.outdir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    phoSimGalaxyCatalog.write_images(nameRoot=os.path.join(output_dir, 'e-image_')+visitIDString)

if __name__ == "__main__":
    main(sys.argv)
