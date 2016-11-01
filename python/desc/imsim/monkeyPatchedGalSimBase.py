"""
Module of replacement methods to lsst.sims.GalSimInterface.GalSimBase
in order to enable running GalSim on a PhoSim instance catalog as
input.
"""
from __future__ import absolute_import, print_function, division
import sys
import os
import copy
import logging
import numpy

from lsst.sims.photUtils import Sed, Bandpass
try:
    from lsst.sims.catalogs.measures.instance import is_null
except ImportError:
    from lsst.sims.catalogs.definitions import is_null

from lsst.sims.GalSimInterface.galSimCelestialObject import GalSimCelestialObject
from lsst.sims.GalSimInterface import ExampleCCDNoise
from lsst.sims.GalSimInterface import SNRdocumentPSF

from lsst.sims.coordUtils import chipNameFromRaDec

from lsst.sims.utils import altAzPaFromRaDec
from lsst.sims.utils import pupilCoordsFromRaDec
from lsst.sims.utils import ObservationMetaData

__all__ = ['phoSimInitializer', 'get_phoSimInstanceCatalog',
           'phoSimCalculateGalSimSeds']


class MemoryTracker(object):
    def __init__(self, pid=None, logger=None):
        try:
            import psutil
            if pid is None:
                pid = os.getpid()
            self.process = psutil.Process(pid)
        except ImportError:
            self.process = None
        if logger is None:
            logging.basicConfig(format="%(message)s", level=logging.INFO,
                                stream=sys.stdout)
            logger = logging.getLogger()
        self.logger = logger
    def print_usage(self, message=''):
        if self.process is None:
            return
        self.logger.debug(message)
        self.logger.debug("%.3f GB"
                          % (self.process.memory_full_info().uss/1024.**3))
        self.logger.debug('')



class DummyDB(object):
    """
    We don't use the built in databases in this program but the intepreter
    still wants to use this setting stored in the database.  For right now
    just make a dummy database object with what it needs.
    """
    epoch = 2000


def phoSimInitializer(self, phoSimDataBase, obs_metadata=None, logger=None):
    '''
    This function is used replace the standard __init__ class in the standard
    catalog classes so that we can initilize them with the parsed PhoSim
    database.  Currently, it is monkey patched into GalSimBase.  This should be
    is a hack which should be remedied by a class redesign.  Since in our case
    we always want the same PSF and background for all of our different objects
    we go ahead and define those here too.

    The original definition and docstring is here:

    def __init__(self, phoSimDataBase, obs_metadata=None):
        """
        @param [in] obs_metadata is an instantiation of the ObservationMetaData
        class characterizing a specific telescope observation
        """
    '''

    self._monkeyPatchLogger = logger
    if obs_metadata is not None:
        if not isinstance(obs_metadata, ObservationMetaData):
            raise ValueError("You passed InstanceCatalog something that was not ObservationMetaData")

        self.obs_metadata = copy.deepcopy(obs_metadata)
    else:
        self.obs_metadata = ObservationMetaData()

    self.phoSimDataBase = phoSimDataBase

    # This is stupid but needed for now.  The epoch in calculations is
    # carried in the db_obj.  Even though we don't need it we make a Dummy
    # class just to have db_obj.epoch avaliable.
    self.db_obj = DummyDB()

    # Add noise and sky background
    self.noise_and_background = ExampleCCDNoise(addNoise=True,
                                                addBackground=True)

    # Add a PSF.  This one Taken from equation 30 of
    # www.astro.washington.edu/users/ivezic/Astr511/LSST_SNRdoc.pdf
    #
    # Set seeing from self.obs_metadata.
    self.PSF = \
        SNRdocumentPSF(self.obs_metadata.seeing[self.obs_metadata.bandpass])

    # Add bandpasses to simulate over.
    self.bandpassNames = list(self.obs_metadata.bandpass)


def get_phoSimInstanceCatalog(self):
    """
    This is a modified version of the getter from galSimCatalogs.  Instead of
    returning a catalog of values taken from our databases it returns a catalog
    built from the parsing of a PhoSim instance file. For now this routine will
    be monkey patched to replace the routine get_fitsFiles. In the future, this
    class structure should be redisigned.  I (CWW) have also changed some
    formatting in order to respect PEP 8.

    Original Text below:
    This getter returns a column listing the names of the detectors whose
    corresponding FITS files contain the object in question.  The detector names
    will be separated by a '//' This getter also passes objects to the
    GalSimInterpreter to actually draw the FITS images.
    """
    mem_tracker = MemoryTracker(logger=self._monkeyPatchLogger)
    mem_tracker.print_usage('entered get_phoSimInstanceCatalog')
    # EXAMPLE star from an input file
    #
    # object 1046817878020
    # RA 31.2400746
    # DEC -10.09365
    # MAG 29.3370237
    # sedName starSED/phoSimMLT/lte033-4.5-1.0a+0.4.BT-Settl.spec.gz
    # REDSHIFT 0
    # GAMMA1 0
    # GAMMA2 0
    # KAPPA 0
    # DELTARA 0
    # DELTADEC 0
    # TYPE point
    # DUST REST CCM
    # DUSTPAR1 0.0635117705
    # DUSTPAR2 3.1
    # DUSTLAB none
    # Calculated x and y pupil: [-0.0008283] [-0.00201296]

    # This class was given the obs_metadata when the catalog was instantiated.
    # First calculate the position on the pupil.

    objectNames = self.phoSimDataBase['objectID'].values
    raICRS = self.phoSimDataBase['ra'].values
    decICRS = self.phoSimDataBase['dec'].values
    halfLight = self.phoSimDataBase['halfLightRadius'].values
    minorAxis = self.phoSimDataBase['halfLightSemiMinor'].values
    majorAxis = self.phoSimDataBase['halfLightSemiMajor'].values
    positionAngle = self.phoSimDataBase['positionAngle'].values
    sindex = self.phoSimDataBase['sersicIndex'].values

    if 0:
        objectIDs = slice(0, 5)
        alt, az, pa = altAzPaFromRaDec(raICRS, decICRS, self.obs_metadata)
        print('ra, dec, alt, az of our source: ', raICRS[objectIDs],
              decICRS[objectIDs], alt[objectIDs], az[objectIDs])

        chipName = chipNameFromRaDec(raICRS, decICRS,
                                     camera=self.camera,
                                     obs_metadata=self.obs_metadata,
                                     epoch=2000.0)

        print('chip on which the ', objectIDs, 'th source falls: ',
              chipName[objectIDs])

        sys.exit(-1)

    (xPupil, yPupil) = pupilCoordsFromRaDec(raICRS, decICRS,
                                            obs_metadata=self.obs_metadata,
                                            epoch=2000.0)

    # correct the SEDs for redshift, dust, etc.  Return a list of Sed objects as
    # defined in sims_photUtils/../../Sed.py
    sedList = self._calculateGalSimSeds()

    mem_tracker.print_usage('calling _initializeGalSimCatlog')
    if self.hasBeenInitialized is False and len(objectNames) > 0:
        # This needs to be here in case, instead of writing the whole catalog
        # with write_catalog(), the user wishes to iterate through the catalog
        # with InstanceCatalog.iter_catalog(), which will not call
        # write_header()
        self._initializeGalSimCatalog()

        if not hasattr(self, 'bandpassDict'):
            raise RuntimeError('ran initializeGalSimCatalog but do not have bandpassDict')

    mem_tracker.print_usage('about to enter loop over objects')
    output = []
    for (name, ra, dec, xp, yp, hlr, minor, major, pa, ss, sn) in \
        zip(objectNames, raICRS, decICRS, xPupil, yPupil, halfLight,
            minorAxis, majorAxis, positionAngle, sedList, sindex):
        flux_dict = {}
        for bb in self.bandpassNames:
            adu = ss.calcADU(self.bandpassDict[bb], self.photParams)
            flux_dict[bb] = adu*self.photParams.gain

        gsObj = GalSimCelestialObject(self.galsim_type, ss, ra, dec, xp, yp,
                                      hlr, minor, major, pa, sn, flux_dict)

        # Actually draw the object
        detectorsString = self.galSimInterpreter.drawObject(gsObj)
        output.append(detectorsString)

    mem_tracker.print_usage('finihsed loop over objects')
    return numpy.array(output)


def phoSimCalculateGalSimSeds(self):
    """
    This modified version of _calculateGalSimSeds gets it's information from
    the Pandas dataFrame.  I have also applied a few bug fixes.  This routine
    is monkey patched into GalSimBase.

    Original Text Below:

    Apply any physical corrections to the objects' SEDS (redshift them, apply
    dust, etc.). Return a list of Sed objects containing the SEDS
    """

    sedList = []

    actualSEDnames = self.phoSimDataBase['sedName']
    redshift = self.phoSimDataBase['redShift']
    internalAv = self.phoSimDataBase['internalAv']
    internalRv = self.phoSimDataBase['internalRv']
    galacticAv = self.phoSimDataBase['galacticAv']
    galacticRv = self.phoSimDataBase['galacticRv']
    magNorm = self.phoSimDataBase['magNorm']

    # Original code below.
    # actualSEDnames = self.column_by_name('sedFilepath')
    # redshift = self.column_by_name('redshift')
    # internalAv = self.column_by_name('internalAv')
    # internalRv = self.column_by_name('internalRv')
    # galacticAv = self.column_by_name('galacticAv')
    # galacticRv = self.column_by_name('galacticRv')
    # magNorm = self.column_by_name('magNorm')

    # For setting magNorm
    imsimband = Bandpass()
    imsimband.imsimBandpass()

    for (sedName, zz, iAv, iRv, gAv, gRv, norm) in zip(actualSEDnames, redshift,
                                                       internalAv, internalRv,
                                                       galacticAv, galacticRv,
                                                       magNorm):

        if is_null(sedName):
            sedList.append(None)
        else:
            if sedName in self.uniqueSeds:
                # we have already read in this file; no need to do it again
                sed = Sed(wavelen=self.uniqueSeds[sedName].wavelen,
                          flambda=self.uniqueSeds[sedName].flambda,
                          fnu=self.uniqueSeds[sedName].fnu,
                          name=self.uniqueSeds[sedName].name)
            else:
                # load the SED of the object
                sed = Sed()
                sedFile = os.path.join(self.sedDir, sedName)
                sed.readSED_flambda(sedFile)

                flambdaCopy = copy.deepcopy(sed.flambda)

                # If the SED is zero inside of the bandpass, GalSim raises an
                # error. This sets a minimum flux value of 1.0e-30 so that the
                # SED is never technically zero inside of the bandpass.
                sed.flambda = numpy.array([ff if ff > 1.0e-30 else 1.0e-30 for ff in flambdaCopy])
                sed.fnu = None

                # copy the unnormalized file to uniqueSeds so we don't have to read it in again
                sedCopy = Sed(wavelen=sed.wavelen, flambda=sed.flambda,
                              fnu=sed.fnu, name=sed.name)
                self.uniqueSeds[sedName] = sedCopy

            # normalize the SED
            # Consulting the file sed.py in GalSim/galsim/ it appears that
            # GalSim expects its SEDs to ultimately be in units of ergs/nm so
            # that, when called, they can be converted to photons/nm (see the
            # function __call__() and the assignment of self._rest_photons in
            # the __init__() of galsim's sed.py file).  Thus, we need to read in
            # our SEDs, normalize them, and then multiply by the exposure time
            # and the effective area to get from ergs/s/cm^2/nm to ergs/nm.
            #
            # The gain parameter should convert between photons and ADU (so: it
            # is the traditional definition of "gain" -- electrons per ADU --
            # multiplied by the quantum efficiency of the detector).  Because we
            # fold the quantum efficiency of the detector into our
            # total_[u,g,r,i,z,y].dat bandpass files (see the readme in the
            # THROUGHPUTS_DIR/baseline/), we only need to multiply by the
            # electrons per ADU gain.
            #
            # We will take these parameters from an instantiation of the
            # PhotometricParameters class (which can be reassigned by defining a
            # daughter class of this class)
            #
            fNorm = sed.calcFluxNorm(norm, imsimband)
            sed.multiplyFluxNorm(fNorm)

            # Apply dust extinction (internal)
            if iAv != 0.0 and iRv != 0.0:
                a_int, b_int = sed.setupCCMab()
                sed.addCCMDust(a_int, b_int, A_v=iAv, R_v=iRv)

            # 22 June 2015
            # apply redshift; there is no need to apply the distance modulus
            # from sims/photUtils/CosmologyWrapper; magNorm takes that into
            # account however, magNorm does not take into account cosmological
            # dimming
            if zz != 0.0:
                sed.redshiftSED(zz, dimming=True)

            # Apply dust extinction (galactic)
            # This if statement was missing! Talk to Scott.
            if gAv != 0.0 and gRv != 0.0:
                a_int, b_int = sed.setupCCMab()
                sed.addCCMDust(a_int, b_int, A_v=gAv, R_v=gRv)

            sedList.append(sed)

    return sedList
