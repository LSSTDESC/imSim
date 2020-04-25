"""
This file defines GalSimBase, which is a daughter of InstanceCatalog designed
to interface with GalSimInterpreter and generate images using GalSim.

It also defines daughter classes of GalSimBase designed for specific
classes of astronomical objects:

GalSimGalaxies
GalSimAgn
GalSimStars
"""

from builtins import zip
from builtins import str
import numpy as np
import os

import lsst.utils
from lsst.sims.utils import arcsecFromRadians
from lsst.sims.catalogs.definitions import InstanceCatalog
from lsst.sims.catalogs.decorators import cached
from lsst.sims.catUtils.mixins import (CameraCoords, AstrometryGalaxies, AstrometryStars,
                                       EBVmixin)
#from lsst.sims.GalSimInterface import GalSimInterpreter, GalSimDetector, GalSimCelestialObject
#from lsst.sims.GalSimInterface import GalSimCameraWrapper
#from lsst.sims.GalSimInterface import make_galsim_detector
from . import GalSimInterpreter, GalSimDetector, GalSimCelestialObject
from . import GalSimCameraWrapper
from . import make_galsim_detector
from lsst.sims.photUtils import (Sed, Bandpass, BandpassDict,
                                 PhotometricParameters)
from lsst.afw.cameraGeom import DetectorType

__all__ = ["GalSimGalaxies", "GalSimAgn", "GalSimStars", "GalSimRandomWalk"]


def _is_null(argument):
    """
    Return True if 'argument' is some null value
    (i.e. 'Null', None, nan).
    False otherwise.
    This is used by InstanceCatalog.write_catalog() to identify rows
    with null values in key columns.
    """
    try:
        str_class = basestring
    except:
        str_class = str

    if argument is None:
        return True
    elif isinstance(argument, str_class):
        if argument.strip().lower() == 'null':
            return True
        elif argument.strip().lower() == 'nan':
            return True
        elif argument.strip().lower() == 'none':
            return True
    elif np.isnan(argument):
        return True

    return False


class GalSimBase(InstanceCatalog, CameraCoords):
    """
    The catalog classes in this file use the InstanceCatalog infrastructure to construct
    FITS images for each detector-filter combination on a simulated camera.  This is done by
    instantiating the class GalSimInterpreter.  GalSimInterpreter is the class which
    actually generates the FITS images.  As the GalSim InstanceCatalogs are iterated over,
    each object in the catalog is passed to the GalSimInterpeter, which adds the object
    to the appropriate FITS images.  The user can then write the images to disk by calling
    the write_images method in the GalSim InstanceCatalog.

    Objects are passed to the GalSimInterpreter by the get_fitsFiles getter function, which
    adds a column to the InstanceCatalog indicating which detectors' FITS files contain each
    object.

    Note: because each GalSim InstanceCatalog has its own GalSimInterpreter, it means
    that each GalSimInterpreter will only draw FITS images containing one type of object
    (whatever type of object is contained in the GalSim InstanceCatalog).  If the user
    wishes to generate FITS images containing multiple types of object, the method
    copyGalSimInterpreter allows the user to pass the GalSimInterpreter from one
    GalSim InstanceCatalog to another (so, the user could create a GalSim Instance
    Catalog of stars, generate that InstanceCatalog, then create a GalSim InstanceCatalog
    of galaxies, pass the GalSimInterpreter from the star catalog to this new catalog,
    and thus create FITS images that contain both stars and galaxies; see galSimCompoundGenerator.py
    in the examples/ directory of sims_catUtils for an example).

    This class (GalSimBase) is the base class for all GalSim InstanceCatalogs.  Daughter
    classes of this class need to behave like ordinary InstanceCatalog daughter classes
    with the following exceptions:

    1) If they re-define column_outputs, they must be certain to include the column
    'fitsFiles.'  The getter for this column (defined in this class) calls all of the
    GalSim image generation infrastructure

    2) Daughter classes of this class must define a member variable galsim_type that is either
    'sersic' or 'pointSource'.  This variable tells the GalSimInterpreter how to draw the
    object (to allow a different kind of image profile, define a new method in the GalSimInterpreter
    class similar to drawPoinSource and drawSersic)

    3) The variables bandpass_names (a list of the form ['u', 'g', 'r', 'i', 'z', 'y']),
    bandpass_directory, and bandpass_root should be defined to tell the GalSim InstanceCatalog
    where to find the files defining the bandpasses to be used for these FITS files.
    The GalSim InstanceCatalog will look for bandpass files in files with the names

    for bpn in bandpass_names:
        name = self.bandpass_directory+'/'+self.bandpass_root+'_'+bpn+'.dat'

    one should also define the following member variables:

        componentList is a list of files ins banpass_directory containing the response
        curves for the different components of the camera, e.g.
        ['detector.dat', 'm1.dat', 'm2.dat', 'm3.dat', 'lens1.dat', 'lens2.dat', 'lens3.dat']

        atomTransmissionName is the name of the file in bandpass_directory that contains the
        atmostpheric transmissivity, e.g. 'atmos_std.dat'

    4) Telescope parameters such as exposure time, area, and gain are stored in the
    GalSim InstanceCatalog member variable photParams, which is an instantiation of
    the class PhotometricParameters defined in sims_photUtils.

    Daughter classes of GalSimBase will generate both FITS images for all of the detectors/filters
    in their corresponding cameras and InstanceCatalogs listing all of the objects
    contained in those images.  The catalog is written using the normal write_catalog()
    method provided for all InstanceClasses.  The FITS files are drawn using the write_images()
    method that is unique to GalSim InstanceCatalogs.  The FITS file will be named something like:

    DetectorName_FilterName.fits

    (a typical LSST fits file might be R_0_0_S_1_0_y.fits)

    Note: If you call write_images() before iterating over the catalog (either by calling
    write_catalog() or using the iterator returned by InstanceCatalog.iter_catalog()),
    you will get empty or incomplete FITS files.  Objects are only added to the GalSimInterpreter
    in the course of iterating over the InstanceCatalog.
    """

    seed = 42

    # This is sort of a hack; it prevents findChipName in coordUtils from dying
    # if an object lands on multiple science chips.
    allow_multiple_chips = True

    # There is no point in writing things to the InstanceCatalog that do not have SEDs and/or
    # do not land on any detectors
    cannot_be_null = ['sedFilepath']

    column_outputs = ['galSimType', 'uniqueId', 'raICRS', 'decICRS',
                      'chipName', 'x_pupil', 'y_pupil', 'sedFilepath',
                      'majorAxis', 'minorAxis', 'sindex', 'halfLightRadius',
                      'npoints', 'positionAngle', 'fitsFiles']

    transformations = {'raICRS': np.degrees,
                       'decICRS': np.degrees,
                       'x_pupil': arcsecFromRadians,
                       'y_pupil': arcsecFromRadians,
                       'halfLightRadius': arcsecFromRadians}

    default_formats = {'S': '%s', 'f': '%.9g', 'i': '%i'}

    # This is used as the delimiter because the names of the detectors printed in the fitsFiles
    # column contain both ':' and ','
    delimiter = '; '

    sedDir = lsst.utils.getPackageDir('sims_sed_library')

    bandpassNames = None
    bandpassDir = os.path.join(lsst.utils.getPackageDir('throughputs'), 'baseline')
    bandpassRoot = 'filter_'
    componentList = ['detector.dat', 'm1.dat', 'm2.dat', 'm3.dat',
                     'lens1.dat', 'lens2.dat', 'lens3.dat']
    atmoTransmissionName = 'atmos_std.dat'

    # allowed_chips is a list of the names of the detectors we actually want to draw.
    # If 'None', then all chips are drawn.
    allowed_chips = None

    # This member variable will define a PSF to convolve with the sources.
    # See the classes PSFbase and DoubleGaussianPSF in
    # galSimUtilities.py for more information
    PSF = None

    # This member variable can store a GalSim noise model instantiation
    # which will be applied to the FITS images when they are created
    noise_and_background = None

    # Stores the gain and readnoise
    photParams = PhotometricParameters()

    # This must be an instantiation of the GalSimCameraWrapper class defined in
    # galSimCameraWrapper.py
    _camera_wrapper = None

    hasBeenInitialized = False

    galSimInterpreter = None  # the GalSimInterpreter instantiation for this catalog

    totalDrawings = 0
    totalObjects = 0

    @property
    def camera_wrapper(self):
        return self._camera_wrapper

    @camera_wrapper.setter
    def camera_wrapper(self, val):
        self._camera_wrapper = val
        self.camera = val.camera

    def _initializeGalSimCatalog(self):
        """
        Initializes an empty list of objects that have already been drawn to FITS images.
        We do not want to accidentally draw an object twice.

        Also initializes the GalSimInterpreter by calling self._initializeGalSimInterpreter()

        Objects are stored based on their uniqueId values.
        """
        self.objectHasBeenDrawn = set()
        self._initializeGalSimInterpreter()
        self.hasBeenInitialized = True

    @cached
    def get_sedFilepath(self):
        """
        Maps the name of the SED as stored in the database to the file stored in
        sims_sed_library
        """
        # copied from the phoSim catalogs
        return np.array([self.specFileMap[k] if k in self.specFileMap else None
                         for k in self.column_by_name('sedFilename')])

    def _calcSingleGalSimSed(self, sedName, zz, iAv, iRv, gAv, gRv, norm):
        """
        correct the SED for redshift, dust, etc.  Return an Sed object as defined in
        sims_photUtils/../../Sed.py
        """
        if _is_null(sedName):
            return None
        sed = Sed()
        sed.readSED_flambda(os.path.join(self.sedDir, sedName))
        imsimband = Bandpass()
        imsimband.imsimBandpass()
        # normalize the SED
        # Consulting the file sed.py in GalSim/galsim/ it appears that GalSim expects
        # its SEDs to ultimately be in units of ergs/nm so that, when called, they can
        # be converted to photons/nm (see the function __call__() and the assignment of
        # self._rest_photons in the __init__() of galsim's sed.py file).  Thus, we need
        # to read in our SEDs, normalize them, and then multiply by the exposure time
        # and the effective area to get from ergs/s/cm^2/nm to ergs/nm.
        #
        # The gain parameter should convert between photons and ADU (so: it is the
        # traditional definition of "gain" -- electrons per ADU -- multiplied by the
        # quantum efficiency of the detector).  Because we fold the quantum efficiency
        # of the detector into our total_[u,g,r,i,z,y].dat bandpass files
        # (see the readme in the THROUGHPUTS_DIR/baseline/), we only need to multiply
        # by the electrons per ADU gain.
        #
        # We will take these parameters from an instantiation of the PhotometricParameters
        # class (which can be reassigned by defining a daughter class of this class)
        #
        fNorm = sed.calcFluxNorm(norm, imsimband)
        sed.multiplyFluxNorm(fNorm)

        # apply dust extinction (internal)
        if iAv != 0.0 and iRv != 0.0:
            a_int, b_int = sed.setupCCM_ab()
            sed.addDust(a_int, b_int, A_v=iAv, R_v=iRv)

        # 22 June 2015
        # apply redshift; there is no need to apply the distance modulus from
        # sims/photUtils/CosmologyWrapper; magNorm takes that into account
        # however, magNorm does not take into account cosmological dimming
        if zz != 0.0:
            sed.redshiftSED(zz, dimming=True)

        # apply dust extinction (galactic)
        if gAv != 0.0 and gRv != 0.0:
            a_int, b_int = sed.setupCCM_ab()
            sed.addDust(a_int, b_int, A_v=gAv, R_v=gRv)
        return sed

    def _calculateGalSimSeds(self):
        """
        Apply any physical corrections to the objects' SEDS (redshift them, apply dust, etc.).

        Return a generator that serves up the Sed objects in order.
        """
        actualSEDnames = self.column_by_name('sedFilepath')
        redshift = self.column_by_name('redshift')
        internalAv = self.column_by_name('internalAv')
        internalRv = self.column_by_name('internalRv')
        galacticAv = self.column_by_name('galacticAv')
        galacticRv = self.column_by_name('galacticRv')
        magNorm = self.column_by_name('magNorm')

        return (self._calcSingleGalSimSed(*args) for args in
                zip(actualSEDnames, redshift, internalAv, internalRv,
                    galacticAv, galacticRv, magNorm))

    @cached
    def get_fitsFiles(self, checkpoint_file=None, nobj_checkpoint=1000):
        """
        This getter returns a column listing the names of the detectors whose corresponding
        FITS files contain the object in question.  The detector names will be separated by a '//'

        This getter also passes objects to the GalSimInterpreter to actually draw the FITS
        images.

        WARNING: do not include 'fitsFiles' in the cannot_be_null list of non-null columns.
        If you do that, this method will be called several times by the catalog, as it
        attempts to determine which rows are actually in the catalog.  That will cause
        your images to have too much flux in them.
        """
        if self.bandpassNames is None:
            if isinstance(self.obs_metadata.bandpass, list):
                self.bandpassNames = [self.obs_metadata.bandpass]
            else:
                self.bandpassNames = self.obs_metadata.bandpass

        objectNames = self.column_by_name('uniqueId')
        xPupil = self.column_by_name('x_pupil')
        yPupil = self.column_by_name('y_pupil')
        halfLight = self.column_by_name('halfLightRadius')
        minorAxis = self.column_by_name('minorAxis')
        majorAxis = self.column_by_name('majorAxis')
        positionAngle = self.column_by_name('positionAngle')
        sindex = self.column_by_name('sindex')
        npoints = self.column_by_name('npoints')
        gamma1 = self.column_by_name('gamma1')
        gamma2 = self.column_by_name('gamma2')
        kappa = self.column_by_name('kappa')

        sedList = self._calculateGalSimSeds()

        if self.hasBeenInitialized is False and len(objectNames) > 0:
            # This needs to be here in case, instead of writing the whole catalog with write_catalog(),
            # the user wishes to iterate through the catalog with InstanceCatalog.iter_catalog(),
            # which will not call write_header()
            self._initializeGalSimCatalog()
            if not hasattr(self, 'bandpassDict'):
                raise RuntimeError('ran initializeGalSimCatalog but do not have bandpassDict')
            self.galSimInterpreter.checkpoint_file = checkpoint_file
            self.galSimInterpreter.nobj_checkpoint = nobj_checkpoint
            self.galSimInterpreter.restore_checkpoint(self._camera_wrapper,
                                                      self.photParams,
                                                      self.obs_metadata,
                                                      epoch=self.db_obj.epoch)

        output = []
        for (name, xp, yp, hlr, minor, major, pa, ss, sn, npo, gam1, gam2, kap) in \
            zip(objectNames, xPupil, yPupil, halfLight,
                 minorAxis, majorAxis, positionAngle, sedList, sindex, npoints,
                 gamma1, gamma2, kappa):

            if name in self.objectHasBeenDrawn:
                raise RuntimeError('Trying to draw %s more than once ' % str(name))
            elif ss is None:
                raise RuntimeError('Trying to draw an object with SED == None')
            else:

                self.objectHasBeenDrawn.add(name)

                if name not in self.galSimInterpreter.drawn_objects:

                    gsObj = GalSimCelestialObject(self.galsim_type, xp, yp,
                                                  hlr, minor, major, pa, sn,
                                                  ss, self.bandpassDict, self.photParams,
                                                  npo, None, None, None,
                                                  gam1, gam2, kap, uniqueId=name)

                    # actually draw the object
                    detectorsString = self.galSimInterpreter.drawObject(gsObj)
                else:
                    # For objects that have already been drawn in the
                    # checkpointed data, use a blank string.
                    detectorsString = ''

                output.append(detectorsString)

        # Force checkpoint at the end (if a checkpoint file has been specified).
        if self.galSimInterpreter is not None:
            self.galSimInterpreter.write_checkpoint(force=True)
        return np.array(output)

    def setPSF(self, PSF):
        """
        Set the PSF of this GalSimCatalog after instantiation.

        @param [in] PSF is an instantiation of a GalSimPSF class.
        """
        self.PSF = PSF
        if self.galSimInterpreter is not None:
            self.galSimInterpreter.setPSF(PSF=PSF)

    def copyGalSimInterpreter(self, otherCatalog):
        """
        Copy the camera, GalSimInterpreter, from another GalSim InstanceCatalog
        so that multiple types of object (stars, AGN, galaxy bulges, galaxy disks, etc.)
        can be drawn on the same FITS files.

        @param [in] otherCatalog is another GalSim InstanceCatalog that already has
        an initialized GalSimInterpreter

        See galSimCompoundGenerator.py in the examples/ directory of sims_catUtils for
        an example of how this is used.
        """
        self.camera_wrapper = otherCatalog.camera_wrapper
        self.photParams = otherCatalog.photParams
        self.PSF = otherCatalog.PSF
        self.noise_and_background = otherCatalog.noise_and_background
        if otherCatalog.hasBeenInitialized:
            self.bandpassDict = otherCatalog.bandpassDict
            self.galSimInterpreter = otherCatalog.galSimInterpreter

    def _initializeGalSimInterpreter(self):
        """
        This method creates the GalSimInterpreter (if it is None)

        This method reads in all of the data about the camera and pass it into
        the GalSimInterpreter.

        This method calls _getBandpasses to construct the paths to
        the files containing the bandpass data.
        """

        if not isinstance(self.camera_wrapper, GalSimCameraWrapper):
            raise RuntimeError("GalSimCatalog.camera_wrapper must be an instantiation of "
                               "GalSimCameraWrapper or one of its daughter classes\n"
                               "It is actually of type %s" % str(type(self.camera_wrapper)))

        if self.galSimInterpreter is None:

            # This list will contain instantiations of the GalSimDetector class
            # (see galSimInterpreter.py), which stores detector information in a way
            # that the GalSimInterpreter will understand
            detectors = []

            for dd in self.camera_wrapper.camera:
                if dd.getType() == DetectorType.WAVEFRONT or dd.getType() == DetectorType.GUIDER:
                    # This package does not yet handle the 90-degree rotation
                    # in WCS that occurs for wavefront or guide sensors
                    continue

                if self.allowed_chips is None or dd.getName() in self.allowed_chips:
                    detectors.append(make_galsim_detector(self.camera_wrapper, dd.getName(),
                                                          self.photParams, self.obs_metadata,
                                                          epoch=self.db_obj.epoch))

            if not hasattr(self, 'bandpassDict'):
                if self.noise_and_background is not None:
                    if self.obs_metadata.m5 is None:
                        raise RuntimeError('WARNING  in GalSimCatalog; you did not specify m5 in your '
                                           'obs_metadata. m5 is required in order to '
                                           'add noise to your images')

                    for name in self.bandpassNames:
                        if name not in self.obs_metadata.m5:
                            raise RuntimeError('WARNING in GalSimCatalog; your obs_metadata does not have ' +
                                               'm5 values for all of your bandpasses \n' +
                                               'bandpass has: %s \n' % self.bandpassNames.__repr__() +
                                               'm5 has: %s ' % list(self.obs_metadata.m5.keys()).__repr__())

                    if self.obs_metadata.seeing is None:
                        raise RuntimeError('WARNING  in GalSimCatalog; you did not specify seeing in your '
                                           'obs_metadata.  seeing is required in order to add '
                                           'noise to your images')

                    for name in self.bandpassNames:
                        if name not in self.obs_metadata.seeing:
                            raise RuntimeError('WARNING in GalSimCatalog; your obs_metadata does not have ' +
                                               'seeing values for all of your bandpasses \n' +
                                               'bandpass has: %s \n' % self.bandpassNames.__repr__() +
                                               'seeing has: %s ' % list(self.obs_metadata.seeing.keys()).__repr__())

                (self.bandpassDict,
                 hardwareDict) = BandpassDict.loadBandpassesFromFiles(bandpassNames=self.bandpassNames,
                                                                      filedir=self.bandpassDir,
                                                                      bandpassRoot=self.bandpassRoot,
                                                                      componentList=self.componentList,
                                                                      atmoTransmission=os.path.join(self.bandpassDir,
                                                                                                    self.atmoTransmissionName))

            self.galSimInterpreter = GalSimInterpreter(obs_metadata=self.obs_metadata,
                                                       epoch=self.db_obj.epoch,
                                                       detectors=detectors,
                                                       bandpassDict=self.bandpassDict,
                                                       noiseWrapper=self.noise_and_background,
                                                       seed=self.seed)

            self.galSimInterpreter.setPSF(PSF=self.PSF)



    def write_images(self, nameRoot=None):
        """
        Writes the FITS images associated with this InstanceCatalog.

        Cannot be called before write_catalog is called.

        @param [in] nameRoot is an optional string prepended to the names
        of the FITS images.  The FITS images will be named

        @param [out] namesWritten is a list of the names of the FITS files generated

        nameRoot_DetectorName_FilterName.fits

        (e.g. myImages_R_0_0_S_1_1_y.fits for an LSST-like camera with
        nameRoot = 'myImages')
        """
        namesWritten = self.galSimInterpreter.writeImages(nameRoot=nameRoot)

        return namesWritten


class GalSimGalaxies(GalSimBase, AstrometryGalaxies, EBVmixin):
    """
    This is a GalSimCatalog class for galaxy components (i.e. objects that are shaped
    like Sersic profiles).

    See the docstring in GalSimBase for explanation of how this class should be used.
    """

    catalog_type = 'galsim_galaxy'
    galsim_type = 'sersic'
    default_columns = [('galacticAv', 0.1, float),
                       ('galacticRv', 3.1, float),
                       ('galSimType', 'sersic', str, 6),
                       ('npoints', 0, int),
                       ('gamma1', 0.0, float),
                       ('gamma2', 0.0, float),
                       ('kappa', 0.0, float)]

class GalSimRandomWalk(GalSimBase, AstrometryGalaxies, EBVmixin):
    """
    This is a GalSimCatalog class for galaxy components (i.e. objects that are shaped
    like Sersic profiles).

    See the docstring in GalSimBase for explanation of how this class should be used.
    """

    catalog_type = 'galsim_random_walk'
    galsim_type = 'RandomWalk'
    default_columns = [('galacticAv', 0.1, float),
                       ('galacticRv', 3.1, float),
                       ('galSimType', 'RandomWalk', str, 10),
                       ('sindex', 0.0, float),
                       ('gamma1', 0.0, float),
                       ('gamma2', 0.0, float),
                       ('kappa', 0.0, float)]

class GalSimAgn(GalSimBase, AstrometryGalaxies, EBVmixin):
    """
    This is a GalSimCatalog class for AGN.

    See the docstring in GalSimBase for explanation of how this class should be used.
    """
    catalog_type = 'galsim_agn'
    galsim_type = 'pointSource'
    default_columns = [('galacticAv', 0.1, float),
                       ('galacticRv', 3.1, float),
                       ('galSimType', 'pointSource', str, 11),
                       ('majorAxis', 0.0, float),
                       ('minorAxis', 0.0, float),
                       ('sindex', 0.0, float),
                       ('npoints', 0, int),
                       ('positionAngle', 0.0, float),
                       ('halfLightRadius', 0.0, float),
                       ('internalAv', 0.0, float),
                       ('internalRv', 0.0, float),
                       ('gamma1', 0.0, float),
                       ('gamma2', 0.0, float),
                       ('kappa', 0.0, float)]


class GalSimStars(GalSimBase, AstrometryStars):
    """
    This is a GalSimCatalog class for stars.

    See the docstring in GalSimBase for explanation of how this class should be used.
    """
    catalog_type = 'galsim_stars'
    galsim_type = 'pointSource'
    default_columns = [('galacticAv', 0.1, float),
                       ('galacticRv', 3.1, float),
                       ('galSimType', 'pointSource', str, 11),
                       ('internalAv', 0.0, float),
                       ('internalRv', 0.0, float),
                       ('redshift', 0.0, float),
                       ('majorAxis', 0.0, float),
                       ('minorAxis', 0.0, float),
                       ('sindex', 0.0, float),
                       ('npoints', 0, int),
                       ('positionAngle', 0.0, float),
                       ('halfLightRadius', 0.0, float),
                       ('gamma1', 0.0, float),
                       ('gamma2', 0.0, float),
                       ('kappa', 0.0, float)]
