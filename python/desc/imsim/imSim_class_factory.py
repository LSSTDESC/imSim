"""
Code to generate imSim subclasses of GalSimBase subclasses.
"""
from __future__ import absolute_import, print_function, division
import gc

from lsst.sims.utils import pupilCoordsFromRaDec

from lsst.sims.GalSimInterface import GalSimStars, GalSimGalaxies
from lsst.sims.GalSimInterface import SNRdocumentPSF
from desc.imsim.skyModel import ESOSkyModel

__all__ = ['ImSimStars', 'ImSimGalaxies']


def imSim_class_factory(galsim_subclass):
    """
    Return a subclass of galsim_subclass that takes a pandas DataFrame
    instead of a CatalogDBObject.
    """
    imSim_class_name = galsim_subclass.__name__.replace('GalSim', 'ImSim')
    imSim_class = type(imSim_class_name,
                       (galsim_subclass,),
                       dict([('column_by_name', imSim_column_by_name),
                             ('__init__', imSim__init__),
                             ('__name__', imSim_class_name)]))
    imSim_class.__imSim_class__ = imSim_class
    return imSim_class


def imSim__init__(self, phosim_objects, obs_metadata, catalog_db=None):
    """
    ImSim* subclass constructor.

    Parameters
    ----------
    phosim_objects : pandas.DataFrame
        A DataFrame containing the instance catalog object data.
    obs_metadata : lsst.sims.utils.ObservationMetaData
        Object containing the telescope observation parameters.
    catalog_db : lsst.sims.catalogs.db.CatalogDBObject, optional
        CatalogDBObject to pass to InstanceCatalog superclass.
    """
    if catalog_db is not None:
        super(self.__imSim_class__, self).__init__(catalog_db,
                                                   obs_metadata=obs_metadata)
    else:
        self.obs_metadata = obs_metadata

    xPupil, yPupil = pupilCoordsFromRaDec(phosim_objects['raJ2000'].values,
                                          phosim_objects['decJ2000'].values,
                                          pm_ra = phosim_objects['properMotionRa'].values,
                                          pm_dec = phosim_objects['properMotionDec'].values,
                                          parallax = phosim_objects['parallax'].values,
                                          v_rad = phosim_objects['radialVelocity'].values,
                                          obs_metadata=obs_metadata,
                                          epoch=2000.0)

    self.phosim_objects = phosim_objects.assign(x_pupil=xPupil, y_pupil=yPupil)
    gc.collect()

    self.db_obj = type('DummyDB', (), dict(epoch=2000))

    # Add noise and sky background
    # The simple code using the default lsst-GalSim interface would be:
    #
    #    self.noise_and_background = ExampleCCDNoise(addNoise=True,
    #                                                addBackground=True)
    #
    # But, we need a more realistic sky model and we need to pass more than
    # this basic info to use Peter Y's ESO sky model.
    # We must pass obs_metadata, chip information etc...
    self.noise_and_background = ESOSkyModel(obs_metadata, addNoise=True,
                                            addBackground=True)

    # Add a PSF.  This one is taken from equation 30 of
    # www.astro.washington.edu/users/ivezic/Astr511/LSST_SNRdoc.pdf .
    #
    # Set seeing from self.obs_metadata.
    self.PSF = \
        SNRdocumentPSF(self.obs_metadata.seeing[self.obs_metadata.bandpass])

    # Add bandpasses to simulate over.
    self.bandpassNames = list(self.obs_metadata.bandpass)


def imSim_column_by_name(self, colname):
    """
    Function to overload InstanceCatalog.column_by_name.

    Parameters
    ----------
    colname : str
        The name of the column to return.

    Returns
    -------
    np.array
    """
    if colname not in self.phosim_objects:
        return super(self.__imSim_class__, self).column_by_name(colname)
    return self.phosim_objects[colname].values

ImSimStars = imSim_class_factory(GalSimStars)
ImSimGalaxies = imSim_class_factory(GalSimGalaxies)
