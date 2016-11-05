"""
Code to generate imSim subclasses of GalSimBase subclasses.
"""
from __future__ import absolute_import, print_function, division
import gc
#from lsst.sims.catalogs.db import CatalogDBObject
from lsst.sims.GalSimInterface import ExampleCCDNoise, SNRdocumentPSF
from lsst.sims.utils import pupilCoordsFromRaDec

__all__ = ['imSim_class_factory']

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

def imSim__init__(self, phosim_objects, obs_metadata):
    """
    ImSim* subclass constructor.

    Parameters
    ----------
    phosim_objects : pandas.DataFrame
        A DataFrame containing the instance catalog object data.
    obs_metadata : lsst.sims.utils.ObservationMetaData
        Object containing the telescope observation parameters.
    """
#    super(self.__imSim_class__, self).__init__(CatalogDBObject(),
#                                               obs_metadata=obs_metadata)
    self.obs_metadata = obs_metadata

    xPupil, yPupil = pupilCoordsFromRaDec(phosim_objects['raICRS'].values,
                                          phosim_objects['decICRS'].values,
                                          obs_metadata=obs_metadata,
                                          epoch=2000.0)
    phosim_objects = phosim_objects.assign(x_pupil=xPupil)
    self.phosim_objects = phosim_objects.assign(y_pupil=yPupil)
    gc.collect()

    self.db_obj = type('DummyDB', (), dict(epoch=2000))

    # Add noise and sky background
    self.noise_and_background = ExampleCCDNoise(addNoise=True,
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
