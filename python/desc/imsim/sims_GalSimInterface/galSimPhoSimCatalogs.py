from builtins import object
from lsst.sims.catUtils.exampleCatalogDefinitions import PhoSimCatalogPoint
from lsst.sims.catUtils.exampleCatalogDefinitions import PhoSimCatalogZPoint
from lsst.sims.catUtils.exampleCatalogDefinitions import PhoSimCatalogSersic2D

#from lsst.sims.GalSimInterface import GalSimStars, GalSimGalaxies, GalSimAgn
from . import GalSimStars, GalSimGalaxies, GalSimAgn

from lsst.sims.catalogs.decorators import compound
from lsst.sims.utils import _observedFromICRS


__all__ = ["GalSimPhoSimStars", "GalSimPhoSimGalaxies", "GalSimPhoSimAgn"]


class GalSimAstrometryStars(object):

    @compound('raPhoSim','decPhoSim')
    def get_phoSimCoordinates(self):
        ff = self.column_by_name('fitsFiles') # to force the catalog to draw the GalSim images
        raObs, decObs = self.observedStellarCoordinates(includeRefraction = False)
        return self._dePrecess(raObs, decObs, self.obs_metadata)


class GalSimAstrometryGalaxies(object):

    @compound('raPhoSim','decPhoSim')
    def get_phoSimCoordinates(self):
        ff = self.column_by_name('fitsFiles') # to force the catalog to draw the GalSim images
        ra = self.column_by_name('raJ2000')
        dec = self.column_by_name('decJ2000')
        raObs, decObs =  _observedFromICRS(ra, dec, includeRefraction = False, obs_metadata=self.obs_metadata,
                                           epoch=self.db_obj.epoch)
        return self._dePrecess(raObs, decObs, self.obs_metadata)


class GalSimPhoSimStars(GalSimAstrometryStars, PhoSimCatalogPoint, GalSimStars):
    """
    This InstanceCatalog class is written so that the write_catalog() method produces
    and InstanceCatalog formatted appropriately for input to PhoSim.  The write_images()
    method can then be called as in other GalSimCatalogs to produce images with GalSim.
    """

    default_columns = [('redshift', 0., float),('gamma1', 0., float), ('gamma2', 0., float),
                       ('kappa', 0., float), ('raOffset', 0., float), ('decOffset', 0., float),
                       ('galacticExtinctionModel', 'CCM', (str,3)),
                       ('internalExtinctionModel', 'none', (str,4)),
                       ('galacticAv', 0.1, float),
                       ('galacticRv', 3.1, float),
                       ('galSimType', 'pointSource', (str,11)),
                       ('internalAv', 0.0, float),
                       ('internalRv', 0.0, float),
                       ('majorAxis', 0.0, float),
                       ('minorAxis', 0.0, float),
                       ('sindex', 0.0, float),
                       ('npoints', 0, int),
                       ('positionAngle', 0.0, float),
                       ('halfLightRadius', 0.0, float)]


class GalSimPhoSimGalaxies(GalSimAstrometryGalaxies, PhoSimCatalogSersic2D, GalSimGalaxies):
    """
    This InstanceCatalog class is written so that the write_catalog() method produces
    and InstanceCatalog formatted appropriately for input to PhoSim.  The write_images()
    method can then be called as in other GalSimCatalogs to produce images with GalSim.
    """

    default_columns = [('gamma1', 0., float), ('gamma2', 0., float), ('kappa', 0., float),
                       ('raOffset', 0., float), ('decOffset', 0., float),
                       ('galacticAv', 0.1, float), ('galacticRv', 3.1, float),
                       ('galacticExtinctionModel', 'CCM', (str,3)),
                       ('internalExtinctionModel', 'CCM', (str,3)), ('internalAv', 0., float),
                       ('internalRv', 3.1, float),
                       ('npoints', 0, int),
                       ('galSimType', 'sersic', (str, 6))]


class GalSimPhoSimAgn(GalSimAstrometryGalaxies, PhoSimCatalogZPoint, GalSimAgn):
    """
    This InstanceCatalog class is written so that the write_catalog() method produces
    and InstanceCatalog formatted appropriately for input to PhoSim.  The write_images()
    method can then be called as in other GalSimCatalogs to produce images with GalSim.
    """

    default_columns = [('gamma1', 0., float), ('gamma2', 0., float), ('kappa', 0., float),
                       ('raOffset', 0., float), ('decOffset', 0., float), ('spatialmodel', 'ZPOINT', (str, 6)),
                       ('galacticExtinctionModel', 'CCM', (str,3)),
                       ('galacticAv', 0.1, float),
                       ('galacticRv', 3.1, float),
                       ('internalExtinctionModel', 'none', (str,4)),
                       ('galSimType', 'pointSource', (str,11)),
                       ('majorAxis', 0.0, float),
                       ('minorAxis', 0.0, float),
                       ('sindex', 0.0, float),
                       ('npoints', 0, int),
                       ('positionAngle', 0.0, float),
                       ('halfLightRadius', 0.0, float),
                       ('internalAv', 0.0, float),
                       ('internalRv', 0.0, float)]
