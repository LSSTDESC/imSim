"""
Interface to obtain objects from skyCatalogs.
"""
import os
import numpy as np
import galsim
from galsim.config import InputLoader, RegisterInputType, RegisterValueType, \
    RegisterObjectType
from desc.skycatalogs import skyCatalogs
from .instcat import get_radec_limits


class SkyCatalogInterface:
    """Interface to skyCatalogs package."""
    # Rubin effective area computed using numbers at
    # https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
    _eff_area = 0.25 * np.pi * 649**2  # cm^2
    def __init__(self, file_name, wcs, bandpass, xsize, ysize, obj_types=None,
                 skycatalog_root=None, edge_pix=100, logger=None):
        """Parameters
        ----------
        file_name : str
            Name of skyCatalogs yaml config file.
        wcs : galsim.WCS
            WCS of the image to render.
        bandpass : galsim.Bandpass
            Bandpass to use for flux calculations.
        xsize : int
            Size in pixels of CCD in x-direction.
        ysize : int
            Size in pixels of CCD in y-direction.
        obj_types : list-like [None]
            List or tuple of object types to render, e.g., ('star', 'galaxy').
            If None, then consider all object types.
        skycatalog_root : str [None]
            Root directory containing skyCatalogs files.  If None,
            then set skycatalog_root to the same directory containing
            the yaml config file.
        edge_pix : float [100]
            Size in pixels of the buffer region around nominal image
            to consider objects.
        logger : logging.Logger
            Logger object.

        """
        logger = galsim.config.LoggerWrapper(logger)
        if obj_types is not None:
            logger.warning(f'Object types restricted to {obj_types}')
        self.file_name = file_name
        self.wcs = wcs
        self.bandpass = bandpass
        if skycatalog_root is None:
            skycatalog_root = os.path.dirname(os.path.abspath(file_name))
        sky_cat = skyCatalogs.open_catalog(file_name,
                                           skycatalog_root=skycatalog_root)
        region = skyCatalogs.Box(*get_radec_limits(wcs, xsize, ysize, logger, edge_pix)[:4])
        self.objects = sky_cat.get_objects_by_region(region,
                                                     obj_type_set=obj_types)

    def getNObjects(self):
        """
        Return the number of GSObjects to render, where each subcomponent
        (e.g., bulge, disk, etc.) of each skyCatalog object is a distinct
        GSObject.
        """
        return len(self.objects)

    def getWorldPos(self, index):
        """
        Return the sky coordinates of the skyCatalog object
        corresponding to the specified index.

        Parameters
        ----------
        index : int
            Index of the (object_index, subcomponent) combination.

        Returns
        -------
        galsim.CelestialCoord
        """
        skycat_obj = self.objects[index]
        ra, dec = skycat_obj.ra, skycat_obj.dec
        return galsim.CelestialCoord(ra*galsim.degrees, dec*galsim.degrees)

    def getObj(self, index, gsparams=None, rng=None, exp_time=30):
        """
        Return the galsim object for the skyCatalog object
        corresponding to the specified index.  If the skyCatalog
        object is a galaxy, the returned galsim object will be
        a galsim.Sum.

        Parameters
        ----------
        index : int
            Index of the object in the self.objects catalog.

        Returns
        -------
        galsim.GSObject
        """
        skycat_obj = self.objects[index]
        gsobjs = skycat_obj.get_gsobject_components(gsparams, rng)
        seds = skycat_obj.get_observer_sed_components()

        gs_obj_list = []
        for component in gsobjs:
            if component in seds:
                gs_obj_list.append(gsobjs[component]*seds[component]
                                   *exp_time*self._eff_area)

        if not gs_obj_list:
            return None

        if len(gs_obj_list) == 1:
            gs_object = gs_obj_list[0]
        else:
            gs_object = galsim.Add(gs_obj_list)

        # Compute the flux or get the cached value.
        gs_object.flux \
            = skycat_obj.get_flux(self.bandpass)*exp_time*self._eff_area

        return gs_object


class SkyCatalogLoader(InputLoader):
    """
    Class to load SkyCatalogInterface object.
    """
    def getKwargs(self, config, base, logger):
        req = {'file_name': str}
        opt = {
               'edge_pix' : float,
               'obj_types' : list
              }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req,
                                                  opt=opt)
        wcs = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        kwargs['wcs'] = wcs
        kwargs['bandpass'] = base['bandpass']
        kwargs['xsize'] = base['xsize']
        kwargs['ysize'] = base['ysize']
        kwargs['logger'] = galsim.config.GetLoggerProxy(logger)
        return kwargs, safe


def SkyCatObj(config, base, ignore, gsparams, logger):
    """
    Build an object according to info in the sky catalog.
    """
    skycat = galsim.config.GetInputObj('sky_catalog', config, base, 'SkyCatObj')

    # Setup the indexing sequence if it hasn't been specified.  The
    # normal thing with a catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do
    # it for them.
    galsim.config.SetDefaultIndex(config, skycat.getNObjects())

    req = { 'index' : int }
    opt = { 'num' : int }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    index = kwargs['index']

    rng = galsim.config.GetRNG(config, base, logger, 'SkyCatObj')
    exp_time = base.get('exp_time', None)

    obj = skycat.getObj(index, gsparams=gsparams, rng=rng, exp_time=exp_time)
    base['object_id'] = skycat.objects[index].id

    return obj, safe


def SkyCatWorldPos(config, base, value_type):
    """Return a value from the object part of the skyCatalog
    """
    skycat = galsim.config.GetInputObj('sky_catalog', config, base,
                                       'SkyCatWorldPos')

    # Setup the indexing sequence if it hasn't been specified.  The
    # normal thing with a catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do
    # it for them.
    galsim.config.SetDefaultIndex(config, skycat.getNObjects())

    req = { 'index' : int }
    opt = { 'num' : int }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    index = kwargs['index']

    pos = skycat.getWorldPos(index)
    return pos, safe

RegisterInputType('sky_catalog',
                  SkyCatalogLoader(SkyCatalogInterface, has_nobj=True))
RegisterObjectType('SkyCatObj', SkyCatObj, input_type='sky_catalog')
RegisterValueType('SkyCatWorldPos', SkyCatWorldPos, [galsim.CelestialCoord],
                  input_type='sky_catalog')
