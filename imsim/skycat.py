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


def get_skycat_objects(sky_cat, wcs, xsize, ysize, edge_pix, logger, obj_types):
    """
    Find skycat objects that land on the CCD + edge_pix buffer region.

    Parameters
    ----------
    sky_cat : desc.skycatalogs.SkyCatalog
        SkyCatalog object.
    wcs : GalSim.GSFitsWCS
        WCS object for the current CCD.
    xsize : int
        Size in pixels of CCD in x-direction.
    ysize : int
        Size in pixels of CCD in y-direction.
    edge_pix : float
        Size in pixels of the buffer region around nominal image
        to consider objects.
    logger : logging.Logger
        Logger object.
    obj_types : list-like
        List or tuple of object types to render, e.g., ('star', 'galaxy').
        If None, then consider all object types.

    Returns
    -------
    list of skyCatalogs objects, CCD center as galsim.CelestialCoord
    """
    # Get range of ra, dec values given CCD size + edge_pix buffer.
    radec_limits = get_radec_limits(wcs, xsize, ysize, logger, edge_pix)

    # Initial pass using skyCatalogs.Box in ra, dec.
    region = skyCatalogs.Box(*radec_limits[:4])
    candidates = sky_cat.get_objects_by_region(region, obj_type_set=obj_types)

    # Compute pixel coords of candidate objects and downselect in pixel
    # coordinates.
    min_x, min_y, max_x, max_y = radec_limits[4:]
    objects = []
    for candidate in candidates:
        sky_coords = galsim.CelestialCoord(candidate.ra*galsim.degrees,
                                           candidate.dec*galsim.degrees)
        pixel_coords = wcs.toImage(sky_coords)
        if ((min_x < pixel_coords.x < max_x) and
            (min_y < pixel_coords.y < max_y)):
            objects.append(candidate)

    ccd_center = wcs.toWorld(galsim.PositionD(xsize/2.0, ysize/2.0))
    return objects, ccd_center


class SkyCatalogInterface:
    """Interface to skyCatalogs package."""
    # Rubin effective area computed using numbers at
    # https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
    _eff_area = 0.25 * np.pi * 649**2  # cm^2
    def __init__(self, file_name, wcs, band, xsize=4096, ysize=4096, obj_types=None,
                 skycatalog_root=None, edge_pix=100, max_flux=None, logger=None):
        """
        Parameters
        ----------
        file_name : str
            Name of skyCatalogs yaml config file.
        wcs : galsim.WCS
            WCS of the image to render.
        band : str
            LSST band, one of ugrizy
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
        max_flux : float [None]
            If object flux exceeds max_flux, the return None for that object.
            if max_flux == None, then don't apply a maximum flux cut.
        logger : logging.Logger
            Logger object.

        """
        logger = galsim.config.LoggerWrapper(logger)
        if obj_types is not None:
            logger.warning(f'Object types restricted to {obj_types}')
        self.file_name = file_name
        self.wcs = wcs
        self.band = band
        self.max_flux = max_flux
        if skycatalog_root is None:
            skycatalog_root = os.path.dirname(os.path.abspath(file_name))
        sky_cat = skyCatalogs.open_catalog(file_name,
                                           skycatalog_root=skycatalog_root)

        self.objects, self.ccd_center \
            = get_skycat_objects(sky_cat, wcs, xsize, ysize,
                                 edge_pix, logger, obj_types)
        if not self.objects and logger is not None:
            logger.warning("No objects found on image")

    def get_ccd_center(self):
        """
        Return the CCD center.
        """
        return self.ccd_center

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
        if not self.objects:
            raise RuntimeError("Trying to get an object from an empty sky catalog")

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
            = skycat_obj.get_LSST_flux(self.band)*exp_time*self._eff_area

        if self.max_flux is not None and gs_object.flux > self.max_flux:
            return None

        return gs_object


class SkyCatalogLoader(InputLoader):
    """
    Class to load SkyCatalogInterface object.
    """
    def getKwargs(self, config, base, logger):
        req = {'file_name': str}
        opt = {
               'edge_pix' : float,
               'obj_types' : list,
               'max_flux' : float
              }
        meta = galsim.config.GetInputObj('opsim_meta_dict', config, base,
                                         'SkyCatalogLoader')
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req,
                                                  opt=opt)
        wcs = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        kwargs['wcs'] = wcs
        kwargs['xsize'] = base['xsize']
        kwargs['ysize'] = base['ysize']
        # The skyCatalogs code will use the LSST bandpasses from the
        # throughputs distribution, so just need the LSST band (=filter)
        # from the opsim metadata object.
        kwargs['band'] = meta.get('band')
        kwargs['logger'] = galsim.config.GetLoggerProxy(logger)

        # Sky catalog object lists are created per CCD, so they are
        # not safe to reuse.
        safe = False
        return kwargs, safe


def SkyCatObj(config, base, ignore, gsparams, logger):
    """
    Build an object according to info in the sky catalog.
    """
    skycat = galsim.config.GetInputObj('sky_catalog', config, base, 'SkyCatObj')

    # Ensure that this sky catalog matches the CCD being simulated by
    # comparing center locations on the sky.
    world_center = base['world_center']
    ccd_center = skycat.get_ccd_center()
    sep = ccd_center.distanceTo(base['world_center'])/galsim.arcsec
    # Centers must agree to within at least 1 arcsec:
    if sep > 1.0:
        message = ("skyCatalogs selection and CCD center do not agree: \n"
                   "skycat.ccd_center: "
                   f"{ccd_center.ra/galsim.degrees:.5f}, "
                   f"{ccd_center.dec/galsim.degrees:.5f}\n"
                   "world_center: "
                   f"{world_center.ra/galsim.degrees:.5f}, "
                   f"{world_center.dec/galsim.degrees:.5f} \n"
                   f"Separation: {sep:.2e} arcsec")
        raise RuntimeError(message)

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
