"""
Interface to obtain objects from skyCatalogs.
"""
import os
import numpy as np
import galsim
from galsim.config import InputLoader, RegisterInputType, RegisterValueType, \
    RegisterObjectType
from skycatalogs import skyCatalogs


class SkyCatalogInterface:
    """Interface to skyCatalogs package."""
    # Rubin effective area computed using numbers at
    # https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
    _eff_area = 0.25 * np.pi * 649**2  # cm^2

    def __init__(self, file_name, wcs, band, mjd, xsize=4096, ysize=4096,
                 obj_types=None, skycatalog_root=None, edge_pix=100,
                 max_flux=None, logger=None, apply_dc2_dilation=False,
                 approx_nobjects=None):
        """
        Parameters
        ----------
        file_name : str
            Name of skyCatalogs yaml config file.
        wcs : galsim.WCS
            WCS of the image to render.
        band : str
            LSST band, one of ugrizy
        mjd : float
            MJD of the midpoint of the exposure.
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
        logger : logging.Logger [None]
            Logger object.
        apply_dc2_dilation : bool [False]
            Flag to increase object sizes by a factor sqrt(a/b) where
            a, b are the semi-major and semi-minor axes, respectively.
            This has the net effect of using the semi-major axis as the
            sersic half-light radius when building the object.  This will
            only be applied to galaxies.
        approx_nobjects : int [None]
            Approximate number of objects per CCD used by galsim to
            set up the image processing.  If None, then the actual number
            of objects found by skyCatalogs, via .getNObjects, will be used.
        """
        self.file_name = file_name
        self.wcs = wcs
        self.band = band
        self.mjd = mjd
        self.xsize = xsize
        self.ysize = ysize
        self.obj_types = obj_types
        if skycatalog_root is None:
            self.skycatalog_root = os.path.dirname(os.path.abspath(file_name))
        else:
            self.skycatalog_root = skycatalog_root
        self.edge_pix = edge_pix
        self.max_flux = max_flux
        self.logger = galsim.config.LoggerWrapper(logger)
        self.apply_dc2_dilation = apply_dc2_dilation
        self.approx_nobjects = approx_nobjects

        if obj_types is not None:
            self.logger.warning(f'Object types restricted to {obj_types}')
        self.ccd_center = wcs.toWorld(galsim.PositionD(xsize/2.0, ysize/2.0))
        self._objects = None

    @property
    def objects(self):
        if self._objects is None:
            # Select objects from polygonal region bounded by CCD edges
            corners = ((-self.edge_pix, -self.edge_pix),
                       (self.xsize + self.edge_pix, -self.edge_pix),
                       (self.xsize + self.edge_pix, self.ysize + self.edge_pix),
                       (-self.edge_pix, self.ysize + self.edge_pix))
            vertices = []
            for x, y in corners:
                sky_coord = self.wcs.toWorld(galsim.PositionD(x, y))
                vertices.append((sky_coord.ra/galsim.degrees,
                                 sky_coord.dec/galsim.degrees))
            region = skyCatalogs.PolygonalRegion(vertices)
            sky_cat = skyCatalogs.open_catalog(
                self.file_name, skycatalog_root=self.skycatalog_root)
            self._objects = sky_cat.get_objects_by_region(
                region, obj_type_set=self.obj_types, mjd=self.mjd)
            if not self._objects:
                self.logger.warning("No objects found on image.")
        return self._objects

    def get_ccd_center(self):
        """
        Return the CCD center.
        """
        return self.ccd_center

    def getNObjects(self):
        """
        Return the number of GSObjects to render
        """
        return len(self.objects)

    def getApproxNObjects(self):
        """
        Return the approximate number of GSObjects to render, as set in
        the class initializer.
        """
        if self.approx_nobjects is not None:
            return self.approx_nobjects
        return self.getNObjects()

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

    def getObj(self, index, gsparams=None, rng=None, exptime=30):
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

        if self.apply_dc2_dilation and skycat_obj.object_type == 'galaxy':
            # Apply DC2 dilation to the individual galaxy components.
            for component, gsobj in gsobjs.items():
                comp = component if component != 'knots' else 'disk'
                a = skycat_obj.get_native_attribute(f'size_{comp}_true')
                b = skycat_obj.get_native_attribute(f'size_minor_{comp}_true')
                scale = np.sqrt(a/b)
                gsobjs[component] = gsobj.dilate(scale)

        seds = skycat_obj.get_observer_sed_components(mjd=self.mjd)

        gs_obj_list = []
        for component in gsobjs:
            if component in seds:
                gs_obj_list.append(gsobjs[component]*seds[component]
                                   *exptime*self._eff_area)

        if not gs_obj_list:
            return None

        if len(gs_obj_list) == 1:
            gs_object = gs_obj_list[0]
        else:
            gs_object = galsim.Add(gs_obj_list)

        # Compute the flux or get the cached value.
        gs_object.flux \
            = skycat_obj.get_LSST_flux(self.band, mjd=self.mjd)*exptime*self._eff_area

        if self.max_flux is not None and gs_object.flux > self.max_flux:
            return None

        return gs_object


class SkyCatalogLoader(InputLoader):
    """
    Class to load SkyCatalogInterface object.
    """
    def getKwargs(self, config, base, logger):
        req = {'file_name': str, 'band':str}
        opt = {
               'edge_pix' : float,
               'obj_types' : list,
               'max_flux' : float,
               'apply_dc2_dilation': bool,
               'approx_nobjects': int,
               'mjd': float,
              }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req,
                                                  opt=opt)
        wcs = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        kwargs['wcs'] = wcs
        kwargs['xsize'] = base.get('det_xsize', 4096)
        kwargs['ysize'] = base.get('det_ysize', 4096)
        kwargs['logger'] = logger

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
    # Centers must agree to within at least 10 arcsec:
    # (If the sizes here are the default 4096, the real ysize might be 4004, so center could
    #  be off by as much as 1/2 * 92 * 0.2 arcsec < 10 arcsec.  This can happen if det_xsize
    #  and det_ysize aren't set by the time the SkyCatatalog is loaded.)
    if sep > 10.0:
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
    exptime = base.get('exptime', 30)

    obj = skycat.getObj(index, gsparams=gsparams, rng=rng, exptime=exptime)
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
