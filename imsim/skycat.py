"""
Interface to obtain objects from skyCatalogs.
"""
import os
import math
import numpy as np
import astropy.units as u
from dust_extinction.parameter_averages import F19
from galsim.config import InputLoader, RegisterInputType, RegisterValueType, \
    RegisterObjectType
from galsim import CelestialCoord
import galsim
from desc.skycatalogs import skyCatalogs
from .instcat import get_radec_limits


class SkyCatalogInterface:
    """Interface to skyCatalogs package."""
    _bp500 = galsim.Bandpass(galsim.LookupTable([499,500,501],[0,1,0]),
                             wave_type='nm').withZeropoint('AB')

    # Using area-weighted effective aperture over FOV
    # from https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
    _rubin_area = 0.25 * np.pi * 649**2  # cm^2

    def __init__(self, file_name, wcs, band, obj_types=None, edge_pix=100,
                 logger=None):
        """
        Parameters
        ----------
        file_name : str
            Name of skyCatalogs yaml config file.
        wcs : galsim.WCS
            WCS of the image to render.
        band : str
            LSST band, e.g., 'u', 'g', 'r', 'i', 'z', or 'y'
        obj_types : list-like [None]
            List or tuple of object types to render, e.g., ('star', 'galaxy').
            If None, then consider all object types.
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
        self.band = band
        sky_cat = skyCatalogs.open_catalog(file_name)
        region = skyCatalogs.Box(*get_radec_limits(wcs, logger, edge_pix)[:4])
        self.objects = sky_cat.get_objects_by_region(region,
                                                     obj_type_set=obj_types)

    def getNObjects(self):
        """
        Return the number of GSObjects to render, where each subcomponent
        (e.g., bulge, disk, etc.) of each skyCatalog object is a distinct
        GSObject.
        """
        return len(self.objects)

    @property
    def nobjects(self):
        return self.getNObjects()

    def getSED_info(self, skycat_obj, component):
        """
        Return the SED and magnorm value of the skyCatalog object
        corresponding to the specified index.

        Parameters
        ----------
        skycat_obj : skycatalogs.BaseObject
            The skycatalogs object, e.g., star or galaxy, which can
            have several subcomponents, e.g., disk, bulge, knots.
        component : str
            Name of the subcomponent.

        Returns
        -------
        (galsim.SED, float)
        """

        wl, flambda, magnorm \
            = skycat_obj.get_sed(component=component)
        if np.isinf(magnorm):
            # Galaxy subcomponents, e.g., bulge components, can
            # have zero-valued SEDs.  The skyCatalogs code returns
            # magnorm=inf for these components.
            return None, magnorm
        sed_lut = galsim.LookupTable(wl, flambda)
        sed = galsim.SED(sed_lut, wave_type='nm', flux_type='flambda')
        sed = sed.withMagnitude(0, self._bp500)
        iAv, iRv, mwAv, mwRv = self.getDust(skycat_obj)

        # TODO: apply internal extinction here

        # Apply redshift.
        if 'redshift' in skycat_obj.native_columns:
            redshift = skycat_obj.get_native_attribute('redshift')
            sed = sed.atRedshift(redshift)

        # Apply Milky Way extinction
        extinction = F19(Rv=mwRv)
        wl = np.linspace(300, 1200, 901)
        ext = extinction.extinguish(wl*u.nm, Av=mwAv)
        spec = galsim.LookupTable(wl, ext)
        mw_ext = galsim.SED(spec, wave_type='nm', flux_type='1')
        sed = sed*mw_ext

        sed = sed.thin()

        return sed, magnorm

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

    def getLens(self, skycat_obj):
        """
        Return the weak lensing parameters for the skyCatalog object
        corresponding to the specified index.

        Parameters
        ----------
        skycat_obj : skycatalogs.BaseObject
            The skycatalogs object, e.g., star or galaxy, which can
            have several subcomponents, e.g., disk, bulge, knots.

        Returns
        -------
        (g1, g2, mu)
        """
        gamma1 = skycat_obj.get_native_attribute('shear_1')
        gamma2 = skycat_obj.get_native_attribute('shear_2')
        kappa =  skycat_obj.get_native_attribute('convergence')
        # Return reduced shears and magnification.
        g1 = gamma1/(1. - kappa)    # real part of reduced shear
        g2 = gamma2/(1. - kappa)    # imaginary part of reduced shear
        mu = 1./((1. - kappa)**2 - (gamma1**2 + gamma2**2)) # magnification
        return g1, g2, mu

    def getDust(self, skycat_obj):
        """
        Return the extinction parameters for the skyCatalog object
        corresponding to the specified index.

        Parameters
        ----------
        skycat_obj : skycatalogs.BaseObject
            The skycatalogs object, e.g., star or galaxy, which can
            have several subcomponents, e.g., disk, bulge, knots.

        Returns
        -------
        (internal_av, internal_rv, galactic_av, galactic_rv)
        """
        # For all objects, internal extinction is already part of the SED,
        # so Milky Way dust is the only source of reddening.
        internal_av = 0
        internal_rv = 1.
        MW_av_colname = f'MW_av_lsst_{self.band}'
        galactic_av = skycat_obj.get_native_attribute(MW_av_colname)
        galactic_rv = skycat_obj.get_native_attribute('MW_rv')
        return internal_av, internal_rv, galactic_av, galactic_rv

    def get_gsobject(self, skycat_obj, component, gsparams, rng, exp_time):
        """
        Return a galsim.GSObject for the specifed skyCatalogs object
        and component.

        Parameters
        ----------
        skycat_obj : skyCatalogs.BaseObject
            A skyCatalogs object, e.g., a star or galaxy.
        component : str
            The name of the sub-component of the skyCatalogs object
            to consider.
        """
        sed, magnorm = self.getSED_info(skycat_obj, component)
        if sed is None or magnorm >= 50:
            return None

        if gsparams is not None:
            gsparams = galsim.GSParams(**gsparams)

        if skycat_obj.object_type == 'star':
            obj = galsim.DeltaFunction(gsparams=gsparams)
        elif (skycat_obj.object_type == 'galaxy' and
              component in ('bulge', 'disk', 'knots')):
            my_component = component
            if my_component == 'knots':
                my_component = 'disk'
            a = skycat_obj.get_native_attribute(f'size_{my_component}_true')
            b = skycat_obj.get_native_attribute(f'size_minor_{my_component}_true')
            assert a >= b
            pa = skycat_obj.get_native_attribute('position_angle_unlensed')
            beta = float(90 + pa)*galsim.degrees
            hlr = (a*b)**0.5   # approximation for half-light radius
            if component == 'knots':
                npoints = skycat_obj.get_native_attribute('n_knots')
                assert npoints > 0
                obj =  galsim.RandomKnots(npoints=npoints,
                                          half_light_radius=hlr, rng=rng,
                                          gsparams=gsparams)
            else:
                n = skycat_obj.get_native_attribute(f'sersic_{component}')
                # Quantize the n values at 0.05 so that galsim can
                # possibly amortize sersic calculations from the previous
                # galaxy.
                n = round(n*20.)/20.
                obj = galsim.Sersic(n=n, half_light_radius=hlr,
                                    gsparams=gsparams)
            shear = galsim.Shear(q=b/a, beta=beta)
            obj = obj._shear(shear)
            g1, g2, mu = self.getLens(skycat_obj)
            obj = obj._lens(g1, g2, mu)
        else:
            raise RuntimeError("Do not know how to handle object type: %s" %
                               component)

        # The seds are normalized to correspond to magnorm=0.
        # The flux for the given magnorm is 10**(-0.4*magnorm)
        # The constant here, 0.9210340371976184 = 0.4 * log(10)
        flux = math.exp(-0.9210340371976184 * magnorm)

        # This gives the normalization in photons/cm^2/sec.
        # Multiply by area and exptime to get photons.
        fAt = flux * self._rubin_area * exp_time
        return obj.withFlux(fAt) * sed

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
        subcomponents = skycat_obj.subcomponents
        if not subcomponents:
            # Stars have an empty list as their subcomponents
            # attribute, indicating they are not composite objects
            # like galaxies, and can be returned directly.
            return self.get_gsobject(skycat_obj, None, gsparams, rng,  exp_time)
        gs_objs = []
        for component in subcomponents:
            gs_obj = self.get_gsobject(skycat_obj, component, gsparams, rng,
                                       exp_time)
            if gs_obj is not None:
                gs_objs.append(gs_obj)

        if not gs_objs:
            return None

        return galsim.Add(gs_objs)


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
        meta = galsim.config.GetInputObj('opsim_meta_dict', config, base,
                                         'SkyCatObj')
        wcs = galsim.config.BuildWCS(base['image'], 'wcs', base, logger=logger)
        kwargs['wcs'] = wcs
        kwargs['band'] = meta.get('band')
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
RegisterValueType('SkyCatWorldPos', SkyCatWorldPos, [CelestialCoord],
                  input_type='sky_catalog')
