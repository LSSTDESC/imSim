from collections import namedtuple
import math
import numpy as np
from galsim.config import InputLoader, RegisterInputType, RegisterValueType, \
    RegisterObjectType
from galsim.config import RegisterSEDType, RegisterBandpassType
from galsim import CelestialCoord
import galsim
from desc.skycatalogs import skyCatalogs
from .instcat import getHourAngle, get_radec_limits, InstCatalogLoader

SED_info = namedtuple('SED_info', ['sed', 'magnorm'])

class SkyCatalogInterface:
    """Interface to skyCatalogs package."""
    _bp500 = galsim.Bandpass(galsim.LookupTable([499,500,501],[0,1,0]),
                             wave_type='nm').withZeropoint('AB')

    # Using area-weighted effective aperture over FOV
    # from https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
    _rubin_area = 0.25 * np.pi * 649**2  # cm^2

    def __init__(self, file_name, wcs, edge_pix=100,
                 sort_mag=True, flip_g2=True, min_source=None,
                 skip_invalid=True, logger=None):
        logger = galsim.config.LoggerWrapper(logger)
        self.file_name = file_name
        self.wcs = wcs
        self.flip_g2 = flip_g2
        sky_cat = skyCatalogs.open_catalog(file_name)
        region = skyCatalogs.Box(*get_radec_limits(wcs, logger, edge_pix)[:4])
        obj_type_set = set(('star',))
        self.objects = sky_cat.get_objects_by_region(region,
                                                     obj_type_set=obj_type_set)
        self._sed_cache = dict()

    @staticmethod
    def get_subcomponents(skycat_obj):
        subcomponents = skycat_obj.subcomponents
        if not subcomponents:
            subcomponents = [None]
        return subcomponents

    def _update_sed_cache(self, index):
        skycat_obj = self.objects[index]
        for component in self.get_subcomponents(skycat_obj):
            wl, flambda, magnorm = skycat_obj.get_sed(component=component)
            if np.isinf(magnorm):
                flambda = np.zeros(len(wl))
            sed_lut = galsim.LookupTable(wl, flambda)
            sed = galsim.SED(sed_lut, wave_type='nm', flux_type='flambda')
            sed = sed.withMagnitude(0, self._bp500)
            self._sed_cache[(index, component)] = SED_info(sed, magnorm)

    def getNObjects(self, logger=None):
        return len(self.objects)

    @staticmethod
    def getWorldPos(skycat_obj):
        ra, dec = skycat_obj.ra, skycat_obj.dec
        return galsim.CelestialCoord(ra*galsim.degrees, dec*galsim.degrees)

    def getImagePos(self, world_pos):
        return self.wcs.toImage(world_pos)

    def getMagNorm(self, index, component):
        if (index, component) not in self._sed_cache:
            self._update_sed_cache(index)
        return self._sed_cache[(index, component)].magnorm

    def getSED(self, index, component):
        if (index, component) not in self._sed_cache:
            self._update_sed_cache(index)
        return self._sed_cache[(index, component)].sed

    @staticmethod
    def getLens(skycat_obj):
        gamma1 = skycat_obj.get_native_attribute('shear_1')
        gamma2 = skycat_obj.get_native_attribute('shear_2')
        kappa =  skycat_obj.get_native_attribute('convergence')
        # Return reduced shears and magnification.
        g1 = gamma1/(1. - kappa)    # real part of reduced shear
        g2 = gamma2/(1. - kappa)    # imaginary part of reduced shear
        mu = 1./((1. - kappa)**2 - (gamma1**2 + gamma2**2)) # magnification
        return g1, g2, mu

    def getDust(self, skycat_obj, band='i'):
        # For all objects, internal extinction is already part of SED,
        # so Milky Way dust is the only source of reddening.
        internal_av = 0
        internal_rv = 1.
        MW_av_colname = f'MW_av_lsst_{band}'
        galactic_av = skycat_obj.get_native_attribute(MW_av_colname)
        galactic_rv = skycat_obj.get_native_attribute('MW_rv')
        return internal_av, internal_rv, galactic_av, galactic_rv

    def getObj(self, index, gsparams=None, rng=None, bandpass=None,
               chromatic=False, exp_time=30):
        skycat_obj = self.objects[index]
        # Just return the first component for now.
        component = self.get_subcomponents(skycat_obj)[0]
        magnorm = self.getMagNorm(index, component=component)
        if magnorm >= 50:
            return None

        if gsparams is not None:
            gsparams = galsim.GSParams(**gsparams)

        if skycat_obj.object_type == 'star':
            obj = galsim.DeltaFunction(gsparams=gsparams)
        elif skycat_obj.object_type == 'galaxy':
            a = skycat_obj.get_native_attribute(f'size_{component}_true')
            b = skycat_obj.get_native_attribute(f'size_minor_{component}_true')
            assert a >= b
            pa = skycat_obj.get_native_attribute('position_angle_unlensed')
            if self.flip_g2:
                beta = float(90 - pa)*galsim.degrees
            else:
                beta = float(90 + pa)*galsim.degrees
            n = skycat_obj.get_native_attribute(f'sersic_{component}')
            # quantize the n values at 0.05 so that galsim can
            # possibly amortize sersic calculations from previous
            # galaxy.
            n = round(n*20.)/20.
            hlr = (a*b)**2   # approximation of half-light radius
            obj = galsim.Sersic(n=n, half_light_radius=hlr, gsparams=gsparams)
            shear = galsim.Shear(q=b/a, beta=beta)
            obj = obj._shear(shear)
            g1, g2, mu = self.getLens(skycat_obj)
            obj = obj._lens(g1, g2, mu)

        # The seds are normalized to correspond to magnorm=0.
        # The flux for the given magnorm is 10**(-0.4*magnorm)
        # The constant here, 0.9210340371976184 = 0.4 * log(10)
        flux = math.exp(-0.9210340371976184 * magnorm)

        # This gives the normalization in photons/cm^2/sec.
        # Multiply by area and exptime to get photons.
        fAt = flux * self._rubin_area * exp_time
        sed = self.getSED(index, component)
        if chromatic:
            return obj.withFlux(fAt) * sed

        flux = sed.calculateFlux(bandpass) * fAt
        return obj.withFlux(flux)

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

    skycat_obj = skycat.objects[index]
    pos = skycat.getWorldPos(skycat_obj)
    return pos, safe


def SkyCatObj(config, base, ignore, gsparams, logger):
    """
    Build an object according to info in the sky catalog.
    """
    skycat = galsim.config.GetInputObj('sky_catalog', config, base, 'SkyCat')

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
    bp = base['bandpass']
    exp_time = base.get('exp_time', None)

    obj = skycat.getObj(index, gsparams=gsparams, rng=rng, bandpass=bp,
                        exp_time=exp_time)
    return obj, safe


class SkyCatSEDBuilder(galsim.config.SEDBuilder):
    """A class for loading an SED from the sky catalog.
    """
    def buildSED(self, config, base, logger):
        """Build the SED based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the SED type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed SED object.
        """
        skycat = galsim.config.GetInputObj('sky_catalog', config, base,
                                           'SkyCatWorldPos')

        galsim.config.SetDefaultIndex(config, skycat.getNObjects())

        req = { 'index' : int }
        opt = { 'num' : int }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
        index = kwargs['index']
        skycat_obj = skycat.objects[index]
        component = skycat.get_subcomponents(skycat_obj)[0]
        sed = skycat.getSED(index, component)
        return sed, safe


RegisterInputType('sky_catalog',
                  InstCatalogLoader(SkyCatalogInterface, has_nobj=True))
RegisterValueType('SkyCatWorldPos', SkyCatWorldPos, [CelestialCoord],
                  input_type='sky_catalog')
RegisterObjectType('SkyCatObj', SkyCatObj, input_type='sky_catalog')
RegisterSEDType('SkyCatSED', SkyCatSEDBuilder(), input_type='sky_catalog')
