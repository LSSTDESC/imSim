"""
Wrapper class for SkyCatalog objects.
"""
import numpy as np
import astropy.units as u
from dust_extinction.parameter_averages import F19
import galsim
from desc.skycatalogs import skyCatalogs

__all__ = ['SkyCatalogObjectWrapper']

class SkyCatalogObjectWrapper:
    """
    Class to wrap skyCatalog objects with an interface that creates
    galsim.SEDs and galsim.GSObjects based on the catalog column values.
    """
    _bp500 = galsim.Bandpass(galsim.LookupTable([499, 500, 501],[0, 1, 0]),
                             wave_type='nm').withZeropoint('AB')

    def __init__(self, skycat_obj, band, bandpass=None, eff_area=None):
        """
        Parameters
        ----------
        skycat_obj : desc.skycatalogs.objects.BaseObject
            The skyCatalogs object to be rendered, e.g., a star or galaxy.
        band : str
            LSST band, one of 'ugrizy'
        bandpass : galsim.Bandpass [None]
            Bandpass object.  If None, then the galsim versions of the
            LSST throughputs will be used.
        eff_area : float [None]
            Area-weighted effective aperture over FOV.  If None, then
            LSST value computed from
            https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
            will be used.
        """
        self.skycat_obj = skycat_obj
        self.band = band
        if bandpass is None:
            self.bandpass = galsim.Bandpass(f'LSST_{band}.dat', wave_type='nm')
        else:
            self.bandpass = bandpass
        if eff_area is None:
            self.eff_area = 0.25 * np.pi * 649**2  # cm^2, Rubin value
        else:
            self.eff_area = eff_area

    def get_dust(self):
        """Return the Av, Rv parameters for internal and Milky Way extinction."""
        internal_av = 0
        internal_rv = 1.
        MW_av_colname = f'MW_av_lsst_{self.band}'
        galactic_av = self.skycat_obj.get_native_attribute(MW_av_colname)
        galactic_rv = self.skycat_obj.get_native_attribute('MW_rv')
        return internal_av, internal_rv, galactic_av, galactic_rv

    def get_wl_params(self):
        """Return the weak lensing parameters, g1, g2, mu."""
        gamma1 = self.skycat_obj.get_native_attribute('shear_1')
        gamma2 = self.skycat_obj.get_native_attribute('shear_2')
        kappa =  self.skycat_obj.get_native_attribute('convergence')
        # Compute reduced shears and magnification.
        g1 = gamma1/(1. - kappa)    # real part of reduced shear
        g2 = gamma2/(1. - kappa)    # imaginary part of reduced shear
        mu = 1./((1. - kappa)**2 - (gamma1**2 + gamma2**2)) # magnification
        return g1, g2, mu

    def get_gsobject_components(self, gsparams=None, rng=None):
        """
        Return a dictionary of the GSObject components for the
        SkyCatalogs object, keyed by component name.
        """
        if gsparams is not None:
            gsparams = galsim.GSParams(**gsparams)
        if self.skycat_obj.object_type == 'star':
            return {None: galsim.DeltaFunction(gsparams=gsparams)}
        if self.skycat_obj.object_type != 'galaxy':
            raise RuntimeError("Do not know how to handle object type "
                               f"{self.skycat_obj.object_type}")
        obj_dict = {}
        for component in self.skycat_obj.subcomponents:
            # knots use the same major/minor axes as the disk component.
            my_component = 'disk' if component != 'bulge' else 'bulge'
            a = self.skycat_obj.get_native_attribute(
                f'size_{my_component}_true')
            b = self.skycat_obj.get_native_attribute(
                f'size_minor_{my_component}_true')
            assert a >= b
            pa = self.skycat_obj.get_native_attribute('position_angle_unlensed')
            beta = float(90 + pa)*galsim.degrees
            hlr = (a*b)**0.5   # approximation for half-light radius
            if component == 'knots':
                npoints = self.skycat_obj.get_native_attribute('n_knots')
                assert npoints > 0
                obj = galsim.RandomKnots(npoints=npoints,
                                         half_light_radius=hlr, rng=rng,
                                         gsparams=gsparams)
            else:
                n = self.skycat_obj.get_native_attribute(f'sersic_{component}')
                # Quantize the n values at 0.05 so that galsim can
                # possibly amortize sersic calculations from the previous
                # galaxy.
                n = round(n*20.)/20.
                obj = galsim.Sersic(n=n, half_light_radius=hlr,
                                    gsparams=gsparams)
            shear = galsim.Shear(q=b/a, beta=beta)
            obj = obj._shear(shear)
            g1, g2, mu = self.get_wl_params()
            obj_dict[component] = obj._lens(g1, g2, mu)
        return obj_dict

    def get_sed_component(self, component):
        """
        Return the SED for the specified subcomponent of the SkyCatalog
        object, applying internal extinction, redshift, and Milky Way
        extinction.

        For Milky Way extinction, the Fitzpatrick, et al. (2019) (F19)
        model, as implemented in the dust_extinction package, is used.
        """
        wl, flambda, magnorm = self.skycat_obj.get_sed(component=component)
        if np.isinf(magnorm):
            # This subcomponent has zero emission so return None.
            return None

        # Create a galsim.SED for this subcomponent.
        sed_lut = galsim.LookupTable(wl, flambda)
        sed = galsim.SED(sed_lut, wave_type='nm', flux_type='flambda')
        sed = sed.withMagnitude(0, self._bp500)

        # Apply magnorm and multiply by the Rubin effective area so that
        # the SED is in units of photons/nm/s.
        flux_500 = np.exp(-0.9210340371976184 * magnorm)
        sed = sed*flux_500*self.eff_area

        iAv, iRv, mwAv, mwRv = self.get_dust()
        if iAv > 0:
            # Apply internal extinction model, which is assumed
            # to be the same for all subcomponents.
            pass  #TODO add implementation for internal extinction.

        if 'redshift' in self.skycat_obj.native_columns:
            redshift = self.skycat_obj.get_native_attribute('redshift')
            sed = sed.atRedshift(redshift)

        # Apply Milky Way extinction.
        extinction = F19(Rv=mwRv)
        # Use SED wavelengths
        wl = sed.wave_list
        # Restrict to the range where F19 can be evaluated. F19.x_range is
        # in units of 1/micron, so convert to nm.
        wl_min = 1e3/F19.x_range[1]
        wl_max = 1e3/F19.x_range[0]
        wl = wl[np.where((wl_min < wl) & (wl < wl_max))]
        ext = extinction.extinguish(wl*u.nm, Av=mwAv)
        spec = galsim.LookupTable(wl, ext)
        mw_ext = galsim.SED(spec, wave_type='nm', flux_type='1')
        sed = sed*mw_ext

        return sed

    def get_sed_components(self):
        """
        Return a dictionary of the SEDs, keyed by component name.
        """
        sed_components = {}
        subcomponents = [None] if not self.skycat_obj.subcomponents \
            else self.skycat_obj.subcomponents
        for component in subcomponents:
            sed_components[component] = self.get_sed_component(component)
        return sed_components

    def get_total_sed(self):
        """
        Return the SED summed over SEDs for the individual SkyCatalog
        components.
        """
        sed = None
        for sed_component in self.get_sed_components().values():
            if sed is None:
                sed = sed_component
            else:
                sed += sed_component
        if 'shear_1' in self.skycat_obj.native_columns:
            _, _, mu = self.get_wl_params()
            sed *= mu
        return sed

    def get_flux(self):
        """
        Return the total object flux over the bandpass in photons/sec.
        """
        sed = self.get_total_sed()
        return sed.calculateFlux(self.bandpass)

if __name__ == '__main__':
    skycat_file = '../tests/data/sky_cat_9683.yaml'
    skycat = skyCatalogs.open_catalog(skycat_file)
    objs = skycat.get_objects_by_hp(9683)

    band = 'i'
    for obj in objs[0:10]:
        foo = SkyCatalogObjectWrapper(obj, band)
        print(foo.get_flux())
