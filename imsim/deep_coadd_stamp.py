import numpy as np
import galsim
from galsim.config import StampBuilder, RegisterStampType
from .stamp_utils import get_stamp_size


__all__ = ["RubinDeepCoaddStampBuilder"]


class RubinDeepCoaddStampBuilder(StampBuilder):
    _pixel_scale = 0.2
    _tiny_flux = 10.0
    _Nmax = 4096

    def setup(self, config, base, xsize, ysize, ignore, logger):
        # Use base class setup to find default xsize, ysize (and object
        # position).
        xsize, ysize, image_pos, world_pos \
            = super().setup(config, base, xsize, ysize, ignore, logger)

        obj = galsim.config.BuildGSObject(base, 'gal', logger=logger)[0]
        if obj is None:
            raise galsim.config.SkipThisObject(
                'gal is None (invalid parameters)')
        self.obj = obj

        self.rng = galsim.config.GetRNG(config, base, logger, "RubinDeepCoadd")
        self.image = base['current_image']
        bandpass = base['bandpass']
        if not hasattr(obj, 'flux'):
            obj.flux = obj.calculateFlux(bandpass)
        self.nominal_flux = obj.flux
        self.phot_flux = galsim.PoissonDeviate(self.rng, mean=obj.flux)()

        # Save values for output truth catalog
        base['nominal_flux'] = self.nominal_flux
        base['phot_flux'] = self.phot_flux
        base['realized_flux'] = 0  # This will be updated by .drawImage

        if self.phot_flux == 0:
            raise galsim.config.SkipThisObject('phot_flux==0')

        if xsize > 0 and ysize > 0:
            # Use the stamp size from the config.
            pass
        elif 'size' in config:
            # Get the stamp size from the size config entry
            xsize = ysize = galsim.config.ParseValue(
                config, 'size', base, int)[0]
        elif self.nominal_flux < self._tiny_flux:
            xsize = ysize = 32
        else:
            # Determine the stamp size from the object flux.
            base['current_noise_image'] = base['current_image']
            noise_var = galsim.config.CalculateNoiseVariance(base)
            obj_achrom = obj.evaluateAtWavelength(bandpass.effective_wavelength)
            stamp_size = get_stamp_size(
                obj_achrom=obj_achrom,
                nominal_flux=self.nominal_flux,
                noise_var=noise_var,
                Nmax=self._Nmax,
                pixel_scale=self._pixel_scale,
                logger=logger
            )
            xsize = ysize = stamp_size

        logger.info('Object %d will use stamp size = %s, %s and nominal flux %s',
                    base.get('obj_num',0), xsize, ysize, self.nominal_flux)
        return xsize, ysize, image_pos, world_pos

    def draw(self, prof, image, method, offset, config, base, logger):
        if prof is None:
            image.photons = galsim.PhotonArray(0)
            return image

        # Prof is normally a convolution here with obj_list being
        # [gal, psf1, psf2,...] for some number of component PSFs.
        gal, *psfs = prof.obj_list if hasattr(prof, 'obj_list') else [prof]
        obj_num = base.get('obj_num',0)
        bandpass = base['bandpass']

        max_flux_simple = config.get('max_flux_simple', 100)
        faint = self.nominal_flux < max_flux_simple

        if faint:
            logger.info("Flux = %.0f  Using trivial sed", self.obj.flux)
            for profile_wl in (bandpass.effective_wavelength,
                               bandpass.red_limit,
                               bandpass.blue_limit):
                sed_value = gal.sed(profile_wl)
                if sed_value != 0:
                    break
            if sed_value == 0:
                # We can't evaluate the profile for this object, so skip it.
                obj_num = base.get('obj_num')
                object_id = base.get('object_id')
                logger.warning("Zero-valued SED for faint object %d, "
                               "object_id %s.  Skipping.", obj_num, object_id)
                return image
            gal = gal.evaluateAtWavelength(profile_wl)
            gal = gal * self._trivial_sed
        else:
            self._fix_seds(gal, bandpass, logger)

        image.wcs = base['wcs']

        gal = gal.withFlux(self.phot_flux, bandpass)

        if not faint and 'photon_ops' in config:
            photon_ops = galsim.config.BuildPhotonOps(
                config, 'photon_ops', base, logger)
        else:
            photon_ops = []

        # Put the psfs at the start of the photon_ops.
        # Probably a little better to put them a bit later than the
        # start in some cases (e.g. after TimeSampler,
        # PupilAnnulusSampler), but leave that as a todo for now.
        photon_ops = psfs + photon_ops
        sensor = None
        image = gal.drawImage(bandpass,
                              method='phot',
                              offset=offset,
                              rng=self.rng,
                              n_photons=self.phot_flux,
                              image=image,
                              sensor=None,
                              photon_ops=photon_ops,
                              add_to_image=True,
                              poisson_flux=False)
        base['realized_flux'] = image.added_flux
        logger.debug('After .drawImage(...), image.added_flux %s', image.added_flux)
        return image

    @classmethod
    def _fix_seds(cls, prof, bandpass, logger):
        # If any SEDs are not currently using a LookupTable for the
        # function or if they are using spline interpolation, then the
        # codepath is quite slow.  Better to fix them before doing
        # WavelengthSampler.
        if (isinstance(prof, galsim.SimpleChromaticTransformation) and
            (not isinstance(prof._flux_ratio._spec, galsim.LookupTable)
             or prof._flux_ratio._spec.interpolant != 'linear')):
            if not cls._sed_logged:
                logger.warning(
                    "Warning: Chromatic drawing is most efficient when SEDs have "
                    "interpolant='linear'. Switching LookupTables to use 'linear'."
                )
                cls._sed_logged = True
            sed = prof._flux_ratio
            wave_list, _, _ = galsim.utilities.combine_wave_list(sed, bandpass)
            f = np.broadcast_to(sed(wave_list), wave_list.shape)
            new_spec = galsim.LookupTable(wave_list, f, interpolant='linear')
            new_sed = galsim.SED(
                new_spec,
                'nm',
                'fphotons' if sed.spectral else '1'
            )
            prof._flux_ratio = new_sed

        # Also recurse onto any components.
        if isinstance(prof, galsim.ChromaticObject):
            if hasattr(prof, 'obj_list'):
                for obj in prof.obj_list:
                    cls._fix_seds(obj, bandpass, logger)
            if hasattr(prof, 'original'):
                cls._fix_seds(prof.original, bandpass, logger)


RegisterStampType('RubinDeepCoadd', RubinDeepCoaddStampBuilder())
