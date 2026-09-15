import galsim
from galsim.config import StampBuilder, RegisterStampType
from .stamp import LSST_SiliconBuilder
from .stamp_utils import get_stamp_size


__all__ = ["RubinDeepCoaddStampBuilder"]


class RubinDeepCoaddStampBuilder(LSST_SiliconBuilder):

    def setup(self, config, base, xsize, ysize, ignore, logger):
        # Use StampBuilder.setup(...) to set the object position.
        ignore = ignore + ['fft_sb_thresh', 'max_flux_simple']
        _, _, image_pos, world_pos = StampBuilder.setup(
            self, config, base, xsize, ysize, ignore, logger)

        obj = galsim.config.BuildGSObject(base, 'gal', logger=logger)[0]
        if obj is None:
            raise galsim.config.SkipThisObject('obj is None')
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

        logger.info('Object %d will use stamp size %s, %s and nominal flux %s',
                    base.get('obj_num',0), xsize, ysize, self.nominal_flux)
        return xsize, ysize, image_pos, world_pos

    def buildPSF(self, config, base, gsparams, logger):
        psf = galsim.config.BuildGSObject(
            base, 'psf', gsparams=gsparams, logger=logger)[0]

        # Check if fft rendering should be used.
        if 'fft_sb_thresh' in config:
            fft_sb_thresh = galsim.config.ParseValue(
                config, 'fft_sb_thresh', base, float)[0]
        else:
            fft_sb_thresh = None

        self.use_fft = (fft_sb_thresh is not None
                        and self.nominal_flux > fft_sb_thresh)
        return psf

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

        if method == 'fft':
            gal = gal.withFlux(self.nominal_flux, bandpass)
            fft_image = image.copy()
            fft_offset = offset
            kwargs = dict(
                method='fft',
                offset=fft_offset,
                image=fft_image
            )
            if not faint and config.get('fft_photon_ops'):
                fft_photon_ops = galsim.config.BuildPhotonOps(
                    config, 'fft_photon_ops', base, logger)
                kwargs.update({
                    "photon_ops": fft_photon_ops,
                    "rng": self.rng,
                    "n_subsample": 1,
                })

            # Go back to a combined convolution for fft drawing.
            prof = galsim.Convolve([gal] + psfs)
            try:
                fft_image = prof.drawImage(bandpass, **kwargs)
            except galsim.errors.GalSimFFTSizeError as e:
                # I think this shouldn't happen with the updates I
                # made to how the image size is calculated, even for
                # extremely bright things.  So it should be ok to just
                # report what happened, give some extra information to
                # diagonose the problem and raise the error.
                logger.error('Caught error trying to draw using FFT:')
                logger.error('%s', e)
                logger.error('You may need to add a gsparams field with '
                             'maximum_fft_size to either')
                logger.error('the psf or gal field to allow larger FFTs.')
                logger.info('prof = %r', prof)
                logger.info('fft_image = %s', fft_image)
                logger.info('offset = %r', offset)
                raise
            # Some pixels can end up negative from FFT numerics.  Just
            # set them to 0.
            fft_image.array[fft_image.array < 0] = 0.
            fft_image.addNoise(galsim.PoissonNoise(rng=self.rng))
            # In case we had to make a bigger image, just copy the
            # part we need.
            image += fft_image[image.bounds]
            base['realized_flux'] = fft_image.added_flux
            logger.debug('After .drawImage(...), fft_image.added_flux %s',
                         fft_image.added_flux)
        else:
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
            logger.debug('After .drawImage(...), image.added_flux %s',
                         image.added_flux)

        return image


RegisterStampType('RubinDeepCoadd', RubinDeepCoaddStampBuilder())
