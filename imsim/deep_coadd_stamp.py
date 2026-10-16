import astropy.units as u
import galsim
from galsim.config import StampBuilder, RegisterStampType
from .stamp import LSST_SiliconBuilder
from .stamp_utils import get_stamp_size


__all__ = ["RubinDeepCoaddStampBuilder"]


class DeepCoaddWeighting(galsim.PhotonOp):
    """
    Weight photons so that final object fluxes are in nJy.
    """
    def __init__(self, mag, nphotons):
        """
        Parameters
        ----------
        mag: float
            The bandpass magnitude for the object being rendered.
        nphotons: float
            The number of photons to simulate.  This should be
            the total number over all visits contributing to the
            object's counts. Nominally, this is given by
            nphotons = photon_flux * num_visits * exptime_per_visit * pupil_area
        """
        # Convert magnitude to nJy:
        fnu = 1e9 * (mag*u.ABmag).to(u.Jy).to_value()
        self.weight = fnu / nphotons

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        photon_array.flux *= self.weight


class RubinDeepCoaddStampBuilder(LSST_SiliconBuilder):

    def setup(self, config, base, xsize, ysize, ignore, logger):
        # Always disable fft rendering:
        self.use_fft = False

        # Use StampBuilder.setup(...) to set the object position.
        ignore = ignore + ['max_flux_simple', 'band', 'pupil_area', 'exptime']
        _, _, image_pos, world_pos = StampBuilder.setup(
            self, config, base, xsize, ysize, ignore, logger)

        obj = galsim.config.BuildGSObject(base, 'gal', logger=logger)[0]
        if obj is None:
            raise galsim.config.SkipThisObject('obj is None')
        self.obj = obj

        self.rng = galsim.config.GetRNG(config, base, logger, "RubinDeepCoadd")
        self.image = base['current_image']

        deep_coadds = galsim.config.GetInputObj('deep_coadds', config, base,
                                                'RubinDeepCoaddStampBuilder')
        band = galsim.config.ParseValue(config, 'band', base, str)[0]
        num_visits = deep_coadds.getNumVisits(world_pos, band)

        bandpass = base['bandpass']
        if not hasattr(obj, 'flux'):
            obj.flux = obj.calculateFlux(bandpass)

        # To get the AB magnitude, we need to divide out the pupil
        # area and single visit exposure time from the object SED as
        # provided from skyCatalogs.
        pupil_area = galsim.config.ParseValue(config, 'pupil_area', base, float)[0]
        exptime = galsim.config.ParseValue(config, 'exptime', base, float)[0]
        sed = obj.sed / pupil_area / exptime
        mag = sed.calculateMagnitude(bandpass)

        # Scale the single-visit object flux by the number of visits
        # in order to get the correct signal variance.  The deep coadd
        # weighting ensures that the final flux is in units of nJy.
        self.simulated_flux = obj.flux * num_visits
        self.deep_coadd_phot_op = DeepCoaddWeighting(mag, self.simulated_flux)
        self.phot_flux = galsim.PoissonDeviate(self.rng, mean=self.simulated_flux)()

        # Save values for output truth catalog
        base['simulated_flux'] = self.simulated_flux
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
        elif self.simulated_flux < self._tiny_flux:
            xsize = ysize = 32
        else:
            # Determine the stamp size from the object flux.
            noise_var = galsim.config.CalculateNoiseVariance(base)
            obj_achrom = obj.evaluateAtWavelength(bandpass.effective_wavelength)
            stamp_size = get_stamp_size(
                obj_achrom=obj_achrom,
                nominal_flux=self.simulated_flux,
                noise_var=noise_var,
                Nmax=self._Nmax,
                pixel_scale=self._pixel_scale,
                logger=logger
            )
            xsize = ysize = stamp_size

        base['stamp_size'] = xsize
        logger.info('Object %d will use stamp size %s, %s and simulated flux %s',
                    base.get('obj_num',0), xsize, ysize, self.simulated_flux)
        return xsize, ysize, image_pos, world_pos

    def buildPSF(self, config, base, gsparams, logger):
        psf = galsim.config.BuildGSObject(
            base, 'psf', gsparams=gsparams, logger=logger)[0]
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
        faint = self.simulated_flux < max_flux_simple

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

        photon_ops = [self.deep_coadd_phot_op]

        # Set limit on the size of photons batches to consider when
        # calling gsobject.drawImage.
        maxN = int(1e6)
        if 'maxN' in config:
            maxN = galsim.config.ParseValue(config, 'maxN', base, int)[0]

        gal = gal.withFlux(self.phot_flux, bandpass)
        if not faint and 'photon_ops' in config:
            photon_ops = photon_ops + galsim.config.BuildPhotonOps(
                config, 'photon_ops', base, logger)

        # Put the psfs at the start of the photon_ops.
        # Probably a little better to put them a bit later than the
        # start in some cases (e.g. after TimeSampler,
        # PupilAnnulusSampler), but leave that as a todo for now.
        photon_ops = psfs + photon_ops
        image = gal.drawImage(bandpass,
                              method='phot',
                              offset=offset,
                              rng=self.rng,
                              maxN=maxN,
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


RegisterStampType('RubinDeepCoaddStamp', RubinDeepCoaddStampBuilder())
