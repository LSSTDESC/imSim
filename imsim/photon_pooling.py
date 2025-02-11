import galsim
import numpy as np
import dataclasses
from galsim.config import RegisterImageType
from galsim.config.extra import RegisterExtraOutput
from galsim.config.extra_truth import TruthBuilder
from galsim.config.value import ParseValue, GetCurrentValue
from galsim.sensor import Sensor
from galsim.wcs import PixelScale
from galsim.errors import GalSimConfigValueError, GalSimConfigError
from galsim.utilities import basestring

from .stamp import ProcessingMode, ObjectInfo, build_obj
from .lsst_image import LSST_ImageBuilderBase


class LSST_PhotonPoolingImageBuilder(LSST_ImageBuilderBase):
    """Pools photon from all objects in `nbatch` batches.
    Photons from faint objects only appear in one of the batches randomly.
    """

    def setup(self, config, base, image_num, obj_num, ignore, logger):
        # Check we're using the correct stamp type before calling base setup method.
        if base['stamp']['type'] != 'LSST_Photons':
            raise GalSimConfigValueError("Must use stamp.type = LSST_Photons with LSST_PhotonPoolingImage.", base['stamp']['type'])
        return super().setup(config, base, image_num, obj_num, ignore, logger)

    def buildImage(self, config, base, image_num, _obj_num, logger):
        """Build the Image.

        In contrast to LSST_ImageBuilder, fluxes of all objects are precomputed
        before rendering to determine how each object is rendered (FFT / photon shooting).
        FFT objects will be handled before the photon shooting objects.
        Batching is done over objects for FFT objects and over photons for
        all photon shooting objects.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            logger:     A logger object to log progress.

        Returns:
            the final image and the current noise variance in the image as a tuple
        """
        self._set_config_image_pos(config, base)

        # For cases where there is noise in individual stamps, we need to keep track of the
        # stamp bounds and their current variances.  When checkpointing, we don't need to
        # save the pixel values for this, just the bounds and the current_var value of each.
        all_stamps = []
        all_vars = []
        all_obj_nums = []
        current_photon_batch_num = 0

        full_image = None

        if self.checkpoint is not None:
            chk_name = "buildImage_photonpooling_" + self.det_name
            full_image, all_vars, all_stamps, all_obj_nums, current_photon_batch_num = self.load_checkpoint(self.checkpoint, chk_name, base, logger)
        remaining_obj_nums = sorted(frozenset(range(self.nobjects)) - frozenset(all_obj_nums))

        if full_image is None:
            full_image = self._create_full_image(config, base)

        sensor = base.get('sensor', None)
        rng = galsim.config.GetRNG(config, base, logger, "LSST_Silicon")
        if sensor is not None:
            sensor.updateRNG(rng)

        # Create partitions each containing one of the three classes of object.
        fft_objects, phot_objects, faint_objects = self.partition_objects(self.load_objects(remaining_obj_nums, config, base, logger), self.nbatch)
        logger.info("Found %d FFT objects, %d photon shooting objects and %d faint objects", len(fft_objects), len(phot_objects), len(faint_objects))
        # Ensure 1 <= nbatch <= len(fft_objects)
        nbatch = max(min(self.nbatch_fft, len(fft_objects)), 1)
        if self.checkpoint is not None:
            if not fft_objects:
                logger.warning('All FFT objects already rendered for this image.')
            else:
                logger.warning("%d objects already rendered", len(all_obj_nums))

        # Handle FFT objects first:
        for batch_num, batch in enumerate(self.make_batches(fft_objects, nbatch), start=1):
            if nbatch > 1:
                logger.warning("Start FFT batch %d/%d with %d objects",
                               batch_num, nbatch, len(batch))
            stamps, current_vars = self.build_stamps(base, logger, batch)
            base['index_key'] = 'image_num'

            for stamp_obj, stamp in zip(batch, stamps):
                bounds = self.stamp_bounds(stamp, full_image.bounds)
                if bounds is None:
                    continue
                logger.debug('image %d: full bounds = %s', image_num, str(full_image.bounds))
                logger.debug('image %d: stamp %d bounds = %s',
                             image_num, stamp_obj.index, str(stamp.bounds))
                logger.debug('image %d: Overlap = %s', image_num, str(bounds))
                full_image[bounds] += stamp[bounds]
                all_obj_nums.append(stamp_obj.index)

            # Note: in typical imsim usage, all current_vars will be 0. So this normally doens't
            # add much to the checkpointing data.
            nz_var = np.nonzero(current_vars)[0]
            all_stamps.extend([stamps[k] for k in nz_var])
            all_vars.extend([current_vars[k] for k in nz_var])

            if self.checkpoint is not None:
                self.save_checkpoint(self.checkpoint, chk_name, base, full_image, all_stamps, all_vars, all_obj_nums, current_photon_batch_num)
                logger.warning('File %d: Completed batch %d, and wrote '
                               'checkpoint data to %s',
                               base.get('file_num', 0), batch_num,
                               self.checkpoint.file_name)

        # Ensure 1 <= nbatch <= len(phot_objects) and make batches.
        nbatch = max(min(self.nbatch, len(phot_objects)), 1)
        phot_batches = self.make_photon_batches(
                config, base, logger, phot_objects, faint_objects, nbatch
            )

        if current_photon_batch_num > 0:
            logger.warning(
                "Photon batches [0, %d) / %d already rendered - skipping",
                current_photon_batch_num,
                nbatch,
            )
            phot_batches = phot_batches[current_photon_batch_num:]

        base["image_pos"] = None
        base["stamp_center"] = None
        photon_ops_cfg = {"photon_ops": base.get("stamp", {}).get("photon_ops", [])}
        photon_ops = galsim.config.BuildPhotonOps(photon_ops_cfg, 'photon_ops', base, logger)
        local_wcs = base['wcs'].local(full_image.true_center)
        resume = False  # Initial call to accumulate will need to do some setup for SiliconSensors.
        if sensor is None:
            sensor = Sensor()
        # Create the image view here and reuse as we accumulate.
        imview = full_image._view()
        imview._shift(-full_image.center)  # equiv. to setCenter(), but faster
        imview.wcs = PixelScale(1.0)
        for batch_num, batch in enumerate(phot_batches, start=current_photon_batch_num):
            if not batch:
                continue
            if nbatch > 1:
                logger.warning("Starting photon batch %d/%d.",
                               batch_num+1, nbatch)

            base['index_key'] = 'image_num'
            stamps, current_vars = self.build_stamps(base, logger, batch)
            photons = self.merge_photon_arrays(stamps)
            for op in photon_ops:
                op.applyTo(photons, local_wcs, rng)
            # Shift photon positions to be relative to full_image.center.
            # This is necessary as all photons will now be in the pool together, and there
            # is no longer any way to distinguish which belong to which object.
            photons.x -= full_image.center.x
            photons.y -= full_image.center.y
            self.accumulate_photons(photons, imview, sensor, resume=resume)
            # Later iterations can skip any setup in sensor accumulation.
            resume = True

            # Note: in typical imsim usage, all current_vars will be 0. So this normally doesn't
            # add much to the checkpointing data.
            nz_var = np.nonzero(current_vars)[0]
            all_vars.extend([current_vars[k] for k in nz_var])

            if self.checkpoint is not None:
                self.save_checkpoint(self.checkpoint, chk_name, base, full_image, all_stamps, all_vars, all_obj_nums, batch_num+1)

        # Bring the image so far up to a flat noise variance
        current_var = galsim.config.FlattenNoiseVariance(
                base, full_image, all_stamps, tuple(all_vars), logger)

        return full_image, current_var

    @staticmethod
    def merge_photon_arrays(stamps):
        """Given a list of stamps of photon objects, return a single merged photon array.

        Parameters:
            stamps: list of galsim.PhotonArrays to be merged into one.

        Returns:
            merged: A single PhotonArray containing all the photons.
        """
        n_tot = sum(len(stamp.photons) for stamp in stamps)
        merged = galsim.PhotonArray(n_tot)
        start = 0
        for stamp in stamps:
            merged.copyFrom(stamp.photons, slice(start, start+stamp.photons.size()))
            start += len(stamp.photons)
        return merged

    @staticmethod
    def accumulate_photons(photons, imview, sensor, resume=False):
        """Accumulate a photon array onto a sensor.

        Parameters:
            photons: A PhotonArray containing the photons to be accumulated.
            imview: The image view to which we draw the accumulated photons.
            sensor: Sensor to use for accumulation. If None, a temporary sensor is created here.
            resume: Resume accumulating following an earlier call for some extra performance. Default False.
        """
        if imview.dtype in (np.float32, np.float64):
            sensor.accumulate(photons, imview, imview.center, resume=resume)
        else:
            # Need a temporary
            im1 = galsim.image.ImageD(bounds=imview.bounds)
            sensor.accumulate(photons, im1, imview.center)
            imview += im1

    @staticmethod
    def make_batches(objects, nbatch: int):
        """Generator converting an input list of objects to batches.

        Parameters:
            objects: List of objects to be yielded in batches.
            nbatch: The number of batches to create.

        Yields:
            A single batch made up of a list of object numbers.
        """
        base_per_batch = len(objects) // nbatch
        per_batch_remainder = len(objects) % nbatch
        o_iter = iter(objects)
        for i in range(nbatch):
            # Add extra objects to early batches if per_batch_remainder > 0.
            if i < per_batch_remainder:
                nobj_per_batch = base_per_batch + 1
            else:
                nobj_per_batch = base_per_batch
            yield [obj for _, obj in zip(range(nobj_per_batch), o_iter)]

    @staticmethod
    def build_stamps(base, logger, objects: list[ObjectInfo]):
        """Create stamps for a list of ObjectCaches.

        Parameters:
            base: The base configuration dictionary.
            logger: Logger object.
            objects: List of ObjectCaches for which we will build stamps.

        Returns:
            images: Tuple of the output stamps. In normal PhotonPooling usage these will actually be
                PhotonArrays to be processed into images after they have been pooled following this step.
            current_vars: Tuple of variance for each stamp (noise etc).
        """
        if not objects:
            return [], []
        base["_objects"] = {obj.index: obj for obj in objects}

        images, current_vars = zip(
            *(
                galsim.config.BuildStamp(
                    base, obj.index, xsize=0, ysize=0, do_noise=False, logger=logger
                )
                for obj in objects
            )
        )
        return images, current_vars

    @staticmethod
    def make_photon_batches(config, base, logger, phot_objects: list[ObjectInfo], faint_objects: list[ObjectInfo], nbatch: int):
        """Create a set of nbatch batches of photon objects.
        The bright objects in phot_objects are replicated across all batches but at 1/nbatch their original
        flux, while the faint objects in faint_objects are randomly placed in batches at full flux.

        Parameters:
            config: The configuration dictionary for the image field.
            base: The base configuration dictionary.
            logger: Logger to record progress.
            phot_objects: The list of ObjectCaches representing the bright photon objects to be batched.
            faint_objects: The list of ObjectCaches representing faint photon objects to be batched.
            nbatch: The integer number of photon batches to create.

        Returns:
            batches: A list of batches, each itself a list of the objects to be drawn.
        """
        if not phot_objects and not faint_objects:
            return []

        # Each batch is a copy of the original list of photon objects at
        # 1/nbatch the original flux. Total flux will conserve the original.
        batches = [
            [dataclasses.replace(obj,
                                 phot_flux=(obj.phot_flux*(i+1))//nbatch - (obj.phot_flux*i)//nbatch
                                 ) for obj in phot_objects]
            for i in range(nbatch)]

        # Shuffle whole faint objects into random batches.
        rng = galsim.config.GetRNG(config, base, logger, "LSST_Silicon")
        ud = galsim.UniformDeviate(rng)
        for obj in faint_objects:
            batch_index = int(ud() * nbatch)
            batches[batch_index].append(obj)

        return batches

    @staticmethod
    def stamp_bounds(stamp, full_image_bounds):
        """Check bounds overlap between an object's stamp and the full image.

        Parameters:
            stamp: An object's stamp, potentially None.
            full_image_bounds: The full image's galsim.BoundsI.
        Returns: 
            bounds: The overlapping bounds
            or None if the stamp and image do not overlap
            or None if the object was not drawn (i.e. does not have a stamp).
        """
        if stamp is None:
            return None
        bounds = stamp.bounds & full_image_bounds
        if not bounds.isDefined():  # pragma: no cover
            # These normally show up as stamp==None, but technically it is possible
            # to get a stamp that is off the main image, so check for that here to
            # avoid an error.  But this isn't covered in the imsim test suite.
            return None
        return bounds

    @staticmethod
    def partition_objects(objects, nbatch):
        """Given a list of objects, return three lists containing only the objects to
        be processed as FFT, photon or faint objects.

        Parameters:
            objects: a list of ObjectCaches
            nbatch: the number of batches. The actual value used may differ, but crucially will not be greater than this value.

        Returns:
            A tuple of three lists respectively containing the objects to be processed
            with FFTs, photons and as faint photon objects.
        """
        objects_by_mode = {
            ProcessingMode.FFT: [],
            ProcessingMode.PHOT: [],
            ProcessingMode.FAINT: [],
        }
        for obj in objects:
            if obj.phot_flux < nbatch and obj.mode == ProcessingMode.PHOT:
                # In the special case of PHOT objects with fewer photons than there are batches,
                # batch them alongside FAINT objects - i.e. draw in a single batch rather than pool.
                mode = ProcessingMode.FAINT
            else:
                # Other objects dealt with as their mode indicates.
                mode = obj.mode
            objects_by_mode[mode].append(obj)
        return (
            objects_by_mode[ProcessingMode.FFT],
            objects_by_mode[ProcessingMode.PHOT],
            objects_by_mode[ProcessingMode.FAINT],
        )

    @staticmethod
    def load_objects(obj_numbers, config, base, logger):
        """Convert the objects in the base configuration to ObjectCaches. Their
        fluxes are calculated at this stage and then stored in the ObjectCaches
        for reuse later on.

        Parameters:
            obj_numbers: a list of the object numbers in the config that are to be drawn.
            config: The configuration dictionary for the image field.
            base: The base configuration dictionary.
            logger: A Logger object to track progress.
        
        Yields:
            obj: An ObjectCacheobject containing the obj ID and flux.
        """
        gsparams = {}
        stamp = base['stamp']
        if 'gsparams' in stamp:
            gsparams = galsim.gsobject.UpdateGSParams(gsparams, stamp['gsparams'], config)

        for obj_num in obj_numbers:
            galsim.config.SetupConfigObjNum(base, obj_num, logger)
            obj = build_obj(stamp, base, logger)
            if obj is not None:
                yield obj

    @staticmethod
    def load_checkpoint(checkpoint, chk_name, base, logger):
        """Load a checkpoint from file.

        Parameters:
            checkpoint: A Checkpointer object.
            chk_name: The checkpoint record's name.
            base: The base configuration dictionary.
            logger: A Logger to provide information.

        Returns:
            full_image: The full image as saved to checkpoint, or None.
            all_vars: List of variables e.g. noise levels, or [].
            all_stamps: List of stamps created as of the time of the checkpoint, or [].
            all_obj_nums: List of object IDs drawn as of the time of the checkpoint, or [].
            current_photon_batch_num: The photon batch from which to start working, or 0.
        """
        saved = checkpoint.load(chk_name)
        if saved is not None:
            # If the checkpoint exists, get the stored information and prepare it for use.
            full_image, all_bounds, all_vars, all_obj_nums, extra_builder, current_photon_batch_num = saved
            if extra_builder is not None:
                base['extra_builder'] = extra_builder
            # Create stamps from the bounds provided by the checkpoint.
            all_stamps = [galsim._Image(np.array([]), b, full_image.wcs) for b in all_bounds]
            logger.warning('File %d: Loaded checkpoint data from %s.',
                        base.get('file_num', 0), checkpoint.file_name)
            return full_image, all_vars, all_stamps, all_obj_nums, current_photon_batch_num
        else:
            # Return empty objects if the checkpoint doesn't yet exist.
            return None, [], [], [], 0

    @staticmethod
    def save_checkpoint(checkpoint, chk_name, base, full_image, all_stamps, all_vars, all_obj_nums, current_photon_batch_num):
        """Save a checkpoint to file.

        Parameters:
            checkpoint: A Checkpointer object.
            chk_name: The record name with which to save the checkpoint.
            base: The base configuration dictionary.
            full_image: The current state of the GalSim image containing the full field.
            all_stamps: List of the stamps drawn so far -- note that only their bounds are saved.
            all_vars: List of variables e.g. noise levels.
            all_obj_nums: List of the objects which have been drawn so far. 
            current_photon_batch_num: The photon batch number from which drawing should begin
                if this checkpoint is loaded.
        """
        # Don't save the full stamps.  All we need for FlattenNoiseVariance is the bounds.
        # Everything else about the stamps has already been handled above.
        all_bounds = [stamp.bounds for stamp in all_stamps]
        data = (full_image, all_bounds, all_vars, all_obj_nums,
                base.get('extra_builder',None), current_photon_batch_num)
        checkpoint.save(chk_name, data)


RegisterImageType('LSST_PhotonPoolingImage', LSST_PhotonPoolingImageBuilder())


class PhotonPoolingTruthBuilder(TruthBuilder):
    """Build an output truth catalog with user-defined columns, typically taken
    from current values of various quantities for each constructed object.
    This is for use with the photon pooling code which provides the
    incident_flux values. Most of this is inherited from the GalSim
    TruthBuilder, but this takes into account the many times that a photon
    object will have its stamp drawn. The first time, the full row is created.
    Later calls only update the incident_flux value. create each object's across
    the many calls made for it and update only incident_flux.
    """
    # The function to call at the end of building each stamp
    def processStamp(self, obj_num, config, base, logger):
        cols = config['columns']
        row = []
        types = []
        # Most of the row information can be created on the first call for each
        # object and then left alone. Only the incident_flux, if it's to be
        # written, should be updated on subsequent calls. Determine this by
        # checking if obj_num is already in scratch -- it will only be absent on
        # the first call for this object.
        if obj_num not in self.scratch:
            for i, name in enumerate(cols):
                key = cols[name]
                if isinstance(key, dict):
                    # Then the "key" is actually something to be parsed in the normal way.
                    # Caveat: We don't know the value_type here, so we give None.  This allows
                    # only a limited subset of the parsing.  Usually enough for truth items, but
                    # not fully featured.
                    value = ParseValue(cols,name,base,None)[0]
                elif not isinstance(key,basestring):
                    # The item can just be a constant value.
                    value = key
                elif key[0] == '$':
                    # This can also be handled by ParseValue
                    value = ParseValue(cols,name,base,None)[0]
                elif key[0] == '@':
                    # Pop off an initial @ if there is one.
                    value = GetCurrentValue(str(key[1:]), base)
                else:
                    # str(key) handles the possibility of unicode.  In particular, this happens with
                    # JSON files.
                    value = GetCurrentValue(str(key), base)
                row.append(value)
                types.append(self._type(value))
                if name == 'incident_flux' and 'incident_flux_index' not in self.scratch:
                    # Remember which column the incident_flux is added to.
                    self.scratch['incident_flux_index'] = i
            if 'types' not in self.scratch:
                self.scratch['types'] = types
            elif self.scratch['types'] != types:
                logger.error("Type mismatch found when building truth catalog at object %d",
                             base['obj_num'])
                for name, t1, t2 in zip(cols, types, self.scratch['types']):
                    if t1 != t2:
                        logger.error("%s has type %s, but previously had type %s"%(
                            name,t1.__name__,t2.__name__))
                raise GalSimConfigError("Type mismatch found when building truth catalog.")
            self.scratch[obj_num] = row
        else:
            # It's a later call for this object, so we already know
            # incident_flux_index. Update it within the row. FFT batches process
            # each object once anyway so don't need any special handling.
            self.scratch[obj_num][self.scratch['incident_flux_index']] += base['incident_flux']

    def finalize(self, config, base, main_data, logger):
        # Remove the incident_flux_index from scratch, if this is being output, then
        # carry on with the original truth output's finalize method.
        self.scratch.pop('incident_flux_index', None)
        return super().finalize(config, base, main_data, logger)


# Register photon_pooling_truth as a valid extra output
RegisterExtraOutput('photon_pooling_truth', PhotonPoolingTruthBuilder())
