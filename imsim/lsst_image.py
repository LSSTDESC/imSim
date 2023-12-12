import logging
import dataclasses
import numpy as np
import hashlib
import galsim
from galsim.config import RegisterImageType, GetAllParams, GetSky, AddNoise
from galsim.config.image_scattered import ScatteredImageBuilder
from galsim.wcs import PixelScale
from galsim.sensor import Sensor

from .sky_model import SkyGradient, CCD_Fringing
from .camera import get_camera
from .vignetting import Vignetting
from .stamp import StellarObject, ProcessingMode, build_obj

def merge_photon_arrays(arrays):
    n_tot = sum(len(arr) for arr in arrays)
    merged = galsim.PhotonArray(n_tot)
    start = 0
    for arr in arrays:
        merged.assignAt(start, arr)
        start += len(arr)
    return merged


class LSST_ImageBuilderBase(ScatteredImageBuilder):
    def setup(self, config, base, image_num, obj_num, ignore, logger):
        """Do the initialization and setup for building the image.

        This figures out the size that the image will be, but doesn't actually build it yet.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here. i.e. it won't be an error if these parameters are present.
            logger:     A logger object to log progress.

        Returns:
            xsize, ysize
        """
        # This is mostly a copy of the ScatteredImageType.setup function.
        # The new bits are setting up vignetting and checkpointing.

        logger.debug('image %d: Building LSST_Image: image, obj = %d,%d',
                     image_num,image_num,obj_num)

        self.nobjects = self.getNObj(config, base, image_num, logger=logger)
        if 'nobjects' in config:
            # User specified nobjects.
            # Make sure it's not more than what any input catalog can
            # handle (we don't want repeated objects).
            input_nobj = galsim.config.ProcessInputNObjects(base)
            if input_nobj is not None:
                self.nobjects = min(self.nobjects, input_nobj)
        logger.info('image %d: nobj = %d', image_num, self.nobjects)

        # These are allowed for LSST_Image, but we don't use them here.
        extra_ignore = [ 'image_pos', 'world_pos', 'stamp_size', 'stamp_xsize', 'stamp_ysize',
                         'nobjects' ]
        req = { 'det_name': str }
        opt = { 'size': int , 'xsize': int , 'ysize': int, 'dtype': None,
                 'apply_sky_gradient': bool, 'apply_fringing': bool,
                 'boresight': galsim.CelestialCoord, 'camera': str, 'nbatch': int,
                 'nbatch_per_checkpoint': int}
        params = GetAllParams(config, base, req=req, opt=opt, ignore=ignore+extra_ignore)[0]

        # Let the user override the image size
        size = params.get('size',0)
        xsize = params.get('xsize',size)
        ysize = params.get('ysize',size)

        self.det_name = params['det_name']
        self.camera_name = params.get('camera', 'LsstCam')

        # If not overridden, then get size from the camera.
        camera = get_camera(self.camera_name)
        if xsize == 0 or ysize == 0:
            # Get detector size in pixels.
            det_bbox = camera[self.det_name].getBBox()
            xsize = det_bbox.width
            ysize = det_bbox.height
            base['det_xsize'] = xsize
            base['det_ysize'] = ysize

        try:
            self.vignetting = galsim.config.GetInputObj('vignetting', config,
                                                        base, 'LSST_Image')
        except galsim.config.GalSimConfigError:
            self.vignetting = None

        self.apply_sky_gradient = params.get('apply_sky_gradient', False)
        self.apply_fringing = params.get('apply_fringing', False)
        if self.apply_fringing:
            if 'boresight' not in params:
                raise galsim.config.GalSimConfigError(
                    "Boresight is missing in image config dict. This is required for fringing.")
            else:
                self.boresight = params.get('boresight')

        self.camera_name = params.get('camera', 'LsstCam')

        # Batching is also useful for memory reasons, to limit the number of stamps held
        # in memory before adding them all to the main image.  So even if not checkpointing,
        # keep the default value as 100.
        try:
            self.checkpoint = galsim.config.GetInputObj('checkpoint', config, base, 'LSST_Image')
            self.nbatch = params.get('nbatch', 100)
            self.nbatch_per_checkpoint = params.get('nbatch_per_checkpoint', 1)
        except galsim.config.GalSimConfigError:
            self.checkpoint = None

        return xsize, ysize

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        """Add the final noise and sky.

        Parameters:
            image:          The image onto which to add the noise.
            config:         The configuration dict for the image field.
            base:           The base configuration dict.
            image_num:      The current image number.
            obj_num:        The first object number in the image.
            current_var:    The current noise variance in each postage stamps.
            logger:         A logger object to log progress.
        """
        base['current_noise_image'] = base['current_image']

        if logger.isEnabledFor(logging.INFO):
            skyCoord = base['world_center']
            # Normally skyCoord is a CelestialCoord, but if not make sure this doesn't bork.
            try:
                ra, dec = skyCoord.ra.deg, skyCoord.dec.deg
            except AttributeError:
                ra, dec = 0,0
            sky1 = GetSky(config, base, full=False)
            logger.info("Setting sky level to %.2f photons/arcsec^2 "
                        "at (ra, dec) = %s, %s", sky1, ra, dec)

        sky = GetSky(config, base, full=True)

        if ((self.apply_sky_gradient or self.apply_fringing or self.vignetting is not None)
            and not isinstance(sky, galsim.Image)):
            # Handle the case where a full image isn't returned by
            # GetSky, i.e., when the sky level is constant and the wcs
            # is uniform.
            sky = galsim.Image(bounds=image.bounds, wcs=image.wcs, init_value=sky)

        if self.apply_sky_gradient:
            ny, nx = sky.array.shape
            sky_model = galsim.config.GetInputObj('sky_model', config, base,
                                                  'LSST_ImageBuilder')
            sky_gradient = SkyGradient(sky_model, image.wcs,
                                       base['world_center'], nx)
            logger.info("Applying sky gradient = %s", sky_gradient)
            xarr, yarr = np.meshgrid(range(nx), range(ny))
            sky.array[:] *= sky_gradient(xarr, yarr)

        if self.vignetting is not None:
            camera = get_camera(self.camera_name)
            logger.info("Applying vignetting according to radial spline model.")
            radii = Vignetting.get_pixel_radii(camera[self.det_name])
            sky.array[:] *= self.vignetting.apply_to_radii(radii)

        if self.apply_fringing:
            # Use the hash value of the serial number as random seed number to
            # make sure the height map of the same sensor remains unchanged for different exposures.
            camera = get_camera(self.camera_name)
            det_name = base['det_name']
            serial_number = camera[det_name].getSerial()
            # Note: the regular Python hash function is non-deterministic, which is not good.
            # Instead we use hashlib.sha256, which is deterministic and convert that to an integer.
            # https://stackoverflow.com/questions/27954892/deterministic-hashing-in-python-3
            seed = int(hashlib.sha256(serial_number.encode('UTF-8')).hexdigest(), 16) & 0xFFFFFFFF
            # Only apply fringing to e2v sensors.
            if serial_number[:3] == 'E2V':
                ccd_fringing = CCD_Fringing(true_center=image.wcs.toWorld(image.true_center),
                                            boresight=self.boresight,
                                            seed=seed, spatial_vary=True)
                ny, nx = sky.array.shape
                xarr, yarr = np.meshgrid(range(nx), range(ny))
                logger.info("Apply fringing")
                fringing_map = ccd_fringing.calculate_fringe_amplitude(xarr,yarr)
                sky.array[:] *= fringing_map

        image += sky
        AddNoise(base,image,current_var,logger)


class LSST_ImageBuilder(LSST_ImageBuilderBase):
    """This is mostly the same as the GalSim "Scattered" image type.
    So far the only change is in the sky background image."""

    def buildImage(self, config, base, image_num, obj_num, logger):
        """Build the Image.

        This is largely the same as the GalSim Scattered image type.
        The main difference is that we add checkpointing capabilities to occasionally
        write out the image so far, so if interrupted, the process can be restarted
        from the last checkpoint.  This feature requires an input.checkpoint object.
        The number of batches is controlled with the nbatch option of LSST_Image.
        (Default when checkpointing is 10 batches.)

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            logger:     A logger object to log progress.

        Returns:
            the final image and the current noise variance in the image as a tuple
        """
        set_config_image_pos(config, base)

        full_image = None
        start_num = obj_num

        # For cases where there is noise in individual stamps, we need to keep track of the
        # stamp bounds and their current variances.  When checkpointing, we don't need to
        # save the pixel values for this, just the bounds and the current_var value of each.
        all_stamps = []
        all_vars = []

        if self.checkpoint is not None:
            chk_name = 'buildImage_%s'%(self.det_name)
            saved = self.checkpoint.load(chk_name)
            if saved is not None:
                full_image, all_bounds, all_vars, start_num, extra_builder = saved
                if extra_builder is not None:
                    base['extra_builder'] = extra_builder
                all_stamps = [galsim._Image(np.array([]), b, full_image.wcs) for b in all_bounds]
                logger.warning('File %d: Loaded checkpoint data from %s.',
                               base.get('file_num', 0), self.checkpoint.file_name)
                if start_num == obj_num + self.nobjects:
                    logger.warning('All objects already rendered for this image.')
                else:
                    logger.warning("Objects %d..%d already rendered", obj_num, start_num-1)
                    logger.warning('Starting at obj_num %d', start_num)
        nobj_tot = self.nobjects - (start_num - obj_num)

        if full_image is None:
            full_image = create_full_image(config, base)

        # Ensure 1 <= nbatch <= nobj_tot
        nbatch = max(min(self.nbatch, nobj_tot), 1)
        for batch in range(nbatch):
            start_obj_num = start_num + (nobj_tot * batch // nbatch)
            end_obj_num = start_num + (nobj_tot * (batch+1) // nbatch)
            nobj_batch = end_obj_num - start_obj_num
            if nbatch > 1:
                logger.warning("Start batch %d/%d with %d objects [%d, %d)",
                               batch+1, nbatch, nobj_batch, start_obj_num, end_obj_num)
            stamps, current_vars = galsim.config.BuildStamps(
                    nobj_batch, base, logger=logger, obj_num=start_obj_num, do_noise=False)
            base['index_key'] = 'image_num'

            for k in range(nobj_batch):
                # This is our signal that the object was skipped.
                if stamps[k] is None:
                    continue
                bounds = stamps[k].bounds & full_image.bounds
                if not bounds.isDefined():  # pragma: no cover
                    # These noramlly show up as stamp==None, but technically it is possible
                    # to get a stamp that is off the main image, so check for that here to
                    # avoid an error.  But this isn't covered in the imsim test suite.
                    continue

                logger.debug('image %d: full bounds = %s', image_num, str(full_image.bounds))
                logger.debug('image %d: stamp %d bounds = %s',
                        image_num, k+start_obj_num, str(stamps[k].bounds))
                logger.debug('image %d: Overlap = %s', image_num, str(bounds))
                full_image[bounds] += stamps[k][bounds]

            # Note: in typical imsim usage, all current_vars will be 0. So this normally doens't
            # add much to the checkpointing data.
            nz_var = np.nonzero(current_vars)[0]
            all_stamps.extend([stamps[k] for k in nz_var])
            all_vars.extend([current_vars[k] for k in nz_var])

            if self.checkpoint is not None:
                # Don't save the full stamps.  All we need for FlattenNoiseVariance is the bounds.
                # Everything else about the stamps has already been handled above.
                all_bounds = [stamp.bounds for stamp in all_stamps]
                data = (full_image, all_bounds, all_vars, end_obj_num,
                        base.get('extra_builder',None))
                self.checkpoint.save(chk_name, data)
                logger.warning('File %d: Completed batch %d with objects [%d, %d), and wrote '
                               'checkpoint data to %s',
                               base.get('file_num', 0), batch+1, start_obj_num, end_obj_num,
                               self.checkpoint.file_name)

        # Bring the image so far up to a flat noise variance
        current_var = galsim.config.FlattenNoiseVariance(
                base, full_image, all_stamps, tuple(all_vars), logger)

        return full_image, current_var


class LSST_PhotonPoolingImageBuilder(LSST_ImageBuilderBase):
    """Pools photon from all objects in `nbatch` batches.
    Photons from faint objects only appear in one of the batches randomly.
    """

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
        set_config_image_pos(config, base)

        # For cases where there is noise in individual stamps, we need to keep track of the
        # stamp bounds and their current variances.  When checkpointing, we don't need to
        # save the pixel values for this, just the bounds and the current_var value of each.
        all_stamps = []
        all_vars = []
        all_obj_nums = []
        photon_batch_num = 0

        full_image = None

        if self.checkpoint is not None:
            chk_name = "buildImage_" + self.det_name
            full_image, all_vars, all_stamps, all_obj_nums, photon_batch_num = load_checkpoint(self.checkpoint, chk_name, base, logger)
        remaining_obj_nums = sorted(frozenset(range(self.nobjects)) - frozenset(all_obj_nums))

        if full_image is None:
            full_image = create_full_image(config, base)

        # Ensure 1 <= nbatch <= len(remaining_obj_nums)
        n_fft_batch = max(min(self.nbatch, len(remaining_obj_nums)), 1)
        sensor = base.get('sensor', None)
        rng = galsim.config.GetRNG(config, base, logger, "LSST_Silicon")
        if sensor is not None:
            sensor.updateRNG(rng)
        fft_objects, phot_objects, faint_objects = partition_objects(load_objects(remaining_obj_nums, config, base, logger))
        logger.info("Found %d FFT objects, %d photon shooting objects and %d faint objects", len(fft_objects), len(phot_objects), len(faint_objects))
        if self.checkpoint is not None:
            if not fft_objects:
                logger.warning('All FFT objects already rendered for this image.')
            else:
                logger.warning("%d objects already rendered", len(all_obj_nums))

        # Handle FFT objects first:
        for batch_num, batch in enumerate(make_batches(fft_objects, n_fft_batch), start=1):
            if n_fft_batch > 1:
                logger.warning("Start FFT batch %d/%d with %d objects",
                               batch_num, n_fft_batch, len(batch))
            stamps, current_vars = build_stamps(base, logger, batch, stamp_type="LSST_Silicon")
            base['index_key'] = 'image_num'

            for stamp_obj, stamp in zip(batch, stamps):
                bounds = stamp_bounds(stamp, full_image.bounds)
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
                save_checkpoint(self.checkpoint, chk_name, base, full_image, all_stamps, all_vars, all_obj_nums, photon_batch_num)
                logger.warning('File %d: Completed batch %d, and wrote '
                               'checkpoint data to %s',
                               base.get('file_num', 0), batch_num,
                               self.checkpoint.file_name)
        # Handle photons:
        phot_batches = make_photon_batches(
                config, base, logger, phot_objects, faint_objects, self.nbatch
            )

        if photon_batch_num > 0:
            logger.warning(
                "Photon batches [0, %d) / %d already rendered - skipping",
                photon_batch_num,
                self.nbatch,
            )
            phot_batches = phot_batches[photon_batch_num:]
        base["image_pos"].x = full_image.center.x
        base["image_pos"].y = full_image.center.y
        photon_ops_cfg = {"photon_ops": base.get("stamp", {}).get("photon_ops", [])}
        photon_ops = galsim.config.BuildPhotonOps(photon_ops_cfg, 'photon_ops', base, logger)
        local_wcs = base["wcs"].local(galsim.position._PositionD(0., 0.))
        for batch_num, batch in enumerate(phot_batches, start=photon_batch_num):
            if not batch:
                continue
            base['index_key'] = 'image_num'
            stamps, current_vars = build_stamps(base, logger, batch, stamp_type="PhotonStampBuilder")
            photons = merge_photon_arrays(stamps)
            for op in photon_ops:
                op.applyTo(photons, local_wcs, rng)
            accumulate_photons(photons, full_image, sensor, full_image.center)

            # Note: in typical imsim usage, all current_vars will be 0. So this normally doens't
            # add much to the checkpointing data.
            nz_var = np.nonzero(current_vars)[0]
            all_vars.extend([current_vars[k] for k in nz_var])

            if self.checkpoint is not None:
                save_checkpoint(self.checkpoint, chk_name, base, full_image, all_stamps, all_vars, all_obj_nums, batch_num+1)

        # Bring the image so far up to a flat noise variance
        current_var = galsim.config.FlattenNoiseVariance(
                base, full_image, all_stamps, tuple(all_vars), logger)

        return full_image, current_var



def accumulate_photons(photons, image, sensor, center):
    if sensor is None:
        sensor = Sensor()
    imview = image._view()
    imview._shift(-center)  # equiv. to setCenter(), but faster
    imview.wcs = PixelScale(1.0)
    if imview.dtype in (np.float32, np.float64):
        sensor.accumulate(photons, imview, imview.center)
    else:
        # Need a temporary
        im1 = galsim.image.ImageD(bounds=imview.bounds)
        sensor.accumulate(photons, im1, imview.center)
        imview += im1

def make_batches(objects, nbatch: int):
    per_batch = len(objects) // nbatch
    o_iter = iter(objects)
    for _ in range(nbatch):
        yield [obj for _, obj in zip(range(per_batch), o_iter)]


def build_stamps(base, logger, objects: list[StellarObject], stamp_type: str):
    base["stamp"]["type"] = stamp_type
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


def make_photon_batches(config, base, logger, phot_objects: list[StellarObject], faint_objects: list[StellarObject], nbatch: int):
    if not phot_objects and not faint_objects:
        return []
    batches = [
        [dataclasses.replace(obj, phot_flux=obj.phot_flux / nbatch) for obj in phot_objects]
    ] * nbatch
    rng = galsim.config.GetRNG(config, base, logger, "LSST_Silicon")
    ud = galsim.UniformDeviate(rng)
    # Shuffle faint objects into the batches randomly:
    for obj in faint_objects:
        batch_index = int(ud() * nbatch)
        batches[batch_index].append(obj)

    return batches

def stamp_bounds(stamp, full_image_bounds):
    if stamp is None:
        return None
    bounds = stamp.bounds & full_image_bounds
    if not bounds.isDefined():  # pragma: no cover
        # These noramlly show up as stamp==None, but technically it is possible
        # to get a stamp that is off the main image, so check for that here to
        # avoid an error.  But this isn't covered in the imsim test suite.
        return None
    return bounds


def partition_objects(objects):
    objects_by_mode = {
        ProcessingMode.FFT: [],
        ProcessingMode.PHOT: [],
        ProcessingMode.FAINT: [],
    }
    for obj in objects:
        objects_by_mode[obj.mode].append(obj)
    return (
        objects_by_mode[ProcessingMode.FFT],
        objects_by_mode[ProcessingMode.PHOT],
        objects_by_mode[ProcessingMode.FAINT],
    )


def load_objects(obj_numbers, config, base, logger):
    gsparams = {}
    stamp = base['stamp']
    if 'gsparams' in stamp:
        gsparams = galsim.gsobject.UpdateGSParams(gsparams, stamp['gsparams'], config)

    for obj_num in obj_numbers:
        galsim.config.SetupConfigObjNum(base, obj_num, logger)
        obj = build_obj(stamp, base, logger)
        if obj is not None:
            yield build_obj(stamp, base, logger)

def create_full_image(config, base):
    if galsim.__version_info__ < (2,5):
        # GalSim 2.4 required a bit more work here.
        from galsim.config.stamp import _ParseDType

        full_xsize = base['image_xsize']
        full_ysize = base['image_ysize']
        wcs = base['wcs']

        dtype = _ParseDType(config, base)

        full_image = galsim.Image(full_xsize, full_ysize, dtype=dtype)
        full_image.setOrigin(base['image_origin'])
        full_image.wcs = wcs
        full_image.setZero()
        base['current_image'] = full_image
    else:
        # In GalSim 2.5+, the image is already built and available as 'current_image'
        full_image = base['current_image']
    return full_image


def set_config_image_pos(config, base):
    if 'image_pos' in config and 'world_pos' in config:
        raise galsim.config.GalSimConfigValueError(
            "Both image_pos and world_pos specified for LSST_Image.",
            (config['image_pos'], config['world_pos']))

    if ('image_pos' not in config and 'world_pos' not in config and
            not ('stamp' in base and
                ('image_pos' in base['stamp'] or 'world_pos' in base['stamp']))):
        full_xsize = base['image_xsize']
        full_ysize = base['image_ysize']
        xmin = base['image_origin'].x
        xmax = xmin + full_xsize-1
        ymin = base['image_origin'].y
        ymax = ymin + full_ysize-1
        config['image_pos'] = {
            'type' : 'XY' ,
            'x' : { 'type' : 'Random' , 'min' : xmin , 'max' : xmax },
            'y' : { 'type' : 'Random' , 'min' : ymin , 'max' : ymax }
        }


def load_checkpoint(checkpoint, chk_name, base, logger):
    saved = checkpoint.load(chk_name)
    if saved is not None:
        full_image, all_bounds, all_vars, all_obj_nums, extra_builder, photon_batch_num = saved
        if extra_builder is not None:
            base['extra_builder'] = extra_builder
        all_stamps = [galsim._Image(np.array([]), b, full_image.wcs) for b in all_bounds]
        logger.warning('File %d: Loaded checkpoint data from %s.',
                       base.get('file_num', 0), checkpoint.file_name)
        return full_image, all_vars, all_stamps, all_obj_nums, photon_batch_num
    return (None,)*5

def save_checkpoint(checkpoint, chk_name, base, full_image, all_stamps, all_vars, all_obj_nums, photon_batch_num):
    # Don't save the full stamps.  All we need for FlattenNoiseVariance is the bounds.
    # Everything else about the stamps has already been handled above.
    all_bounds = [stamp.bounds for stamp in all_stamps]
    data = (full_image, all_bounds, all_vars, all_obj_nums,
            base.get('extra_builder',None), photon_batch_num)
    checkpoint.save(chk_name, data)


RegisterImageType('LSST_Image', LSST_ImageBuilder())
RegisterImageType('LSST_PhotonPoolingImage', LSST_PhotonPoolingImageBuilder())
