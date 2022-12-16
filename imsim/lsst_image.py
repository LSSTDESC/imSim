import os
import json
import numpy as np
import scipy
import logging
import galsim
from galsim.config import RegisterImageType, GetAllParams, GalSimConfigError, GetSky, AddNoise
from galsim.config.image_scattered import ScatteredImageBuilder
from lsst.afw import cameraGeom
import lsst.geom
from .sky_model import SkyGradient
from .camera import get_camera
from .meta_data import data_dir


class Vignetting:
    """
    Compute vignetting using a 1D spline to model the radial profile
    of the vignetting function.
    """
    def __init__(self, spline_data_file):
        """
        Parameters
        ----------
        spline_data_file : str
            File containing the 1D spline model.
        """
        if not os.path.isfile(spline_data_file):
            # Check if spline data file is in data_dir.
            data_file = os.path.join(data_dir, spline_data_file)
            if not os.path.isfile(data_file):
                raise OSError(f"Vignetting data file {spline_data_file} not found.")
        else:
            data_file = spline_data_file
        with open(data_file) as fobj:
            spline_data = json.load(fobj)

        self.spline_model = scipy.interpolate.BSpline(*spline_data)

        # Use the value at the center of the focal plane to
        # normalize the vignetting profile.
        self.value_at_zero = self.spline_model(0)

    def __call__(self, det):
        """
        Return the vignetting for each pixel in a CCD.

        Parameters
        ----------
        det : lsst.afw.cameraGeom.Detector
            CCD in the focal plane.
        """
        bbox = det.getBBox()
        nx = bbox.getWidth()
        ny = bbox.getHeight()

        # Compute the location of the lower-left corner pixel of the
        # CCD in focal plane coordinates.
        pix_to_fp = det.getTransform(cameraGeom.PIXELS,
                                     cameraGeom.FOCAL_PLANE)
        pixel_size = det.getPixelSize()
        llc = pix_to_fp.applyForward(lsst.geom.Point2D(0, 0))

        # Generate a grid of pixel x, y locations in the focal plane.
        xarr, yarr = np.meshgrid(np.arange(nx)*pixel_size.x + llc.x,
                                 np.arange(ny)*pixel_size.y + llc.y)

        # Compute the radial distance of each pixel to the focal plane
        # center.
        r = np.sqrt(xarr**2 + yarr**2)

        return self.spline_model(r)/self.value_at_zero


class LSST_ImageBuilder(ScatteredImageBuilder):

    # This is mostly the same as the GalSim "Scattered" image type.
    # So far the only change is in the sky background image.

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
        logger.debug('image %d: nobj = %d',image_num,self.nobjects)

        # These are allowed for LSST_Image, but we don't use them here.
        extra_ignore = [ 'image_pos', 'world_pos', 'stamp_size', 'stamp_xsize', 'stamp_ysize',
                         'nobjects' ]
        opt = { 'size': int , 'xsize': int , 'ysize': int, 'dtype': None,
                'apply_vignetting': bool, 'apply_sky_gradient': bool,
                'vignetting_data_file': str, 'camera': str, 'nbatch': int}
        params = GetAllParams(config, base, opt=opt, ignore=ignore+extra_ignore)[0]

        size = params.get('size',0)
        full_xsize = params.get('xsize',size)
        full_ysize = params.get('ysize',size)

        self.apply_vignetting = params.get('apply_vignetting', False)
        if self.apply_vignetting:
            vignetting_data_file = params.get('vignetting_data_file')
            self.vignetting = Vignetting(vignetting_data_file)

        self.apply_sky_gradient = params.get('apply_sky_gradient', False)

        self.camera_name = params.get('camera')

        try:
            self.checkpoint = galsim.config.GetInputObj('checkpoint', config, base, 'LSST_Image')
            self.nbatch = params.get('nbatch', 10)
        except galsim.GalSimConfigError:
            self.checkpoint = None
            self.nbatch = params.get('nbatch', 1)
            # Note: This will probably also become 10 once we're doing the photon
            #       pooling stuff.  But for now, let it be 1 if not checkpointing.

        if (full_xsize <= 0) or (full_ysize <= 0):
            raise GalSimConfigError(
                "Both image.xsize and image.ysize need to be defined and > 0.")

        # If image_force_xsize and image_force_ysize were set in config, make sure it matches.
        if ( ('image_force_xsize' in base and full_xsize != base['image_force_xsize']) or
             ('image_force_ysize' in base and full_ysize != base['image_force_ysize']) ):
            raise GalSimConfigError(
                "Unable to reconcile required image xsize and ysize with provided "
                "xsize=%d, ysize=%d, "%(full_xsize,full_ysize))

        return full_xsize, full_ysize

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
        from galsim.config.stamp import _ParseDType

        full_xsize = base['image_xsize']
        full_ysize = base['image_ysize']
        wcs = base['wcs']

        dtype = _ParseDType(config, base)

        if 'image_pos' in config and 'world_pos' in config:
            raise GalSimConfigValueError(
                "Both image_pos and world_pos specified for LSST_Image.",
                (config['image_pos'], config['world_pos']))

        if 'image_pos' not in config and 'world_pos' not in config:
            xmin = base['image_origin'].x
            xmax = xmin + full_xsize-1
            ymin = base['image_origin'].y
            ymax = ymin + full_ysize-1
            config['image_pos'] = {
                'type' : 'XY' ,
                'x' : { 'type' : 'Random' , 'min' : xmin , 'max' : xmax },
                'y' : { 'type' : 'Random' , 'min' : ymin , 'max' : ymax }
            }

        full_image = None
        start_num = obj_num
        if self.checkpoint is not None:
            chk_name = 'buildImage_%s'%(base.get('det_name',''))
            saved = self.checkpoint.load(chk_name)
            if saved is not None:
                full_image, current_var, start_num, base['extra_builder'] = saved
                logger.warning('File %d: Loaded checkpoint data from %s.',
                               base.get('file_num', 0), self.checkpoint.file_name)
                if start_num == obj_num + self.nobjects:
                    logger.warning('All objects already rendered for this image.')
                else:
                    logger.warning("Objects %d..%d already rendered", obj_num, start_num-1)
                    logger.warning('Starting at obj_num %d', start_num)
        nobj_tot = self.nobjects - (start_num - obj_num)

        if full_image is None:
            full_image = galsim.Image(full_xsize, full_ysize, dtype=dtype)
            full_image.setOrigin(base['image_origin'])
            full_image.wcs = wcs
            full_image.setZero()
            start_batch = 0
        base['current_image'] = full_image

        nbatch = min(self.nbatch, nobj_tot)
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
                if stamps[k] is None: continue
                bounds = stamps[k].bounds & full_image.bounds
                logger.debug('image %d: full bounds = %s', image_num, str(full_image.bounds))
                logger.debug('image %d: stamp %d bounds = %s',
                        image_num, k+start_obj_num, str(stamps[k].bounds))
                logger.debug('image %d: Overlap = %s', image_num, str(bounds))
                if bounds.isDefined():
                    full_image[bounds] += stamps[k][bounds]
                else:
                    logger.info(
                        "Object centered at (%d,%d) is entirely off the main image, "
                        "whose bounds are (%d,%d,%d,%d)."%(
                            stamps[k].center.x, stamps[k].center.y,
                            full_image.bounds.xmin, full_image.bounds.xmax,
                            full_image.bounds.ymin, full_image.bounds.ymax))

            # Bring the image so far up to a flat noise variance
            # Note: This is pretty sub-optimal when nbatch > 1, but it's only relevant when
            #       drawing RealGalaxy objects, which we usually don't do in imSim.  We might
            #       need to rethink this a bit if people want to use RealGalaxy's with
            #       checkpointing (or in general nbatch>1).
            current_var = galsim.config.FlattenNoiseVariance(
                    base, full_image, stamps, current_vars, logger)

            if self.checkpoint is not None:
                data = (full_image, current_var, end_obj_num, base['extra_builder'])
                self.checkpoint.save(chk_name, data)
                logger.warning('File %d: Completed batch %d with objects [%d, %d), and wrote '
                               'checkpoint data to %s',
                               base.get('file_num', 0), batch+1, start_obj_num, end_obj_num,
                               self.checkpoint.file_name)

        return full_image, current_var

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
            sky1 = GetSky(config, base, full=False)
            logger.info("Setting sky level to %.2f photons/arcsec^2 "
                        "at (ra, dec) = %s, %s", sky1,
                        skyCoord.ra.deg, skyCoord.dec.deg)

        sky = GetSky(config, base, full=True)

        if ((self.apply_sky_gradient or self.apply_vignetting)
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

        if self.apply_vignetting:
            det_name = base['det_name']
            camera = get_camera(self.camera_name)
            logger.info("Applying vignetting according to radial spline model.")
            sky.array[:] *= self.vignetting(camera[det_name])

        image += sky

        AddNoise(base,image,current_var,logger)

RegisterImageType('LSST_Image', LSST_ImageBuilder())
