import os
import json
import numpy as np
import scipy
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
            logger:     If given, a logger object to log progress.

        Returns:
            xsize, ysize
        """
        # This is mostly a copy of the ScatteredImageType.setup function.
        logger.debug('image %d: Building LSST_Image: image, obj = %d,%d',
                     image_num,image_num,obj_num)

        self.nobjects = self.getNObj(config, base, image_num, logger=logger)
        logger.debug('image %d: nobj = %d',image_num,self.nobjects)

        # These are allowed for LSST_Image, but we don't use them here.
        extra_ignore = [ 'image_pos', 'world_pos', 'stamp_size', 'stamp_xsize', 'stamp_ysize',
                         'nobjects' ]
        opt = { 'size': int , 'xsize': int , 'ysize': int, 'dtype': None,
                'apply_vignetting': bool, 'apply_sky_gradient': bool,
                'vignetting_data_file': str, 'camera': str}
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

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        """Add the final noise to a Scattered image

        Parameters:
            image:          The image onto which to add the noise.
            config:         The configuration dict for the image field.
            base:           The base configuration dict.
            image_num:      The current image number.
            obj_num:        The first object number in the image.
            current_var:    The current noise variance in each postage stamps.
            logger:         If given, a logger object to log progress.
        """
        base['current_noise_image'] = base['current_image']
        sky = GetSky(config, base, full=True)

        skyCoord = base['world_center']
        logger.info("Setting sky level to %.2f photons/arcsec^2 "
                    "at (ra, dec) = %s, %s", sky,
                    skyCoord.ra.deg, skyCoord.dec.deg)

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