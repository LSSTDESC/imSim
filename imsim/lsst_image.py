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
    # _relative_fp_coords caches the x, y positions in mm of each pixel
    # for a ITL or e2V CCDs in focal plane coordinates relative to the
    # lower-left corner of the CCD.  To obtain arrays of focal plane
    # coordinates relative to the center of the focal plane, an
    # overall translation is applied.
    _relative_fp_coords = {}

    def __init__(self, spline_data_file, logger):
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
        self.value_at_zero = self.spline_model(0)
        self.logger = logger

    @staticmethod
    def _get_relative_fp_coords(det, pix_to_fp=None):
        """Compute the focal plane locations of each pixel of the
        CCD relative to its lower-left corner."""
        if pix_to_fp is None:
            pix_to_fp = det.getTransform(cameraGeom.PIXELS,
                                         cameraGeom.FOCAL_PLANE)
        bbox = det.getBBox()
        nx = bbox.getWidth()
        ny = bbox.getHeight()

        xarr = np.array([list(range(nx))]*ny).flatten()
        yarr = np.array([[_]*nx for _ in range(ny)]).flatten()
        pixel_coords = [lsst.geom.Point2D(x, y) for x, y in zip(xarr, yarr)]

        fp_coords = pix_to_fp.applyForward(pixel_coords)

        fp_xarr = np.array([_.x for _ in fp_coords]).reshape(ny, nx)
        fp_yarr = np.array([_.y for _ in fp_coords]).reshape(ny, nx)

        return fp_xarr - fp_xarr[0, 0], fp_yarr - fp_yarr[0, 0]

    def _get_fp_coords(self, det):
        """
        Get arrays of x and y coordinates in focal plane coordinates for
        the pixels in the requested CCD.
        """
        pix_to_fp = det.getTransform(cameraGeom.PIXELS,
                                     cameraGeom.FOCAL_PLANE)
        vendor = det.getSerial()[:3]
        # Retrieve the cached values of pixel coordinates for a CCD of
        # this vendor type (either 'ITL' or 'E2V') relative to its llc.
        if vendor not in self._relative_fp_coords:
            # If values aren't cached for this vendor type, do the calculation.
            # This takes about ~1 minute.
            self.logger.info("Computing relative focal plane coordinates "
                             f"of pixels for {vendor} CCDs.")
            self._relative_fp_coords[vendor] \
                = self._get_relative_fp_coords(det, pix_to_fp=pix_to_fp)
        xarr0, yarr0 = self._relative_fp_coords[vendor]

        # Compute absolute location of llc, apply this offset, and
        # return the x- and y-arrays.
        llc = pix_to_fp.applyForward(lsst.geom.Point2D(0, 0))
        return xarr0 + llc.x, yarr0 + llc.y

    def __call__(self, det):
        """Return the vignetting for each pixel in the CCD."""
        xarr, yarr = self._get_fp_coords(det)
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
                'vignetting_data_file': str}
        params = GetAllParams(config, base, opt=opt, ignore=ignore+extra_ignore)[0]

        size = params.get('size',0)
        full_xsize = params.get('xsize',size)
        full_ysize = params.get('ysize',size)

        self.apply_vignetting = params.get('apply_vignetting', False)
        vignetting_data_file =  params.get('vignetting_data_file', False)
        if self.apply_vignetting:
            self.vignetting = Vignetting(vignetting_data_file, logger)

        self.apply_sky_gradient = params.get('apply_sky_gradient', False)

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

        if sky:
            if self.apply_sky_gradient:
                ny, nx = sky.array.shape
                sky_model = galsim.config.GetInputObj('sky_model', config, base,
                                                      'LSST_ImageBuilder')
                sky_gradient = SkyGradient(sky_model, config['wcs']['current'][0],
                                           base['world_center'], nx)
                xarr = np.array([list(range(nx))]*ny)
                yarr = np.array([[_]*nx for _ in range(ny)])
                sky.array[:] += sky_gradient(xarr, yarr)

            if self.apply_vignetting:
                det_name = base['det_name']
                camera = get_camera(base['camera'])
                sky.array[:] *= self.vignetting(camera[det_name])

            image += sky

        AddNoise(base,image,current_var,logger)

RegisterImageType('LSST_Image', LSST_ImageBuilder())
