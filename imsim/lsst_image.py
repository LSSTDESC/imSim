import os
import numpy as np
import galsim
from galsim.config import RegisterImageType, GetAllParams, GalSimConfigError, GetSky, AddNoise
from galsim.config.image_scattered import ScatteredImageBuilder
from .sky_model import SkyGradient


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
                'apply_vignetting': bool, 'apply_sky_gradient': bool }
        params = GetAllParams(config, base, opt=opt, ignore=ignore+extra_ignore)[0]

        size = params.get('size',0)
        full_xsize = params.get('xsize',size)
        full_ysize = params.get('ysize',size)
        self.apply_vignetting = params.get('apply_vignetting', False)
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
        if self.apply_vignetting:
            # TODO
            pass
        if self.apply_sky_gradient:
            ny, nx = sky.array.shape
            sky_model = galsim.config.GetInputObj('sky_model', config, base,
                                                  'LSST_ImageBuilder')
            sky_gradient = SkyGradient(sky_model, config['wcs']['current'][0],
                                       base['world_center'], nx)
            xarr = np.array([list(range(nx))]*ny)
            yarr = np.array([[_]*nx for _ in range(ny)])
            sky.array[:] += sky_gradient(xarr, yarr)
        if sky:
            image += sky
        AddNoise(base,image,current_var,logger)

RegisterImageType('LSST_Image', LSST_ImageBuilder())

