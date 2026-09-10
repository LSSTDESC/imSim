import galsim
from galsim.config import GetAllParams, AddNoise, RegisterImageType
from .lsst_image import LSST_ImageBuilderBase


__all__ = ['RubinCoaddImageBuilder']


class RubinCoaddImageBuilder(LSST_ImageBuilderBase):  # noqa: N801
    def setup(self, config, base, image_num, obj_num, ignore, logger):
        """
        Do the initialization and setup for a Rubin/LSST deep coadd image.

        Parameters:
            config:    The configuration dict for the image field.
            base:      The base configuration dict.
            image_num: The current image number.
            obj_num:   The first object number in the image.
            ignore:    A list of parameters that are allowed to be in config
                       that we can ignore here.
            logger:    A logger object to log progress.

        Returns:
            xsize, ysize
        """
        self.nobjects = self.getNObj(config, base, image_num, logger=logger)
        if 'nobjects' in config:
            # User specified nobjects.
            # Make sure it's not more than what any input catalog can
            # handle (we don't want repeated objects).
            input_nobj = galsim.config.ProcessInputNObjects(base)
            if input_nobj is not None:
                self.nobjects = min(self.nobjects, input_nobj)
        logger.info('image %d: nobj = %d', image_num, self.nobjects)

        req = { 'tract': int, 'patch': int, 'band': str }
        opt = { 'nbatch': int, 'nsubbatch': int, 'nbatch_fft': int,
                'nbatch_per_checkpoint': int}
        extra_ignore = ['image_pos', 'world_pos', 'stamp_size',
                        'stamp_xsize', 'stamp_ysize', 'nobjects' ]
        params = GetAllParams(config, base, req=req, opt=opt,
                              ignore=ignore+extra_ignore)[0]

        self.tract = params['tract']
        self.patch = params['patch']
        self.band = params['band']

        self.add_noise = True

        self.nbatch = params.get('nbatch', 10)
        self.nbatch_per_checkpoint = params.get('nbatch_per_checkpoint', 1)
        self.nsubbatch = params.get('nsubbatch', 50)
        self.nbatch_fft = params.get('nbatch_fft', 1)
        try:
            self.checkpoint = galsim.config.GetInputObj(
                'checkpoint', config, base, 'RubinCoaddImageBuilder')
        except galsim.config.GalSimConfigError:
            self.checkpoint = None

        deep_coadd = galsim.config.GetInputObj(
            'deep_coadd', config, base, 'LSSTC_CoaddImageBuilder')
        bbox = deep_coadd.skymap[self.tract][self.patch].getOuterBBox()
        xsize, ysize = bbox.width, bbox.height

        return xsize, ysize

    def addNoise(self, image, config, base, image_num, obj_num, current_var,
                 logger):
        """Add image noise

        Parameters:
            image:          The image onto which to add the noise.
            config:         The configuration dict for the image field.
            base:           The base configuration dict.
            image_num:      The current image number.
            obj_num:        The first object number in the image.
            current_var:    The current noise variance in each postage stamps.
            logger:         A logger object to log progress.
        """
        if self.add_noise:
            AddNoise(base, image, current_var, logger)


RegisterImageType('RubinDeepCoadd', RubinCoaddImageBuilder())
