import os
import galsim
from galsim.config import OutputBuilder, RegisterOutputType
from .cosmic_rays import CosmicRays
from .meta_data import data_dir
from .camera import get_camera

class LSST_CCDBuilder(OutputBuilder):
    """This runs the overall generation of an LSST CCD file.

    Most of the defaults work fine.  There are a few extra things we do that are LSST-specific.
    """

    def setup(self, config, base, file_num, logger):
        """Do any necessary setup at the start of processing a file.

        Parameters:
            config:     The configuration dict for the output type.
            base:       The base configuration dict.
            file_num:   The current file_num.
            logger:     If given, a logger object to log progress.
        """
        # This is a copy of the base class code
        seed = galsim.config.SetupConfigRNG(base, logger=logger)
        logger.debug('file %d: seed = %d',file_num,seed)

        # Figure out the detector name for the file name.
        detnum = galsim.config.ParseValue(config, 'det_num', base, int)[0]
        camera = get_camera(config['camera'])
        det_name = camera[detnum].getName()
        base['det_name'] = det_name
        if 'eval_variables' not in base:
            base['eval_variables'] = {}
        if 'sdet_name' not in base['eval_variables']:
            base['eval_variables']['sdet_name'] = det_name

        base['exp_time'] = float(config.get('exp_time', 30))

    def getNFiles(self, config, base, logger=None):
        """Returns the number of files to be built.

        nfiles can be specified if you want.
        
        But the default is 189, not 1.

        Parameters:
            config:     The configuration dict for the output field.
            base:       The base configuration dict.

        Returns:
            the number of files to build.
        """
        if 'nfiles' in config:
            return galsim.config.ParseValue(config, 'nfiles', base, int)[0]
        else:
            return 189

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images for output.

        Parameters:
            config:     The configuration dict for the output field.
            base:       The base configuration dict.
            file_num:   The current file_num.
            image_num:  The current image_num.
            obj_num:    The current obj_num.
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here.  i.e. it won't be an error if they are present.
            logger:     If given, a logger object to log progress.

        Returns:
            a list of the images built
        """
        # This is basically the same as the base class version.  Just a few extra things to
        # add to the ignore list.
        ignore += [ 'file_name', 'dir', 'nfiles', 'checkpoint', 'det_num',
                    'readout', 'exp_time', 'camera' ]

        opt = {
            'cosmic_ray_rate': float,
            'cosmic_ray_catalog': str
        }
        params, safe = galsim.config.GetAllParams(config, base, opt=opt, ignore=ignore)

        image = galsim.config.BuildImage(base, image_num, obj_num, logger=logger)

        # Add cosmic rays.
        cosmic_ray_rate = params.get('cosmic_ray_rate', 0)
        if cosmic_ray_rate > 0:
            cosmic_ray_catalog = params.get('cosmic_ray_catalog', None)
            if cosmic_ray_catalog is None:
                cosmic_ray_catalog = os.path.join(data_dir, 'cosmic_rays_itl_2017.fits.gz')
            if not os.path.isfile(cosmic_ray_catalog):
                raise FileNotFoundError(f'{cosmic_ray_catalog} not found')

            logger.info('Adding cosmic rays with rate %f using %s.',
                        cosmic_ray_rate, cosmic_ray_catalog)
            exp_time = base['exp_time']
            det_name = base['det_name']
            cosmic_rays = CosmicRays(cosmic_ray_rate, cosmic_ray_catalog)
            rng = galsim.config.GetRNG(config, base)
            cosmic_rays.paint(image.array, rng, exptime=exp_time)

        return [ image ]

RegisterOutputType('LSST_CCD', LSST_CCDBuilder())

