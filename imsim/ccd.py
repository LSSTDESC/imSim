import os
import warnings
import astropy.time
import galsim
from galsim.config import OutputBuilder, RegisterOutputType
from .cosmic_rays import CosmicRays
from .meta_data import data_dir
from .camera import get_camera
from .opsim_meta import get_opsim_md


# Add `xsize` and `ysize` to the list of preset variables. These are
# evaluated below in LSST_CCDBuilder.setup.
galsim.config.eval_base_variables.extend(('xsize', 'ysize'))


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
        if 'only_dets' in config:
            only_dets = config['only_dets']
            det_name = only_dets[detnum]
        else:
            det_name = camera[detnum].getName()
        base['det_name'] = det_name
        if 'eval_variables' not in base:
            base['eval_variables'] = {}
        base['eval_variables']['sdet_name'] = det_name

        # Get detector size in pixels.
        det_bbox = camera[det_name].getBBox()
        base['xsize'] = det_bbox.width
        base['ysize'] = det_bbox.height

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
        ignore += [ 'file_name', 'dir', 'nfiles', 'det_num',
                    'only_dets', 'readout', 'exp_time', 'camera' ]

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

        # Add header keywords for various values written to the primary
        # header of the simulated raw output file, so that all the needed
        # information is in the eimage file.
        opsim_md = get_opsim_md(config, base)
        image.header = galsim.FitsHeader()
        exp_time = base['exp_time']
        image.header['EXPTIME'] = exp_time
        image.header['DET_NAME'] = base['det_name']
        # MJD is the midpoint of the exposure.  51444 = Jan 1, 2000, which is
        # not a real observing date.
        image.header['MJD'] = opsim_md.get('mjd', 51444)
        # MJD-OBS is the start of the exposure
        mjd_obs = opsim_md.get('observationStartMJD', 51444)
        mjd_end =  mjd_obs + exp_time/86400.
        image.header['MJD-OBS'] = mjd_obs
        dayobs = astropy.time.Time(mjd_obs, format='mjd').strftime('%Y%m%d')
        image.header['DAYOBS'] = dayobs
        seqnum = opsim_md.get('seqnum', opsim_md.get('snap', 0))
        image.header['SEQNUM'] = seqnum
        image.header['CONTRLLR'] = 'P'  # For simulated data.
        image.header['OBSID'] = f"IM_P_{dayobs}_{seqnum:06d}"
        image.header['IMGTYPE'] = opsim_md.get('image_type', 'SKYEXP')
        image.header['REASON'] = opsim_md.get('reason', 'survey')
        ratel = opsim_md.get('fieldRA', 0.)
        dectel = opsim_md.get('fieldDec', 0.)
        image.header['RATEL'] = ratel
        image.header['DECTEL'] = dectel
        with warnings.catch_warnings():
            # Silence FITS warning about long header keyword
            warnings.simplefilter('ignore')
            image.header['ROTTELPOS'] = opsim_md.get('rotTelPos', 0.)
        image.header['FILTER'] = opsim_md.get('band')
        image.header['CAMERA'] = base['output']['camera']
        image.header['HASTART'] = opsim_md.getHourAngle(mjd_obs, ratel)
        image.header['HAEND'] = opsim_md.getHourAngle(mjd_end, ratel)
        image.header['AMSTART'] = opsim_md.get('airmass', 'N/A')
        image.header['AMEND'] = image.header['AMSTART']  # XXX: This is not correct. Does anyone care?
        return [ image ]

RegisterOutputType('LSST_CCD', LSST_CCDBuilder())

