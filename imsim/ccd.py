import copy
import os
import warnings
import astropy.time
import galsim
from galsim.config import OutputBuilder, RegisterOutputType
from .cosmic_rays import CosmicRays
from .meta_data import data_dir
from .camera import get_camera
from .opsim_data import get_opsim_data


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

        if 'det_num' not in config:
            config['det_num'] = { 'type': 'Sequence', 'nitems': 189 }

        # Figure out the detector name for the file name.
        detnum = galsim.config.ParseValue(config, 'det_num', base, int)[0]
        if 'camera' in config:
            camera_name = galsim.config.ParseValue(config, 'camera', base, str)[0]
        else:
            camera_name = 'LsstCam'
        camera = get_camera(camera_name)
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

        if 'exptime' in config:
            base['exptime'] = galsim.config.ParseValue(
                config, 'exptime', base, float
            )[0]
        else:
            base['exptime'] = 30.0

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
                    'only_dets', 'readout', 'exptime', 'camera' ]

        opt = {
            'cosmic_ray_rate': float,
            'cosmic_ray_catalog': str,
            'header': dict
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
            exptime = base['exptime']
            det_name = base['det_name']
            cosmic_rays = CosmicRays(cosmic_ray_rate, cosmic_ray_catalog)
            rng = galsim.config.GetRNG(config, base)
            cosmic_rays.paint(image.array, rng, exptime=exptime)

        # Add header keywords for various values written to the primary
        # header of the simulated raw output file, so that all the needed
        # information is in the eimage file.
        image.header = galsim.FitsHeader()
        exptime = base['exptime']
        image.header['EXPTIME'] = exptime
        image.header['DET_NAME'] = base['det_name']

        header_vals = copy.deepcopy(params.get('header', {}))
        opsim_data = get_opsim_data(config, base)

        # Helper function to parse a value with priority:
        # 1. from header_vals (popped from dict if present)
        # 2. from opsim_data
        # 3. specified default
        def parse(item, type, default):
            if item in header_vals:
                val = galsim.config.ParseValue(header_vals, item, base, type)[0]
                del header_vals[item]
            else:
                val = opsim_data.get(item, default)
            return val

        # Get a few items needed more than once first
        mjd = parse('mjd', float, 51444.0)
        mjd_obs = parse('observationStartMJD', float, mjd)
        mjd_end =  mjd_obs + exptime/86400.0
        seqnum = parse('seqnum', int, 0)
        ratel = parse('fieldRA', float, 0.0)
        dectel = parse('fieldDec', float, 0.0)
        airmass = parse('airmass', float, 'N/A')

        # Now construct the image header
        image.header['MJD'] = mjd
        image.header['MJD-OBS'] = mjd_obs, 'Start of exposure'
        # NOTE: Should this day be the current day,
        # or the day at the time of the most recent noon?
        dayobs = astropy.time.Time(mjd_obs, format='mjd').strftime('%Y%m%d')
        image.header['DAYOBS'] = dayobs
        image.header['SEQNUM'] = seqnum
        image.header['CONTRLLR'] = 'P', 'simulated data'
        image.header['RUNNUM'] = parse('observationId', int, -999)
        image.header['OBSID'] = f"IM_P_{dayobs}_{seqnum:06d}"
        image.header['IMGTYPE'] = parse('image_type', str, 'SKYEXP')
        image.header['REASON'] = parse('reason', str, 'survey')
        image.header['RATEL'] = ratel
        image.header['DECTEL'] = dectel
        with warnings.catch_warnings():
            # Silence FITS warning about long header keyword
            warnings.simplefilter('ignore')
            image.header['ROTTELPOS'] = parse('rotTelPos', float, 0.0)
        image.header['FILTER'] = parse('band', str, 'N/A/')
        image.header['CAMERA'] = base['output']['camera']
        image.header['HASTART'] = opsim_data.getHourAngle(mjd_obs, ratel)
        image.header['HAEND'] = opsim_data.getHourAngle(mjd_end, ratel)
        image.header['AMSTART'] = airmass
        image.header['AMEND'] = airmass  # wrong, does anyone care?

        # If there's anything left in header_vals, add it to the header.
        for k in header_vals:
            image.header[k] = galsim.config.ParseValue(
                header_vals, k, base, None
            )[0]

        return [ image ]

RegisterOutputType('LSST_CCD', LSST_CCDBuilder())

