
import galsim
from galsim.config import OutputBuilder, RegisterOutputType

class LSST_CCDBuilder(OutputBuilder):
    """This runs the overall generation of an LSST CCD file.

    Most of the defaults work fine.  There are a few extra things we do that are LSST-specific.
    """
    rafts = [        'R01', 'R02', 'R03',
              'R10', 'R11', 'R12', 'R13', 'R14',
              'R20', 'R21', 'R22', 'R23', 'R24',
              'R30', 'R31', 'R32', 'R33', 'R34',
                     'R41', 'R42', 'R43'
            ]
    sensors = [ 'S00', 'S01', 'S02',
                'S10', 'S11', 'S12',
                'S20', 'S21', 'S22'
              ]

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
        # Detectors have names Rij_Smn
        # We need to run through them in a predictable way for detnum = 0..188
        raft = detnum // 9   # 0..20
        sensor = detnum % 9  # 0..8
        raft = self.rafts[raft]
        sensor = self.sensors[sensor]
        det_name = raft + '-' + sensor
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
                    'cosmic_rays', 'readout', 'exp_time' ]
        galsim.config.CheckAllParams(config, ignore=ignore)

        image = galsim.config.BuildImage(base, image_num, obj_num, logger=logger)
        return [ image ]


RegisterOutputType('LSST_CCD', LSST_CCDBuilder())

