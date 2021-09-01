
import yaml
import galsim
from galsim.config import WCSBuilder, RegisterWCSType

class DictWCS(WCSBuilder):
    def __init__(self):
        self.d = {}  # Empty dict means we haven't read the file yet.

    def buildWCS(self, config, base, logger):
        """Build the TanWCS based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the wcs type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed WCS object.
        """
        req = { "file_name": str,
                "key": str
              }
        opt = { "fix_ab": bool,
              }
        params, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)

        file_name = params['file_name']
        key = params['key']
        fix_ab = params.get('fix_ab', True)
        logger.info("Finding WCS for %s",key)

        if not self.d or file_name != self.file_name:
            with open(file_name) as f:
                self.d = yaml.load(f.read(), Loader=yaml.SafeLoader)
            self.file_name = file_name
        logger.debug("Using WCS: %s",self.d[key])

        # The dict stores the WCS as its repr.  Eval it to make an actual WCS.
        wcs = galsim.utilities.math_eval(self.d[key])

        # I changed the internal storage of the ab matrices in GalSim, so the reprs I have
        # in the yaml file are wrong.  This fixes them.
        # TODO: Fix the yaml file and get rid of this hack.
        if fix_ab:
            if wcs.ab is not None:
                wcs.ab[0,1,0] += 1
                wcs.ab[1,0,1] += 1
            if wcs.abp is not None:
                wcs.abp[0,1,0] += 1
                wcs.abp[1,0,1] += 1

        return wcs

RegisterWCSType('Dict', DictWCS())
