
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
        params, safe = galsim.config.GetAllParams(config, base, req=req)

        file_name = params['file_name']
        key = params['key']
        logger.info("Finding WCS for %s",key)

        if not self.d or file_name != self.file_name:
            with open(file_name) as f:
                self.d = yaml.load(f.read(), Loader=yaml.SafeLoader)
            self.file_name = file_name
        logger.debug("Using WCS: %s",self.d[key])

        # The dict stores the WCS as its repr.  Eval it to make an actual WCS.
        return galsim.utilities.math_eval(self.d[key])

RegisterWCSType('Dict', DictWCS())
