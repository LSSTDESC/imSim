
import os
import galsim
from galsim.config import RegisterBandpassType

__all__ = ['RubinBandpass']


def RubinBandpass(band, logger=None):
    """Return one of the Rubin bandpasses, specified by the single-letter name.

    The zeropoint is automatically set to the AB zeropoint normalization.

    Parameters
    ----------
    band : `str`
        The name of the bandpass.  One of u,g,r,i,z,y.
    logger : logging.Logger
        If provided, a logger for logging debug statements.
    """
    # This uses the baseline throughput files from lsst.sims
    sims_dir = os.getenv("RUBIN_SIM_DATA_DIR")
    file_name = os.path.join(sims_dir, "throughputs", "baseline", f"total_{band}.dat")
    if not os.path.isfile(file_name):
        # If the user doesn't have the RUBIN_SIM_DATA_DIR defined, or if they don't have
        # the correct files installed, revert to the GalSim files.
        logger = galsim.config.LoggerWrapper(logger)
        logger.warning("Warning: Using the old bandpass files from GalSim, not lsst.sims")
        file_name = f"LSST_{band}.dat"
    bp = galsim.Bandpass(file_name, wave_type='nm')
    bp = bp.truncate(relative_throughput=1.e-3)
    bp = bp.thin()
    bp = bp.withZeropoint('AB')
    return bp

class RubinBandpassBuilder(galsim.config.BandpassBuilder):
    """A class for building a RubinBandpass in the config file
    """
    def buildBandpass(self, config, base, logger):
        """Build the Bandpass object based on the LSST filter name.

        Parameters:
            config:     The configuration dict for the bandpass type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed Bandpass object.
        """
        req = { 'band' : str }
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
        kwargs['logger'] = logger
        bp = RubinBandpass(**kwargs)
        logger.debug('bandpass = %s', bp)
        return bp, safe

RegisterBandpassType('RubinBandpass', RubinBandpassBuilder())
