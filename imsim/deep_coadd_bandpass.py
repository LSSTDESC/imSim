import galsim
from galsim.config import BandpassBuilder, RegisterBandpassType


__all__ = ['RubinDeepCoaddBandpassBuilder']


class RubinDeepCoaddBandpassBuilder(BandpassBuilder):
    """Build the Rubin bandpass to be used with deep coadd sims
    from standard passbands."""
    def buildBandpass(self, config, base, logger):
        """
        Parameters:
            config:     The configuration dict for the bandpass type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.
        Returns:
            the constructed Bandpass object.
        """
        req = { 'band': str }
        params, safe = galsim.config.GetAllParams(config, base, req=req)
        deep_coadds = galsim.config.GetInputObj('deep_coadd', config, base,
                                                'DeepCoaddBandpass')
        return deep_coadds.getBandpass(params['band']), safe


RegisterBandpassType('RubinDeepCoaddBandpass', RubinDeepCoaddBandpassBuilder())
