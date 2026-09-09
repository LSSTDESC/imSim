"""
WCS builder for LSST deep_coadds
"""
from galsim.config import WCSBuilder, GetAllParams, GetInputObj, RegisterWCSType


__all__ = ["LSSTDeepCoaddWCSBuilder"]


class LSSTDeepCoaddWCSBuilder(WCSBuilder):

    def buildWCS(self, config, base, logger):
        """Build an AstropyWCS from a deep_coadd

        Parameters:
            config:     The configuration dict for the wcs type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            The constructed WCS object (a galsim.AstropyWCS instance).
        """
        req = {
            "tract": int,
            "patch": int,
            "band": str,
        }
        params, _ = GetAllParams(config, base, req=req)
        data_id = { key: params[key] for key in req }
        deep_coadd = GetInputObj("deep_coadd", config, base,
                                 "LSSTDeepCoaddWCSBuilder")
        return deep_coadd.getWcs(data_id)


RegisterWCSType("LSSTDeepCoaddWcs", LSSTDeepCoaddWCSBuilder())
