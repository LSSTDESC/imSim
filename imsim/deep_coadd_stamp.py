import galsim
from galsim.config import (ParseValue, ParseWorldPos,
                           StampBuilder, RegisterStampType, GetAllParams)


__all__ = ["RubinDeepCoaddStampBuilder"]


class RubinDeepCoaddStampBuilder(StampBuilder):

    def setup(self, config, base, xsize, ysize, ignore, logger):

        xsize, ysize, image_pos, world_pos \
            = super().setup(config, base, xsize, ysize, ignore, logger)
        return xsize, ysize, image_pos, world_pos

#        if 'image_pos' in config:
#            image_pos = ParseValue(config, 'image_pos', base,
#                                   galsim.PositionD)[0]
#        else:
#            image_pos = None
#
#        if 'world_pos' in config:
#            world_pos = ParseWorldPos(config, 'world_pos', base, logger)
#        else:
#            world_pos = None


RegisterStampType('RubinDeepCoadd', RubinDeepCoaddStampBuilder())
