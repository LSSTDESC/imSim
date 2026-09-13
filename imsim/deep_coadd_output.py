import galsim
from galsim.config import (OutputBuilder, RegisterOutputType, ParseValue,
                           GetAllParams, GetInputObj, BuildImage,
                           eval_base_variables)


__all__ = ['RubinDeepCoaddOutputBuilder']


class RubinDeepCoaddOutputBuilder(OutputBuilder):

    _added_eval_base_variables = False

    def setup(self, config, base, file_num, logger):
        coadd_num = ParseValue(config, 'coadd_num', base, int)[0]
        if not self._added_eval_base_variables:
            eval_base_variables.append('coadd_num')
            self._added_eval_base_variables = True
        base['coadd_num'] = coadd_num
        deep_coadds = GetInputObj('deep_coadd', config, base,
                                  'RubinDeepCoaddOutputBuilder')
        data_id = deep_coadds.data_ids[coadd_num]
        patch = deep_coadds.skymap[data_id['tract']][data_id['patch']]
        bbox = patch.getOuterBBox()
        base['det_xsize'] = bbox.width
        base['det_ysize'] = bbox.height

    def getNFiles(self, config, base, logger=None):
        if 'nfiles' in config:
            return ParseValue(config, 'nfiles', base, int)[0]
        else:
            deep_coadds = GetInputObj('deep_coadd', config, base,
                                      'RubinDeepCoaddOutputBuilder')
            return len(deep_coadds.data_ids)

    def buildImages(self, config, base, file_num, image_num, obj_num,
                    ignore, logger):
        ignore += [ 'file_name', 'dir', 'nfiles', 'coadd_num' ]
        params, _ = GetAllParams(config, base, ignore=ignore)

        image = BuildImage(base, image_num, obj_num, logger=logger)
        image.header = galsim.FitsHeader()
        image.wcs.header = {}
        return [ image ]


RegisterOutputType('RubinDeepCoaddOutput', RubinDeepCoaddOutputBuilder())
