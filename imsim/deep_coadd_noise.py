import numpy as np
import galsim
from galsim.config import (GaussianNoiseBuilder, NoiseBuilder, GetAllParams,
                           GetInputObj, RegisterNoiseType)


__all__ = ["RubinDeepCoaddGaussianNoiseBuilder",
           "RubinDeepCoaddNoiseRealizationBuilder"]


class RubinDeepCoaddGaussianNoiseBuilder(GaussianNoiseBuilder):
    """
    Treat the variance of the coadd noise realization plane
    as Gaussian noise for a specified deep_coadd image.
    """

    def getNoiseVariance(self, config, base, full=False):
        req = { 'tract': int, 'patch': int, 'band': str }
        params = GetAllParams(config, base, req=req)[0]
        deep_coadds = galsim.config.GetInputObj('deep_coadds', config, base,
                                                'RubinDeepCoaddGaussianNoiseBuilder')
        return deep_coadds.getNoiseVar(params['tract'], params['patch'],
                                       params['band'])


class RubinDeepCoaddNoiseRealizationBuilder(NoiseBuilder):
    """
    Use one of the deep_coadd noise realization planes as the noise to
    add to the simulated image.
    """

    def addNoise(self, config, base, im, rng, current_var, draw_method, logger):
        req = { 'tract': int, 'patch': int, 'band': str }
        opt = { 'noise_plane': int }
        params = GetAllParams(config, base, req=req, opt=opt)[0]
        data_id = dict(tract=params['tract'], patch=params['patch'],
                       band=params['band'])
        noise_plane = params.get('noise_plane', 0)
        deep_coadds = GetInputObj('deep_coadds', config, base,
                                  'RubinDeepCoaddNoiseRealizationBuilder')
        im.array += (deep_coadds.get(data_id=data_id)
                     .noise_realizations[noise_plane].array)

    def getNoiseVariance(self, config, base, full=False):
        req = { 'tract': int, 'patch': int, 'band': str }
        ignore = { 'noise_plane': int }
        params = GetAllParams(config, base, req=req, ignore=ignore)[0]
        deep_coadds = GetInputObj('deep_coadds', config, base,
                                  'RubinDeepCoaddNoiseRealizationBuilder')
        return deep_coadds.getNoiseVar(params['tract'], params['patch'],
                                       params['band'])


RegisterNoiseType('RubinDeepCoaddGaussianNoise',
                  RubinDeepCoaddGaussianNoiseBuilder())
RegisterNoiseType('RubinDeepCoaddNoiseRealization',
                  RubinDeepCoaddNoiseRealizationBuilder())
