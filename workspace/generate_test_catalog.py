import os
import copy
import numpy as np
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from lsst.sims.catUtils.exampleCatalogDefinitions import PhoSimCatalogPoint
from lsst.sims.catUtils.exampleCatalogDefinitions import PhoSimCatalogSersic2D
from lsst.sims.catUtils.exampleCatalogDefinitions import DefaultPhoSimHeaderMap
from lsst.sims.catalogs.definitions import parallelCatalogWriter
from lsst.sims.catalogs.definitions import InstanceCatalog
from lsst.sims.catUtils.mixins import AstrometryStars, CameraCoords
from lsst.sims.catUtils.mixins import AstrometryGalaxies, EBVmixin
from lsst.sims.catUtils.baseCatalogModels import StarObj
from lsst.sims.catUtils.baseCatalogModels import GalaxyBulgeObj
from lsst.sims.catalogs.decorators import cached


class FilteringMixin(object):

    @cached
    def get_sedFilepath(self):
        raw_output =  np.array([self.specFileMap[k] if k in self.specFileMap else None
                               for k in self.column_by_name('sedFilename')])

        if not hasattr(self, '_filter_rng'):
            self._filter_rng = np.random.RandomState(112)

        filter_val = self._filter_rng.random_sample(len(raw_output))

        return np.where(filter_val<0.001, raw_output, None)


class StellarTruthCatalog(FilteringMixin, AstrometryStars,
                         CameraCoords, InstanceCatalog):

    cannot_be_null = ['sedFilepath']

    column_outputs = ['uniqueId', 'x_pupil', 'y_pupil',
                      'sedFilepath', 'magNorm',
                      'raJ2000', 'decJ2000',
                      'properMotionRa',
                      'properMotionDec',
                      'parallax',
                      'radialVelocity',
                      'galacticAv',
                      'galacticRv']
    delimiter = ';'
    default_formats = {'f': '%.12e'}
    override_formats = {'magNorm': '%.7f'}
    default_columns = [('galacticRv', 3.1, float)]


class StarPhoSimCatalog(FilteringMixin, PhoSimCatalogPoint):
    pass


class GalaxyTestMixin(object):
    def _init_gal_test(self):
        if not hasattr(self, '_galaxy_test_rng'):
            self._galaxy_test_rng = np.random.RandomState(88)

    @cached
    def get_gamma1(self):
        self._init_gal_test()
        n_obj = len(self.column_by_name('raJ2000'))
        return self._galaxy_test_rng.random_sample(n_obj)

    @cached
    def get_gamma2(self):
        self._init_gal_test()
        n_obj = len(self.column_by_name('raJ2000'))
        return self._galaxy_test_rng.random_sample(n_obj)

    @cached
    def get_kappa(self):
        self._init_gal_test()
        n_obj = len(self.column_by_name('raJ2000'))
        return self._galaxy_test_rng.random_sample(n_obj)

    @cached
    def get_internalAv(self):
        self._init_gal_test()
        n_obj = len(self.column_by_name('raJ2000'))
        return self._galaxy_test_rng.random_sample(n_obj)

    @cached
    def get_internalRv(self):
        self._init_gal_test()
        n_obj = len(self.column_by_name('raJ2000'))
        return self._galaxy_test_rng.random_sample(n_obj)+2.1


class GalaxyTruthCatalog(FilteringMixin, GalaxyTestMixin,
                         EBVmixin, AstrometryGalaxies,
                         CameraCoords, InstanceCatalog):

    cannot_be_null = ['sedFilepath']

    column_outputs = ['uniqueId', 'x_pupil', 'y_pupil',
                      'sedFilepath', 'magNorm',
                      'raJ2000', 'decJ2000',
                      'redshift',
                      'gamma1',
                      'gamma2',
                      'kappa',
                      'galacticAv',
                      'galacticRv',
                      'internalAv',
                      'internalRv',
                      'minorAxis',
                      'majorAxis',
                      'positionAngle',
                      'sindex']
    delimiter = ';'
    default_formats = {'f': '%.12e'}
    override_formats = {'magNorm': '%.7f',
                        'redshift': '%.9f'}
    default_columns = [('galacticRv', 3.1, float)]


class GalaxyPhoSimCatalog(FilteringMixin, GalaxyTestMixin,
                          PhoSimCatalogSersic2D):
    pass


if __name__ == "__main__":

    star_db = StarObj(database='LSSTCATSIM',
                      host='fatboy.phys.washington.edu',
                      port=1433,
                      driver='mssql+pymssql')

    galaxy_db = GalaxyBulgeObj(database='LSSTCATSIM',
                               host='fatboy.phys.washington.edu',
                               port=1433,
                               driver='mssql+pymssql')

    opsimdb = os.path.join('/Users', 'danielsf', 'physics', 'lsst_150412',
                           'Development', 'garage', 'OpSimData',
                           'minion_1016_sqlite.db')

    assert os.path.exists(opsimdb)
    obs_gen = ObservationMetaDataGenerator(database=opsimdb)
    obs_list = obs_gen.getObservationMetaData(obsHistID=230)
    obs = obs_list[0]
    obs.boundLength=1.5

    phosim_header_map = copy.deepcopy(DefaultPhoSimHeaderMap)
    phosim_header_map['rawSeeing'] = ('rawSeeing', None)
    phosim_header_map['FWHMeff'] = ('FWHMeff', None)
    phosim_header_map['FWHMgeom'] = ('FWHMgeom',None)

    phosim_cat = StarPhoSimCatalog(star_db, obs_metadata=obs)
    phosim_cat.phoSimHeaderMap = phosim_header_map

    truth_cat = StellarTruthCatalog(star_db, obs_metadata=obs)

    cat_dict = {'catalogs/phosim_stars.txt': phosim_cat,
                'catalogs/truth_stars.txt': truth_cat}

    parallelCatalogWriter(cat_dict, chunk_size=10000)

    print('\n\ndone with stars\n\n')

    obs.boundLength = 0.5

    truth_cat = GalaxyTruthCatalog(galaxy_db, obs_metadata=obs)
    phosim_cat = GalaxyPhoSimCatalog(galaxy_db, obs_metadata=obs)
    phosim_cat.phoSimHeaderMap = phosim_header_map

    cat_dict = {'catalogs/phosim_galaxies.txt': phosim_cat,
                'catalogs/truth_galaxies.txt': truth_cat}

    parallelCatalogWriter(cat_dict, chunk_size=10000)
