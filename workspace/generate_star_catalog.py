import os
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator
from lsst.sims.catUtils.exampleCatalogDefinitions import PhoSimCatalogPoint
from lsst.sims.catUtils.exampleCatalogDefinitions import DefaultPhoSimHeaderMap
from lsst.sims.catUtils.baseCatalogModels import StarObj

if __name__ == "__main__":

    db = StarObj(database='LSSTCATSIM', host='fatboy.phys.washington.edu',
                 port=1433, driver='mssql+pymssql')

    opsimdb = os.path.join('/Users', 'danielsf', 'physics', 'lsst_150412',
                           'Development', 'garage', 'OpSimData',
                           'minion_1016_sqlite.db')

    assert os.path.exists(opsimdb)
    obs_gen = ObservationMetaDataGenerator(database=opsimdb)
    obs_list = obs_gen.getObservationMetaData(obsHistID=230)
    obs = obs_list[0]
    obs.boundLength=0.5

    cat = PhoSimCatalogPoint(db, obs_metadata=obs)
    cat.phoSimHeaderMap = DefaultPhoSimHeaderMap
    cat.write_catalog('catalogs/star_catalog.txt', chunk_size=10000)

