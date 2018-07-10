import warnings
from lsst.sims.GalSimInterface import SNRdocumentPSF
import desc.imsim

instcat = '/global/cscratch1/sd/jchiang8/imSim_prs/test_data/catalogs/imsim_cat_197356.txt'

phosim_commands = desc.imsim.metadata_from_file(instcat)

obs_md = desc.imsim.phosim_obs_metadata(phosim_commands)

psf = SNRdocumentPSF(obs_md.OpsimMetaData['FWHMgeom'])

#numRows = None
numRows = 1000
image_simulator = desc.imsim.ImageSimulator(instcat, psf, numRows=numRows)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'Automatic n_photons', UserWarning)
    image_simulator.run(processes=3)
