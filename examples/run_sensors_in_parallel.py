from lsst.sims.GalSimInterface import SNRdocumentPSF
import desc.imsim

instcat = '/global/cscratch1/sd/jchiang8/imSim_prs/test_data/catalogs/imsim_cat_197356.txt'

phosim_commands = desc.imsim.metadata_from_file(instcat)

obs_md = desc.imsim.phosim_obs_metadata(phosim_commands)

psf = SNRdocumentPSF(obs_md.OpsimMetaData['FWHMgeom'])

image_simulator = desc.imsim.ImageSimulator(instcat, psf, numRows=30)

image_simulator.run(processes=3)
