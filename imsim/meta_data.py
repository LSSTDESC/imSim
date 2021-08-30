
import os

imsim_dir = os.path.split(os.path.realpath(__file__))[0]
install_dir = os.path.split(imsim_dir)[0]

if 'IMSIM_DATA_DIR' in os.environ: # pragma: no cover
    data_dir = os.environ['IMSIM_DATA_DIR']
else:
    data_dir = os.path.join(install_dir, 'imsim', 'data')
