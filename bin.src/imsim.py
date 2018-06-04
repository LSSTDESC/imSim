#!/usr/bin/env python
"""
This is the imSim program, used to drive GalSim to simulate the LSST.  Written
for the DESC collaboration and LSST project.  This version of the program can
read phoSim instance files as is. It leverages the LSST Sims GalSim interface
code found in sims_GalSimInterface.
"""
import argparse
import desc.imsim

parser = argparse.ArgumentParser()
parser.add_argument('file', help="The instance catalog")
parser.add_argument('-n', '--numrows', default=None, type=int,
                    help="Read the first numrows of the file.")
parser.add_argument('--outdir', type=str, default='fits',
                    help='Output directory for eimage file')
parser.add_argument('--sensors', type=str, default=None,
                    help='Sensors to simulate, e.g., "R:2,2 S:1,1^R:2,2 S:1,0".' +
                    'If None, then simulate all sensors with sources on them')
parser.add_argument('--config_file', type=str, default=None,
                    help="Config file. If None, the default config will be used.")
parser.add_argument('--log_level', type=str,
                    choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'],
                    default='INFO', help='Logging level. Default: "INFO"')
parser.add_argument('--psf', type=str, default='Kolmogorov',
                    choices=['DoubleGaussian', 'Kolmogorov', 'Atmospheric'],
                    help="PSF model to use.")
parser.add_argument('--disable_sensor_model', default=False,
                    action='store_true',
                    help='disable sensor effects')
parser.add_argument('--file_id', type=str, default=None,
                    help='ID string to use for checkpoint and centroid filenames.')
parser.add_argument('--seed', type=int, default=267,
                    help='integer used to seed random number generator')
parser.add_argument('--processes', type=int, default=1,
                    help='number of processes to use in multiprocessing mode')
args = parser.parse_args()

commands = desc.imsim.metadata_from_file(args.file)

obs_md = desc.imsim.phosim_obs_metadata(commands)

psf = desc.imsim.make_psf(args.psf, obs_md, log_level=args.log_level)

sensor_list = args.sensors.split('^') if args.sensors is not None \
              else args.sensors

apply_sensor_model = not args.disable_sensor_model

image_simulator \
    = desc.imsim.ImageSimulator(args.file, psf,
                                numRows=args.numrows,
                                config=args.config_file,
                                seed=args.seed,
                                outdir=args.outdir,
                                sensor_list=sensor_list,
                                apply_sensor_model=apply_sensor_model,
                                file_id=args.file_id,
                                log_level=args.log_level)

image_simulator.run(processes=args.processes)
