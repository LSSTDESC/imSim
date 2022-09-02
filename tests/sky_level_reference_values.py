"""Script to generate reference values for test_sky_model.py"""
import warnings
import numpy as np
import json
import galsim
from rubin_sim import skybrightness


RUBIN_AREA = 0.25 * np.pi * 649**2  # cm^2

ra = 54.9348753510528
dec = -35.8385705255579
rottelpos = 341.776422048124
mjd = 60232.3635999295
exptime = 30.
band = 'r'
camera_name = 'LsstCam'

sky_model = skybrightness.SkyModel()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    sky_model.setRaDecMjd(ra, dec, mjd, degrees=True)

sky_levels = {}
for band in 'ugrizy':
    bandpass = galsim.Bandpass(f'LSST_{band}.dat', wave_type='nm')
    wave, spec = sky_model.returnWaveSpec()
    lut = galsim.LookupTable(wave, spec[0])
    sed = galsim.SED(lut, wave_type='nm', flux_type='flambda')
    sky_levels[band] = sed.calculateFlux(bandpass)*RUBIN_AREA*exptime

with open('data/reference_sky_levels.json', 'w') as fobj:
    json.dump(sky_levels, fobj)
