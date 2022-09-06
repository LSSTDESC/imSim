import os
from pathlib import Path
import numpy as np
import json
import galsim
from imsim import SkyModel


DATA_DIR = Path(__file__).parent / 'data'


def test_sky_model():
    """
    Test the sky_model code for a random viable pointing, comparing to
    sky background values computed by hand using the
    rubin_sim.skybrightness code.
    """
    # Pointing info for observationId=11873 from the
    # baseline_v2.0_10yrs.db cadence file at
    # http://astro-lsst-01.astro.washington.edu:8080/
    ra = 54.9348753510528
    dec = -35.8385705255579
    skyCoord = galsim.CelestialCoord(ra*galsim.degrees, dec*galsim.degrees)
    mjd = 60232.3635999295
    exptime = 30.

    # Load expected sky bg values obtained from running the
    # rubin_sim.skybrightness code using sky_level_reference_values.py script.
    with open(os.path.join(DATA_DIR, 'reference_sky_levels.json')) as fobj:
        expected_sky_levels = json.load(fobj)

    for band in 'ugrizy':
        bandpass = galsim.Bandpass(f'LSST_{band}.dat', wave_type='nm')
        sky_model = SkyModel(exptime, mjd, bandpass)
        sky_level = sky_model.get_sky_level(skyCoord)
        np.testing.assert_approx_equal(sky_level, expected_sky_levels[band],
                                       significant=5)
