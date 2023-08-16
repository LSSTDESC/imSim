import numpy as np
import galsim
from imsim import RubinBandpass

def test_rubin_bandpass():
    """Check the RubinBandpass functions to check that they are similar to GalSim lsst_*.dat files.
    """
    for band in "ugrizy":
        rubin_bp = RubinBandpass(band)
        galsim_bp = galsim.Bandpass(f"LSST_{band}.dat", wave_type='nm').thin()
        print(f'Band = {band}')
        print('Rubin_bp: ',rubin_bp.blue_limit, rubin_bp.effective_wavelength, rubin_bp.red_limit)
        print('galsim_bp: ',galsim_bp.blue_limit, galsim_bp.effective_wavelength, galsim_bp.red_limit)
        # These are quite close.  The only exception is the blue_limit of the u band, where the
        # old GalSim bandpass is too red.
        if band == 'u':
            assert np.isclose(rubin_bp.blue_limit, galsim_bp.blue_limit, atol=20)
        else:
            assert np.isclose(rubin_bp.blue_limit, galsim_bp.blue_limit, atol=1)
        assert np.isclose(rubin_bp.red_limit, galsim_bp.red_limit, atol=1)
        assert np.isclose(rubin_bp.effective_wavelength, galsim_bp.effective_wavelength, atol=3)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
