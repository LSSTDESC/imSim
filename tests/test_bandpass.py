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
        #print('Rubin_bp: ',repr(rubin_bp))

        # These are reasonably close.  Especially in the effective wavelength.
        # The limits of the actual filters are a bit different in some cases though from what was
        # estimated many years ago.  Especially for u and y bands.
        if band in 'uy':
            assert np.isclose(rubin_bp.blue_limit, galsim_bp.blue_limit, atol=30)
            assert np.isclose(rubin_bp.red_limit, galsim_bp.red_limit, atol=10)
            assert np.isclose(rubin_bp.effective_wavelength, galsim_bp.effective_wavelength, atol=5)
        else:
            assert np.isclose(rubin_bp.blue_limit, galsim_bp.blue_limit, atol=15)
            assert np.isclose(rubin_bp.red_limit, galsim_bp.red_limit, atol=8)
            assert np.isclose(rubin_bp.effective_wavelength, galsim_bp.effective_wavelength, atol=3)


if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
