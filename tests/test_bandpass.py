import numpy as np
import galsim
from imsim import RubinBandpass
from unittest import mock
from imsim_test_helpers import CaptureLog

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


        # If RUBIN_SIM_DATA isn't availble, then RubinBandpass reverts to the galsim files,
        # and everything should match exactly.
        with mock.patch('os.getenv', return_value=''):
            with CaptureLog() as cl:
                alt_rubin_bp = RubinBandpass(band, logger=cl.logger)
        assert "Using the old bandpass files" in cl.output
        print('Alt_Rubin_bp: ',alt_rubin_bp.blue_limit, alt_rubin_bp.effective_wavelength,
              alt_rubin_bp.red_limit)
        galsim_bp = galsim.Bandpass(f"LSST_{band}.dat", wave_type='nm')
        galsim_bp = galsim_bp.truncate(relative_throughput=1.e-3)
        galsim_bp = galsim_bp.thin()
        galsim_bp = galsim_bp.withZeropoint('AB')
        assert alt_rubin_bp == galsim_bp



if __name__ == "__main__":
    testfns = [v for k, v in vars().items() if k[:5] == 'test_' and callable(v)]
    for testfn in testfns:
        testfn()
