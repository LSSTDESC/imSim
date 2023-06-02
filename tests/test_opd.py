import textwrap
from tempfile import TemporaryDirectory
from pathlib import Path

import yaml
import numpy as np
from astropy.io import fits
import galsim

import imsim


DATA_DIR = Path(__file__).parent / 'data'


def test_opd():
    with TemporaryDirectory() as d:
        config = textwrap.dedent(
            f"""
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        M2:
                            shift: [100.e-6, 0.0, 0.0]
            output:
                dir: {d}
                opd:
                    file_name: opd.fits
                    wavelength: 694.0  # nm
                    projection: zemax
                    fields:
                        - {{thx: 1.121 deg, thy: 1.231 deg}}
            """
        )

        config = yaml.safe_load(config)
        galsim.config.ProcessInput(config)
        galsim.config.SetupExtraOutput(config)
        # Need to mock file_num for following method to be happy:
        config['file_num'] = 0
        galsim.config.extra.WriteExtraOutputs(config, None)

        # Check the contents of the output file against Zemax references
        fn = Path(d) / "opd.fits"
        hdu = fits.open(fn)[0]

    # First, check annular Zernike coefficients in fits header
    with open(
        DATA_DIR / "LSST_AZ_v3.3_c3_f6_w3_M2_dx_100um.txt",
        encoding='utf-16-le'
    ) as f:
        zk = np.genfromtxt(f, skip_header=32, usecols=(2))
    for j in range(1, 29):
        np.testing.assert_allclose(
            hdu.header[f'AZ_{j:03d}'],
            zk[j-1]*694.0,  # waves to nm
            atol=0.2,  # nm
            rtol=1e-3
        )

    # Next, check the OPD image data
    with open(
        DATA_DIR / "LSST_WF_v3.3_c3_f6_w3_M2_dx_100um.txt",
        encoding='utf-16-le'
    ) as f:
        opd = np.genfromtxt(f, skip_header=16)
    opd = np.flipud(opd)  # Zemax has opposite y-axis convention
    opd = opd[1:, 1:]  # Zemax has 1-pixel border of zeros
    opd *= 694.0  # waves to nm

    # Zemax obnoxiously uses 0.0 for vignetted pixels; batoid/imSim uses NaN
    # Also, the Zemax model includes the spider.
    # So only compare non-zero and non-nan pixels.

    w = opd != 0.0
    w &= ~np.isnan(hdu.data)

    np.testing.assert_allclose(
        hdu.data[w],
        opd[w],
        atol=0.01,  # nm
        rtol=1e-5
    )


if __name__ == '__main__':
    test_opd()
