import textwrap
from tempfile import TemporaryDirectory
from pathlib import Path

import yaml
import numpy as np
from astropy.io import fits
import galsim

import imsim


def test_sag():
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
                sag:
                    file_name: sag.fits
                    nx: 127
            """
        )

        config = yaml.safe_load(config)
        galsim.config.ProcessInput(config)
        galsim.config.SetupExtraOutput(config)
        galsim.config.SetupConfigFileNum(config, 0, 0, 0)
        galsim.config.extra.WriteExtraOutputs(config, None)

        telescope = galsim.config.GetInputObj(
            'telescope',
            config,
            config,
            'opd'
        ).fiducial

        # Check the contents of the output file against Zemax references
        fn = Path(d) / "sag.fits"

        hdulist = fits.open(fn)
        for hdu in hdulist:
            hdr = hdu.header
            sag = hdu.data
            optic = telescope[hdr['name']]

            # Verify sag
            xs = np.arange(hdr['NAXIS1'])*hdr['dx']
            ys = np.arange(hdr['NAXIS2'])*hdr['dy']
            xs -= np.mean(xs)
            ys -= np.mean(ys)
            xx, yy = np.meshgrid(xs, ys)
            ww = ~np.isnan(sag)
            np.testing.assert_allclose(
                sag[ww],
                optic.surface.sag(xx[ww], yy[ww]),
            )

            # Verify coordinate system
            np.testing.assert_allclose(
                [hdr['x0'], hdr['y0'], hdr['z0']],
                optic.coordSys.origin
            )

            np.testing.assert_allclose(
                [[hdr['R00'], hdr['R01'], hdr['R02']],
                 [hdr['R10'], hdr['R11'], hdr['R12']],
                 [hdr['R20'], hdr['R21'], hdr['R22']]],
                optic.coordSys.rot
            )
            assert hdu.header['telescop'] == "LSST"
            # Test GalSim wrote reasonable WCS keywords
            scale = xs[1] - xs[0]
            np.testing.assert_allclose(
                hdu.header['GS_SCALE'], scale,
                rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(
                hdu.header['CD1_1'], scale,
                rtol=1e-10, atol=1e-10
            )
            np.testing.assert_allclose(
                hdu.header['CD2_2'], scale,
                rtol=1e-10, atol=1e-10
            )
            assert hdu.header['CD1_2'] == 0.0
            assert hdu.header['CD2_1'] == 0.0
