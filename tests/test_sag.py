import textwrap
from tempfile import TemporaryDirectory
from pathlib import Path

import yaml
import numpy as np
from astropy.io import fits
import galsim

import imsim


def test_sag():
    for nx in [127, 128]:  # Make sure works with both odd and even nx
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
                        nx: {nx}
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
            hdu_list = fits.open(fn)

            for ihdu in range(len(hdu_list)):
                img = galsim.fits.read(hdu_list=hdu_list, hdu=ihdu, read_header=True)
                sag = img.array
                hdr = img.header
                optic = telescope[hdr['name']]

                # Verify sag
                x, y = img.get_pixel_centers()
                xx, yy = img.wcs.toWorld(x, y)
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
                assert hdr['telescop'] == "LSST"
                # Test GalSim wrote reasonable WCS keywords
                scale = xx[0,1] - xx[0,0]
                np.testing.assert_allclose(
                    hdr['GS_SCALE'], scale,
                    rtol=1e-10, atol=1e-10
                )
                np.testing.assert_allclose(
                    hdr['CD1_1'], scale,
                    rtol=1e-10, atol=1e-10
                )
                np.testing.assert_allclose(
                    hdr['CD2_2'], scale,
                    rtol=1e-10, atol=1e-10
                )
                assert hdr['CD1_2'] == 0.0
                assert hdr['CD2_1'] == 0.0


if __name__ == '__main__':
    test_sag()
