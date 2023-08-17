import textwrap
from tempfile import TemporaryDirectory
from pathlib import Path

import yaml
import numpy as np
from astropy.io import fits
import galsim

import imsim


DATA_DIR = Path(__file__).parent / 'data'


def test_opd_zemax():
    for nx in [255, 256]:
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
                        nx: {nx}
                        fields:
                            - {{thx: 1.121 deg, thy: 1.231 deg}}
                """
            )

            config = yaml.safe_load(config)
            galsim.config.ProcessInput(config)
            galsim.config.SetupExtraOutput(config)
            galsim.config.SetupConfigFileNum(config, 0, 0, 0)
            galsim.config.extra.WriteExtraOutputs(config, None)

            # Check the contents of the output file against Zemax references
            fn = Path(d) / "opd.fits"
            img = galsim.fits.read(str(fn), read_header=True)
            opd = img.array
            hdr = img.header

        if nx == 255:
            # Compare computed annular Zernike coefficients in OPD header to values
            # from Zemax.  For the Zemax run, I used the design file from
            # https://docushare.lsstcorp.org/docushare/dsweb/View/Collection-2097
            # specifically, the file
            # LSST_Ver_3.3_Baseline_Design_Spiders_Baffles.ZMX
            # with the following modifications:
            # - I set the stop surface to the baffle just above M1
            # - I decentered M2 by 100 microns in x
            with open(
                DATA_DIR / "LSST_AZ_v3.3_c3_f6_w3_M2_dx_100um.txt",
                encoding='utf-16-le'
            ) as f:
                zk = np.genfromtxt(f, skip_header=32, usecols=(2))
            for j in range(1, 29):
                np.testing.assert_allclose(
                    hdr[f'AZ_{j:03d}'],
                    zk[j-1]*694.0,  # waves to nm
                    atol=0.2,  # nm
                    rtol=1e-3
                )

            # Compare the OPD image to Zemax reference.  The setup is the same as above.
            with open(
                DATA_DIR / "LSST_WF_v3.3_c3_f6_w3_M2_dx_100um.txt",
                encoding='utf-16-le'
            ) as f:
                opd_zemax = np.genfromtxt(f, skip_header=16)
            opd_zemax = np.flipud(opd_zemax)  # Zemax has opposite y-axis convention
            opd_zemax = opd_zemax[1:, 1:]  # Zemax has 1-pixel border of zeros
            opd_zemax *= 694.0  # waves to nm

            # Zemax obnoxiously uses 0.0 for vignetted pixels; batoid/imSim uses NaN
            # Also, the Zemax model includes the spider.
            # So only compare non-zero and non-nan pixels.

            w = opd_zemax != 0.0
            w &= ~np.isnan(opd)

            np.testing.assert_allclose(
                opd[w],
                opd_zemax[w],
                atol=0.01,  # nm
                rtol=1e-5
            )

        # Verify that other data made it into the header
        x, y = img.get_pixel_centers()
        xx, yy = img.wcs.toWorld(x, y)
        assert hdr['units'] == 'nm'
        scale = xx[0,1] - xx[0,0]
        np.testing.assert_allclose(
            hdr['dx'], scale,
            rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            hdr['dy'], scale,
            rtol=1e-10, atol=1e-10
        )
        assert hdr['thx'] == 1.121
        assert hdr['thy'] == 1.231
        assert hdr['r_thx'] == hdr['thx']  # rotator not engaged
        assert hdr['r_thy'] == hdr['thy']
        assert hdr['wavelen'] == 694.0
        assert hdr['prjct'] == 'zemax'
        assert hdr['sph_ref'] == 'chief'
        assert hdr['eps'] == 0.612
        assert hdr['jmax'] == 28
        assert hdr['telescop'] == "LSST"
        # Test GalSim wrote reasonable WCS keywords
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


def test_opd_wavelength():
    with TemporaryDirectory() as d:
        # Write out an OPD at the bandpass effective wavelength
        config = textwrap.dedent(
            f"""
            input:
                telescope:
                    file_name: LSST_r.yaml
            image:
                bandpass:
                    file_name: LSST_r.dat
                    wave_type: nm
            output:
                dir: {d}
                opd:
                    file_name: opd1.fits
                    fields:
                        - {{thx: 1.121 deg, thy: 1.231 deg}}
            """
        )
        config = yaml.safe_load(config)
        galsim.config.ProcessInput(config)
        galsim.config.SetupExtraOutput(config)
        galsim.config.SetupConfigFileNum(config, 0, 0, 0)
        galsim.config.extra.WriteExtraOutputs(config, None)
        fn = Path(d) / "opd1.fits"
        hdu1 = fits.open(fn)[0]

        # Now repeat but remove the bandpass and explicitly specify the
        # wavelength
        config = textwrap.dedent(
            f"""
            input:
                telescope:
                    file_name: LSST_r.yaml
            output:
                dir: {d}
                opd:
                    file_name: opd2.fits
                    fields:
                        - {{thx: 1.121 deg, thy: 1.231 deg}}
                    wavelength: {hdu1.header['wavelen']}
            """
        )
        config = yaml.safe_load(config)
        galsim.config.ProcessInput(config)
        galsim.config.SetupExtraOutput(config)
        galsim.config.SetupConfigFileNum(config, 0, 0, 0)
        galsim.config.extra.WriteExtraOutputs(config, None)
        fn = Path(d) / "opd2.fits"
        hdu2 = fits.open(fn)[0]

        np.testing.assert_allclose(
            hdu1.data, hdu2.data,
            rtol=1e-16, atol=1e-16
        )
        for j in range(1, 29):
            np.testing.assert_allclose(
                hdu1.header[f'AZ_{j:03d}'], hdu2.header[f'AZ_{j:03d}'],
                atol=1e-16, rtol=1e-16
            )

        # It's an error to not have either bandpass or wavelength
        config = textwrap.dedent(
            f"""
            input:
                telescope:
                    file_name: LSST_r.yaml
            output:
                dir: {d}
                opd:
                    file_name: opd3.fits
                    fields:
                        - {{thx: 1.121 deg, thy: 1.231 deg}}
            """
        )
        config = yaml.safe_load(config)
        with np.testing.assert_raises(ValueError):
            galsim.config.ProcessInput(config)
            galsim.config.SetupExtraOutput(config)


def test_opd_phase():
    """Test that we can add a phase that zeros out the OPD"""
    with TemporaryDirectory() as d:
        # Write out an OPD at the bandpass effective wavelength
        config = textwrap.dedent(
            f"""
            input:
                telescope:
                    file_name: LSST_r.yaml
            image:
                bandpass:
                    file_name: LSST_r.dat
                    wave_type: nm
            output:
                dir: {d}
                opd:
                    file_name: opd1.fits
                    fields:
                        - {{thx: 1.121 deg, thy: 1.231 deg}}
                    eps: 0.612
            """
        )
        config = yaml.safe_load(config)
        galsim.config.ProcessInput(config)
        galsim.config.SetupExtraOutput(config)
        galsim.config.SetupConfigFileNum(config, 0, 0, 0)
        galsim.config.extra.WriteExtraOutputs(config, None)
        fn = Path(d) / "opd1.fits"
        hdr = fits.getheader(fn)

        zk = np.zeros(29)
        for i in range(1, 29):
            zk[i] = hdr[f'AZ_{i:03d}']

        # Write a new config with phases to zero out the OPD
        config = textwrap.dedent(
            f"""
            input:
                telescope:
                    file_name: LSST_r.yaml
                    fea:
                        extra_zk:
                            zk: {(zk*1e-9).tolist()}
                            eps: 0.612
            image:
                bandpass:
                    file_name: LSST_r.dat
                    wave_type: nm
            output:
                dir: {d}
                opd:
                    file_name: opd2.fits
                    fields:
                        - {{thx: 1.121 deg, thy: 1.231 deg}}
            """
        )
        config = yaml.safe_load(config)
        galsim.config.ProcessInput(config)
        galsim.config.SetupExtraOutput(config)
        galsim.config.SetupConfigFileNum(config, 0, 0, 0)
        galsim.config.extra.WriteExtraOutputs(config, None)
        fn2 = Path(d) / "opd2.fits"
        hdr2 = fits.getheader(fn2)

        zk2 = np.zeros(29)
        for i in range(1, 29):
            zk2[i] = hdr2[f'AZ_{i:03d}']

        # Tricky to zero-out tip and tilt since these also depend strongly on
        # the position of the chief ray.  But other zks should be close to zero
        # now.  Test to 0.03 nm, which is much smaller than the initial zks
        # which were in the ~10-100 nm range.
        np.testing.assert_allclose(
            zk2[4:], 0.0,
            atol=0.03, rtol=0.0
        )


if __name__ == '__main__':
    test_opd_zemax()
    test_opd_wavelength()
    test_opd_phase()
