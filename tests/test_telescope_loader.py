import yaml
import imsim
import galsim
import textwrap
import batoid


def test_config_shift():
    """Test that we can shift a telescope.
    """
    telescope = imsim.load_telescope("LSST_r.yaml")
    shifted_ref = (telescope
        .withLocallyShiftedOptic('M1', [1e-3, 1e-3, 0.0])
        .withLocallyShiftedOptic('LSSTCamera', [0.0, 0.0, -1e-3])
    )

    shifted_configs = [
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        M1:
                            shift: [1.e-3, 1.e-3, 0.0]
                        LSSTCamera:
                            shift: [0.0, 0.0, -1.e-3]
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        -
                            M1:
                                shift: [1.e-3, 1.e-3, 0.0]
                            LSSTCamera:
                                shift: [0.0, 0.0, -1.e-3]
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        -
                            M1:
                                shift: [1.e-3, 1.e-3, 0.0]
                        -
                            LSSTCamera:
                                shift: [0.0, 0.0, -1.e-3]
            """
        ),
    ]

    for config in shifted_configs:
        config = yaml.safe_load(config)
        galsim.config.ProcessInput(config)
        shifted = galsim.config.GetInputObj(
            'telescope',
            config['input']['telescope'],
            config,
            'telescope'
        )['base']
        assert shifted == shifted_ref


def test_config_rot():
    """Test that we can rotate a telescope.
    """
    telescope = imsim.load_telescope("LSST_r.yaml")
    rotated_ref = (telescope
        .withLocallyRotatedOptic('M2', batoid.RotX(1e-3))
        .withLocallyRotatedOptic('LSSTCamera', batoid.RotY(1e-3))
    )

    rotated_configs = [
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        M2:
                            rotX: 1.e-3
                        LSSTCamera:
                            rotY: 1.e-3
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        -
                            M2:
                                rotX: 1.e-3
                            LSSTCamera:
                                rotY: 1.e-3
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        -
                            M2:
                                rotX: 1.e-3
                        -
                            LSSTCamera:
                                rotY: 1.e-3
            """
        ),
    ]

    for config in rotated_configs:
        config = yaml.safe_load(config)
        galsim.config.ProcessInput(config)
        rotated = galsim.config.GetInputObj(
            'telescope',
            config['input']['telescope'],
            config,
            'telescope'
        )['base']
        assert rotated == rotated_ref


def test_config_zernike_perturbation():
    """Test that we can perturb an optical surface with Zernikes.
    """
    telescope = imsim.load_telescope("LSST_r.yaml")
    ref = (telescope
        .withSurface(
            'M1',
            batoid.Sum([
                telescope['M1'].surface,
                batoid.Zernike(
                    [0.0]*4 + [1e-7],
                    R_outer=telescope['M1'].obscuration.original.outer,
                    R_inner=telescope['M1'].obscuration.original.inner
                )
            ])
        )
        .withSurface(
            'M2',
            batoid.Sum([
                telescope['M2'].surface,
                batoid.Zernike(
                    [0.0]*5 + [2e-7],
                    R_outer=telescope['M2'].obscuration.original.outer,
                    R_inner=telescope['M2'].obscuration.original.inner
                )
            ])
        )
    )

    perturb_configs = [
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        M1:
                            Zernike:
                                coef: [0.0, 0.0, 0.0, 0.0, 1.e-7]  # Z4
                        M2:
                            Zernike:
                                coef: [0.0, 0.0, 0.0, 0.0, 0.0, 2.e-7]  # Z5
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        -
                            M1:
                                Zernike:
                                    coef: [0.0, 0.0, 0.0, 0.0, 1.e-7]  # Z4
                            M2:
                                Zernike:
                                    coef: [0.0, 0.0, 0.0, 0.0, 0.0, 2.e-7]  # Z5
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        -
                            M1:
                                Zernike:
                                    coef: [0.0, 0.0, 0.0, 0.0, 1.e-7]  # Z4
                        -
                            M2:
                                Zernike:
                                    coef: [0.0, 0.0, 0.0, 0.0, 0.0, 2.e-7]  # Z5
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        M1:
                            Zernike:
                                idx: 4
                                val: 1.e-7
                        M2:
                            Zernike:
                                idx: 5
                                val: 2.e-7
            """
        ),
    ]

    for config in perturb_configs:
        config = yaml.safe_load(config)
        galsim.config.ProcessInput(config)
        perturbed = galsim.config.GetInputObj(
            'telescope',
            config['input']['telescope'],
            config,
            'telescope'
        )['base']
        assert perturbed == ref

    # Test that we can set more than one Zernike at a time,
    # and that we can override the default R_outer and R_inner.
    telescope = imsim.load_telescope("LSST_r.yaml")
    ref = (telescope
        .withSurface(
            'M1',
            batoid.Sum([
                telescope['M1'].surface,
                batoid.Zernike(
                    [0.0]*4 + [1e-7, 3e-7],
                    R_outer=1.2,
                    R_inner=0.6
                )
            ])
        )
    )

    perturbed_configs = [
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        M1:
                            Zernike:
                                coef: [0.0, 0.0, 0.0, 0.0, 1.e-7, 3.e-7]  # Z4, Z5
                                R_outer: 1.2
                                R_inner: 0.6
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturbations:
                        M1:
                            Zernike:
                                idx: [4, 5]
                                val: [1.e-7, 3.e-7]
                                R_outer: 1.2
                                R_inner: 0.6
            """
        )
    ]

    for config in perturbed_configs:
        config = yaml.safe_load(config)
        galsim.config.ProcessInput(config)
        perturbed = galsim.config.GetInputObj(
            'telescope',
            config['input']['telescope'],
            config,
            'telescope'
        )['base']
        assert perturbed == ref
