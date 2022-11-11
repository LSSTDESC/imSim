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
                    perturb:
                        shift:
                            M1: [1.e-3, 1.e-3, 0.0]
                            LSSTCamera: [0.0, 0.0, -1.e-3]
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturb:
                        -
                            shift:
                                M1: [1.e-3, 1.e-3, 0.0]
                                LSSTCamera: [0.0, 0.0, -1.e-3]
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturb:
                        -
                            shift:
                                M1: [1.e-3, 1.e-3, 0.0]
                        -
                            shift:
                                LSSTCamera: [0.0, 0.0, -1.e-3]
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
        )
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
                    perturb:
                        rotX:
                            M2: 1.e-3
                        rotY:
                            LSSTCamera: 1.e-3
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturb:
                        -
                            rotX:
                                M2: 1.e-3
                            rotY:
                                LSSTCamera: 1.e-3
            """
        ),
        textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    perturb:
                        -
                            rotX:
                                M2: 1.e-3
                        -
                            rotY:
                                LSSTCamera: 1.e-3
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
        )
        assert rotated == rotated_ref
