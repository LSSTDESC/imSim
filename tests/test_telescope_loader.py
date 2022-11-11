import yaml
import imsim
import galsim
import textwrap

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
