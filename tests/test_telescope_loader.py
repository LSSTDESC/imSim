import yaml
import imsim
import galsim
import textwrap
import batoid
import numpy as np


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
        ).fiducial
        assert shifted == shifted_ref

    # Test out exceptions
    config = yaml.safe_load(textwrap.dedent(
        """
        input:
            telescope:
                file_name: LSST_r.yaml
                perturbations:
                    M1:
                        shift: [1.e-3, 1.e-3]
        """
    ))
    with np.testing.assert_raises(ValueError):
        galsim.config.ProcessInput(config)

    config = yaml.safe_load(textwrap.dedent(
        """
        input:
            telescope:
                file_name: LSST_r.yaml
                perturbations:
                    M1:
                        shift: ["a", "b", "c"]
        """
    ))
    with np.testing.assert_raises(ValueError):
        galsim.config.ProcessInput(config)


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
                            rotX: 1.e-3 rad
                        LSSTCamera:
                            rotY: 1.e-3 rad
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
                                rotX: 1.e-3 rad
                            LSSTCamera:
                                rotY: 1.e-3 rad
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
                                rotX: 1.e-3 rad
                        -
                            LSSTCamera:
                                rotY: 1.e-3 rad
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
        ).fiducial
        assert rotated == rotated_ref

    # Check that we can rotate the camera using rotTelPos
    config = yaml.safe_load(textwrap.dedent(
        """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    rotTelPos: 30 deg
        """
    ))
    galsim.config.ProcessInput(config)
    rotated = galsim.config.GetInputObj(
        'telescope',
        config['input']['telescope'],
        config,
        'telescope'
    ).fiducial
    assert (
        rotated ==
        telescope.withLocallyRotatedOptic(
            'LSSTCamera',
            batoid.RotZ(np.deg2rad(30))
        )
    )


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
        ).fiducial
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
        ).fiducial
        assert perturbed == ref


def test_config_fea():
    """Test that we can use batoid_rubin package to add FEA perturbations.
    """

    # It's difficult to directly add a bending mode or printthrough to a
    # surface figure, but we can at least test that we can manipulate rigid
    # body degrees-of-freedom either directly or through the FEA interface.

    # First standard degree of freedom is M2 -dz in microns
    telescope = imsim.load_telescope("LSST_r.yaml")
    ref = (telescope
        .withGloballyShiftedOptic(
            'M2',
            [0, 0, 1e-6]
        )
    )

    perturb_config = textwrap.dedent(
        f"""
        input:
            telescope:
                file_name: LSST_r.yaml
                fea:
                    aos_dof:
                        dof: '$[-1.0]+[0.0]*49'
        """
    )

    config = yaml.safe_load(perturb_config)
    galsim.config.ProcessInput(config)
    perturbed = galsim.config.GetInputObj(
        'telescope',
        config['input']['telescope'],
        config,
        'telescope'
    ).fiducial
    assert ref == perturbed

    # Next is M2 -dx and +dy
    telescope = imsim.load_telescope("LSST_r.yaml")
    ref = (telescope
        .withGloballyShiftedOptic(
            'M2',
            [2e-6, 3e-6, 0]
        )
    )
    dof = [0.0, -2.0, +3.0] + [0.0]*47
    dofstr = ",".join(str(s) for s in dof)

    perturb_config = textwrap.dedent(
        f"""
        input:
            telescope:
                file_name: LSST_r.yaml
                fea:
                    aos_dof:
                        dof: [{dofstr}]
        """
    )

    config = yaml.safe_load(perturb_config)
    galsim.config.ProcessInput(config)
    perturbed = galsim.config.GetInputObj(
        'telescope',
        config['input']['telescope'],
        config,
        'telescope'
    ).fiducial
    assert ref == perturbed

    # Next is M2 -Rx and Ry in arcsec
    telescope = imsim.load_telescope("LSST_r.yaml")
    ref = (telescope
        .withLocallyRotatedOptic(
            'M2',
            batoid.RotX(np.deg2rad(2./3600))@batoid.RotY(np.deg2rad(3./3600))
        )
    )
    dof = [0.0]*3 + [-2.0, 3.0] + [0.0]*45
    dofstr = ",".join(str(s) for s in dof)

    perturb_config = textwrap.dedent(
        f"""
        input:
            telescope:
                file_name: LSST_r.yaml
                fea:
                    aos_dof:
                        dof: [{dofstr}]
        """
    )

    config = yaml.safe_load(perturb_config)
    galsim.config.ProcessInput(config)
    perturbed = galsim.config.GetInputObj(
        'telescope',
        config['input']['telescope'],
        config,
        'telescope'
    ).fiducial
    # Just get close for this one since it depends slightly on order of applying
    # X,Y rotations.
    np.testing.assert_allclose(
        ref['M2'].coordSys.rot,
        perturbed['M2'].coordSys.rot,
        atol=1e-12, rtol=0
    )

    # Repeat of above for Camera shifts and tilts.
    telescope = imsim.load_telescope("LSST_r.yaml")
    ref = (telescope
        .withGloballyShiftedOptic(
            'LSSTCamera',
            [0, 0, 1e-6]
        )
    )
    dof = [0.0]*5+[-1.0]+[0.0]*44
    dofstr = ",".join(str(s) for s in dof)

    perturb_config = textwrap.dedent(
        f"""
        input:
            telescope:
                file_name: LSST_r.yaml
                fea:
                    aos_dof:
                        dof: [{dofstr}]
        """
    )

    config = yaml.safe_load(perturb_config)
    galsim.config.ProcessInput(config)
    perturbed = galsim.config.GetInputObj(
        'telescope',
        config['input']['telescope'],
        config,
        'telescope'
    ).fiducial
    assert ref == perturbed

    # Next is camera -dx and +dy
    telescope = imsim.load_telescope("LSST_r.yaml")
    ref = (telescope
        .withGloballyShiftedOptic(
            'LSSTCamera',
            [2e-6, 3e-6, 0]
        )
    )
    dof = [0.0]*5 + [0.0, -2.0, +3.0] + [0.0]*42
    dofstr = ",".join(str(s) for s in dof)

    perturb_config = textwrap.dedent(
        f"""
        input:
            telescope:
                file_name: LSST_r.yaml
                fea:
                    aos_dof:
                        dof: [{dofstr}]
        """
    )

    config = yaml.safe_load(perturb_config)
    galsim.config.ProcessInput(config)
    perturbed = galsim.config.GetInputObj(
        'telescope',
        config['input']['telescope'],
        config,
        'telescope'
    ).fiducial
    assert ref == perturbed

    # Next is camera -Rx and Ry in arcsec
    telescope = imsim.load_telescope("LSST_r.yaml")
    ref = (telescope
        .withLocallyRotatedOptic(
            'LSSTCamera',
            batoid.RotX(np.deg2rad(2./3600))@batoid.RotY(np.deg2rad(3./3600))
        )
    )
    dof = [0.0]*8 + [-2.0, 3.0] + [0.0]*40
    dofstr = ",".join(str(s) for s in dof)

    perturb_config = textwrap.dedent(
        f"""
        input:
            telescope:
                file_name: LSST_r.yaml
                fea:
                    aos_dof:
                        dof: [{dofstr}]
        """
    )

    config = yaml.safe_load(perturb_config)
    galsim.config.ProcessInput(config)
    perturbed = galsim.config.GetInputObj(
        'telescope',
        config['input']['telescope'],
        config,
        'telescope'
    ).fiducial
    # Just get close for this one since it depends slightly on order of applying
    # X,Y rotations.
    np.testing.assert_allclose(
        ref['LSSTCamera'].coordSys.rot,
        perturbed['LSSTCamera'].coordSys.rot,
        atol=1e-12, rtol=0
    )
    # Sub items should also agree in orientation and position
    np.testing.assert_allclose(
        ref['Filter'].coordSys.origin,
        perturbed['Filter'].coordSys.origin,
        atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        ref['Filter'].coordSys.rot,
        perturbed['Filter'].coordSys.rot,
        atol=1e-12, rtol=0
    )


    # Finally, just make sure we can construct a telescope with (at least most)
    # of the FEA perturbations turned on.
    perturb_config = textwrap.dedent(
        f"""
        eval_variables:
            azenith: &zenith 30 deg
            arot: &rot 15 deg
        input:
            telescope:
                file_name: LSST_r.yaml
                fea:
                    m1m3_gravity:
                        zenith: *zenith
                    m1m3_temperature:
                        m1m3_TBulk: 0.0
                        m1m3_TxGrad: 0.1
                        m1m3_TyGrad: 0.1
                        m1m3_TzGrad: 0.1
                        m1m3_TrGrad: 0.1
                    m1m3_lut:
                        zenith: *zenith
                        error: 0.01
                        seed: 1
                    m2_gravity:
                        zenith: *zenith
                    m2_temperature:
                        m2_TzGrad: 0.1
                        m2_TrGrad: 0.1
                    camera_gravity:
                        zenith: *zenith
                        rotation: *rot
                    camera_temperature:
                        camera_TBulk: 0.1
        """
    )
    config = yaml.safe_load(perturb_config)
    galsim.config.ProcessInput(config)
    perturbed = galsim.config.GetInputObj(
        'telescope',
        config['input']['telescope'],
        config,
        'telescope'
    ).fiducial


def test_config_fea_file_num():
    """Test that we can set perturbations using the file number"""

    fiducial = batoid.Optic.fromYaml("LSST_r.yaml")
    for file_num in [1, 2]:
        config = textwrap.dedent(
            f"""
            eval_variables:
                izk: $file_num+4
            input:
                telescope:
                    file_name: LSST_r.yaml
                    fea:
                        extra_zk:
                            zk: $[0]*zk+[1e-6]
                            eps: 0.61
            """
        )
        image_num = file_num
        obj_num = 1
        logger = None
        config = yaml.safe_load(config)
        galsim.config.SetupConfigFileNum(config, file_num, image_num, obj_num, logger)
        galsim.config.ProcessInput(config)

        reference = fiducial.withInsertedOptic(
            before='M1',
            item=batoid.OPDScreen(
                name='Screen',
                surface=batoid.Plane(),
                screen=batoid.Zernike(
                    [0]*(4+file_num)+[1e-6],
                    R_outer=fiducial['M1'].R_outer,
                    R_inner=fiducial['M1'].R_outer*0.61
                ),
                coordSys=fiducial.stopSurface.coordSys,
                obscuration=fiducial['M1'].obscuration,
            )
        )

        configd = galsim.config.GetInputObj(
            'telescope',
            config['input']['telescope'],
            config,
            'telescope'
        ).fiducial

        assert configd == reference

def test_config_focus_z():
    """Test that focusZ is properly set from configs."""

    telescope = imsim.load_telescope("LSST_r.yaml")
    shifted_ref = (telescope
        .withLocallyShiftedOptic('LSSTCamera', [0.0, 0.0, 1e-3])
    )

    focusz_shifted_function_call = imsim.load_telescope("LSST_r.yaml", focusZ=1e-3)

    assert shifted_ref == focusz_shifted_function_call

    shifted_config = textwrap.dedent(
            """
            input:
                telescope:
                    file_name: LSST_r.yaml
                    focusZ : 1e-3
            """
        )

    config = yaml.safe_load(shifted_config)
    galsim.config.ProcessInput(config)
    focusz_shifted_config = galsim.config.GetInputObj(
        'telescope',
        config['input']['telescope'],
        config,
        'telescope'
    ).fiducial

    assert shifted_ref == focusz_shifted_config


if __name__ == "__main__":
    test_config_shift()
    test_config_rot()
    test_config_zernike_perturbation()
    test_config_fea()
    test_config_fea_file_num()
