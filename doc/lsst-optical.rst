.. _optical-system-label:

========================
The Rubin Optical System
========================

Controlling Optical Perturbations
---------------------------------

Examples of perturbations dicts (for this documentation we should convert this to YAML file)

.. code-block:: py

    # Shift M2 in x and y by 1 mm
        {'M2': {'shift': [1e-3, 1e-3, 0.0]}}

    # Rotate M3 about the local x axis by 1 arcmin
        {'M3': {'rotX': 1*galim.arcmin}}

    # Apply 1 micron of the Z6 Zernike aberration to M1
    # using list of coefficients indexed by Noll index (starting at 0).
        {'M1': {'Zernike': {'coef': [0.0]*6+[1e-6]}}}
    # or specify Noll index and value
        {'M1': {'Zernike': {'idx': 6, 'val': 1e-6}}}

    # Apply 1 micron of Z6 and 2 microns of Z4 to M1
        {'M1': {'Zernike': {'coef': [0.0]*4 + [2e-6], 0.0, 1e-6]}}}
    # or
        {'M1': {'Zernike': {'idx': [4, 6], 'val': [2e-6, 1e-6]}}}

    # By default, Zernike inner and outer radii are inferred from the
    # optic's obscuration, but you can also manually override them.
        {'M1': {
            'Zernike': {
                'coef': [0.0]*4+[2e-6, 0.0, 1e-6],
                'R_outer': 4.18,
                'R_inner': 2.558
            }
        }}


    # You can specify multiple perturbations in a single dict
        {
            'M2': {'shift':[1e-3, 1e-3, 0.0]},
            'M3': {'rotX':1*galim.arcmin}
        }

    # The telescope loader will preserve the order of multiple perturbations,
    # but to help disambiguate non-commuting perturbations, you can also use a
    # list:
        [
            {'M3': {'rotX':1*galim.arcmin}},  # X-rot is applied first
            {'M3': {'rotY':1*galim.arcmin}}
        ]

    # is the same as
        [
            {'M3': {
                'rotX':1*galim.arcmin},
                'rotY':1*galim.arcmin}
                }
            }
        ]

Specifying FEA parameters
-------------------------

Examples of fea config dicts

.. code-block:: yaml

    # Set M1M3 gravitational perturbations.  This requires a zenith angle
    # be supplied.
    fea:
      m1m3_gravity:
        zenith: 30 deg


    # Set M1M3 temperature induced figure perturbations.  This requires
    # the bulk temperature and 4 temperature gradients be supplied.
    fea:
      m1m3_temperature:
        m1m3_TBulk: 0.1  # Celsius
        m1m3_TxGrad: 0.01  # Celsius/meter
        m1m3_TyGrad: 0.01  # Celsius/meter
        m1m3_TzGrad: 0.01  # Celsius/meter
        m1m3_TrGrad: 0.01  # Celsius/meter

    # Engage M1M3 lookup table.  Requires zenith angle and optionally a
    # fractional random error to apply to each force actuator.
    fea:
      m1m3_lut:
        zenith: 39 deg
        error: 0.01  # fractional random error to apply to each actuator
        seed: 1  # random seed for error above

    # Set M2 gravitational perturbations.  Requires zenith angle.
    fea:
      m2_gravity:
        zenith: 30 deg

    # Set M2 temperature gradient induced figure errors.  Requires 2 temperature
    # gradients (in the z and radial directions).
    fea:
      m2_temperature:
        m2_TzGrad: 0.01  # Celsius/meter
        m2_TrGrad: 0.01  # Celsius/meter

    # Set camera gravitational perturbations.  Requires zenith angle and camera
    # rotator angle.
    fea:
      camera_gravity:
        zenith: 30 deg
        rotation: -25 deg

    # Set camera temperature-induced perturbations.  Requires the bulk
    # temperature of the camera.
    fea:
      camera_temperature:
        camera_TBulk: 0.1  # Celsius

    # Set the Active Optics degrees of freedom.  There are 50 baseline degrees
    # of freedom, so we won't copy them all here, but you can imagine a list of
    # 50 floats as the specifications for each degree of freedom.
    fea:
      aos_dof:
        dof: list-of-50-floats
