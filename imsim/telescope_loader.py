import os
from galsim.config import (
    InputLoader, RegisterInputType, GetAllParams, get_cls_params, ParseValue,
    LoggerWrapper
)
from galsim import Angle
import batoid
from collections.abc import Sequence
from .camera import get_camera


def infer_optic_radii(optic):
    """Infer the inner/outer radii of an optic from its obscuration.

    Parameters
    ----------
    optic : batoid.Optic
        The optic from which to infer radii.

    Returns
    -------
    R_outer : float
        The outer radius of the optic.
    R_inner : float
        The inner radius of the optic.
    """
    obsc = optic.obscuration
    if obsc is not None:
        if isinstance(obsc, batoid.obscuration.ObscNegation):
            obsc = obsc.original
        if isinstance(obsc, batoid.obscuration.ObscAnnulus):
            return obsc.outer, obsc.inner
        if isinstance(obsc, batoid.obscuration.ObscCircle):
            return obsc.radius, 0.0

    raise ValueError(
        f"Cannot infer radii for optic {optic.name}"
    )


def parse_xyz(xyz, base):
    if not isinstance(xyz, list) or len(xyz) != 3:
        raise ValueError("Expecting a list of 3 elements")
    parsed_xyz, safe = zip(*[ParseValue(xyz, i, base, float) for i in range(3)])
    return parsed_xyz, all(safe)


def parse_fea(config, base, telescope):
    """ Parse a finite element analysis config dict.  Used to add detailed
    effects like mirror print through or temperature originating figure
    perturbations.  Also facilitates direct manipulation of standard active
    optics degrees of freedom.

    Parameters
    ----------
    config : dict
        The configuration dictionary to parse.
    base : dict
        Base configuration dictionary.
    telescope : batoid.Optic
        Telescope to perturb

    Examples of fea config dicts
    ----------------------------

    # Set M1M3 gravitational perturbations.  This requires a zenith angle
    # be supplied.
    fea:
      m1m3_gravity:
        zenith_angle: 30 deg


    # Set M1M3 temperature induced figure perturbations.  This requires
    # the bulk temperature and 4 temperature gradients be supplied.
    fea:
      m1m3_temperature:
        m1m3_TBulk: 0.1  # Kelvin
        m1m3_TxGrad: 0.01  # Kelvin/meter
        m1m3_TyGrad: 0.01  # Kelvin/meter
        m1m3_TzGrad: 0.01  # Kelvin/meter
        m1m3_TrGrad: 0.01  # Kelvin/meter

    # Engage M1M3 lookup table.  Requires zenith angle and optionally a
    # fractional random error to apply to each force actuator.
    fea:
      m1m3_lut:
        zenith_angle: 39 deg
        error: 0.01  # fractional random error to apply to each actuator
        seed: 1  # random seed for error above

    # Set M2 gravitational perturbations.  Requires zenith angle.
    fea:
      m2_gravity:
        zenith_angle: 30 deg

    # Set M2 temperature gradient induced figure errors.  Requires 2 temperature
    # gradients (in the z and radial directions).
    fea:
      m2_temperature:
        m2_TzGrad: 0.01  # Kelvin/meter
        m2_TrGrad: 0.01  # Kelvin/meter

    # Set camera gravitational perturbations.  Requires zenith angle and camera
    # rotator angle.
    fea:
      camera_gravity:
        zenith_angle: 30 deg
        rotation_angle: -25 deg

    # Set camera temperature-induced perturbations.  Requires the bulk
    # temperature of the camera.
    fea:
      camera_temperature:
        camera_TBulk: 0.1

    # Set the Active Optics degrees of freedom.  There are 50 baseline degrees
    # of freedom, so we won't copy them all here, but you can imagine a list of
    # 50 floats as the specifications for each degree of freedom.
    fea:
      aos_dof:
        dof: list-of-50-floats

    Notes
    -----
    The implementation of FEA degrees of freedom in ImSim uses the batoid_rubin
    package.  The available perturbations are algorithmically determined from
    the `with_*` methods of the LSSTBuilder class there.  The arguments to each
    `with_*` method are passed in as **kwargs from the config dict.  The only
    additional processing is that arguments with names ending in `_angle` are
    parsed using the normal galsim config parsing architecture, which allows one
    to use unit-ful string "30 deg" or "25.2 arcsec" in addition to a bare
    float.  Future additions to the `batoid_rubin` package may include new or
    changed APIs to the examples above.
    """
    from batoid_rubin import LSSTBuilder
    base = {} if base is None else base
    fea_dir = config.pop("fea_dir", "fea_legacy")
    bend_dir = config.pop("bend_dir", "bend_legacy")
    builder = LSSTBuilder(telescope, fea_dir=fea_dir, bend_dir=bend_dir)
    for k, v in config.items():
        method = getattr(builder, "with_"+k)
        # parse angles
        for kk, vv in v.items():
            if kk.endswith("_angle"):
                v[kk], safe = ParseValue(v, kk, base, Angle)
            elif kk == 'dof':
                v[kk], safe = ParseValue(v, kk, base, None)
            else:
                v[kk], safe = ParseValue(v, kk, base, float)
        builder = method(**v)
    return builder.build()


def load_telescope(
    file_name,
    perturbations=(),
    fea=None,
    rotTelPos=None,
    cameraName="LSSTCamera",
    base=None
):
    """ Load a telescope.

    Parameters
    ----------
    file_name : str
        File name describing batoid Optic in yaml format.
    perturbations : (list of) dict of dict
        (List of) dict of dict describing perturbations to apply to the
        telescope in order.  Each outer dict should have keys indicating
        optics to be perturbed and values indicating the perturbations
        to apply (each perturbation is a dictionary keyed by the type of
        perturbation and value the magnitude of the perturbation.)  See notes
        for details.
    fea : dict, optional
        Finite element analysis perturbations.  See parse_fea docstring for
        details.
    rotTelPos : galsim.Angle, optional
        Rotator angle to apply.
    cameraName : str, optional
        The name of the camera to rotate.

    Examples of perturbations dicts:
    --------------------------------
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
    """
    telescope = batoid.Optic.fromYaml(file_name)
    if not isinstance(perturbations, Sequence):
        perturbations = [perturbations]
    for group in perturbations:
        for optic, perturbs in group.items():
            for ptype, pval in perturbs.items():
                if ptype == 'shift':
                    shift, safe = parse_xyz(pval, base)
                    telescope = telescope.withLocallyShiftedOptic(
                        optic, shift
                    )
                elif ptype.startswith('rot'):
                    angle, safe = ParseValue(perturbs, ptype, base, Angle)
                    if ptype == 'rotX':
                        rotMat = batoid.RotX(angle)
                    elif ptype == 'rotY':
                        rotMat = batoid.RotY(angle)
                    elif ptype == 'rotZ':
                        rotMat = batoid.RotZ(angle)
                    telescope = telescope.withLocallyRotatedOptic(
                        optic, rotMat
                    )
                elif ptype == 'Zernike':
                    R_outer = pval.get('R_outer', None)
                    R_inner = pval.get('R_inner', None)
                    if (R_outer is None) != (R_inner is None):
                        raise ValueError(
                            "Must specify both or neither of R_outer and R_inner"
                        )
                    if not R_outer or not R_inner:
                        R_outer, R_inner = infer_optic_radii(telescope[optic])

                    if 'coef' in pval and 'idx' in pval:
                        raise ValueError(
                            "Cannot specify both coef and idx for Zernike perturbation"
                        )
                    if 'coef' in pval:
                        coef = pval['coef']
                    if 'idx' in pval:
                        idx = pval['idx']
                        if not isinstance(idx, Sequence):
                            idx = [idx]
                        val = pval['val']
                        if not isinstance(val, Sequence):
                            val = [val]
                        coef = [0.0]*(max(idx)+1)
                        for i, v in zip(idx, val):
                            coef[i] = v
                    telescope = telescope.withPerturbedSurface(
                        optic,
                        batoid.Zernike(coef, R_outer, R_inner)
                    )
    if fea is not None:
        telescope = parse_fea(fea, base, telescope)

    if rotTelPos is not None:
        telescope = telescope.withLocallyRotatedOptic(
            cameraName,
            batoid.RotZ(rotTelPos.rad)
        )
    return telescope


class DetectorTelescope:
    """
    Produce a batoid telescope instance appropriate for a particular detector,
    optionally with a shifted detector position.
    """
    _req_params = { 'file_name' : str }
    _opt_params = {
        'rotTelPos': Angle,
        'cameraName': str,
        'fea': None,
    }

    def __init__(
        self,
        file_name,
        perturbations=(),
        rotTelPos=None,
        cameraName='LSSTCamera',
        fea=None,
        logger=None
    ):
        self.fiducial = load_telescope(
            file_name, perturbations=perturbations,
            rotTelPos=rotTelPos, cameraName=cameraName,
            fea=fea
        )
        self.logger = logger

    def get_shifted_det(self, z_offset):
        """Get a potentially detector-shifted version of the telescope with the given z_offset.
        """
        return self.fiducial.withLocallyShiftedOptic(
            "Detector",
            [0, 0, -z_offset]  # batoid convention is opposite of DM
        )

class TelescopeLoader(InputLoader):
    """Load a telescope from a yaml file.
    """
    def getKwargs(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(self.init_func)
        perturbations = config.pop('perturbations', ())
        fea = config.pop('fea', None)
        kwargs, safe = GetAllParams(config, base, req=req, opt=opt, single=single)
        kwargs['perturbations'] = perturbations
        kwargs['fea'] = fea
        kwargs['logger'] = logger
        return kwargs, True

    def setupImage(self, input_obj, config, base, logger=None):
        """Set up the telescope for the current image."""
        logger = LoggerWrapper(logger)
        camera = get_camera(base['output']['camera'])
        det_name = base['det_name']

        ccd_orientation = camera[det_name].getOrientation()
        if hasattr(ccd_orientation, 'getHeight'):
            z_offset = ccd_orientation.getHeight()*1.0e-3  # Convert to meters.
        else:
            z_offset = 0
        det_telescope = input_obj.get_shifted_det(z_offset)
        base['det_telescope'] = det_telescope

RegisterInputType('telescope', TelescopeLoader(DetectorTelescope))
