from batoid_rubin import LSSTBuilder
from galsim.config import (
    InputLoader, RegisterInputType, GetAllParams, get_cls_params, ParseValue,
    LoggerWrapper, GetCurrentValue
)
from galsim import Angle
import batoid
from collections.abc import Sequence
from .camera import get_camera


def parse_xyz(xyz, base):
    if not isinstance(xyz, list) or len(xyz) != 3:
        raise ValueError("Expecting a list of 3 elements")
    parsed_xyz, safe = zip(*[ParseValue(xyz, i, base, float) for i in range(3)])
    return parsed_xyz, all(safe)


def apply_fea(
    fea_perturbations, telescope, **kwargs
):
    """ Parse a finite element analysis config dict.  Used to add detailed
    effects like mirror print through or temperature originating figure
    perturbations.  Also facilitates direct manipulation of standard active
    optics degrees of freedom.

    Parameters
    ----------
    fea_perturbations : dict
        The perturbations dictionary to parse.
    telescope : batoid.Optic
        Telescope to perturb
    **kwargs: dict
        Additional arguments to pass to the LSSTBuilder constructor.

    Examples of fea config dicts
    ----------------------------

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
    builder = LSSTBuilder(telescope, **kwargs)
    for k, v in fea_perturbations.items():
        method = getattr(builder, "with_"+k)
        builder = method(**v)
    return builder.build()


def load_telescope(
    telescope,
    builder_kwargs=None,
    perturbations=(),
    fea_perturbations=None,
    rotTelPos=None,
    cameraName="LSSTCamera",
    focusZ=None,
):
    """ Load a telescope.

    Parameters
    ----------
    telescope : batoid.Optic or str
        Either the fiducial telescope to modify, or a string giving the
        name of a yaml file describing the telescope.
    builder_kwargs: dict, optional
        Additional arguments to pass to the LSSTBuilder constructor.
    perturbations : (list of) dict of dict
        (List of) dict of dict describing perturbations to apply to the
        telescope in order.  Each outer dict should have keys indicating
        optics to be perturbed and values indicating the perturbations
        to apply (each perturbation is a dictionary keyed by the type of
        perturbation and value the magnitude of the perturbation.)  See notes
        for details.
    fea_perturbations : dict, optional
        Finite element analysis perturbations.  See apply_fea docstring for
        details.
    rotTelPos : galsim.Angle, optional
        Rotator angle to apply.
    cameraName : str, optional
        The name of the camera to rotate.
    focusZ : float, optional
        Distance to intentionally defocus the camera for AOS analysis.
        Units are meters.

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
    if isinstance(telescope, str):
        telescope = batoid.Optic.fromYaml(telescope)
    if not isinstance(perturbations, Sequence):
        perturbations = [perturbations]
    for group in perturbations:
        for optic, perturbs in group.items():
            for ptype, pval in perturbs.items():
                if ptype == 'shift':
                    telescope = telescope.withLocallyShiftedOptic(
                        optic, pval
                    )
                elif ptype.startswith('rot'):
                    if ptype == 'rotX':
                        rotMat = batoid.RotX(pval)
                    elif ptype == 'rotY':
                        rotMat = batoid.RotY(pval)
                    elif ptype == 'rotZ':
                        rotMat = batoid.RotZ(pval)
                    telescope = telescope.withLocallyRotatedOptic(
                        optic, rotMat
                    )
                elif ptype == 'Zernike':
                    R_outer = pval['R_outer']
                    R_inner = pval['R_inner']
                    coef = pval['coef']
                    telescope = telescope.withPerturbedSurface(
                        optic,
                        batoid.Zernike(coef, R_outer, R_inner)
                    )
    if fea_perturbations is not None:
        telescope = apply_fea(fea_perturbations, telescope, **builder_kwargs)

    if rotTelPos is not None:
        telescope = telescope.withLocallyRotatedOptic(
            cameraName,
            batoid.RotZ(rotTelPos.rad)
        )

    if focusZ is not None:
        telescope = telescope.withLocallyShiftedOptic(cameraName, [0.0, 0.0, focusZ])
    return telescope


def _parse_fea(config, base, logger):
    req, opt, single, takes_rng = get_cls_params(LSSTBuilder)
    LSSTBuilder_kwargs, safe = GetAllParams(
        config,
        base,
        req=req,
        opt=opt,
        single=single,
        ignore=LSSTBuilder._ignore_params
    )

    perturbations = {}

    skip = list(req.keys()) + list(opt.keys()) + list(single.keys())
    for k, v in config.items():
        if k in skip or k[0] == '_':
            continue
        method = getattr(LSSTBuilder, "with_"+k)
        req = getattr(method, "_req_params", {})
        opt = getattr(method, "_opt_params", {})
        single = getattr(method, "_single_params", {})
        ignore = getattr(method, "_ignore_params", {})

        kwargs, safe1 = GetAllParams(
            v, base,
            req=req, opt=opt, single=single, ignore=ignore
        )
        safe &= safe1
        perturbations[k] = {}
        perturbations[k].update(kwargs)
    return LSSTBuilder_kwargs, perturbations, safe


def _parse_perturbations(config, base, telescope, logger):
    safe = True
    out = ()
    if not isinstance(config, Sequence):
        config = (config,)
    for group in config:
        outgroup = {}
        for optic, perturbs in group.items():
            outperturbs = {}
            for ptype, pval in perturbs.items():
                if ptype == 'shift':
                    shift, safe1 = parse_xyz(pval, base)
                    safe &= safe1
                    outperturbs['shift'] = shift
                elif ptype.startswith('rot'):
                    angle, safe1 = ParseValue(perturbs, ptype, base, Angle)
                    safe &= safe1
                    outperturbs[ptype] = angle
                elif ptype == 'Zernike':
                    R_outer = None
                    if 'R_outer' in pval:
                        R_outer, safe1 = ParseValue(pval, 'R_outer', base, float)
                        safe &= safe1
                    R_inner = None
                    if 'R_inner' in pval:
                        R_inner, safe1 = ParseValue(pval, 'R_inner', base, float)
                        safe &= safe1
                    if (R_outer is None) != (R_inner is None):
                        raise ValueError(
                            "Must specify both or neither of R_outer and R_inner"
                        )
                    if not R_outer:
                        R_outer = telescope[optic].R_outer
                        R_inner = telescope[optic].R_inner
                    if 'coef' in pval and 'idx' in pval:
                        raise ValueError(
                            "Cannot specify both coef and idx for Zernike perturbation"
                        )

                    if 'coef' in pval:
                        coef, safe1 = ParseValue(pval, 'coef', base, list)
                        safe &= safe1
                    if 'idx' in pval:
                        try:
                            idx, safe1 = ParseValue(pval, 'idx', base, list)
                        except Exception:
                            idx, safe1 = ParseValue(pval, 'idx', base, int)
                            idx = [idx]
                        safe &= safe1
                        try:
                            val, safe1 = ParseValue(pval, 'val', base, list)
                        except Exception:
                            val, safe1 = ParseValue(pval, 'val', base, float)
                            val = [val]
                        safe &= safe1
                        coef = [0.0]*(max(idx)+1)
                        for i, v in zip(idx, val):
                            coef[i] = v
                    outperturbs['Zernike'] = {
                        'coef': coef,
                        'R_outer': R_outer,
                        'R_inner': R_inner
                    }
            outgroup[optic] = outperturbs
        out += (outgroup,)
    return out, safe


class DetectorTelescope:
    """
    Produce a batoid telescope instance appropriate for a particular detector,
    optionally with a shifted detector position.
    """
    _req_params = {
        'file_name' : str,
    }
    _opt_params = {
        'rotTelPos': Angle,
        'camera': str,
        'focusZ': float,
    }

    def __init__(
        self,
        file_name,
        perturbations=(),
        rotTelPos=None,
        camera='LsstCam',
        builder_kwargs=None,
        fea_perturbations=None,
        focusZ=None,
        logger=None
    ):
        # Batoid has a different name for LsstCam than DM code.  So we need to switch it here.
        match camera:
            case 'LsstCam':
                cameraName = 'LSSTCamera'
            case 'LsstComCamSim':
                cameraName = 'ComCam'
            case _:
                cameraName = camera
        self.fiducial = load_telescope(
            telescope=file_name,
            perturbations=perturbations,
            builder_kwargs=builder_kwargs,
            fea_perturbations=fea_perturbations,
            rotTelPos=rotTelPos,
            cameraName=cameraName,
            focusZ=focusZ
        )
        self.camera = camera
        self.logger = logger

    def get_telescope(self, z_offset):
        """Get a potentially detector-shifted version of the telescope with the given z_offset.
        """
        return self.fiducial.withLocallyShiftedOptic(
            "Detector",
            [0, 0, -z_offset]  # batoid convention is opposite of DM
        )

    def calculate_z_offset(self, det_name):
        camera = get_camera(self.camera)

        ccd_orientation = camera[det_name].getOrientation()
        if hasattr(ccd_orientation, 'getHeight'):
            z_offset = ccd_orientation.getHeight()*1.0e-3  # Convert to meters.
        else:
            z_offset = 0
        return z_offset

class TelescopeLoader(InputLoader):
    """Load a telescope from a yaml file.
    """
    def getKwargs(self, config, base, logger):
        safe = True
        req, opt, single, takes_rng = get_cls_params(self.init_func)
        kwargs, safe1 = GetAllParams(
            config, base,
            req=req, opt=opt, single=single,
            ignore=['fea', 'perturbations']
        )
        safe &= safe1
        telescope = batoid.Optic.fromYaml(kwargs['file_name'])

        perturbations = config.get('perturbations', ())
        if perturbations:
            perturbations, safe1 = _parse_perturbations(
                perturbations, base, telescope, logger
            )
            safe &= safe1
        fea = config.get('fea', None)
        if fea:
            builder_kwargs, fea_perturbations, safe1 = _parse_fea(
                fea, base, logger
            )
            safe &= safe1
        else:
            builder_kwargs = {}
            fea_perturbations = None

        kwargs['perturbations'] = perturbations
        kwargs['fea_perturbations'] = fea_perturbations
        kwargs['builder_kwargs'] = builder_kwargs
        kwargs['logger'] = logger
        return kwargs, safe

    def setupImage(self, input_obj, config, base, logger=None):
        """Set up the telescope for the current image."""
        logger = LoggerWrapper(logger)
        if 'det_name' in base.get('image',{}):
            det_name = GetCurrentValue('image.det_name', base)
            logger.info('Setting up det_telescope for detector %s', det_name)
            z_offset = input_obj.calculate_z_offset(det_name)
            det_telescope = input_obj.get_telescope(z_offset)
        else:
            det_telescope = input_obj.get_telescope(0)
        base['det_telescope'] = det_telescope


RegisterInputType('telescope', TelescopeLoader(DetectorTelescope, file_scope=True))
