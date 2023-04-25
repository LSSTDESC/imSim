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


def load_telescope(
    file_name,
    perturbations=(),
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
    rotTelPos : galsim.Angle, optional
        Rotator angle to apply.
    cameraName : str, optional
        The name of the camera to rotate.

    Examples of perturbations dicts:
    --------------------------
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

    if rotTelPos is not None:
        telescope = telescope.withLocallyRotatedOptic(
            cameraName,
            batoid.RotZ(rotTelPos.rad)
        )
    return telescope


class Telescopes:
    """
    This input class stores one or more batoid telescope instances.

    There is always a base telescope, which is created at the start.
    This is accessible via telescopes.get('base')

    Whenever `set_shifted_det` is called, then a shifted telescope will
    be set in the 'det' key, so it will be accessible via telescopes.get('det').
    """
    # Note: We don't use a simple dict for this,  because subscripting doesn't
    # work correctly through proxies, which will be required whenever using this
    # in multiprocessing.

    _req_params = { 'file_name' : str }
    _opt_params = {
        'rotTelPos': Angle,
        'cameraName': str
    }

    def __init__(self, file_name, perturbations=(), rotTelPos=None, cameraName='LSSTCamera'):
        self._d = {
            # Always start with the base telescope
            'base': load_telescope(file_name, perturbations=perturbations,
                                   rotTelPos=rotTelPos, cameraName=cameraName)
        }

    def set_shifted_det(self, z_offset):
        """Set the 'det' key to a shifted optics version with the given z_offset
        """
        self._d['det'] = self._d['base'].withLocallyShiftedOptic(
                "Detector",
                [0, 0, -z_offset]  # batoid convention is opposite of DM
            )

    def get(self, key):
        return self._d.get(key)


class TelescopeLoader(InputLoader):
    """Load a telescope from a yaml file.
    """
    def getKwargs(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(self.init_func)
        perturbations = config.pop('perturbations', ())
        kwargs, safe = GetAllParams(config, base, req=req, opt=opt, single=single)
        kwargs['perturbations'] = perturbations
        return kwargs, True

    def setupImage(self, input_obj, config, base, logger=None):
        """Set up the telescope for the current image."""
        logger = LoggerWrapper(logger)
        camera = get_camera(base['output']['camera'])
        det_name = base['det_name']

        ccd_orientation = camera[det_name].getOrientation()
        if hasattr(ccd_orientation, 'getHeight'):
            z_offset = ccd_orientation.getHeight()*1.0e-3  # Convert to meters.
            logger.info("Setting CCD z-offset to %.2e m", z_offset)
        else:
            z_offset = 0
        input_obj.set_shifted_det(z_offset)


RegisterInputType('telescope', TelescopeLoader(Telescopes))
