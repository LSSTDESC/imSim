from galsim.config import InputLoader, RegisterInputType, GetAllParams, get_cls_params
from galsim import Angle
import batoid
from collections.abc import Sequence


def load_telescope(
    file_name, perturb=(), rotTelPos=None, cameraName="LSSTCamera"
):
    """ Load a telescope.

    Parameters
    ----------
    file_name : str
        File name describing batoid Optic in yaml format.
    perturb : (list of) dict
        (List of) dict describing perturbations to apply to the telescope in
        order.  Each dict should have a key that is the type of perturbation,
        and a value that is a dict indicating which optic to perturb and the
        magnitude of the perturbation.  See notes for details.
    rotTelPos : galsim.Angle, optional
        Rotator angle to apply.
    cameraName : str, optional
        The name of the camera to rotate.

    Examples of perturb dicts:
    --------------------------
    # Shift M2 in x and y by 1 mm
        {'shift': {'M2': [1e-3, 1e-3, 0.0]}}

    # Rotate M3 about the local x axis by 1 arcmin
        {'rotX': {'M3': (1*galim.arcmin).rad}}

    # Apply 1 micron of the Z6 Zernike aberration to M1
    # using list of coefficients indexed by Noll index (starting at 0).
        {'Zernike': {'M1': {'coef': [0.0]*6+[1e-6]}}}
    # or specify Noll index and value
        {'Zernike': {'M1': {'idx': 6, 'val': 1e-6}}}

    # Apply 1 micron of Z6 and 2 microns of Z4 to M1
        {'Zernike': {'M1': {'coef': [0.0]*4 + [2e-6, 0.0, 1e-6]}}}
    # or
        {'Zernike': {'M1': {'idx': [4, 6], 'val': [2e-6, 1e-6]}}}

    # By default, Zernike inner and outer radii are inferred from the
    # optic's obscuration, but you can also manually override them.
        {'Zernike': {
            'M1': {
                'coef': [0.0]*4+[2e-6, 0.0, 1e-6],
                'R_outer': 4.18,
                'R_inner': 2.558
            }
        }}

    # You can specify multiple types of perturbations in a single dict
        {
            'shift': {'M2': [1e-3, 1e-3, 0.0]},
            'rotX': {'M3': (1*galim.arcmin).rad}
        }
    # or multiple optics for a single type of perturbation.
        {'Zernike':
            'M1': {'coef': [0.0]*4+[2e-6, 0.0, 1e-6]},
            'M2': {'idx': 3, 'val': 3e-6}
        }
    """
    telescope = batoid.Optic.fromYaml(file_name)
    if not isinstance(perturb, Sequence):
        perturb = [perturb]
    for group in perturb:
        for ptype, pvals in group.items():
            if ptype == 'shift':
                for optic, shift in pvals.items():
                    telescope = telescope.withLocallyShiftedOptic(
                        optic, shift
                    )
            elif ptype == 'rotX':
                for optic, angle in pvals.items():
                    telescope = telescope.withLocallyRotatedOptic(
                        optic, batoid.RotX(angle)
                    )
            elif ptype == 'rotY':
                for optic, angle in pvals.items():
                    telescope = telescope.withLocallyRotatedOptic(
                        optic, batoid.RotY(angle)
                    )
            elif ptype == 'rotZ':
                for optic, angle in pvals.items():
                    telescope = telescope.withLocallyRotatedOptic(
                        optic, batoid.RotZ(angle)
                    )
    if rotTelPos is not None:
        telescope = telescope.withLocallyRotatedOptic(
            cameraName,
            batoid.RotZ(rotTelPos.rad)
        )
    return telescope

load_telescope._req_params = { 'file_name' : str }
load_telescope._opt_params = {
    'rotTelPos': Angle,
    'cameraName': str
}


class TelescopeLoader(InputLoader):
    """Load a telescope from a yaml file.
    """
    def getKwargs(self, config, base, logger):
        req, opt, single, takes_rng = get_cls_params(self.init_func)
        perturb = config.pop('perturb', ())
        kwargs, safe = GetAllParams(config, base, req=req, opt=opt, single=single)
        kwargs['perturb'] = perturb
        return kwargs, False


RegisterInputType('telescope', TelescopeLoader(load_telescope))
