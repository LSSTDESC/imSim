from galsim.config import InputLoader, RegisterInputType
from galsim import Angle
import batoid


def load_telescope(file_name, rotTelPos=None, cameraName="LSSTCamera"):
    """ Load a telescope.

    Parameters
    ----------
    file_name : str
        File name describing batoid Optic in yaml format.
    rotTelPos : galsim.Angle, optional
        Rotator angle to apply.
    cameraName : str, optional
        The name of the camera to rotate.
    """
    telescope = batoid.Optic.fromYaml(file_name)
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

RegisterInputType('telescope', InputLoader(load_telescope))
