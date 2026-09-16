import galsim
from galsim.config import GetInputObj, RegisterObjectType


__all__ = ["BuildRubinDeepCoaddPSF"]


def BuildRubinDeepCoaddPSF(config, base, ignore, gsparams, logger):
    """
    Build PSFs from Rubin deep_coadds.  The deep_coadd image will be
    retrieved from the data repository using the butler for the
    tract-patch combination corresponding to the object's sky position.

    Assuming the input.deep_coadd object is defined, to use this
    PSF, add the following to the config yaml:

    input.atm_psf: ""  # disable the atmospheric PSF
    psf:
        type: RubinDeepCoaddPSF
    """
    deep_coadds = GetInputObj('deep_coadds', config, base, 'RubinDeepCoaddPSF')
    coadd_num = base['coadd_num']
    assert (coadd_num >= 0 and coadd_num < len(deep_coadds.data_ids))
    data_id = deep_coadds.data_ids[coadd_num]
    image_pos = base['image_pos']
    celestial_coord = base['wcs'].toWorld(image_pos)
    ra = celestial_coord.ra / galsim.degrees
    dec = celestial_coord.dec / galsim.degrees
    band = data_id['band']
    safe = False
    return deep_coadds.getPSF(ra, dec, band), safe


RegisterObjectType('RubinDeepCoaddPSF', BuildRubinDeepCoaddPSF,
                   input_type='deep_coadds')
