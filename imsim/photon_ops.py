"""PhotonOp classes for imsim"""

import numpy as np

import batoid
from batoid import Optic
from galsim import PhotonArray, PhotonOp
from galsim.config import RegisterPhotonOpType, PhotonOpBuilder, GetAllParams

from galsim.celestial import CelestialCoord
from galsim.config.util import get_cls_params
from .camera import get_camera
from .utils import focal_to_pixel


class LsstOptics(PhotonOp):
    """A photon operator that performs raytracing through the LSST optics.

    Parameters
    ----------
    telescope : batoid.Optic
        The telescope to trace through.
    boresight : galsim.CelestialCoord
        The ICRF coordinate of light that reaches the boresight.  Note that this
        is distinct from the spherical coordinates of the boresight with respect
        to the ICRF axes.
    sky_pos : galsim.CelestialCoord
    image_pos : galsim.PositionD
    icrf_to_field : galsim.GSFitsWCS
    det_name : str
    shift_optics : dict[str, list[float]]
        A dict mapping optics keys to shifts represented by a list of 3 floats.
        The corresponding optics will be displaced by the specified corrdinates.
        Example config for perturbed+defocused telescope to obtain a donut:
        -
            type: lsst_optics
            ...
            shift_optics:
              Detector: [0, 0, 1.5e-3]
              M2: [3.0e-3, 0, 0]


    """

    _req_params = {"telescope": Optic, "band": str, "boresight": CelestialCoord}
    _opt_params = {"shift_optics": dict}

    def __init__(
        self,
        telescope,
        boresight,
        sky_pos,
        image_pos,
        icrf_to_field,
        det_name,
        shift_optics=None,
    ):
        if shift_optics is not None:
            for optics_key, shift in shift_optics.items():
                telescope = telescope.withGloballyShiftedOptic(optics_key, shift)
        self.telescope = telescope
        self.detector = get_camera()[det_name]
        self.boresight = boresight
        self.sky_pos = sky_pos
        self.image_pos = image_pos
        self.icrf_to_field = icrf_to_field

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the photon operator to a PhotonArray.

        Here, we assume that the photon array has passed through
        `imsim.atmPSF.AtmosphericPSF` which stores sampled pupil
        locations in photon_array.pupil_u and .pupil_v.

        Parameters
        ----------
        photon_array:   A `PhotonArray` to apply the operator to.
        local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                        bundle in case the operator needs this information.  [default: None]
        rng:            A random number generator to use if needed. [default: None]
        """
        # x/y to u/v
        u = local_wcs._u(photon_array.x, photon_array.y)
        v = local_wcs._v(photon_array.x, photon_array.y)
        # u/v to ICRF
        u_rad = np.deg2rad(u / 3600)
        v_rad = np.deg2rad(v / 3600)
        ra, dec = self.sky_pos.deproject_rad(u_rad, v_rad)
        # ICRF to field
        thx, thy = self.icrf_to_field.radecToxy(ra, dec, units="rad")

        vx, vy, vz = batoid.utils.gnomonicToDirCos(thx, thy)
        # Adjust for refractive index of air
        wavelength = photon_array.wavelength * 1e-9
        n = self.telescope.inMedium.getN(wavelength)
        vx /= n
        vy /= n
        vz /= n

        x, y = photon_array.pupil_u, photon_array.pupil_v
        z = self.telescope.stopSurface.surface.sag(x, y)
        ray_vec = batoid.RayVector._directInit(
            x,
            y,
            z,
            vx,
            vy,
            vz,
            t=np.zeros_like(x),
            wavelength=wavelength,
            flux=photon_array.flux,
            vignetted=np.zeros_like(x, dtype=bool),
            failed=np.zeros_like(x, dtype=bool),
            coordSys=self.telescope.stopSurface.coordSys,
        )
        traced = self.telescope.trace(ray_vec)
        ray_vector_to_photon_array(traced, detector=self.detector, out=photon_array)
        photon_array.x -= self.image_pos.x
        photon_array.y -= self.image_pos.y

    def __str__(self):
        return "imsim.LsstPhotonOp()"

    def __repr__(self):
        return str(self)


def ray_vector_to_photon_array(
    ray_vector: batoid.RayVector, detector, out: PhotonArray
) -> PhotonArray:
    """Converts a batoid.RayVector to a galsim.PhotonArray

    Stores into an already existing galsim.PhotonArray out"""

    assert all(np.abs(ray_vector.z) < 1.0e-15)
    out.x, out.y = focal_to_pixel(ray_vector.y * 1e3, ray_vector.x * 1e3, detector)

    out.dxdz = ray_vector.vx / ray_vector.vz
    out.dydz = ray_vector.vy / ray_vector.vz
    out.flux[ray_vector.vignetted] = 0.0
    return out


class LsstOpticsFactory(PhotonOpBuilder):
    """Build the LsstOptics PhotonOp.

    Returns:
         the constructed LsstOptics object.
    """

    def buildPhotonOp(self, config, base, _logger):
        req, opt, single, _takes_rng = get_cls_params(LsstOptics)
        kwargs, _safe = GetAllParams(config, base, req, opt, single)

        shift_optics = kwargs.get("shift_optics")
        if shift_optics is None:
            shift_optics = base.get("shift_optics", None)
        return LsstOptics(
            telescope=base["_telescope"],
            boresight=kwargs["boresight"],
            sky_pos=base["sky_pos"],
            image_pos=base["image_pos"],
            icrf_to_field=base["_icrf_to_field"],
            det_name=base["det_name"],
            shift_optics=shift_optics,
        )


RegisterPhotonOpType("lsst_optics", LsstOpticsFactory())
