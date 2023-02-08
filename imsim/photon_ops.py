"""PhotonOp classes for imsim"""

from functools import lru_cache
import numpy as np

import galsim
import batoid
from galsim import PhotonArray, PhotonOp, GaussianDeviate
from galsim.config import RegisterPhotonOpType, PhotonOpBuilder, GetAllParams

from galsim.celestial import CelestialCoord
from galsim.config.util import get_cls_params
from .camera import get_camera
from .utils import focal_to_pixel
from .diffraction import (
    LSST_SPIDER_GEOMETRY,
    apply_diffraction_delta,
)


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
    camera : lsst.afw.cameraGeom.Camera
    """

    _req_params = {
        "boresight": CelestialCoord,
        "camera": str,
    }

    def __init__(
        self,
        telescope,
        boresight,
        sky_pos,
        image_pos,
        icrf_to_field,
        det_name,
        camera,
    ):
        self.telescope = telescope
        self.detector = camera[det_name]
        self.boresight = boresight
        self.sky_pos = sky_pos
        self.image_pos = image_pos
        self.icrf_to_field = icrf_to_field

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the photon operator to a PhotonArray.

        Note that we assume that photon entrance pupil positions and arrival
        times have already been sampled here.  This might be accomplished by
        including an atmospheric PSF component or by explicitly specifying
        TimeSampler or PupilSampler photon operations.

        Parameters
        ----------
        photon_array:   A `PhotonArray` to apply the operator to.
        local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                        bundle in case the operator needs this information.  [default: None]
        rng:            A random number generator to use if needed. [default: None]
        """

        assert photon_array.hasAllocatedPupil()
        assert photon_array.hasAllocatedTimes()
        # Convert xy coordinates to a cartesian 3d velocity vector of the photons
        v = XyToV(local_wcs, self.icrf_to_field, self.sky_pos)(
            photon_array.x, photon_array.y
        )

        # Adjust for refractive index of air
        wavelength = photon_array.wavelength * 1e-9
        n = self.telescope.inMedium.getN(wavelength)
        v /= n[:, None]

        x, y = photon_array.pupil_u, photon_array.pupil_v
        z = self.telescope.stopSurface.surface.sag(x, y)
        ray_vec = batoid.RayVector._directInit(
            x,
            y,
            z,
            v[:, 0],
            v[:, 1],
            v[:, 2],
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
        return f"imsim.{type(self).__name__}()"

    def __repr__(self):
        return str(self)


class LsstDiffraction(PhotonOp):
    """Photon operator that applies statistical diffraction by the
    LSST spider.

    Parameters
    ----------
    telescope : batoid.Optic
        The telescope to trace through.
    latitude : float
        Geographic latitude of telescope. Needed to calculate the field rotation
        for the spider diffraction.
    altitude, azimuth : float
        alt/az coordinates the telescope is pointing to (in degree).
    sky_pos : galsim.CelestialCoord
    icrf_to_field : galsim.GSFitsWCS
    """

    _req_params = {
        "latitude": float,
    }

    def __init__(
        self,
        telescope,
        latitude,
        altitude,
        azimuth,
        sky_pos,
        icrf_to_field,
    ):
        self.telescope = telescope
        self.latitude = latitude
        self.altitude = altitude
        self.azimuth = azimuth
        self.sky_pos = sky_pos
        self.icrf_to_field = icrf_to_field

    def diffraction_rng(self, rng):
        deviate = GaussianDeviate(seed=rng)

        def _rng(phi):
            var = phi**2
            deviate.generate_from_variance(var)
            return var

        return _rng

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        """Apply the photon operator to a PhotonArray.

        Note that we assume that photon entrance pupil positions and arrival
        times have already been sampled here.  This might be accomplished by
        including an atmospheric PSF component or by explicitly specifying
        TimeSampler or PupilSampler photon operations.

        Parameters
        ----------
        photon_array:   A `PhotonArray` to apply the operator to.
        local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                        bundle in case the operator needs this information.  [default: None]
        rng:            A random number generator to use if needed. [default: None]
        """

        assert photon_array.hasAllocatedPupil()
        assert photon_array.hasAllocatedTimes()
        xy_to_v = XyToV(local_wcs, self.icrf_to_field, self.sky_pos)
        # Convert xy coordinates to a cartesian 3d velocity vector of the photons
        v = xy_to_v(photon_array.x, photon_array.y)

        # Adjust for refractive index of air
        wavelength = photon_array.wavelength * 1e-9
        n = self.telescope.inMedium.getN(wavelength)
        v /= n[:, None]

        x, y = photon_array.pupil_u, photon_array.pupil_v
        v = apply_diffraction_delta(
            np.c_[x, y],
            v,
            photon_array.time,
            wavelength,
            lat=self.latitude,
            az=self.azimuth,
            alt=self.altitude,
            geometry=LSST_SPIDER_GEOMETRY,
            distribution=self.diffraction_rng(rng),
        )
        photon_array.x, photon_array.y = xy_to_v.inverse(v)

    def __str__(self):
        return f"imsim.{type(self).__name__}()"

    def __repr__(self):
        return str(self)


def photon_op_type(identifier: str, input_type=None):
    """Decorator which calls RegisterPhotonOpType on a PhotonOp factory,
    defined by a function deserializing a PhotonOp from a dict.

    The decorated function should have the following signature
    > def deserializer(config, base, logger): ...

    config, base and logger will receive the arguments of a call to
    galsim.config.PhotonOpBuilder.buildPhotonOp.
    """

    def decorator(deserializer):
        class Factory(PhotonOpBuilder):
            """Build a PhotonOp generated by a deserializer.

            Returns:
                 the constructed PhotonOp object.
            """

            def buildPhotonOp(self, config, base, logger):
                return deserializer(config, base, logger)

        RegisterPhotonOpType(identifier, Factory(), input_type=input_type)
        return deserializer

    return decorator


def config_kwargs(config, base, cls):
    """Given config and base, extract parameters."""
    req, opt, single, _takes_rng = get_cls_params(cls)
    kwargs, _safe = GetAllParams(config, base, req, opt, single)
    return kwargs


@photon_op_type("lsst_optics", input_type="telescope")
def deserialize_lsst_optics(config, base, _logger):
    kwargs = config_kwargs(config, base, LsstOptics)

    telescope = galsim.config.GetInputObj("telescope", config, base, "telescope")['det']

    return LsstOptics(
        telescope=telescope,
        sky_pos=base["sky_pos"],
        image_pos=base["image_pos"],
        icrf_to_field=base["_icrf_to_field"],
        det_name=base["det_name"],
        camera=get_camera_cached(kwargs.pop("camera")),
        **kwargs,
    )


@lru_cache
def get_camera_cached(camera_name: str):
    return get_camera(camera_name)


@photon_op_type("lsst_diffraction", input_type="telescope")
def deserialize_lsst_diffraction(config, base, _logger):
    kwargs = config_kwargs(config, base, LsstDiffraction)

    opsim_meta = galsim.config.GetInputObj(
        "opsim_meta_dict", config, base, "opsim_meta_dict"
    )

    telescope = galsim.config.GetInputObj("telescope", config, base, "telescope")['det']

    return LsstDiffraction(
        telescope=telescope,
        altitude=opsim_meta.get("altitude"),
        azimuth=opsim_meta.get("azimuth"),
        sky_pos=base["sky_pos"],
        icrf_to_field=base["_icrf_to_field"],
        **kwargs,
    )


class XyToV:
    """Maps image coordinates (x,y) to a 3d cartesian direction
    vector of the photons before entering the telescope.

    The transform is composed of the chain
    (x,y) -> (u,v) -> (ra,dec) -> (thx, thy) -> v.

    The transform takes 2 vectors of shape (n,) and returns
    a column major vector of shape (n,3).
    """

    def __init__(self, local_wcs, icrf_to_field, sky_pos):
        self.local_wcs = local_wcs
        self.icrf_to_field = icrf_to_field
        self.sky_pos = sky_pos

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # x/y to u/v
        u = self.local_wcs._u(x, y)
        v = self.local_wcs._v(x, y)
        # u/v to ICRF
        u_rad = np.deg2rad(u / 3600)
        v_rad = np.deg2rad(v / 3600)
        ra, dec = self.sky_pos.deproject_rad(u_rad, v_rad)
        # ICRF to field
        thx, thy = self.icrf_to_field.radecToxy(ra, dec, units="rad")

        return np.array(batoid.utils.gnomonicToDirCos(thx, thy)).T

    def inverse(self, v_photon: np.ndarray) -> tuple:
        thx, thy = batoid.utils.dirCosToGnomonic(
            v_photon[:, 0], v_photon[:, 1], v_photon[:, 2]
        )
        # field to ICRF
        ra, dec = self.icrf_to_field.xyToradec(thx, thy, units="rad")
        # ICRF to u/v
        u_rad, v_rad = self.sky_pos.project_rad(ra, dec)
        u = 3600 * np.rad2deg(u_rad)
        v = 3600 * np.rad2deg(v_rad)
        # u/v to x/y
        x = self.local_wcs._x(u, v)
        y = self.local_wcs._y(u, v)

        return (x, y)


def ray_vector_to_photon_array(
    ray_vector: batoid.RayVector, detector, out: PhotonArray
) -> PhotonArray:
    """Converts a batoid.RayVector to a galsim.PhotonArray

    Stores into an already existing galsim.PhotonArray out"""

    w = ~ray_vector.vignetted
    assert all(np.abs(ray_vector.z[w]) < 1.0e-15)
    out.x, out.y = focal_to_pixel(ray_vector.y * 1e3, ray_vector.x * 1e3, detector)

    out.dxdz = ray_vector.vx / ray_vector.vz
    out.dydz = ray_vector.vy / ray_vector.vz
    out.flux[ray_vector.vignetted] = 0.0
    return out
