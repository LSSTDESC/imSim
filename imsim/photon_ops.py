"""PhotonOp classes for imsim"""

from functools import lru_cache
import numpy as np

import galsim
import batoid
from galsim import PhotonArray, PhotonOp, GaussianDeviate
from galsim.config import RegisterPhotonOpType, PhotonOpBuilder, GetAllParams
from galsim.celestial import CelestialCoord
from galsim.config.util import get_cls_params
from coord import Angle
from lsst.obs.lsst.translators.lsst import SIMONYI_LOCATION as RUBIN_LOC
from .camera import get_camera
from .utils import focal_to_pixel
from .diffraction import (
    RUBIN_SPIDER_GEOMETRY,
    apply_diffraction_delta,
    apply_diffraction_delta_field_rot,
    prepare_field_rotation_matrix,
)


class RubinOptics(PhotonOp):
    """A photon operator that performs raytracing through the Rubin optics.

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
        logger=None
    ):
        self.telescope = telescope
        self.detector = camera[det_name]
        self.boresight = boresight
        self.sky_pos = sky_pos
        self.image_pos = image_pos
        self.icrf_to_field = icrf_to_field
        self.logger = logger

    def photon_velocity(self, photon_array, local_wcs, rng) -> np.ndarray:
        """Computes the velocity of the photons directly."""

        return photon_velocity(
            photon_array,
            XyToV(local_wcs, self.icrf_to_field, self.sky_pos),
            self.telescope.inMedium.getN,
        )

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

        v = self.photon_velocity(photon_array, local_wcs, rng)

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
            wavelength=photon_array.wavelength * 1e-9,
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


def photon_velocity(photon_array, xy_to_v: "XyToV", get_n) -> np.ndarray:
    """Computes the velocity of a photon array."""

    assert photon_array.hasAllocatedPupil()
    assert photon_array.hasAllocatedTimes()
    # Convert xy coordinates to a cartesian 3d velocity vector of the photons
    v = xy_to_v(photon_array.x, photon_array.y)

    # Adjust for refractive index of air
    wavelength = photon_array.wavelength * 1e-9
    n = get_n(wavelength)
    v /= n[:, None]
    return v


class RubinDiffractionOptics(RubinOptics):
    """Combination of RubinDiffraction and RubinOptics.

    This is for performance reasons, to prevent performing and reversing a transform.

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
    rubin_diffraction : Instance of a RubinDiffraction photon operator
    """

    _req_params = {
        "boresight": CelestialCoord,
        "camera": str,
    }
    _opt_params = {"latitude": Angle, "disable_field_rotation": bool}

    def __init__(
        self,
        telescope,
        boresight,
        sky_pos,
        image_pos,
        icrf_to_field,
        det_name,
        camera,
        rubin_diffraction: "RubinDiffraction",
    ):
        super().__init__(
            telescope, boresight, sky_pos, image_pos, icrf_to_field, det_name, camera
        )
        self.rubin_diffraction = rubin_diffraction

    def photon_velocity(self, photon_array, local_wcs, rng) -> np.ndarray:
        """Computes the velocity of the photons after applying diffraction."""

        return self.rubin_diffraction.photon_velocity(
            photon_array, local_wcs=local_wcs, rng=rng
        )


class RubinDiffraction(PhotonOp):
    """Photon operator that applies statistical diffraction by the
    Rubin spider.

    Parameters
    ----------
    telescope : batoid.Optic
        The telescope to trace through.
    latitude : float
        Geographic latitude of telescope (in rad). Needed to calculate the field rotation
        for the spider diffraction.
    altitude, azimuth : float
        alt/az coordinates the telescope is pointing to (in rad).
    sky_pos : galsim.CelestialCoord
    icrf_to_field : galsim.GSFitsWCS
    """

    _opt_params = {"latitude": Angle, "disable_field_rotation": bool}

    def __init__(
        self,
        telescope,
        latitude,
        altitude,
        azimuth,
        sky_pos,
        icrf_to_field,
        disable_field_rotation: bool = False,
    ):
        self.telescope = telescope
        self.sky_pos = sky_pos
        self.icrf_to_field = icrf_to_field
        if disable_field_rotation:
            self.apply_diffraction_delta = lambda pos, v, _t, wavelength, geometry, distribution: apply_diffraction_delta(
                pos, v, wavelength, geometry, distribution
            )
        else:
            field_rot_matrix = prepare_field_rotation_matrix(
                latitude=latitude,
                azimuth=azimuth,
                altitude=altitude,
            )
            self.apply_diffraction_delta = lambda pos, v, t, wavelength, geometry, distribution: apply_diffraction_delta_field_rot(
                pos, v, t, wavelength, field_rot_matrix, geometry, distribution
            )

    def diffraction_rng(self, rng):
        deviate = GaussianDeviate(seed=rng)

        def _rng(phi):
            var = phi**2
            deviate.generate_from_variance(var)
            return var

        return _rng

    def photon_velocity(self, photon_array, local_wcs, rng) -> np.ndarray:
        """Computes the velocity of the photons after applying diffraction.

        This will not modify the photon array and only return the velocity.
        To be used by the combined photon op: diffraction + raytracing.

        Parameters
        ----------
        photon_array:   A `PhotonArray` to apply the operator to.
        local_wcs:      A `LocalWCS` instance defining the local WCS for the current photon
                        bundle in case the operator needs this information.  [default: None]
        rng:            A random number generator to use if needed. [default: None]

                    Returns: ndarray of shape (n, 3), where n = photon_array.pupil_u.size()
        """
        xy_to_v = XyToV(local_wcs, self.icrf_to_field, self.sky_pos)
        v = photon_velocity(
            photon_array,
            xy_to_v,
            self.telescope.inMedium.getN,
        )
        x, y = photon_array.pupil_u, photon_array.pupil_v
        v = self.apply_diffraction_delta(
            np.c_[x, y],
            v,
            photon_array.time,
            wavelength=photon_array.wavelength * 1.0e-9,
            geometry=RUBIN_SPIDER_GEOMETRY,
            distribution=self.diffraction_rng(rng),
        )
        return v

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
        v = self.apply_diffraction_delta(
            np.c_[x, y],
            v,
            photon_array.time,
            wavelength,
            geometry=RUBIN_SPIDER_GEOMETRY,
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


@photon_op_type("RubinOptics", input_type="telescope")
def deserialize_rubin_optics(config, base, _logger):
    kwargs = config_kwargs(config, base, RubinOptics)
    telescope = base['det_telescope']

    return RubinOptics(
        telescope=telescope,
        sky_pos=base["sky_pos"],
        image_pos=base["image_pos"],
        icrf_to_field=base["_icrf_to_field"],
        det_name=base["det_name"],
        camera=get_camera_cached(kwargs.pop("camera")),
        logger=_logger,
        **kwargs,
    )


@photon_op_type("RubinDiffractionOptics", input_type="telescope")
def deserialize_rubin_diffraction_optics(config, base, _logger):
    kwargs = config_kwargs(config, base, RubinDiffractionOptics)
    opsim_meta = get_opsim_meta(config, base)
    telescope = base['det_telescope']
    rubin_diffraction = RubinDiffraction(
        telescope=telescope,
        latitude=kwargs.pop("latitude", RUBIN_LOC.lat.rad),
        altitude=np.deg2rad(opsim_meta.get("altitude")),
        azimuth=np.deg2rad(opsim_meta.get("azimuth")),
        sky_pos=base["sky_pos"],
        icrf_to_field=base["_icrf_to_field"],
        disable_field_rotation=kwargs.pop("disable_field_rotation", False),
    )

    return RubinDiffractionOptics(
        telescope=telescope,
        sky_pos=base["sky_pos"],
        image_pos=base["image_pos"],
        icrf_to_field=base["_icrf_to_field"],
        det_name=base["det_name"],
        camera=get_camera_cached(kwargs.pop("camera")),
        rubin_diffraction=rubin_diffraction,
        **kwargs,
    )


@lru_cache
def get_camera_cached(camera_name: str):
    return get_camera(camera_name)


@photon_op_type("RubinDiffraction", input_type="telescope")
def deserialize_rubin_diffraction(config, base, _logger):
    kwargs = config_kwargs(config, base, RubinDiffraction)
    opsim_meta = get_opsim_meta(config, base)
    telescope = base['det_telescope']

    return RubinDiffraction(
        telescope=telescope,
        altitude=np.deg2rad(opsim_meta.get("altitude")),
        azimuth=np.deg2rad(opsim_meta.get("azimuth")),
        sky_pos=base["sky_pos"],
        icrf_to_field=base["_icrf_to_field"],
        **kwargs,
    )


def get_opsim_meta(config, base):
    return galsim.config.GetInputObj("opsim_meta_dict", config, base, "opsim_meta_dict")


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
