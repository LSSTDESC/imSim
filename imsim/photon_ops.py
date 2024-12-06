"""PhotonOp classes for imsim"""

from functools import lru_cache
import numpy as np

import batoid
from galsim import Bandpass, PhotonArray, PhotonOp, GaussianDeviate
from galsim.config import RegisterPhotonOpType, PhotonOpBuilder, GetAllParams
from galsim.config import BuildBandpass, GalSimConfigError
from galsim.celestial import CelestialCoord
from galsim.config.util import get_cls_params
from coord import Angle
from lsst.obs.lsst.translators.lsst import SIMONYI_LOCATION as RUBIN_LOC
from .camera import get_camera
from .utils import focal_to_pixel, jac_focal_to_pixel
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
    img_wcs : galsim.BaseWCS
    stamp_center : galsim.PositionD
    icrf_to_field : galsim.GSFitsWCS
    det_name : str
    camera : lsst.afw.cameraGeom.Camera
    shift_photons : Optional, whether to shift photons at start. [default: False]
    """

    _req_params = {
        "boresight": CelestialCoord,
        "camera": str,
        "det_name": str,
    }

    _opt_params = {
        "shift_photons": bool,
    }

    def __init__(
        self,
        telescope,
        boresight,
        img_wcs,
        stamp_center,
        icrf_to_field,
        det_name,
        camera,
        shift_photons=False
    ):
        self.telescope = telescope
        self.detector = camera[det_name]
        self.boresight = boresight
        self.img_wcs = img_wcs
        self.stamp_center = stamp_center
        self.icrf_to_field = icrf_to_field
        self.shift_photons = shift_photons

    def photon_velocity(self, photon_array, rng) -> np.ndarray:
        """Computes the velocity of the photons directly."""

        return photon_velocity(
            photon_array,
            XyToV(self.icrf_to_field, self.img_wcs),
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

        # If a stamp_center has been provided apply it as a shift to the photons. This will
        # probably only ever be when using LSST_Image, as LSST_PhotonPoolingImage
        # positions the photons before they are pooled to be processed together.
        if self.shift_photons and self.stamp_center is not None:
            photon_array.x += self.stamp_center.x
            photon_array.y += self.stamp_center.y

        v = self.photon_velocity(photon_array, rng)

        x = photon_array.pupil_u.copy()
        y = photon_array.pupil_v.copy()
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
        if self.stamp_center is not None:
            photon_array.x -= self.stamp_center.x
            photon_array.y -= self.stamp_center.y

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
    img_wcs : galsim.BaseWCS
    stamp_center : galsim.PositionD
    icrf_to_field : galsim.GSFitsWCS
    det_name : str
    camera : lsst.afw.cameraGeom.Camera
    rubin_diffraction : Instance of a RubinDiffraction photon operator
    shift_photons : Optional, whether to shift photons at start. [default: False]
    """

    _req_params = {
        "boresight": CelestialCoord,
        "camera": str,
        "det_name": str,
        "altitude": Angle,
        "azimuth": Angle,
    }
    _opt_params = {
        "altitude": Angle,
        "azimuth": Angle,
        "latitude": Angle,
        "disable_field_rotation": bool,
        "shift_photons": bool,
    }

    def __init__(
        self,
        telescope,
        boresight,
        stamp_center,
        det_name,
        camera,
        rubin_diffraction: "RubinDiffraction",
        shift_photons=False,
    ):
        super().__init__(
            telescope, boresight, rubin_diffraction.img_wcs, stamp_center, rubin_diffraction.icrf_to_field, det_name, camera, shift_photons
        )
        self.rubin_diffraction = rubin_diffraction

    def photon_velocity(self, photon_array, rng) -> np.ndarray:
        """Computes the velocity of the photons after applying diffraction."""

        return self.rubin_diffraction.photon_velocity(
            photon_array, rng=rng
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
    img_wcs : galsim.BaseWCS
    icrf_to_field : galsim.GSFitsWCS
    """

    _req_params = {"altitude": Angle, "azimuth": Angle, "latitude": Angle}
    _opt_params = {"disable_field_rotation": bool}

    def __init__(
        self,
        telescope,
        latitude,
        altitude,
        azimuth,
        img_wcs,
        icrf_to_field,
        disable_field_rotation: bool = False,
    ):
        self.telescope = telescope
        self.img_wcs = img_wcs
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

    def photon_velocity(self, photon_array, rng) -> np.ndarray:
        """Computes the velocity of the photons after applying diffraction.

        This will not modify the photon array and only return the velocity.
        To be used by the combined photon op: diffraction + raytracing.

        Parameters
        ----------
        photon_array:   A `PhotonArray` to apply the operator to.
        rng:            A random number generator to use if needed. [default: None]

        Returns: ndarray of shape (n, 3), where n = photon_array.pupil_u.size()
        """
        xy_to_v = XyToV(self.icrf_to_field, self.img_wcs)
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
        xy_to_v = XyToV(self.icrf_to_field, self.img_wcs)
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


def config_kwargs(config, base, cls, base_args=()):
    """Given config and base, extract parameters."""
    req, opt, single, _takes_rng = get_cls_params(cls)
    kwargs, _safe = GetAllParams(config, base, req, opt, single)
    kwargs.update({key: base[key] for key in base_args})
    return kwargs


_rubin_optics_base_args = ("stamp_center",)


@photon_op_type("RubinOptics", input_type="telescope")
def deserialize_rubin_optics(config, base, _logger):
    kwargs = config_kwargs(config, base, RubinOptics, base_args=_rubin_optics_base_args)
    telescope = base["det_telescope"]

    return RubinOptics(
        telescope=telescope,
        icrf_to_field=base["_icrf_to_field"],
        img_wcs=base["current_image"].wcs,
        camera=get_camera_cached(kwargs.pop("camera")),
        **kwargs,
    )


@photon_op_type("RubinDiffractionOptics", input_type="telescope")
def deserialize_rubin_diffraction_optics(config, base, _logger):
    kwargs = config_kwargs(config, base, RubinDiffractionOptics, _rubin_optics_base_args)
    telescope = base["det_telescope"]
    rubin_diffraction = RubinDiffraction(
        telescope=telescope,
        latitude=kwargs.pop("latitude", RUBIN_LOC.lat.rad),
        altitude=kwargs.pop("altitude"),
        azimuth=kwargs.pop("azimuth"),
        img_wcs=base["current_image"].wcs,
        icrf_to_field=base["_icrf_to_field"],
        disable_field_rotation=kwargs.pop("disable_field_rotation", False),
    )

    return RubinDiffractionOptics(
        telescope=telescope,
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
    telescope = base["det_telescope"]

    return RubinDiffraction(
        telescope=telescope,
        icrf_to_field=base["_icrf_to_field"],
        img_wcs=base["current_image"].wcs,
        **kwargs,
    )


class XyToV:
    """Maps image coordinates (x,y) to a 3d cartesian direction
    vector of the photons before entering the telescope.

    The transform is composed of the chain
    (x,y) -> (ra,dec) -> (thx, thy) -> v.

    The transform takes 2 vectors of shape (n,) and returns
    a column major vector of shape (n,3).
    """

    def __init__(self, icrf_to_field, img_wcs):
        self.icrf_to_field = icrf_to_field
        self.img_wcs = img_wcs

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # x/y to ra/dec
        ra, dec = self.img_wcs.xyToradec(x, y, units="rad")
        # ICRF to field
        thx, thy = self.icrf_to_field.radecToxy(ra, dec, units="rad")

        return np.array(batoid.utils.gnomonicToDirCos(thx, thy)).T

    def inverse(self, v_photon: np.ndarray) -> tuple:
        thx, thy = batoid.utils.dirCosToGnomonic(
            v_photon[:, 0], v_photon[:, 1], v_photon[:, 2]
        )
        # field to ICRF
        ra, dec = self.icrf_to_field.xyToradec(thx, thy, units="rad")
        return self.img_wcs.radecToxy(ra, dec, units="rad")


def ray_vector_to_photon_array(
    ray_vector: batoid.RayVector, detector, out: PhotonArray
) -> PhotonArray:
    """Converts a batoid.RayVector to a galsim.PhotonArray

    Stores into an already existing galsim.PhotonArray out"""

    w = ~ray_vector.vignetted
    assert all(np.abs(ray_vector.z[w]) < 1.0e-15)
    out.x, out.y = focal_to_pixel(ray_vector.y * 1e3, ray_vector.x * 1e3, detector)
    # Need the jacobian of (x, y) |-> focal_to_pixel(M (x,y)), where M is given below
    M = np.array([[0.0, 1.0e3], [1.0e3, 0.0]])
    jac = M @ jac_focal_to_pixel(0.0, 0.0, detector)  # Jac is constant
    jac /= np.sqrt(np.abs(np.linalg.det(jac)))
    out.dxdz, out.dydz = jac @ np.array([ray_vector.vx, ray_vector.vy]) / ray_vector.vz
    out.dxdz, out.dydz = (out.dxdz.ravel(), out.dydz.ravel())
    out.flux[ray_vector.vignetted] = 0.0
    return out


class BandpassRatio(PhotonOp):
    """Photon operator that reweights photon fluxes to effect
    a specified bandpass from photons initially sampled from
    a different bandpass.
    """
    def __init__(
        self,
        target_bandpass: Bandpass,
        initial_bandpass: Bandpass
    ):
        self.target = target_bandpass
        self.initial = initial_bandpass
        self.ratio = self.target / self.initial

    def applyTo(self, photon_array, local_wcs=None, rng=None):
        photon_array.flux *= self.ratio(photon_array.wavelength)


class BandpassRatioBuilder(PhotonOpBuilder):
    def buildPhotonOp(self, config, base, logger):
        if 'target_bandpass' not in config:
            raise GalSimConfigError("target_bandpass is required for BandpassRatio")
        if 'initial_bandpass' not in config:
            raise GalSimConfigError("initial_bandpass is required for BandpassRatio")
        kwargs = {}
        kwargs['target_bandpass'] = BuildBandpass(config, 'target_bandpass', base, logger)[0]
        kwargs['initial_bandpass'] = BuildBandpass(config, 'initial_bandpass', base, logger)[0]
        return BandpassRatio(**kwargs)


RegisterPhotonOpType('BandpassRatio', BandpassRatioBuilder())
