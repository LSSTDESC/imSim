import numpy as np
import galsim
import batoid
from astropy.time import Time
from astropy import units

from imsim import photon_ops, BatoidWCSFactory, get_camera
from imsim.batoid_utils import load_telescope


def test_lsst_optics() -> None:
    """This just makes sure that the PhotonOp runs. It does not check plausibility
    of results."""
    boresight = galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians)
    fiducial_telescope = load_telescope(telescope="LSST", band="r")
    camera = get_camera()

    factory = BatoidWCSFactory(
        boresight,
        rotTelPos=np.pi / 3,
        obstime=Time("J2020") + 0.5 * units.year,
        fiducial_telescope=fiducial_telescope,
        wavelength=620.0,  # nm
        camera=camera,
        temperature=290.0,
        pressure=70.0,
        H2O_pressure=1.1,
    )

    det_name = "R22_S11"
    lsst_optics = photon_ops.LsstOptics(
        telescope=load_telescope(telescope="LSST", band="r"),
        boresight=boresight,
        sky_pos=galsim.CelestialCoord(0.543 * galsim.radians, -0.174 * galsim.radians),
        image_pos=galsim.PositionD(809.6510740536025, 3432.6477953336625),
        icrf_to_field=factory.get_icrf_to_field(camera[det_name]),
        det_name=det_name,
    )
    photon_array = galsim.PhotonArray(
        5,
        x=np.array([-0.04823635, 0.47023422, -8.53736263, 0.8639109, -3.0237201]),
        y=np.array([1.76626949, -0.89284146, 13.51962823, 0.82503544, -0.1011734]),
        wavelength=np.array(
            [577.67626034, 665.6715595, 564.75533946, 598.74363606, 571.04519139]
        ),
        flux=np.ones(5),
        pupil_u=np.array(
            [-3.60035156, -2.25328125, -2.31042969, 2.56351562, -0.46535156]
        ),
        pupil_v=np.array(
            [1.00417969, 2.73496094, -2.92273437, -1.16746094, 3.09417969]
        ),
    )
    local_wcs = galsim.AffineTransform(
        0.168,
        0.108,
        -0.108,
        0.168,
        origin=galsim.PositionD(x=-0.349, y=-0.352),
        world_origin=galsim.PositionD(x=0.0, y=0.0),
    )
    lsst_optics.applyTo(photon_array, local_wcs=local_wcs)
